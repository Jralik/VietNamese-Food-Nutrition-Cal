"""Experiment B — Mask ablation (pan-bleed filter) on the frozen baseline.

Ablation protocol (approved): everything frozen except the Banh mi mask.
  same images / YOLO detections / bbox / confidence / depth map /
  support plane / ArUco scale / DENSITY_DB (0.45) / volume integration logic
  ONLY change: mask_original → mask_filtered.

Filter variants evaluated (one pipeline.analyze per image; per-item filtered
volumes recomputed with the CAPTURED estimate kwargs — same depth/scale/K):
  original    : SAM2 mask as-is (baseline)
  otsu        : Otsu threshold on masked grays (iteration-1, uncontrolled)
  fixed60     : keep gray > 60 (pan is near-black)
  fixed60cc   : fixed60 + largest connected component
  fixed80cc   : keep gray > 80 + largest CC
  cconly      : largest connected component of the original mask

Per item recorded: mask area, volume_cm3, mass_g for EVERY variant.
Per image + aggregate: Banh mi mass totals per variant vs GT → MAPE.

Writes scratch/experiment_b_records.jsonl + scratch/experiment_b_mask.md.
"""

import json
import os
import statistics

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from eval_full_pipeline import parse_gt, VAL_FILE, CONF
from utils import predict_with_dual_preresize
from food_volume_pipeline import FoodVolumePipeline

AUDITED = "Banh mi (Vietnamese baguette sandwich)"
OUT_JSONL = os.path.join("scratch", "experiment_b_records.jsonl")
OUT_MD = os.path.join("scratch", "experiment_b_mask.md")
VARIANTS = ("original", "otsu", "fixed60", "fixed60cc", "fixed80cc", "cconly")


def otsu_filter(mask, gray):
    vals = gray[mask.astype(bool)]
    if vals.size < 50:
        return mask.copy()
    thr, _ = cv2.threshold(vals.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    keep = mask.astype(bool) & (gray.astype(np.int32) > thr)
    out = np.zeros_like(mask)
    out[keep] = mask[keep]
    return out


def largest_cc(mask):
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.copy()
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = mask[labels == largest]
    return out


def fixed_filter(mask, gray, thr, use_cc):
    keep = mask.astype(bool) & (gray.astype(np.int32) > thr)
    out = np.zeros_like(mask)
    out[keep] = mask[keep]
    if use_cc:
        out = largest_cc(out)
    return out


def apply_variant(variant, mask, gray):
    if variant == "original":
        return mask.copy()
    if variant == "otsu":
        return otsu_filter(mask, gray)
    if variant == "fixed60":
        return fixed_filter(mask, gray, 60, use_cc=False)
    if variant == "fixed60cc":
        return fixed_filter(mask, gray, 60, use_cc=True)
    if variant == "fixed80cc":
        return fixed_filter(mask, gray, 80, use_cc=True)
    if variant == "cconly":
        return largest_cc(mask)
    raise ValueError(variant)


def main():
    entries = {os.path.basename(p): g for p, g in parse_gt(VAL_FILE)
               if "BanhMi" in p}
    model = YOLO("./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt")
    model.overrides["device"] = "cpu"

    import pipeline_config as _pc
    _pc.SEGMENTATION_BACKEND = "sam2"
    pipeline = FoodVolumePipeline()
    orig_estimate = pipeline.volume_estimator.estimate
    captured = []

    def capturing(**kwargs):
        captured.append(dict(kwargs))
        return orig_estimate(**kwargs)

    pipeline.volume_estimator.estimate = capturing

    bm_cid = next(i for i in range(len(class_names))
                  if class_names[i]["name"] == AUDITED)

    records = []
    for fname, gt in entries.items():
        gt_entry = gt.get(bm_cid)
        img_path = next(p for p, _ in parse_gt(VAL_FILE)
                        if os.path.basename(p) == fname)
        rec = {"image": fname,
               "gt_total_g": (gt_entry or {}).get("grams_total"),
               "items": [], "error": None}
        print(f"\n=== {fname} ===", flush=True)
        try:
            upright = ImageOps.exif_transpose(Image.open(img_path))
            w_up, h_up = upright.size
            src = upright.copy()
            src.thumbnail((2000, 2000))
            bbox_scale = (src.width / w_up, src.height / h_up)
            res = predict_with_dual_preresize(model, upright, CONF)[0]
            dets = volume_integration_extract(res)
            if not dets:
                rec["error"] = ("volume skipped: no detections above "
                                "MIN_DETECTION_CONFIDENCE=0.40")
                records.append(rec)
                continue

            img_array = np.asarray(src)
            seg_image = cv2.resize(img_array, (640, 640))
            w, h = src.size
            seg_dets = [{
                "bbox": (int(d["bbox"][0] * 640.0 / w),
                         int(d["bbox"][1] * 640.0 / h),
                         int(d["bbox"][2] * 640.0 / w),
                         int(d["bbox"][3] * 640.0 / h)),
                "class_name": d["class_name"],
                "confidence": d["confidence"],
            } for d in dets]

            captured.clear()
            vol = pipeline.analyze(
                image_rgb=img_array,
                yolo_detections=dets,
                seg_image_rgb=seg_image,
                seg_yolo_detections=seg_dets,
            )

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            n_bm = 0
            for i, e in enumerate(vol.estimations):
                if e.class_name != AUDITED or i >= len(captured):
                    continue
                n_bm += 1
                cap = captured[i]
                mask_o = np.asarray(e.mask)
                item = {"instance": i,
                        "conf": round(float(e.confidence), 3),
                        "gt_share": None,
                        "variants": {}}
                for variant in VARIANTS:
                    fm = apply_variant(variant, mask_o, gray)
                    area = int(np.asarray(fm).sum())
                    if variant == "original":
                        est_v = vol.estimations[i]  # original run result
                    else:
                        est_v = orig_estimate(
                            depth_map_mm=cap["depth_map_mm"],
                            food_mask=fm,
                            class_name=e.class_name,
                            mm_per_pixel=cap["mm_per_pixel"],
                            scale_source=cap["scale_source"],
                            scale_confidence=cap["scale_confidence"],
                            bbox=e.bbox,
                            K=cap["K"],
                            homography=cap["homography"],
                            depth_anchored=cap["depth_anchored"],
                        )
                    item["variants"][variant] = {
                        "area": area,
                        "volume_cm3": round(float(est_v.volume_cm3), 1),
                        "mass_g": round(float(est_v.mass_g), 1),
                    }
                rec["items"].append(item)
                vs = item["variants"]
                print(f"  item {i}: " + " | ".join(
                    f"{v}: {vs[v]['mass_g']}g" for v in VARIANTS), flush=True)
            rec["n_bm_items"] = n_bm
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print("  ERROR:", rec["error"])
        records.append(rec)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print(f"\nrecords -> {OUT_JSONL}")


def volume_integration_extract(res):
    import volume_integration
    return volume_integration.extract_yolo_detections([res], class_names)


if __name__ == "__main__":
    main()
