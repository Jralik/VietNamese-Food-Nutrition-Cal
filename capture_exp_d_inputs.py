"""Capture frozen Experiment D inputs: per-image anchored depth + per-item masks.

Reproduces the EXACT app collection flow (predict_with_dual_preresize @0.30 ->
extract_yolo_detections (0.40 floor) -> FoodVolumePipeline.analyze with the same
seg_image/seg_dets construction as the volume worker) while CAPTURING the exact
kwargs passed to volume_estimator.estimate.

Isolation assertion: detections/est/mass/kcal must be EXACT-equal to
full_eval_records.jsonl (numeric compare; any float serialization difference is
reported with its tolerance, never eyeballed).

Outputs (scratch/exp_d_captures/):
  {stem}_d.npz      : depth_map_mm (anchored, float16), gray, food_mask/bbox per item
  {stem}_params.json: mm_per_pixel, scale_source/confidence, depth_anchored, class_name
  manifest.jsonl    : standard est/detections records (isolation evidence)
"""

import json
import os

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from eval_full_pipeline import parse_gt, VAL_FILE, CONF
from utils import predict_with_dual_preresize
from food_volume_pipeline import FoodVolumePipeline

OUT_DIR = os.path.join("scratch", "exp_d_captures")
MANIFEST = os.path.join(OUT_DIR, "manifest.jsonl")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = parse_gt(VAL_FILE)
    model = YOLO("./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt")
    model.overrides["device"] = "cpu"

    import pipeline_config as _pc
    _pc.SEGMENTATION_BACKEND = "sam2"
    pipeline = FoodVolumePipeline()
    captured = []
    orig_estimate = pipeline.volume_estimator.estimate

    def capturing(**kwargs):
        captured.append(dict(kwargs))
        return orig_estimate(**kwargs)

    pipeline.volume_estimator.estimate = capturing

    manifest = []
    for idx, (img_path, gt) in enumerate(entries):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0].replace(" ", "_")
        rec = {"image": fname, "gt": {c: dict(v) for c, v in gt.items()},
               "est": [], "detections": [], "error": None}
        print(f"[{idx + 1}/{len(entries)}] {fname}", flush=True)
        try:
            upright = ImageOps.exif_transpose(Image.open(img_path))
            w_up, h_up = upright.size
            src = upright.copy()
            src.thumbnail((2000, 2000))
            bbox_scale = (src.width / w_up, src.height / h_up)
            res = predict_with_dual_preresize(model, upright, CONF)[0]
            import volume_integration
            dets = volume_integration.extract_yolo_detections(
                [res], class_names, bbox_scale=bbox_scale)
            rec["detections"] = [
                {"class_name": d["class_name"], "conf": round(d["confidence"], 3)}
                for d in dets]

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

            rec["est"] = [{"class_name": e.class_name,
                           "mass_g": round(float(e.mass_g), 1),
                           "volume_cm3": round(float(e.volume_cm3), 1),
                           "kcal": round(float(e.nutrition.get("Calories", 0.0)), 1)}
                          for e in vol.estimations]

            if captured:
                cap0 = captured[0]
                save = {"depth_map_mm": cap0["depth_map_mm"].astype(np.float16),
                        "gray": cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)}
                for i, cap in enumerate(captured):
                    save[f"mask_{i}"] = cap["food_mask"].astype(np.uint8)
                    save[f"bbox_{i}"] = np.array(cap["bbox"], dtype=np.float32)
                np.savez_compressed(os.path.join(OUT_DIR, f"{stem}_d.npz"), **save)
                params = {"image": fname, "items": []}
                for i, cap in enumerate(captured):
                    params["items"].append({
                        "instance": i,
                        "class_name": cap["class_name"],
                        "mm_per_pixel": float(cap["mm_per_pixel"]),
                        "scale_source": cap["scale_source"],
                        "scale_confidence": cap["scale_confidence"],
                        "depth_anchored": bool(cap["depth_anchored"]),
                        "bbox": [float(x) for x in cap["bbox"]],
                    })
                with open(os.path.join(OUT_DIR, f"{stem}_params.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False, indent=1)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print("  ERROR:", rec["error"])
        manifest.append(rec)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
