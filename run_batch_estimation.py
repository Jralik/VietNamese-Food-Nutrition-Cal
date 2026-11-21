"""
Batch volume/nutrition estimation over a set of images.

Two-stage to avoid native crashes from onnxruntime + torch CUDA sharing
one process (the same reason the WebUI runs the pipeline in volume_worker):

  stage yolo     : onnxruntime-only process, dumps per-image detections JSON
                   (backend-independent — detections are shared by all backends)
  stage pipeline : torch-only process, runs FoodVolumePipeline per image,
                   saves one result JSON per image as it goes (crash-safe,
                   re-runs skip already-done images)
  stage csv      : aggregates every results/<backend>/ tree into one CSV +
                   batch_summary.json (per-folder stats incl. semantic
                   recovery rate for YOLO-miss images)
  stage split    : writes deterministic dev/eval image lists for threshold
                   tuning (tune on dev ONLY, freeze, then run eval)

Research fail-fast: in batch mode a FoodSAM fallback (bridge failure ->
GrabCut masks) or a backend mismatch raises immediately and the result is
NOT written — the eval set can never silently contain degraded results.
(Interactive WebUI keeps the graceful fallback + warning instead.)

Usage:
    python run_batch_estimation.py --stage yolo [--img-dir DIR] [--limit N]
    python run_batch_estimation.py --stage pipeline --seg-backend sam2 --img-dir DIR
    python run_batch_estimation.py --stage pipeline --seg-backend foodsam --img-dir DIR
    python run_batch_estimation.py --stage csv
    python run_batch_estimation.py --stage split
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import cv2
import numpy as np

TEST_IMAGE_DIR = r"C:\Users\huynh\OneDrive\Pictures\test-image"
YOLO_MODEL_PATH = "./model/yolov26/best.onnx"
WORK_DIR = "./test_output/batch"
CONF_DEFAULT = 0.35
DEV_SIZE = 20  # images reserved for threshold tuning (rest -> eval)


def log(msg: str):
    print(msg, flush=True)


def collect_images(img_dir: str = None, images_list: str = None, limit: int = None):
    base = img_dir or TEST_IMAGE_DIR
    exts = (".jpg", ".jpeg", ".png")
    if images_list:
        with open(images_list, encoding="utf-8") as f:
            imgs = [ln.strip() for ln in f if ln.strip()]
        imgs = [p if os.path.isabs(p) else os.path.join(TEST_IMAGE_DIR, p) for p in imgs]
        imgs = [p for p in imgs if os.path.isfile(p)]
    else:
        imgs = sorted(
            os.path.join(base, f)
            for f in os.listdir(base)
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(base, f))
        )
    if limit:
        imgs = imgs[:limit]
    return imgs


def rel_for(image_path: str) -> str:
    """Path relative to TEST_IMAGE_DIR — stable keys for dets/results no
    matter which --img-dir subset is being processed."""
    return os.path.relpath(image_path, TEST_IMAGE_DIR)


def det_file_for(image_path: str) -> str:
    return os.path.join(WORK_DIR, "dets", rel_for(image_path) + ".json")


def backend_tag(args) -> str:
    return args.seg_backend or "default"


def res_file_for(image_path: str, tag: str) -> str:
    return os.path.join(WORK_DIR, "results", tag, rel_for(image_path) + ".json")


def stage_yolo(conf: float, args):
    import onnxruntime as ort
    from class_names import class_names as _cn

    os.makedirs(os.path.join(WORK_DIR, "dets"), exist_ok=True)

    session = ort.InferenceSession(
        YOLO_MODEL_PATH, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    for image_path in collect_images(args.img_dir, args.images_list, args.limit):
        out_path = det_file_for(image_path)
        if os.path.exists(out_path):
            log(f"[skip] {image_path}")
            continue
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            log(f"!! unreadable: {image_path}")
            continue
        h_orig, w_orig = image_bgr.shape[:2]
        img_resized = cv2.resize(image_bgr, (640, 640))
        input_tensor = (
            img_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        input_tensor = np.expand_dims(input_tensor, 0)

        outputs = session.run(output_names, {input_name: input_tensor})
        preds = outputs[0][0]

        scale_x, scale_y = w_orig / 640.0, h_orig / 640.0
        detections = []
        for pred in preds:
            score = float(pred[4])
            if score < conf:
                continue
            class_id = int(pred[5])
            x1 = max(0, min(w_orig, int(pred[0] * scale_x)))
            y1 = max(0, min(h_orig, int(pred[1] * scale_y)))
            x2 = max(0, min(w_orig, int(pred[2] * scale_x)))
            y2 = max(0, min(h_orig, int(pred[3] * scale_y)))
            name = (
                _cn[class_id]["name"]
                if 0 <= class_id < len(_cn)
                else f"Class_{class_id}"
            )
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_name": name,
                    "class_id": class_id,
                    "confidence": score,
                }
            )

        out_path_abs = os.path.join(WORK_DIR, "dets", rel_for(image_path) + ".json")
        os.makedirs(os.path.dirname(out_path_abs), exist_ok=True)
        with open(out_path_abs, "w", encoding="utf-8") as f:
            json.dump({"image_path": image_path, "detections": detections}, f)
        log(
            f"[yolo] {rel_for(image_path)}: "
            f"{len(detections)} detections"
        )


def infer_volume_method(cls: str) -> str:
    """Mirror the dispatch logic in volume_nutrition.estimate()."""
    from pipeline_config import CLASS_TO_DISH_TYPE

    dish_type = CLASS_TO_DISH_TYPE.get(cls, "default")
    if dish_type in {"soup_bowl", "large_bowl", "rice_bowl"}:
        return "bowl_container"
    if dish_type == "ingredient_small":
        return "spheroid_model"
    if dish_type == "side_vegetables":
        return "vegetable_mound"
    return "column_integration"


def stage_pipeline(args):
    from food_volume_pipeline import FoodVolumePipeline

    tag = backend_tag(args)
    backend = args.seg_backend
    if backend:
        import pipeline_config as _pc
        _pc.SEGMENTATION_BACKEND = backend

    os.makedirs(os.path.join(WORK_DIR, "results", tag), exist_ok=True)
    pipeline = FoodVolumePipeline()
    seg = pipeline.segmenter
    log(
        f"Backend={backend or 'default(config)'} "
        f"Segmenter={type(seg).__name__} "
        f"available={seg.is_available} "
        f"Depth={'OK' if pipeline.depth_estimator.is_available else 'UNAVAILABLE'}"
    )
    if backend == "foodsam" and type(seg).__name__ != "FoodSAMSegmenter":
        raise RuntimeError(
            f"Requested --seg-backend foodsam but factory produced {type(seg).__name__}")

    n_new = 0
    for image_path in collect_images(args.img_dir, args.images_list, args.limit):
        res_path = res_file_for(image_path, tag)
        rel = rel_for(image_path)
        if os.path.exists(res_path) and not getattr(args, "force", False):
            log(f"[skip] {rel}")
            continue

        det_path = det_file_for(image_path)
        if os.path.exists(det_path):
            with open(det_path, encoding="utf-8") as f:
                payload = json.load(f)
            detections = [
                {**d, "bbox": tuple(d["bbox"])} for d in payload["detections"]
            ]
        else:
            detections = []
        if not detections:
            log(f"[pipeline] {rel}: no YOLO detections — still running "
                f"(FoodSAM ingredients may recover this image)")

        log(f"[pipeline] {rel} — {len(detections)} YOLO items")
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        seg_image = cv2.resize(image_rgb, (640, 640))
        seg_dets = [
            {
                "bbox": (
                    int(d["bbox"][0] * 640.0 / w),
                    int(d["bbox"][1] * 640.0 / h),
                    int(d["bbox"][2] * 640.0 / w),
                    int(d["bbox"][3] * 640.0 / h),
                ),
                "class_name": d["class_name"],
                "confidence": d["confidence"],
            }
            for d in detections
        ]

        t0 = time.time()
        result = pipeline.analyze(
            image_rgb=image_rgb,
            yolo_detections=detections,
            image_path=image_path,
            generate_visualizations=False,
            seg_image_rgb=seg_image,
            seg_yolo_detections=seg_dets,
        )
        elapsed = round(time.time() - t0, 1)

        # ---- research fail-fast ----
        seg_used = type(pipeline.segmenter).__name__
        fallback_used = bool(getattr(pipeline.segmenter, "fallback_used", False))
        requested = backend or "sam2"
        expected_cls = "FoodSAMSegmenter" if requested == "foodsam" else "SAM2Segmenter"
        if seg_used != expected_cls or fallback_used:
            raise RuntimeError(
                f"[FAIL-FAST] {rel}: requested={requested} but used={seg_used}, "
                f"fallback_used={fallback_used} — result NOT written. "
                f"error={getattr(pipeline.segmenter, 'last_error', None)}")

        sr = result.scale_result
        seg_info = getattr(pipeline.segmenter, "last_info", {}) or {}
        out = {
            "image": rel,
            "backend_requested": requested,
            "backend_used": seg_used,
            "fallback_used": fallback_used,
            "suppressed_dishes": getattr(
                getattr(pipeline.segmenter, "last_info", {}) or {},
                "suppressed_dishes", []),
            "scale_source": sr.scale_source,
            "scale_confidence": sr.confidence,
            "mm_per_pixel": round(sr.mm_per_pixel, 4),
            "perf": {
                "wall_s": elapsed,
                "seg_timing_s": seg_info.get("timing_s"),
                "seg_vram_peak_mib": seg_info.get("vram_peak_mib"),
                "n_sam_masks": seg_info.get("n_sam_masks"),
            },
            "items": [],
        }
        for est in result.estimations:
            cal = est.nutrition.get("Calories", 0.0)
            item = {
                "class_name": est.class_name,
                "det_confidence": round(est.confidence, 2),
                "volume_cm3": round(est.volume_cm3, 1),
                "volume_method": infer_volume_method(est.class_name),
                "mass_g": round(est.mass_g, 1),
                "mass_std_g": round(est.mass_std_g, 1),
                "calories": round(cal, 1),
                "protein_g": round(est.nutrition.get("Protein", 0.0), 1),
                "carbs_g": round(est.nutrition.get("Carbs", 0.0), 1),
                "fat_g": round(est.nutrition.get("Fat", 0.0), 1),
                "estimation_method": est.estimation_method,
                "confidence_level": est.confidence_level,
                "warnings": est.warnings,
                "source": getattr(est, "source", "yolo"),
                "is_ingredient": getattr(est, "is_ingredient", False),
                "component_id": getattr(est, "component_id", ""),
                "mapping_type": getattr(est, "mapping_type", ""),
                "semantic_purity": getattr(est, "semantic_purity", None),
            }
            out["items"].append(item)
            log(
                f"    - {item['class_name']}"
                f"{' [ing]' if item['is_ingredient'] else ''}: "
                f"{item['volume_cm3']} cm3 [{item['volume_method']}], "
                f"{item['mass_g']}+-{item['mass_std_g']} g, "
                f"{item['calories']:.0f} kcal, conf={item['confidence_level']}"
            )
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        n_new += 1
        log(f"    saved ({elapsed}s)")

    log(f"[pipeline] done — {n_new} new results in results/{tag}/")


def collect_csv(args):
    rows = []
    results_root = os.path.join(WORK_DIR, "results")
    if not os.path.isdir(results_root):
        log(f"No results dir: {results_root}")
        return
    summary = {}
    for tag in sorted(os.listdir(results_root)):
        tag_dir = os.path.join(results_root, tag)
        if not os.path.isdir(tag_dir):
            continue
        for root, _dirs, files in os.walk(tag_dir):
            for fname in sorted(files):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(root, fname)
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                rel = data.get("image", "")
                folder = rel.split(os.sep)[0] if os.sep in rel or "/" in rel else "(root)"
                bucket = summary.setdefault((tag, folder), {
                    "images": 0, "zero_yolo": 0, "recovered": 0,
                    "extras": 0, "duplicates": 0,
                    "suppressed": 0, "promotions": 0,
                    "coverage": [],
                })
                bucket["images"] += 1
                sup_list = data.get("suppressed_dishes", [])
                if sup_list:
                    bucket["suppressed"] += len(sup_list)
                    bucket["promotions"] += sum(
                        len(s.get("promoted_labels", [])) for s in sup_list)
                    covs = [s["promoted_area_coverage"] for s in sup_list
                            if s.get("promoted_area_coverage") is not None]
                    if covs:
                        bucket["coverage"].append(sum(covs) / len(covs))
                items = data.get("items", [])
                dish_items = [i for i in items if i.get("source", "yolo") == "yolo"]
                ing_items = [i for i in items if i.get("is_ingredient")]
                bucket["extras"] += len(ing_items)
                if not dish_items:
                    bucket["zero_yolo"] += 1
                    if ing_items:
                        bucket["recovered"] += 1
                if not items:
                    rows.append({
                        "backend": tag, "image": rel,
                        "class_name": "(no detection)", "source": "yolo",
                    })
                    continue
                for item in items:
                    rows.append(
                        {
                            "backend": tag,
                            "image": rel,
                            "folder": folder,
                            "class_name": item["class_name"],
                            "source": item.get("source", "yolo"),
                            "is_ingredient": item.get("is_ingredient", False),
                            "component_id": item.get("component_id", ""),
                            "mapping_type": item.get("mapping_type", ""),
                            "semantic_purity": item.get("semantic_purity"),
                            "det_confidence": item.get("det_confidence"),
                            "volume_cm3": item["volume_cm3"],
                            "volume_method": item["volume_method"],
                            "mass_g": item["mass_g"],
                            "mass_std_g": item["mass_std_g"],
                            "calories": item["calories"],
                            "protein_g": item["protein_g"],
                            "carbs_g": item["carbs_g"],
                            "fat_g": item["fat_g"],
                            "confidence_level": item["confidence_level"],
                            "estimation_method": item["estimation_method"],
                            "scale_source": data.get("scale_source"),
                            "mm_per_pixel": data.get("mm_per_pixel"),
                            "wall_s": data.get("perf", {}).get("wall_s"),
                        }
                    )
    csv_path = os.path.join(WORK_DIR, "batch_results.csv")
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    log(f"CSV: {csv_path} ({len(rows)} rows)")

    # ---- batch_summary.json: per-backend × per-folder stats ----
    summary_out = {}
    for (tag, folder), b in sorted(summary.items()):
        rate = (b["recovered"] / b["zero_yolo"]) if b["zero_yolo"] else None
        summary_out.setdefault(tag, {})[folder] = {
            "images": b["images"],
            "zero_yolo_images": b["zero_yolo"],
            "recovered_by_ingredients": b["recovered"],
            # candidate recovery, NOT accuracy — no ground truth is used
            "semantic_recovery_rate": round(rate, 4) if rate is not None else None,
            "ingredient_extras": b["extras"],
            "suppressed_dishes": b["suppressed"],
            "promoted_labels_total": b["promotions"],
            "promoted_area_coverage_mean": (
                round(sum(b["coverage"]) / len(b["coverage"]), 4)
                if b["coverage"] else None),
        }
    summary_path = os.path.join(WORK_DIR, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    log(f"Summary: {summary_path}")
    for tag, folders in summary_out.items():
        for folder, s in folders.items():
            rr = s["semantic_recovery_rate"]
            log(f"  [{tag}] {folder}: {s['images']} imgs, "
                f"{s['zero_yolo_images']} YOLO-miss, "
                f"recovery={rr if rr is not None else 'n/a'}, "
                f"extras={s['ingredient_extras']} "
                f"suppressed={s['suppressed_dishes']} "
                f"coverage={s['promoted_area_coverage_mean']}")


def stage_split(args):
    """Deterministic dev/eval split over the 4 test folders (tune on dev ONLY)."""
    folders = ["No_ArUcO", "BunBoHue", "BanhMi", "HomeCook"]
    all_rels = []
    for folder in folders:
        fdir = os.path.join(TEST_IMAGE_DIR, folder)
        if not os.path.isdir(fdir):
            log(f"!! missing folder: {fdir}")
            continue
        for f in sorted(os.listdir(fdir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_rels.append(os.path.join(folder, f))
    all_rels.sort()
    rng = random.Random(42)  # fixed seed -> reproducible split
    rng.shuffle(all_rels)
    dev, eval_ = all_rels[:DEV_SIZE], all_rels[DEV_SIZE:]
    split_dir = os.path.join(WORK_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for name, lst in (("dev.txt", dev), ("eval.txt", eval_)):
        with open(os.path.join(split_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lst))
    log(f"split: dev={len(dev)} eval={len(eval_)} -> {split_dir}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["yolo", "pipeline", "csv", "split"],
                        required=True)
    parser.add_argument("--conf", type=float, default=CONF_DEFAULT)
    parser.add_argument("--seg-backend", choices=["sam2", "foodsam"], default=None,
                        help="Segmentation backend override (default: "
                             "pipeline_config.SEGMENTATION_BACKEND). Results are "
                             "stored under results/<backend>/ so backends never "
                             "overwrite each other.")
    parser.add_argument("--img-dir", default=None,
                        help="Folder of loose images (default: TEST_IMAGE_DIR). "
                             "Subfolders of test-image keep their relpath keys.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images (quick runs / sensitivity).")
    parser.add_argument("--images-list", default=None,
                        help="Text file of image paths (dev/eval splits) — "
                             "overrides --img-dir selection.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute results even if the per-image JSON "
                             "already exists (required for threshold tuning "
                             "re-runs on the same images).")
    args = parser.parse_args()

    if args.seg_backend:
        import pipeline_config as _pc
        _pc.SEGMENTATION_BACKEND = args.seg_backend

    if args.stage == "yolo":
        stage_yolo(args.conf, args)
    elif args.stage == "pipeline":
        stage_pipeline(args)
    elif args.stage == "csv":
        collect_csv(args)
    elif args.stage == "split":
        stage_split(args)


if __name__ == "__main__":
    main()
