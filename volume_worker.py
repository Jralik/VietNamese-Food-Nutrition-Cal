"""
Volume pipeline worker — runs in an ISOLATED process, one job per invocation.

Usage:
    python volume_worker.py <job.json> <output.json>

Job JSON: {"image_path": str, "detections": [{"bbox": [x1,y1,x2,y2],
"class_name": str, "confidence": float}, ...]}

The Streamlit process must not load SAM2/torch CUDA alongside onnxruntime
(segmentation faults on some Windows installs), so the heavy pipeline runs
here instead. Any crash is contained: the app falls back to per-serving
nutrition when this process fails.
"""

import base64
import io
import json
import os
import sys
import traceback


def _img_to_b64(img_rgb) -> str:
    """Encode an RGB ndarray as base64 JPEG for transport over JSON."""
    import cv2
    from PIL import Image
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _crop_to_b64(img_rgb, bbox) -> str:
    """Encode a clamped bbox crop of an RGB image as base64 JPEG."""
    import numpy as np
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return _img_to_b64(img_rgb)
    return _img_to_b64(np.ascontiguousarray(img_rgb[y1:y2, x1:x2]))


def main() -> int:
    job_path, out_path = sys.argv[1], sys.argv[2]
    # Surface the pipeline loggers (FoodSAMSegmenter bridge ok/failed, ...) on
    # stderr: volume_integration captures stderr and logs the tail when the
    # worker fails, so INFO lines here make FoodSAM runs debuggable.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    try:
        with open(job_path, encoding="utf-8") as f:
            job = json.load(f)

        # Per-request segmentation backend. Safe as a module-global set: this
        # worker process handles exactly ONE job then exits (spawned fresh by
        # volume_integration per request), and the segmenter factory reads the
        # attribute lazily at first pipeline.segmenter access.
        backend = job.get("backend")
        if backend:
            import pipeline_config
            pipeline_config.SEGMENTATION_BACKEND = backend

        import cv2
        import numpy as np
        from PIL import Image
        import volume_nutrition
        # Phase-C experiment knob (propagated via env; None/absent = baseline)
        cap_env = os.environ.get("VOLUME_SATURATION_CAP_RATIO")
        if cap_env:
            volume_nutrition.SATURATION_CAP_RATIO = float(cap_env)
        from food_volume_pipeline import FoodVolumePipeline

        image = Image.open(job["image_path"]).convert("RGB")
        pipeline = FoodVolumePipeline()

        # SAM2 runs on the 640x640 display copy: its fixed internal resolution
        # starves objects of pixels on large aspect-preserving sources, which
        # visibly degraded masks. Masks are lifted back onto the source grid
        # inside pipeline.analyze so depth/scale integration stays aligned.
        img_array = np.asarray(image)
        seg_image = cv2.resize(img_array, (640, 640))
        w, h = image.size
        seg_dets = [
            {
                "bbox": (int(d["bbox"][0] * 640.0 / w),
                         int(d["bbox"][1] * 640.0 / h),
                         int(d["bbox"][2] * 640.0 / w),
                         int(d["bbox"][3] * 640.0 / h)),
                "class_name": d["class_name"],
                "confidence": d["confidence"],
            }
            for d in job["detections"]
        ]
        result = pipeline.analyze(
            image_rgb=img_array,
            yolo_detections=job["detections"],
            image_path=job["image_path"],
            generate_visualizations=True,
            seg_image_rgb=seg_image,
            seg_yolo_detections=seg_dets,
        )

        # Visualizations for the UI (all encoded as base64 JPEG):
        # mask_overlay is RGB; depth/aruco overlays come back in BGR.
        mask_b64 = _img_to_b64(result.mask_overlay)
        depth_b64 = _img_to_b64(cv2.cvtColor(result.depth_colored, cv2.COLOR_BGR2RGB))
        scale_b64 = _img_to_b64(cv2.cvtColor(result.scale_overlay, cv2.COLOR_BGR2RGB))

        seg = getattr(pipeline, "_segmenter", None)
        payload = {
            "estimations": [
                {
                    "class_name": est.class_name,
                    "confidence": est.confidence,
                    "bbox": list(est.bbox),
                    "volume_cm3": est.volume_cm3,
                    "mass_g": est.mass_g,
                    "mass_std_g": est.mass_std_g,
                    "nutrition": est.nutrition,
                    "nutrition_std": est.nutrition_std,
                    "estimation_method": est.estimation_method,
                    "confidence_level": est.confidence_level,
                    "confidence_note": est.confidence_note,
                    "warnings": est.warnings,
                    "nutrition_per_100g": est.nutrition_per_100g,
                    # provenance (FoodSAM ingredient items carry these)
                    "source": getattr(est, "source", "yolo"),
                    "is_ingredient": getattr(est, "is_ingredient", False),
                    "component_id": getattr(est, "component_id", ""),
                    "mapping_type": getattr(est, "mapping_type", ""),
                    "semantic_purity": getattr(est, "semantic_purity", None),
                    # per-item crop with segmentation mask + label drawn
                    "crop_b64": _crop_to_b64(result.mask_overlay, est.bbox),
                }
                for est in result.estimations
            ],
            "total_nutrition": result.total_nutrition,
            "total_nutrition_std": result.total_nutrition_std,
            "seg_backend_used": type(seg).__name__ if seg is not None else None,
            "seg_fallback_used": bool(getattr(seg, "fallback_used", False)),
            "seg_error": getattr(seg, "last_error", None),
            "suppressed_dishes": getattr(
                getattr(seg, "last_info", {}) or {}, "suppressed_dishes", []),
            "scale": {
                "mm_per_pixel": result.scale_result.mm_per_pixel,
                "scale_source": result.scale_result.scale_source,
                "confidence": result.scale_result.confidence,
            },
            "visualizations": {
                "mask_overlay": mask_b64,
                "depth_colored": depth_b64,
                "scale_overlay": scale_b64,
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return 0

    except Exception:
        # Always hand an error payload back so the parent can log it and
        # fall back gracefully instead of hanging on a missing output file.
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"error": traceback.format_exc()}, f)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
