"""
Volume pipeline worker — runs in an ISOLATED process.

Supports two modes:
1. One-shot mode (legacy/fallback):
     python volume_worker.py <job.json> <output.json>
2. Persistent Daemon mode (Warm worker):
     python volume_worker.py --daemon
   - Communicates over stdin/stdout via line-delimited JSON.
   - All logging, warnings, and diagnostic text are directed strictly to stderr.
   - Pre-loads SAM2 + Depth Anything V2 into GPU VRAM (peak ~623 MiB) so warm
     inference takes ~1.8-2.0s without process spawn or cold model loads.
"""

import base64
import io
import json
import logging
import os
import sys
import time
import traceback
import warnings

# Ensure all warnings and logs go exclusively to stderr so stdout remains
# a pure JSON transport stream.
warnings.filterwarnings("default")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("volume_worker")


def _img_to_b64(img_rgb) -> str:
    """Encode an RGB ndarray as base64 JPEG for transport over JSON."""
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


def process_job(job: dict, pipeline) -> dict:
    """Execute pipeline analysis on a single job dictionary and return result payload."""
    backend = job.get("backend")
    if backend:
        import pipeline_config
        pipeline_config.SEGMENTATION_BACKEND = backend
        # Invalidate cached segmenter if backend changed dynamically
        if hasattr(pipeline, "_segmenter") and pipeline._segmenter is not None:
            current_cls = type(pipeline._segmenter).__name__
            expected_cls = "FoodSAMSegmenter" if backend == "foodsam" else "SAM2Segmenter"
            if current_cls != expected_cls:
                logger.info(f"Switching segmenter from {current_cls} to {expected_cls}")
                pipeline._segmenter = None

    import cv2
    import numpy as np
    from PIL import Image
    import volume_nutrition

    cap_env = os.environ.get("VOLUME_SATURATION_CAP_RATIO")
    if cap_env:
        volume_nutrition.SATURATION_CAP_RATIO = float(cap_env)

    image = Image.open(job["image_path"]).convert("RGB")
    img_array = np.asarray(image)
    seg_image = cv2.resize(img_array, (640, 640))
    w, h = image.size
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
                "source": getattr(est, "source", "yolo"),
                "is_ingredient": getattr(est, "is_ingredient", False),
                "component_id": getattr(est, "component_id", ""),
                "mapping_type": getattr(est, "mapping_type", ""),
                "semantic_purity": getattr(est, "semantic_purity", None),
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
            getattr(seg, "last_info", {}) or {}, "suppressed_dishes", []
        ),
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
    return payload


def _wait_for_cuda(attempts: int = 3, delay_s: float = 3.0) -> None:
    """Brief retry for transient GPU-visibility windows.

    DEVICE is frozen when pipeline_config is imported below. Short windows
    where the GPU disappears from enumeration ("No CUDA GPUs are available",
    observed on this machine's GTX 1650 under heavy CUDA context churn) used
    to silently downgrade the whole daemon session to CPU (~17s/request
    instead of ~2s).
    """
    import torch

    for attempt in range(1, attempts + 1):
        if torch.cuda.is_available():
            if attempt > 1:
                logger.info("CUDA became visible on attempt %d", attempt)
            return
        logger.warning("CUDA unavailable (attempt %d/%d) — retrying in %.0fs "
                       "before freezing DEVICE", attempt, attempts, delay_s)
        time.sleep(delay_s)
    logger.warning("CUDA still unavailable after %d attempts — volume daemon "
                   "will run on CPU (~17s/request instead of ~2s)", attempts)


def run_daemon():
    """Run persistent worker loop over stdin/stdout."""
    logger.info(f"Starting Persistent Volume Worker daemon (PID: {os.getpid()})...")
    _wait_for_cuda()
    t0 = time.time()
    from food_volume_pipeline import FoodVolumePipeline
    import pipeline_config

    # Default warm preload: SAM2 box-prompted + Depth Anything V2 (VRAM peak ~623 MiB)
    pipeline_config.SEGMENTATION_BACKEND = "sam2"
    pipeline = FoodVolumePipeline()
    seg = pipeline.segmenter
    if hasattr(seg, "_load_model"):
        seg._load_model()
    depth = pipeline.depth_estimator
    if hasattr(depth, "_load_model"):
        depth._load_model()
    _ = pipeline.scale_recovery
    _ = pipeline.volume_estimator
    logger.info(f"Daemon pre-warm complete in {time.time() - t0:.2f}s. Entering request loop.")

    # Signal to manager that daemon is initialized and ready
    ready_msg = json.dumps({"status": "ready", "pid": os.getpid()}) + "\n"
    sys.stdout.write(ready_msg)
    sys.stdout.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                logger.info("Daemon stdin closed (EOF received). Exiting cleanly.")
                break
            line = line.strip()
            if not line:
                continue

            # Heartbeat ping
            if line == "__PING__":
                sys.stdout.write(json.dumps({"status": "pong", "pid": os.getpid()}) + "\n")
                sys.stdout.flush()
                continue

            t_req_start = time.time()
            job = json.loads(line)
            payload = process_job(job, pipeline)
            t_elapsed = time.time() - t_req_start

            payload["worker_mode"] = "daemon"
            payload["worker_pid"] = os.getpid()
            payload["worker_fallback"] = False
            payload["worker_elapsed_s"] = round(t_elapsed, 3)

            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
        except Exception as e:
            logger.exception("Error processing daemon request")
            err_payload = {
                "error": traceback.format_exc(),
                "worker_mode": "daemon",
                "worker_pid": os.getpid(),
                "worker_fallback": False,
            }
            sys.stdout.write(json.dumps(err_payload) + "\n")
            sys.stdout.flush()


def run_oneshot(job_path: str, out_path: str) -> int:
    """Legacy one-shot execution mode."""
    t0 = time.time()
    try:
        with open(job_path, encoding="utf-8") as f:
            job = json.load(f)

        from food_volume_pipeline import FoodVolumePipeline
        pipeline = FoodVolumePipeline()
        payload = process_job(job, pipeline)
        t_elapsed = time.time() - t0

        payload["worker_mode"] = "oneshot"
        payload["worker_pid"] = os.getpid()
        payload["worker_fallback"] = False
        payload["worker_elapsed_s"] = round(t_elapsed, 3)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return 0
    except Exception:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "error": traceback.format_exc(),
                        "worker_mode": "oneshot",
                        "worker_pid": os.getpid(),
                        "worker_fallback": False,
                    },
                    f,
                )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        run_daemon()
        sys.exit(0)
    elif len(sys.argv) >= 3:
        sys.exit(run_oneshot(sys.argv[1], sys.argv[2]))
    else:
        print("Usage: volume_worker.py --daemon  OR  volume_worker.py <job.json> <out.json>", file=sys.stderr)
        sys.exit(1)
