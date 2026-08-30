"""
Bridge between the Streamlit app and the volume estimation pipeline.

The pipeline (SAM2 + Depth Anything V2, CUDA) is executed in a SEPARATE
worker process (`volume_worker.py`). Running it in-process alongside
onnxruntime/ultralytics reliably segfaults on some Windows installs
(DLL/CUDA state conflicts), taking the whole Streamlit server down. The
subprocess isolation contains any such crash: if the worker fails or times
out, this module returns None and the app falls back to per-serving
nutrition.

Converts Ultralytics YOLO results into the detection format expected by
FoodVolumePipeline and returns lightweight result proxies.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "volume_worker.py")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

NON_FOOD_CLASS = "Con nguoi (Human)"
# Below this confidence a detection is too unreliable to base a portion on
MIN_DETECTION_CONFIDENCE = 0.40
# First call loads SAM2 + Depth (HF cache); allow generous time
WORKER_TIMEOUT_S = 300


# Reason for the most recent fallback (surfaced in the UI note)
last_error: Optional[str] = None


def volume_pipeline_available() -> bool:
    """Cheap dependency check — never imports the packages themselves.

    Importing transformers here prints docstring-validation noise and pulls
    heavy modules on every rerun; importlib.util.find_spec is side-effect free.
    """
    import importlib.util
    for module_name in ("torch", "sam2", "transformers", "cv2"):
        if importlib.util.find_spec(module_name) is None:
            logger.info(f"Volume pipeline dependency '{module_name}' unavailable")
            return False
    if not os.path.exists(WORKER_SCRIPT):
        logger.info(f"Volume worker script missing: {WORKER_SCRIPT}")
        return False
    return True


def extract_yolo_detections(result, class_names: list, bbox_scale=None) -> List[Dict]:
    """Convert an Ultralytics Results object into pipeline detection dicts.

    Each dict: {"bbox": (x1, y1, x2, y2), "class_name": str, "confidence": float}.
    Non-food classes and low-confidence boxes are skipped.

    `bbox_scale` = (sx, sy) rescales boxes from the YOLO input space (640x640)
    to the aspect-preserving source image given to the volume pipeline.
    """
    detections: List[Dict] = []
    for r in result:
        for box in r.boxes:
            class_id = int(box.cls[0].item())
            if class_id < 0 or class_id >= len(class_names):
                continue
            class_name = class_names[class_id]["name"]
            if class_name == NON_FOOD_CLASS:
                continue
            conf = float(box.conf[0].item())
            if conf < MIN_DETECTION_CONFIDENCE:
                continue
            xyxy = box.xyxy
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            x1, y1, x2, y2 = (float(v) for v in xyxy[0][:4])
            if bbox_scale is not None:
                sx, sy = bbox_scale
                x1, x2 = x1 * sx, x2 * sx
                y1, y2 = y1 * sy, y2 * sy
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "class_name": class_name,
                "confidence": conf,
            })
    return detections


# ─── Lightweight result proxies (what the UI needs; masks are not sent) ────

@dataclass
class ScaleResultProxy:
    mm_per_pixel: float = 0.0
    scale_source: str = "unknown"
    confidence: str = "low"


@dataclass
class EstimationProxy:
    class_name: str
    confidence: float
    bbox: tuple
    volume_cm3: float
    mass_g: float
    mass_std_g: float
    nutrition: Dict[str, float]
    nutrition_std: Dict[str, float]
    estimation_method: str
    confidence_level: str
    confidence_note: str
    warnings: List[str] = field(default_factory=list)
    nutrition_per_100g: Optional[Dict[str, float]] = None
    # bbox crop with segmentation mask + label drawn (base64 JPEG)
    crop_b64: Optional[str] = None


@dataclass
class PipelineResultProxy:
    estimations: List[EstimationProxy]
    scale_result: ScaleResultProxy
    total_nutrition: Dict[str, float] = field(default_factory=dict)
    total_nutrition_std: Dict[str, float] = field(default_factory=dict)
    # Analysis visualizations (base64 JPEG): segmentation overlay,
    # colorized depth map, scale/ArUco overlay
    mask_overlay_b64: Optional[str] = None
    depth_colored_b64: Optional[str] = None
    scale_overlay_b64: Optional[str] = None


def estimate_nutrition_volume(
    image,
    detections: List[Dict],
    image_path: Optional[str] = None,
    timeout_s: int = WORKER_TIMEOUT_S,
) -> Optional[PipelineResultProxy]:
    """Run the volume pipeline in an isolated worker process.

    `image` is a PIL image (RGB). Returns a PipelineResultProxy, or None when
    the worker is unavailable, times out, or crashes — callers should fall
    back to per-serving nutrition.
    """
    if not detections:
        return None

    # Prefer the project's own venv python: sys.executable follows whatever
    # launched Streamlit (PATH may point at another project's venv whose
    # package set differs).
    python_exe = sys.executable
    venv_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_exe):
        python_exe = venv_exe

    global last_error
    last_error = None
    tmp_dir = tempfile.gettempdir()
    job_path = output_path = img_path = None
    try:
        fd, img_path = tempfile.mkstemp(suffix=".png", dir=tmp_dir)
        os.close(fd)
        image.convert("RGB").save(img_path)

        fd, job_path = tempfile.mkstemp(suffix=".json", dir=tmp_dir)
        with os.fdopen(fd, "w") as f:
            json.dump({"image_path": img_path, "detections": detections}, f)

        fd, output_path = tempfile.mkstemp(suffix=".json", dir=tmp_dir)
        os.close(fd)

        proc = subprocess.run(
            [python_exe, WORKER_SCRIPT, job_path, output_path],
            cwd=PROJECT_ROOT,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not os.path.exists(output_path):
            tail = (proc.stderr or "")[-400:]
            last_error = f"worker exit {proc.returncode}: {tail}"
            logger.error(f"Volume worker failed (rc={proc.returncode}): {tail}")
            return None

        with open(output_path, encoding="utf-8") as f:
            payload = json.load(f)
        if "error" in payload:
            last_error = payload["error"][-400:]
            logger.error(f"Volume worker error: {payload['error']}")
            return None
        return _proxy_from_payload(payload)

    except subprocess.TimeoutExpired:
        last_error = f"worker timed out after {timeout_s}s"
        logger.error(last_error)
        return None
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        logger.exception("Volume worker orchestration failed")
        return None
    finally:
        for p in (img_path, job_path, output_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _proxy_from_payload(payload: Dict) -> PipelineResultProxy:
    estimations = [
        EstimationProxy(
            class_name=e["class_name"],
            confidence=e["confidence"],
            bbox=tuple(e["bbox"]),
            volume_cm3=e["volume_cm3"],
            mass_g=e["mass_g"],
            mass_std_g=e["mass_std_g"],
            nutrition=e["nutrition"],
            nutrition_std=e["nutrition_std"],
            estimation_method=e["estimation_method"],
            confidence_level=e["confidence_level"],
            confidence_note=e["confidence_note"],
            warnings=e.get("warnings", []),
            nutrition_per_100g=e.get("nutrition_per_100g"),
            crop_b64=e.get("crop_b64"),
        )
        for e in payload["estimations"]
    ]
    viz = payload.get("visualizations", {})
    return PipelineResultProxy(
        estimations=estimations,
        scale_result=ScaleResultProxy(
            mm_per_pixel=payload["scale"]["mm_per_pixel"],
            scale_source=payload["scale"]["scale_source"],
            confidence=payload["scale"]["confidence"],
        ),
        total_nutrition=payload.get("total_nutrition", {}),
        total_nutrition_std=payload.get("total_nutrition_std", {}),
        mask_overlay_b64=viz.get("mask_overlay"),
        depth_colored_b64=viz.get("depth_colored"),
        scale_overlay_b64=viz.get("scale_overlay"),
    )
