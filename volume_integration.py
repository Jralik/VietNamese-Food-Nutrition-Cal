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
import time
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


def volume_pipeline_available(backend: Optional[str] = None) -> bool:
    """Cheap dependency check — never imports the packages themselves.

    Importing transformers here prints docstring-validation noise and pulls
    heavy modules on every rerun; importlib.util.find_spec is side-effect free.
    `backend` selects which backend the check targets (None = config default).
    """
    import importlib.util
    import pipeline_config as _pc
    if backend is None:
        backend = getattr(_pc, "SEGMENTATION_BACKEND", "sam2")
    for module_name in ("torch", "sam2", "transformers", "cv2"):
        if importlib.util.find_spec(module_name) is None:
            logger.info(f"Volume pipeline dependency '{module_name}' unavailable")
            return False
    if not os.path.exists(WORKER_SCRIPT):
        logger.info(f"Volume worker script missing: {WORKER_SCRIPT}")
        return False
    # FoodSAM backend additionally needs its isolated env (checked on disk —
    # the main .venv keeps no FoodSAM dependency)
    if backend == "foodsam":
        env_python = os.path.join(_pc.FOODSAM_REPO, "env", "Scripts", "python.exe")
        if not (os.path.isfile(env_python) and os.path.isfile(_pc.FOODSAM_INFER_SCRIPT)):
            logger.info(
                f"FoodSAM env not usable (python={env_python}, "
                f"script={_pc.FOODSAM_INFER_SCRIPT})")
            return False
    return True


def extract_yolo_detections(result, class_names: list, bbox_scale=None) -> List[Dict]:
    """Convert an Ultralytics Results object into pipeline detection dicts.

    Each dict: {"bbox": (x1, y1, x2, y2), "class_name": str, "confidence": float}.
    Non-food classes and low-confidence boxes are skipped.

    `bbox_scale` = (sx, sy) rescales boxes from the original image space (the
    coordinates Ultralytics already maps detections back to) to the
    aspect-preserving source image given to the volume pipeline.
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
    # provenance (FoodSAM ingredient items carry these)
    source: str = "yolo"
    is_ingredient: bool = False
    component_id: str = ""
    mapping_type: str = ""
    semantic_purity: Optional[float] = None


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
    # Segmentation diagnostics (what the worker actually ran)
    seg_backend_used: str = ""
    seg_fallback_used: bool = False
    seg_error: Optional[str] = None
    # mixed-plate audit: YOLO dishes removed from accounting by the FoodSAM
    # contradiction heuristic (class_name/confidence/bbox/reason/coverage)
    suppressed_dishes: List[Dict] = field(default_factory=list)
    # Execution provenance diagnostics (transparent reporting for thesis benchmark)
    worker_mode: str = "oneshot"
    worker_fallback: bool = False
    worker_pid: Optional[int] = None
    worker_elapsed_s: float = 0.0


import atexit
import concurrent.futures
import threading


class WorkerDaemonManager:
    """Manages a persistent volume_worker.py --daemon child process over stdin/stdout."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _get_python_exe(self) -> str:
        python_exe = sys.executable
        venv_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
        if os.path.exists(venv_exe):
            python_exe = venv_exe
        return python_exe

    def start(self, timeout_s: int = 60) -> bool:
        """Start or restart the daemon process and wait for ready signal."""
        self.kill()
        python_exe = self._get_python_exe()
        # Ultralytics select_device("cpu") (detector runs on CPU by design)
        # sets CUDA_VISIBLE_DEVICES="-1" process-wide, and child processes
        # inherit it — the volume worker would then see zero GPUs and drop to
        # CPU (~17s/request instead of ~2s). Strip the poison for the worker.
        spawn_env = dict(os.environ)
        if spawn_env.get("CUDA_VISIBLE_DEVICES") == "-1":
            del spawn_env["CUDA_VISIBLE_DEVICES"]
        try:
            logger.info("Starting persistent volume worker daemon...")
            self.proc = subprocess.Popen(
                [python_exe, WORKER_SCRIPT, "--daemon"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,  # direct stderr to console so buffer never deadlocks
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT,
                env=spawn_env,
            )

            def _read_ready():
                return self.proc.stdout.readline()

            fut = self._executor.submit(_read_ready)
            ready_line = fut.result(timeout=timeout_s)
            if not ready_line:
                logger.error("Daemon exited before sending ready signal")
                self.kill()
                return False
            data = json.loads(ready_line.strip())
            if data.get("status") == "ready":
                self.pid = data.get("pid", self.proc.pid)
                logger.info(f"Persistent volume worker ready (PID {self.pid})")
                return True
            else:
                logger.error(f"Unexpected daemon startup response: {ready_line}")
                self.kill()
                return False
        except Exception as e:
            logger.error(f"Failed to start persistent daemon: {e}")
            self.kill()
            return False

    def send_request(self, job: dict, timeout_s: int = 120) -> Optional[dict]:
        """Send job to daemon via stdin and read single JSON line response."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                ok = self.start(timeout_s=60)
                if not ok:
                    return None

            try:
                line = json.dumps(job) + "\n"
                self.proc.stdin.write(line)
                self.proc.stdin.flush()

                def _read_line():
                    return self.proc.stdout.readline()

                fut = self._executor.submit(_read_line)
                resp_line = fut.result(timeout=timeout_s)
                if not resp_line:
                    logger.error("Daemon process returned EOF (crashed)")
                    self.kill()
                    return None

                payload = json.loads(resp_line.strip())
                return payload
            except concurrent.futures.TimeoutError:
                logger.error(f"Daemon request timed out after {timeout_s}s")
                self.kill()
                return None
            except Exception as e:
                logger.error(f"Daemon communication error: {e}")
                self.kill()
                return None

    def kill(self):
        """Clean up child daemon process."""
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.pid = None


_daemon_manager = WorkerDaemonManager()
atexit.register(_daemon_manager.kill)


def _run_oneshot_fallback(
    python_exe: str,
    job: dict,
    timeout_s: int,
    tmp_dir: str,
) -> Optional[dict]:
    """Execute legacy one-shot subprocess when daemon is unavailable or fails."""
    job_path = output_path = None
    try:
        fd, job_path = tempfile.mkstemp(suffix=".json", dir=tmp_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(job, f)

        fd, output_path = tempfile.mkstemp(suffix=".json", dir=tmp_dir)
        os.close(fd)

        t0 = time.time()
        proc = subprocess.run(
            [python_exe, WORKER_SCRIPT, job_path, output_path],
            cwd=PROJECT_ROOT,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0 or not os.path.exists(output_path):
            tail = (proc.stderr or "")[-400:]
            logger.error(f"One-shot volume worker failed (rc={proc.returncode}): {tail}")
            return None

        with open(output_path, encoding="utf-8") as f:
            payload = json.load(f)
        if "error" in payload:
            logger.error(f"One-shot volume worker error: {payload['error']}")
            return None

        payload["worker_mode"] = "oneshot"
        payload["worker_fallback"] = True
        payload["worker_pid"] = proc.args[0] if hasattr(proc, "args") else None
        payload["worker_elapsed_s"] = round(elapsed, 3)
        return payload
    finally:
        for p in (job_path, output_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def estimate_nutrition_volume(
    image,
    detections: List[Dict],
    image_path: Optional[str] = None,
    timeout_s: int = WORKER_TIMEOUT_S,
    backend: Optional[str] = None,
) -> Optional[PipelineResultProxy]:
    """Run the volume pipeline via Persistent Daemon (warm) with One-shot fallback.

    `image` is a PIL image (RGB). `backend` overrides the segmentation backend
    for this request (None = pipeline_config.SEGMENTATION_BACKEND). Returns a
    PipelineResultProxy, or None when all workers fail or time out.
    """
    import pipeline_config as _pc
    if backend is None:
        backend = getattr(_pc, "SEGMENTATION_BACKEND", "sam2")

    if backend == "foodsam":
        timeout_s = max(
            timeout_s,
            WORKER_TIMEOUT_S + int(getattr(_pc, "FOODSAM_TIMEOUT_S", 300)),
        )

    if not detections and backend != "foodsam":
        return None

    python_exe = sys.executable
    venv_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_exe):
        python_exe = venv_exe

    global last_error
    last_error = None
    tmp_dir = tempfile.gettempdir()
    img_path = None
    try:
        fd, img_path = tempfile.mkstemp(suffix=".png", dir=tmp_dir)
        os.close(fd)
        image.convert("RGB").save(img_path)

        job = {
            "image_path": img_path,
            "detections": detections,
            "backend": backend,
        }

        # 1. Primary path: Persistent Daemon Worker (Warm residency)
        t_req_start = time.time()
        payload = _daemon_manager.send_request(job, timeout_s=timeout_s)

        # 2. Fallback path: Legacy one-shot subprocess if daemon failed
        if payload is None or "error" in payload:
            err_detail = payload.get("error", "no response") if payload else "daemon unreachable"
            logger.warning(
                f"Daemon worker returned error or unavailable ({err_detail}); "
                f"initiating transparent one-shot fallback..."
            )
            payload = _run_oneshot_fallback(python_exe, job, timeout_s, tmp_dir)
            if payload is None:
                last_error = f"both daemon and oneshot worker failed: {err_detail}"
                return None

        result = _proxy_from_payload(payload)
        n_extras = sum(1 for e in result.estimations if e.is_ingredient)
        logger.info(
            f"Volume run ok: backend={result.seg_backend_used or '?'} "
            f"mode={result.worker_mode} fallback={result.worker_fallback} "
            f"time={result.worker_elapsed_s}s estimations={len(result.estimations)} "
            f"extras={n_extras} scale={result.scale_result.scale_source}"
        )
        return result

    except subprocess.TimeoutExpired:
        last_error = f"worker timed out after {timeout_s}s"
        logger.error(last_error)
        return None
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        logger.exception("Volume worker orchestration failed")
        return None
    finally:
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
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
            source=e.get("source", "yolo"),
            is_ingredient=e.get("is_ingredient", False),
            component_id=e.get("component_id", ""),
            mapping_type=e.get("mapping_type", ""),
            semantic_purity=e.get("semantic_purity"),
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
        seg_backend_used=payload.get("seg_backend_used") or "",
        seg_fallback_used=bool(payload.get("seg_fallback_used", False)),
        seg_error=payload.get("seg_error"),
        suppressed_dishes=payload.get("suppressed_dishes", []),
        worker_mode=payload.get("worker_mode", "oneshot"),
        worker_fallback=bool(payload.get("worker_fallback", False)),
        worker_pid=payload.get("worker_pid"),
        worker_elapsed_s=float(payload.get("worker_elapsed_s", 0.0)),
    )
