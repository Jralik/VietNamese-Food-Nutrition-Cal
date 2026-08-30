"""
Depth Anything V2 — Metric Depth Estimation Module.

Produces a per-pixel depth map in **millimeters** from an RGB image.

Preferred workflow: use the metric-tuned indoor checkpoint (output is already
in meters → converted to mm). Falls back to relative depth + manual fitting
if the metric model is unavailable.

Camera intrinsics estimation: reads EXIF focal length if available, otherwise
uses a heuristic (fx ≈ max(W, H)) common for smartphone cameras.
"""

import os
import numpy as np
import cv2
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Output of depth estimation."""
    depth_mm: np.ndarray          # shape (H, W), float32, values in mm
    depth_source: str             # "metric_model" | "relative_fitted" | "unavailable"
    encoder: str                  # model variant used
    input_resolution: Tuple[int, int]  # (H, W) of the depth map


def _read_exif_focal_length(image_path: Optional[str]) -> Optional[float]:
    """Try to read focal length in mm from EXIF data.
    
    Returns focal length in mm or None if unavailable.
    """
    if image_path is None:
        return None
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS
        pil_img = PILImage.open(image_path)
        exif_data = pil_img._getexif()
        if exif_data is None:
            return None
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "FocalLength":
                # value is typically a tuple (numerator, denominator)
                if isinstance(value, tuple):
                    return float(value[0]) / float(value[1])
                return float(value)
    except Exception:
        pass
    return None


def estimate_camera_intrinsics(
    image_shape: Tuple[int, int],
    image_path: Optional[str] = None,
    calib_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate camera intrinsic matrix K and distortion coefficients.
    
    Priority:
    1. Load from calibration file (camera_calib.npz) if provided
    2. Use EXIF focal length if available
    3. Heuristic: fx = fy = max(H, W)
    
    Returns (K, dist) where K is 3×3 intrinsic matrix, dist is distortion coeffs.
    """
    h, w = image_shape[:2]

    # 1. Try calibration file
    if calib_file and os.path.exists(calib_file):
        try:
            data = np.load(calib_file)
            K = data["K"].astype(np.float64).copy()
            dist = data["dist"].astype(np.float64).copy()
            if "image_shape" in data:
                calib_w, calib_h = int(data["image_shape"][0]), int(data["image_shape"][1])
                if calib_w > 0 and calib_h > 0 and (calib_w != w or calib_h != h):
                    # Check if orientation is swapped (one is portrait, one is landscape)
                    if (calib_w < calib_h) != (w < h):
                        calib_w, calib_h = calib_h, calib_w
                        K = np.array([
                            [K[1, 1], 0.0, float(w) / 2.0],
                            [0.0, K[0, 0], float(h) / 2.0],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float64)
                    sx = w / float(calib_w)
                    sy = h / float(calib_h)
                    K[0, 0] *= sx
                    K[1, 1] *= sy
                    K[0, 2] = float(w) / 2.0
                    K[1, 2] = float(h) / 2.0
                    logger.info(
                        f"Camera intrinsics scaled from calib shape {calib_w}x{calib_h} "
                        f"to image shape {w}x{h} (sx={sx:.3f}, sy={sy:.3f})"
                    )
            logger.info(f"Camera intrinsics loaded from {calib_file}")
            return K, dist
        except Exception as e:
            logger.warning(f"Failed to load calibration file: {e}")

    # 2. Try EXIF focal length
    focal_mm = _read_exif_focal_length(image_path)
    if focal_mm is not None:
        # Approximate: sensor width ~6.4mm for typical smartphone
        # fx_pixels = focal_mm * image_width / sensor_width
        sensor_width_mm = 6.4  # rough estimate for smartphone sensors
        fx = focal_mm * w / sensor_width_mm
        fy = fx
        logger.info(f"Camera intrinsics estimated from EXIF focal={focal_mm}mm → fx={fx:.0f}px")
    else:
        # 3. Heuristic fallback
        fx = fy = float(max(h, w))
        logger.info(f"Camera intrinsics: heuristic fx=fy={fx:.0f}px")

    cx, cy = w / 2.0, h / 2.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)
    dist = np.zeros(5)
    return K, dist


class DepthEstimator:
    """Estimate metric depth from an RGB image using Depth Anything V2.
    
    Args:
        encoder: Model variant — "vits", "vitb", or "vitl"
        use_metric: Use metric-tuned checkpoint (output in meters)
        device: "cuda" or "cpu"
        max_depth: Maximum depth in meters (for metric model)
    """

    def __init__(
        self,
        encoder: str = None,
        use_metric: bool = None,
        device: str = None,
        max_depth: float = None,
    ):
        from pipeline_config import (
            DEPTH_ENCODER, DEPTH_USE_METRIC, DEVICE,
            DEPTH_MAX_DEPTH, DEPTH_INPUT_SIZE,
        )

        self.encoder = encoder or DEPTH_ENCODER
        self.use_metric = use_metric if use_metric is not None else DEPTH_USE_METRIC
        self.device = device or DEVICE
        self.max_depth = max_depth or DEPTH_MAX_DEPTH
        self.input_size = DEPTH_INPUT_SIZE

        self._model = None
        self._processor = None
        self._available = None
        self._model_type = None  # "huggingface" or "native"

    def _load_model(self):
        """Lazy-load Depth Anything V2 model."""
        if self._model is not None:
            return

        # Try HuggingFace Transformers pipeline first (easier install)
        try:
            self._load_huggingface()
            return
        except Exception as e:
            logger.info(f"HuggingFace Depth Anything V2 not available: {e}")

        # Try native depth_anything_v2 package
        try:
            self._load_native()
            return
        except Exception as e:
            logger.warning(
                f"Depth Anything V2 not available ({e}). "
                f"Depth estimation will be unavailable."
            )
            self._available = False

    def _load_huggingface(self):
        """Load via HuggingFace Transformers (recommended install method)."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
        from pipeline_config import DEPTH_HF_MODELS

        model_id = DEPTH_HF_MODELS.get(self.encoder)
        if model_id is None:
            raise ValueError(f"Unknown encoder '{self.encoder}' for HuggingFace")

        logger.info(f"Loading Depth Anything V2 from HuggingFace: {model_id}")
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self._processor = AutoImageProcessor.from_pretrained(model_id)
        if self.device == "cuda" and torch.cuda.is_available():
            self._model = AutoModelForDepthEstimation.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to("cuda")
        else:
            self._model = AutoModelForDepthEstimation.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
            )
            self._model.to(self.device)
        self._model.eval()
        self._available = True
        self._model_type = "huggingface"
        logger.info(f"Depth Anything V2 loaded: encoder={self.encoder}, device={self.device}")

    def _load_native(self):
        """Load via native depth_anything_v2 package."""
        import torch

        # Try importing the native package
        if self.use_metric:
            from depth_anything_v2.dpt import DepthAnythingV2
        else:
            from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        cfg = model_configs[self.encoder]

        if self.use_metric:
            cfg['max_depth'] = self.max_depth

        self._model = DepthAnythingV2(**cfg)

        # Load checkpoint
        checkpoint_name = f"depth_anything_v2_metric_{'indoor' if self.use_metric else 'outdoor'}_{self.encoder}.pth"
        state_dict = torch.load(checkpoint_name, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()
        self._available = True
        self._model_type = "native"
        logger.info(f"Depth Anything V2 (native) loaded: encoder={self.encoder}")

    @property
    def is_available(self) -> bool:
        """Check if depth model loaded successfully."""
        if self._available is None:
            self._load_model()
        return self._available

    def estimate(self, image_rgb: np.ndarray) -> DepthResult:
        """Estimate depth map from an RGB image.
        
        Args:
            image_rgb: Input image, shape (H, W, 3), uint8, RGB format.
        
        Returns:
            DepthResult with depth_mm array (same spatial dims as input).
        """
        self._load_model()
        h, w = image_rgb.shape[:2]

        if not self._available:
            logger.warning("Depth model unavailable — returning zero depth map")
            return DepthResult(
                depth_mm=np.zeros((h, w), dtype=np.float32),
                depth_source="unavailable",
                encoder=self.encoder,
                input_resolution=(h, w),
            )

        if self._model_type == "huggingface":
            depth_mm = self._estimate_huggingface(image_rgb)
        else:
            depth_mm = self._estimate_native(image_rgb)

        # Resize depth map to match input image if needed
        if depth_mm.shape[:2] != (h, w):
            depth_mm = cv2.resize(depth_mm, (w, h), interpolation=cv2.INTER_LINEAR)

        source = "metric_model" if self.use_metric else "relative_fitted"

        return DepthResult(
            depth_mm=depth_mm,
            depth_source=source,
            encoder=self.encoder,
            input_resolution=(h, w),
        )

    def _estimate_huggingface(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run inference via HuggingFace pipeline."""
        import torch
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image_rgb)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda" and hasattr(self._model, "dtype") and self._model.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth.float()

        # predicted_depth shape: (1, H', W') in meters (metric model)
        depth_map = predicted_depth.squeeze().cpu().numpy()
        del outputs
        del inputs

        if self.use_metric:
            # Metric model output is in meters → convert to mm in-place
            depth_map *= 1000.0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return depth_map.astype(np.float32, copy=False)

    def _estimate_native(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run inference via native depth_anything_v2 package."""
        import torch

        depth_map = self._model.infer_image(image_rgb, self.input_size)

        if self.use_metric:
            depth_mm = depth_map * 1000.0  # meters → mm
        else:
            depth_mm = depth_map

        return depth_mm.astype(np.float32)

    def colorize_depth(
        self, depth_mm: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """Create a colorized visualization of the depth map.
        
        Returns BGR image suitable for display or saving.
        """
        # Normalize to 0-255
        valid = depth_mm[depth_mm > 0]
        if len(valid) == 0:
            return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)

        vmin, vmax = np.percentile(valid, [2, 98])
        depth_norm = np.clip((depth_mm - vmin) / (vmax - vmin + 1e-8), 0, 1)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, colormap)
        return colored
