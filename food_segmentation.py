"""
SAM2 Segmentation Module — YOLO bbox → pixel-accurate food mask.

Uses Meta's SAM 2.1 (Segment Anything Model 2) with YOLO bounding boxes
as spatial prompts to produce accurate food segmentation masks.

Post-processing includes:
  - Boundary erosion to strip plate/bowl rim pixels
  - Oversized-mask flagging (>90% of bbox → likely plate included)
  - Touching-food merge detection (high IoU between masks of different classes)

Fallback: if SAM2 is not available, returns rectangular bbox masks.
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Result from segmenting a single food detection."""
    mask: np.ndarray              # binary mask (H, W), dtype bool
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    warnings: List[str] = field(default_factory=list)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def erode_boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Shrink mask slightly to strip likely plate/bowl rim pixels.
    
    Tune kernel_size empirically against ground-truth masks.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def flag_oversized_mask(mask: np.ndarray, bbox_area: float,
                        threshold: float = 0.90) -> bool:
    """Return True if mask area exceeds threshold fraction of bbox area.
    
    Likely means the plate/bowl got included — flag for manual review.
    """
    mask_area = mask.sum()
    return (mask_area / bbox_area) > threshold if bbox_area > 0 else False


def check_mask_multiplicity(masks: List[np.ndarray],
                             class_names: List[str]) -> List[str]:
    """Heuristic: if two YOLO boxes for different classes produce nearly
    identical masks (IoU > 0.5), flag for review — SAM likely merged them."""
    warnings = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            iou = _mask_iou(masks[i], masks[j])
            if iou > 0.5:
                warnings.append(
                    f"Masks for '{class_names[i]}' and '{class_names[j]}' "
                    f"overlap significantly (IoU={iou:.2f}) — SAM may have "
                    f"merged touching foods."
                )
    return warnings


class SAM2Segmenter:
    """Segment food items using SAM 2.1 prompted by YOLO bounding boxes.
    
    Args:
        model_cfg: SAM2 config yaml path (e.g. "configs/sam2.1/sam2.1_hiera_small.yaml")
        checkpoint: Path to SAM2 checkpoint .pt file
        device: "cuda" or "cpu"
        erode_kernel: kernel size for boundary erosion post-processing
        oversized_threshold: flag masks exceeding this fraction of bbox area
    """

    def __init__(
        self,
        model_cfg: str = None,
        checkpoint: str = None,
        device: str = None,
        erode_kernel: int = None,
        min_mask_threshold: float = None,
        oversized_threshold: float = None,
    ):
        from pipeline_config import (
            SAM2_MODEL_CFG, SAM2_CHECKPOINT, DEVICE,
            SAM2_ERODE_KERNEL_SIZE, SAM2_MIN_MASK_RATIO, SAM2_OVERSIZED_MASK_THRESHOLD,
        )

        self.model_cfg = model_cfg or SAM2_MODEL_CFG
        self.checkpoint = checkpoint or SAM2_CHECKPOINT
        self.device = device or DEVICE
        self.erode_kernel = erode_kernel or SAM2_ERODE_KERNEL_SIZE
        self.min_mask_threshold = min_mask_threshold or SAM2_MIN_MASK_RATIO
        self.oversized_threshold = oversized_threshold or SAM2_OVERSIZED_MASK_THRESHOLD

        self._predictor = None
        self._available = None
        self._use_hf_sam = False

    def _load_model(self):
        """Lazy-load SAM2 model on first use."""
        if self._predictor is not None:
            return

        # 1. Try SAM2 via from_pretrained (HuggingFace Hub)
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            logger.info("Attempting to load SAM2 via from_pretrained('facebook/sam2.1-hiera-small')...")
            self._predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-small",
                device=self.device
            )
            self._available = True
            logger.info(f"SAM2 loaded successfully via from_pretrained, device={self.device}")
            return
        except Exception as e:
            logger.info(f"SAM2 from_pretrained not available: {e}")

        # 2. Try SAM2 via build_sam2 with config
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                self.model_cfg,
                self.checkpoint,
                device=self.device,
            )
            self._predictor = SAM2ImagePredictor(sam2_model)
            self._available = True
            logger.info(
                f"SAM2 loaded successfully: cfg={self.model_cfg}, "
                f"device={self.device}"
            )
            return
        except Exception as e:
            logger.info(f"SAM2 build_sam2 not available: {e}")

        # 3. Try HuggingFace Transformers SAM
        try:
            from transformers import SamModel, SamProcessor
            logger.info("Attempting to load SAM via transformers ('facebook/sam-vit-base')...")
            self._hf_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            self._hf_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
            self._hf_model.eval()
            self._available = True
            self._use_hf_sam = True
            logger.info("HuggingFace SAM loaded successfully!")
            return
        except Exception as e:
            logger.warning(f"Neural segmentation model not available ({e}). Using refined GrabCut/Otsu fallback.")
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if SAM2 model loaded successfully."""
        if self._available is None:
            self._load_model()
        return self._available

    def _bbox_fallback_mask(
        self, image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Create a refined foreground mask using GrabCut/ellipse within the bbox."""
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 4 or bh <= 4:
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask

        # Refined elliptical crop to avoid table background corners of rectangle
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (int(bw * 0.45), int(bh * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # GrabCut refinement within the bbox
        try:
            crop = image_rgb[y1:y2, x1:x2]
            gc_mask = np.zeros(crop.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (max(1, int(bw * 0.05)), max(1, int(bh * 0.05)), int(bw * 0.9), int(bh * 0.9))
            cv2.grabCut(crop, gc_mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
            gc_fg = (gc_mask == 1) | (gc_mask == 3)
            refined_mask = np.zeros((h, w), dtype=bool)
            refined_mask[y1:y2, x1:x2] = gc_fg
            if refined_mask.sum() > 0.15 * (bw * bh):
                return refined_mask
        except Exception:
            pass

        return mask.astype(bool)

    def segment(
        self,
        image_rgb: np.ndarray,
        detections: List[dict],
    ) -> List[SegmentationResult]:
        """Segment all detected food items in the image.
        
        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), uint8
            detections: List of dicts with keys:
                - "bbox": (x1, y1, x2, y2)
                - "class_name": str
                - "confidence": float
        
        Returns:
            List of SegmentationResult, one per detection.
        """
        self._load_model()
        results = []
        all_masks = []
        all_class_names = []

        import torch

        h_orig, w_orig = image_rgb.shape[:2]
        max_dim = 1024
        if max(h_orig, w_orig) > max_dim:
            scale_sam = max_dim / float(max(h_orig, w_orig))
            w_sam, h_sam = int(round(w_orig * scale_sam)), int(round(h_orig * scale_sam))
            image_sam = cv2.resize(image_rgb, (w_sam, h_sam))
        else:
            scale_sam = 1.0
            image_sam = image_rgb

        if self._available:
            # Set the image once for all prompts with inference mode
            with torch.inference_mode():
                self._predictor.set_image(image_sam)

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            confidence = det.get("confidence", 0.0)
            warnings = []

            x1, y1, x2, y2 = bbox
            bbox_area = max(1, (x2 - x1) * (y2 - y1))

            if self._available and not getattr(self, "_use_hf_sam", False):
                try:
                    # Scale bbox for SAM2 predictor
                    sbx1, sby1 = x1 * scale_sam, y1 * scale_sam
                    sbx2, sby2 = x2 * scale_sam, y2 * scale_sam
                    s_bbox_area = max(1, (sbx2 - sbx1) * (sby2 - sby1))

                    with torch.inference_mode():
                        input_box = np.array([[sbx1, sby1, sbx2, sby2]])
                        masks_pred, scores, _ = self._predictor.predict(
                            box=input_box,
                            multimask_output=True,
                        )
                    # SAM returns 3 candidate masks at different granularities
                    candidate_masks = [m.astype(bool) for m in masks_pred]
                    valid_candidates = [m for m in candidate_masks if (m.sum() / s_bbox_area) >= self.min_mask_threshold]
                    if valid_candidates:
                        mask_small = min(valid_candidates, key=lambda m: m.sum())
                    else:
                        mask_small = max(candidate_masks, key=lambda m: m.sum())

                    # If mask is still under threshold, supplement with point prompts
                    if (mask_small.sum() / s_bbox_area) < self.min_mask_threshold:
                        scx, scy = (sbx1 + sbx2) / 2.0, (sby1 + sby2) / 2.0
                        sbw, sbh = float(sbx2 - sbx1), float(sby2 - sby1)
                        point_coords = np.array([
                            [scx, scy],
                            [scx - sbw * 0.2, scy - sbh * 0.2],
                            [scx + sbw * 0.2, scy - sbh * 0.2],
                            [scx - sbw * 0.2, scy + sbh * 0.2],
                            [scx + sbw * 0.2, scy + sbh * 0.2],
                        ])
                        point_labels = np.array([1, 1, 1, 1, 1])
                        with torch.inference_mode():
                            m_pt, _, _ = self._predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                box=input_box,
                                multimask_output=True,
                            )
                        pt_masks = [m.astype(bool) for m in m_pt]
                        best_pt = max(pt_masks, key=lambda m: m.sum())
                        if (best_pt.sum() / s_bbox_area) > (mask_small.sum() / s_bbox_area):
                            mask_small = best_pt

                    # Resize mask back to full original resolution
                    if scale_sam != 1.0:
                        try:
                            mask = cv2.resize(
                                mask_small.astype(np.uint8),
                                (w_orig, h_orig),
                                interpolation=cv2.INTER_NEAREST
                            ) > 0
                        except Exception:
                            from PIL import Image as PILImage
                            pil_m = PILImage.fromarray(mask_small.astype(np.uint8) * 255)
                            mask = np.array(pil_m.resize((w_orig, h_orig), PILImage.NEAREST)) > 0
                    else:
                        mask = mask_small

                    # If still under threshold on full scale, supplement with foreground fallback
                    if (mask.sum() / bbox_area) < self.min_mask_threshold:
                        fallback_fg = self._bbox_fallback_mask(image_rgb, bbox)
                        mask = mask | fallback_fg

                except Exception as e:
                    logger.warning(
                        f"SAM2 predict failed for '{class_name}': {e}. "
                        f"Using refined fallback."
                    )
                    mask = self._bbox_fallback_mask(image_rgb, bbox)
                    warnings.append(f"SAM2 prediction failed: {e}")
            elif self._available and getattr(self, "_use_hf_sam", False):
                try:
                    import torch
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(image_rgb)
                    # input_boxes shape: [[[x1, y1, x2, y2]]]
                    input_boxes = [[[float(x1), float(y1), float(x2), float(y2)]]]
                    inputs = self._hf_processor(pil_img, input_boxes=input_boxes, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self._hf_model(**inputs)
                    masks_pred = self._hf_processor.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
                    )
                    # masks_pred[0] shape: (1, 3, H, W)
                    candidate_masks = [masks_pred[0][0, i].numpy().astype(bool) for i in range(masks_pred[0].shape[1])]
                    valid_candidates = [m for m in candidate_masks if (m.sum() / bbox_area) >= self.min_mask_threshold]
                    mask = min(valid_candidates, key=lambda m: m.sum()) if valid_candidates else max(candidate_masks, key=lambda m: m.sum())
                    if (mask.sum() / bbox_area) < self.min_mask_threshold:
                        mask = mask | self._bbox_fallback_mask(image_rgb, bbox)
                except Exception as e:
                    logger.warning(f"HF SAM predict failed: {e}. Using refined fallback.")
                    mask = self._bbox_fallback_mask(image_rgb, bbox)
                    warnings.append(f"SAM prediction failed: {e}")
            else:
                mask = self._bbox_fallback_mask(image_rgb, bbox)
                warnings.append("Neural segmentation not available — using refined GrabCut/ellipse mask")

            # Post-processing: boundary erosion
            mask = erode_boundary(mask, self.erode_kernel)

            # Post-processing: flag oversized masks
            if flag_oversized_mask(mask, bbox_area, self.oversized_threshold):
                warnings.append(
                    f"Mask for '{class_name}' covers >{self.oversized_threshold*100:.0f}% "
                    f"of bbox — plate/bowl rim may be included."
                )

            results.append(SegmentationResult(
                mask=mask,
                class_name=class_name,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=confidence,
                warnings=warnings,
            ))
            all_masks.append(mask)
            all_class_names.append(class_name)

        # Post-processing: check for merged masks between different classes
        merge_warnings = check_mask_multiplicity(all_masks, all_class_names)
        if merge_warnings:
            for result in results:
                result.warnings.extend(merge_warnings)

        if self._available and hasattr(self, "_predictor") and self._predictor is not None:
            if hasattr(self._predictor, "reset_predictor"):
                try:
                    self._predictor.reset_predictor()
                except Exception:
                    pass

        return results

    def visualize_masks(
        self,
        image_rgb: np.ndarray,
        seg_results: List[SegmentationResult],
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Overlay colored masks on the original image for visualization.
        
        Returns image_rgb with semi-transparent mask overlays and labels.
        """
        overlay = image_rgb.copy()
        # Generate distinct colors for each detection
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 50),  (50, 200, 150),  (150, 50, 200),
        ]

        for i, seg in enumerate(seg_results):
            color = colors[i % len(colors)]
            mask_3ch = np.stack([seg.mask] * 3, axis=-1)
            color_mask = np.zeros_like(overlay)
            color_mask[:] = color
            overlay = np.where(mask_3ch, 
                              (overlay * (1 - alpha) + color_mask * alpha).astype(np.uint8),
                              overlay)

            # Draw bbox and label
            x1, y1, x2, y2 = seg.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{seg.class_name} ({seg.confidence:.0%})"
            cv2.putText(overlay, label, (x1, max(y1 - 8, 12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                       cv2.LINE_AA)

        return overlay
