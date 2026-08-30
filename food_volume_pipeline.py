"""
Food Volume Estimation Pipeline — Orchestrator.

Single entry-point that chains:
  YOLO detections → SAM2 segmentation → Depth Anything V2 → Scale recovery
  → Volume → Mass → Nutrition

Usage:
    from food_volume_pipeline import FoodVolumePipeline

    pipeline = FoodVolumePipeline()
    results = pipeline.analyze(image_rgb, yolo_detections)
    for r in results:
        print(f"{r.class_name}: {r.volume_cm3} cm³, {r.mass_g}g, "
              f"{r.nutrition.get('Calories', 0)} kcal")
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from food_segmentation import SAM2Segmenter, SegmentationResult
from depth_estimation import DepthEstimator, DepthResult, estimate_camera_intrinsics
from scale_recovery import ScaleRecovery, ScaleResult
from volume_nutrition import VolumeNutritionEstimator, NutritionEstimate

logger = logging.getLogger(__name__)


@dataclass
class FoodEstimation:
    """Complete estimation result for a single food item."""
    # Detection info
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    # Segmentation
    mask: np.ndarray                # binary mask (H, W)
    seg_warnings: List[str]

    # Depth
    depth_crop_mm: Optional[np.ndarray]  # depth values within mask region

    # Volume / Mass / Nutrition
    volume_cm3: float
    mass_g: float
    mass_std_g: float
    nutrition: Dict[str, float]
    nutrition_std: Dict[str, float]

    # Metadata
    estimation_method: str          # "aruco_calibrated" | "plate_heuristic" | "bbox_fallback"
    confidence_level: str           # "high" | "medium" | "low"
    confidence_note: str
    warnings: List[str] = field(default_factory=list)

    # Per-100g reference
    nutrition_per_100g: Optional[Dict[str, float]] = None


@dataclass
class PipelineResult:
    """Full pipeline output for an image."""
    estimations: List[FoodEstimation]
    scale_result: ScaleResult
    depth_result: DepthResult
    seg_results: List[SegmentationResult]

    # Aggregate nutrition
    total_nutrition: Dict[str, float] = field(default_factory=dict)
    total_nutrition_std: Dict[str, float] = field(default_factory=dict)

    # Visualizations (populated on demand)
    mask_overlay: Optional[np.ndarray] = None
    depth_colored: Optional[np.ndarray] = None
    scale_overlay: Optional[np.ndarray] = None


class FoodVolumePipeline:
    """End-to-end food volume estimation pipeline.
    
    Chains SAM2 segmentation, Depth Anything V2, scale recovery,
    and volume/mass/nutrition estimation.
    
    All sub-modules are lazy-loaded on first use.
    """

    def __init__(self):
        self._segmenter: Optional[SAM2Segmenter] = None
        self._depth_estimator: Optional[DepthEstimator] = None
        self._scale_recovery: Optional[ScaleRecovery] = None
        self._volume_estimator: Optional[VolumeNutritionEstimator] = None

    @property
    def segmenter(self) -> SAM2Segmenter:
        if self._segmenter is None:
            self._segmenter = SAM2Segmenter()
        return self._segmenter

    @property
    def depth_estimator(self) -> DepthEstimator:
        if self._depth_estimator is None:
            self._depth_estimator = DepthEstimator()
        return self._depth_estimator

    @property
    def scale_recovery(self) -> ScaleRecovery:
        if self._scale_recovery is None:
            self._scale_recovery = ScaleRecovery()
        return self._scale_recovery

    @property
    def volume_estimator(self) -> VolumeNutritionEstimator:
        if self._volume_estimator is None:
            self._volume_estimator = VolumeNutritionEstimator()
        return self._volume_estimator

    def analyze(
        self,
        image_rgb: np.ndarray,
        yolo_detections: List[Dict],
        image_path: Optional[str] = None,
        generate_visualizations: bool = False,
        seg_image_rgb: Optional[np.ndarray] = None,
        seg_yolo_detections: Optional[List[Dict]] = None,
    ) -> PipelineResult:
        """Run the full pipeline on an image with YOLO detections.
        
        Args:
            image_rgb: Input image, shape (H, W, 3), uint8, RGB format
            yolo_detections: List of dicts, each with:
                - "bbox": (x1, y1, x2, y2)  — pixel coords
                - "class_name": str
                - "confidence": float (0-1)
            image_path: Optional path to the image file (for EXIF reading)
            generate_visualizations: If True, create overlay images
            seg_image_rgb: Optional separate image (e.g. the app's 640x640
                display copy) to run SAM2 on; its masks/bboxes are rescaled
                back onto `image_rgb` coordinates. Useful when the aspect-
                preserving source starves SAM2's fixed internal resolution of
                object pixels.
            seg_yolo_detections: Detections matching seg_image_rgb coordinates.

        Returns:
            PipelineResult with all estimations and metadata.
        """
        h, w = image_rgb.shape[:2]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        logger.info(f"Pipeline: processing {len(yolo_detections)} detections "
                    f"on {w}×{h} image")

        # ── Step 1: Camera intrinsics ──
        from pipeline_config import CAMERA_CALIB_FILE
        K, dist = estimate_camera_intrinsics(
            image_shape=(h, w),
            image_path=image_path,
            calib_file=CAMERA_CALIB_FILE,
        )

        # ── Step 2: SAM2 Segmentation ──
        # Segment on seg_image_rgb (if provided), then lift the masks back
        # onto the main image's coordinate grid so depth/scale integration
        # stays aligned.
        if seg_image_rgb is not None and seg_yolo_detections is not None:
            seg_h, seg_w = seg_image_rgb.shape[:2]
            logger.info(f"Step 2: SAM2 segmentation on {seg_w}x{seg_h} input...")
            seg_results = self.segmenter.segment(seg_image_rgb, seg_yolo_detections)
            sx, sy = w / float(seg_w), h / float(seg_h)
            for seg in seg_results:
                seg.bbox = (int(seg.bbox[0] * sx), int(seg.bbox[1] * sy),
                            int(seg.bbox[2] * sx), int(seg.bbox[3] * sy))
                seg.mask = cv2.resize(
                    seg.mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        else:
            logger.info("Step 2: SAM2 segmentation...")
            seg_results = self.segmenter.segment(image_rgb, yolo_detections)

        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── Step 3: Depth Estimation ──
        logger.info("Step 3: Depth Anything V2...")
        depth_result = self.depth_estimator.estimate(image_rgb)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── Step 4: Scale Recovery ──
        logger.info("Step 4: Scale recovery...")
        detected_classes = [d["class_name"] for d in yolo_detections]
        scale_result = self.scale_recovery.recover(
            image_bgr,
            K=K,
            dist=dist,
            detected_classes=detected_classes,
            yolo_detections=yolo_detections,
            seg_results=seg_results,
        )
        logger.info(
            f"Scale: {scale_result.mm_per_pixel:.4f} mm/px "
            f"(source={scale_result.scale_source}, "
            f"confidence={scale_result.confidence})"
        )

        # ── Step 4.5: Anchor Metric Depth to Board Ground-Truth Distance ──
        # Both the ArUco marker (black square = 30mm) and the checkerboard
        # target (30mm squares) yield a solvePnP tvec — the true distance to
        # the board plane — used to remove the metric depth model's bias.
        final_depth_mm = depth_result.depth_mm.copy() if depth_result.depth_mm is not None else None
        if (
            scale_result.scale_source in ("aruco_marker", "checkerboard")
            and scale_result.marker_tvec is not None
            and scale_result.marker_corners is not None
            and final_depth_mm is not None
        ):
            z_marker_true = float(np.ravel(scale_result.marker_tvec)[2])
            corners = scale_result.marker_corners
            
            # Robust polygon median depth sampling across the marker
            marker_mask = np.zeros(final_depth_mm.shape[:2], dtype=np.uint8)
            cv2.fillPoly(marker_mask, [corners.astype(np.int32)], 1)
            marker_depths = final_depth_mm[marker_mask > 0]
            valid_md = marker_depths[marker_depths > 0]
            
            if len(valid_md) > 0:
                d_pred = float(np.median(valid_md))
            else:
                cx = int(np.clip(np.mean(corners[:, 0]), 0, w - 1))
                cy = int(np.clip(np.mean(corners[:, 1]), 0, h - 1))
                d_pred = float(final_depth_mm[cy, cx])
                
            if d_pred > 0 and z_marker_true > 0:
                # Multiplicative depth anchoring: the metric model's bias is
                # primarily multiplicative (Z_meas = b * Z_true), so scaling
                # by z_true/d_pred corrects the absolute distance AND the
                # object height relief in one step (additive anchoring would
                # leave the relief biased).
                scale_anchor = z_marker_true / d_pred
                # Food close-ups sit far outside the metric model's training
                # distribution — measured bias reached 2.5x (model placed the
                # table at 1005 mm vs a true 398 mm). Allow the full correction
                # instead of clamping it away.
                scale_anchor = float(np.clip(scale_anchor, 0.3, 3.0))
                final_depth_mm = np.clip(final_depth_mm * scale_anchor, 100.0, 5000.0)
                scale_result.depth_anchored = True
                logger.info(
                    f"Depth anchored to ArUco marker: z_true={z_marker_true:.1f}mm, "
                    f"d_pred={d_pred:.1f}mm -> scale x{scale_anchor:.3f}"
                )

        # ── Step 5: Volume → Mass → Nutrition for each detection ──
        logger.info("Step 5: Volume/mass/nutrition estimation...")
        estimations = []

        for seg in seg_results:
            # Get per-dish scale (specifically when using plate heuristics)
            dish_scale = self.scale_recovery.get_dish_scale(
                bbox=seg.bbox,
                class_name=seg.class_name,
                global_scale=scale_result
            )
            nutrition_est = self.volume_estimator.estimate(
                depth_map_mm=final_depth_mm,
                food_mask=seg.mask,
                class_name=seg.class_name,
                mm_per_pixel=dish_scale,
                scale_source=scale_result.scale_source,
                scale_confidence=scale_result.confidence,
                bbox=seg.bbox,
                K=K,
                homography=scale_result.homography,
                depth_anchored=scale_result.depth_anchored,
            )

            # Extract depth values within mask for visualization
            depth_crop = None
            if depth_result.depth_mm is not None and seg.mask.any():
                masked_depth = depth_result.depth_mm.copy()
                masked_depth[~seg.mask.astype(bool)] = 0
                depth_crop = masked_depth

            # Combine all warnings
            all_warnings = seg.warnings + nutrition_est.warnings
            if scale_result.notes:
                all_warnings.extend(
                    [n for n in scale_result.notes if "⚠" in n]
                )

            estimations.append(FoodEstimation(
                class_name=seg.class_name,
                confidence=seg.confidence,
                bbox=seg.bbox,
                mask=seg.mask,
                seg_warnings=seg.warnings,
                depth_crop_mm=depth_crop,
                volume_cm3=nutrition_est.volume_cm3,
                mass_g=nutrition_est.mass_g,
                mass_std_g=nutrition_est.mass_std_g,
                nutrition=nutrition_est.nutrition,
                nutrition_std=nutrition_est.nutrition_std,
                estimation_method=nutrition_est.estimation_method,
                confidence_level=nutrition_est.confidence,
                confidence_note=nutrition_est.confidence_note,
                warnings=all_warnings,
                nutrition_per_100g=nutrition_est.nutrition_per_100g,
            ))

        # ── Aggregate totals ──
        total_nutrition: Dict[str, float] = {}
        total_nutrition_std: Dict[str, float] = {}
        for est in estimations:
            for key, value in est.nutrition.items():
                total_nutrition[key] = total_nutrition.get(key, 0) + value
            for key, value in est.nutrition_std.items():
                # Propagate std: sqrt(sum of squares)
                current = total_nutrition_std.get(key, 0)
                total_nutrition_std[key] = (current**2 + value**2) ** 0.5

        # Round totals
        total_nutrition = {k: round(v, 1) for k, v in total_nutrition.items()}
        total_nutrition_std = {k: round(v, 1) for k, v in total_nutrition_std.items()}

        # ── Build result ──
        result = PipelineResult(
            estimations=estimations,
            scale_result=scale_result,
            depth_result=depth_result,
            seg_results=seg_results,
            total_nutrition=total_nutrition,
            total_nutrition_std=total_nutrition_std,
        )

        # ── Generate visualizations if requested ──
        if generate_visualizations:
            result.mask_overlay = self.segmenter.visualize_masks(
                image_rgb, seg_results
            )
            result.depth_colored = self.depth_estimator.colorize_depth(
                depth_result.depth_mm
            )
            result.scale_overlay = self.scale_recovery.draw_aruco_overlay(
                image_bgr, scale_result
            )

        logger.info(
            f"Pipeline complete: {len(estimations)} items, "
            f"total ~{total_nutrition.get('Calories', 0):.0f} kcal"
        )

        return result

    def format_summary(self, result: PipelineResult) -> str:
        """Format pipeline result as a human-readable summary string."""
        lines = [
            "═" * 60,
            "  FOOD VOLUME & NUTRITION ESTIMATION RESULTS",
            "═" * 60,
            f"  Scale: {result.scale_result.mm_per_pixel:.4f} mm/px "
            f"({result.scale_result.scale_source})",
            f"  Depth: {result.depth_result.depth_source} "
            f"(encoder={result.depth_result.encoder})",
            "─" * 60,
        ]

        for i, est in enumerate(result.estimations, 1):
            lines.append(
                f"\n  [{i}] {est.class_name} "
                f"(conf={est.confidence:.0%})"
            )
            lines.append(f"      Volume:  {est.volume_cm3:.1f} cm³")
            lines.append(f"      Mass:    {est.mass_g:.1f} ± {est.mass_std_g:.1f} g")
            lines.append(f"      Method:  {est.estimation_method}")

            if est.nutrition:
                cal = est.nutrition.get('Calories', 0)
                cal_std = est.nutrition_std.get('Calories', 0)
                prot = est.nutrition.get('Protein', 0)
                fat = est.nutrition.get('Fat', 0)
                carb = est.nutrition.get('Carbs', 0)
                lines.append(
                    f"      Nutrition: ~{cal:.0f}±{cal_std:.0f} kcal | "
                    f"P:{prot:.1f}g | F:{fat:.1f}g | C:{carb:.1f}g"
                )

            if est.warnings:
                for w in est.warnings[:3]:  # show max 3 warnings
                    lines.append(f"      ⚠ {w}")

        lines.append("─" * 60)
        total = result.total_nutrition
        total_std = result.total_nutrition_std
        lines.append(
            f"  TOTAL: ~{total.get('Calories', 0):.0f}"
            f"±{total_std.get('Calories', 0):.0f} kcal"
        )
        lines.append("═" * 60)

        return "\n".join(lines)
