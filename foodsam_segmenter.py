"""
FoodSAM Segmentation Module — subprocess bridge to the FoodSAM stack.

Runs FoodSAM (SAM2.1 AMG + SETR-MLA FoodSeg103 semantic) inside its ISOLATED
environment (`FoodSAM\\env`), one image per invocation, and maps results back
onto the pipeline's SegmentationResult contract. The main .venv gains NO new
dependency — the heavy stack lives entirely in the FoodSAM env, mirroring the
volume_worker isolation pattern.

Results returned by segment():
  - one SegmentationResult per YOLO detection (dish mask = SETR semantic core
    + food-labeled SAM masks, closing + hole fill)
  - PLUS one SegmentationResult per accepted ingredient component
    (source="foodsam_ingredient") — components are NEVER merged before
    volume estimation (spheroid/mound heuristics are nonlinear in mask size)

Class identity always comes from YOLO for dishes; ingredient items take
their mapped VietFood68 class (pipeline_config.FOODSAM_INGREDIENT_MAP).

Fallback: when the FoodSAM env or infer script is missing, or the subprocess
fails/times out, dish detections fall back to refined GrabCut/ellipse masks
(same as SAM2Segmenter) and `fallback_used` is set — the batch runner treats
a fallback as a hard error so research results are never silently degraded.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import List, Tuple

import cv2
import numpy as np

from food_segmentation import (
    SegmentationResult,
    check_mask_multiplicity,
    erode_boundary,
    flag_oversized_mask,
)

logger = logging.getLogger(__name__)


class FoodSAMSegmenter:
    """Segment food dishes + ingredient components with the FoodSAM stack."""

    # Receives the SOURCE-resolution image + full-res detections: its SAM2
    # adapter caps the AMG input internally (aspect-preserving, <=1024) and
    # SETR needs full-res semantic detail — the 640 display copy used by the
    # SAM2 backend destroys small food regions on large portraits.
    prefers_full_res = True

    def __init__(
        self,
        erode_kernel: int = None,
        oversized_threshold: float = None,
        points_per_side: int = None,
        timeout_s: int = None,
    ):
        from pipeline_config import (
            FOODSAM_ERODE_KERNEL_SIZE,
            FOODSAM_INFER_SCRIPT,
            FOODSAM_INGREDIENT_LARGE_AREA_FLAG,
            FOODSAM_INGREDIENT_MAP,
            FOODSAM_INGREDIENT_MAX_AREA_RATIO,
            FOODSAM_INGREDIENT_MIN_AREA_RATIO,
            FOODSAM_INGREDIENT_MIN_PURITY,
            FOODSAM_INGREDIENT_MODE,
            FOODSAM_INGREDIENT_OVERLAP_BREAKDOWN,
            FOODSAM_INGREDIENT_OWNER_TIE_MARGIN,
            FOODSAM_INGREDIENT_PROMOTE_MIN_RATIO,
            FOODSAM_MIXED_PLATE_GROUP_MIN_RATIO,
            FOODSAM_MIXED_PLATE_MIN_BBOX_RATIO,
            FOODSAM_MIXED_PLATE_MIN_NONSTAPLE,
            FOODSAM_MIXED_PLATE_SUSPECT_CLASSES,
            FOODSAM_OVERSIZED_MASK_THRESHOLD,
            FOODSAM_POINTS_PER_SIDE,
            FOODSAM_REPO,
            FOODSAM_TIMEOUT_S,
        )

        self.env_python = os.path.join(FOODSAM_REPO, "env", "Scripts", "python.exe")
        self.infer_script = FOODSAM_INFER_SCRIPT
        self.points_per_side = points_per_side or FOODSAM_POINTS_PER_SIDE
        self.timeout_s = timeout_s or FOODSAM_TIMEOUT_S
        self.erode_kernel = erode_kernel or FOODSAM_ERODE_KERNEL_SIZE
        self.oversized_threshold = oversized_threshold or FOODSAM_OVERSIZED_MASK_THRESHOLD
        self.ingredient_mode = FOODSAM_INGREDIENT_MODE
        self.ingredient_map = FOODSAM_INGREDIENT_MAP
        self.suspect_classes = sorted(FOODSAM_MIXED_PLATE_SUSPECT_CLASSES)
        self.ingredient_args = {
            "overlap_breakdown": FOODSAM_INGREDIENT_OVERLAP_BREAKDOWN,
            "min_area_ratio": FOODSAM_INGREDIENT_MIN_AREA_RATIO,
            "large_area_flag": FOODSAM_INGREDIENT_LARGE_AREA_FLAG,
            "max_area_ratio": FOODSAM_INGREDIENT_MAX_AREA_RATIO,
            "min_purity": FOODSAM_INGREDIENT_MIN_PURITY,
            "owner_tie_margin": FOODSAM_INGREDIENT_OWNER_TIE_MARGIN,
            "promote_min_ratio": FOODSAM_INGREDIENT_PROMOTE_MIN_RATIO,
            "mixed_min_nonstaple": FOODSAM_MIXED_PLATE_MIN_NONSTAPLE,
            "mixed_group_min_ratio": FOODSAM_MIXED_PLATE_GROUP_MIN_RATIO,
            "mixed_min_bbox_ratio": FOODSAM_MIXED_PLATE_MIN_BBOX_RATIO,
        }
        self._available = None
        # populated per segment() call — batch runner fail-fast reads these
        self.last_info = {}
        self.fallback_used = False
        self.last_error = None

    @property
    def is_available(self) -> bool:
        """Cheap filesystem check — nothing is imported from the FoodSAM env
        in this process."""
        if self._available is None:
            ok = os.path.isfile(self.env_python) and os.path.isfile(self.infer_script)
            if not ok:
                logger.warning(
                    f"FoodSAM env not usable (python={self.env_python}, "
                    f"script={self.infer_script})")
            self._available = bool(ok)
        return self._available

    @staticmethod
    def _bbox_fallback_mask(image_rgb: np.ndarray,
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Refined GrabCut/ellipse mask inside the bbox (SAM2Segmenter parity)."""
        h, w = image_rgb.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        bw, bh = x2 - x1, y2 - y1
        mask = np.zeros((h, w), dtype=bool)
        if bw <= 4 or bh <= 4:
            mask[y1:y2, x1:x2] = True
            return mask
        try:
            crop = image_rgb[y1:y2, x1:x2]
            gc_mask = np.zeros(crop.shape[:2], np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            rect = (max(1, int(bw * 0.05)), max(1, int(bh * 0.05)),
                    int(bw * 0.9), int(bh * 0.9))
            cv2.grabCut(crop, gc_mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            fg = (gc_mask == 1) | (gc_mask == 3)
            refined = np.zeros((h, w), dtype=bool)
            refined[y1:y2, x1:x2] = fg
            if refined.sum() > 0.15 * (bw * bh):
                return refined
        except Exception:
            pass
        ellipse = np.zeros((h, w), np.uint8)
        cv2.ellipse(ellipse, ((x1 + x2) // 2, (y1 + y2) // 2),
                    (int(bw * 0.45), int(bh * 0.45)), 0, 0, 360, 255, -1)
        return ellipse.astype(bool)

    def segment(self, image_rgb: np.ndarray,
                detections: List[dict]) -> List[SegmentationResult]:
        """Segment dishes (one result per NON-suppressed detection) + ingredient
        components (one result per accepted component). The bridge runs even
        with zero detections — ingredient extraction is YOLO-independent
        (HomeCook). Dishes flagged suppressed_mixed_plate are removed from
        nutrition accounting; the full audit record goes to last_info."""
        self.fallback_used = False
        self.last_error = None

        dish_masks = [None] * len(detections)
        dish_infos = [{"warnings": []} for _ in detections]
        extras = []
        suppressed = []

        if self.is_available:
            dish_masks, dish_infos, extras, suppressed, bridge_fallback = \
                self._run_bridge(image_rgb, detections)
            if bridge_fallback:
                self.fallback_used = True

        suppressed_idx = {s["det_index"] for s in suppressed}

        results: List[SegmentationResult] = []

        # ---- dish results (one per NON-suppressed detection) ----
        for i, det in enumerate(detections):
            if i in suppressed_idx:
                continue  # mixed-plate contradiction — audit record only
            bbox = det["bbox"]
            class_name = det["class_name"]
            confidence = det.get("confidence", 0.0)
            warnings: List[str] = []
            x1, y1, x2, y2 = bbox
            bbox_area = max(1, (x2 - x1) * (y2 - y1))

            mask = dish_masks[i] if dish_masks[i] is not None else None
            info = dish_infos[i] or {}
            if mask is None:
                mask = self._bbox_fallback_mask(image_rgb, bbox)
                self.fallback_used = True
                warnings.append("FoodSAM mask unavailable — using refined "
                                "GrabCut/ellipse bbox mask")
            warnings.extend(info.get("warnings", []))

            pre_erode = mask
            mask = erode_boundary(mask, self.erode_kernel)
            if not mask.any() and pre_erode is not None and pre_erode.any():
                # tiny masks must survive erosion (never revert to a None slot)
                mask = pre_erode
            if flag_oversized_mask(mask, bbox_area, self.oversized_threshold):
                warnings.append(
                    f"Mask for '{class_name}' covers "
                    f">{self.oversized_threshold*100:.0f}% of bbox — plate/bowl "
                    f"rim may be included.")

            results.append(SegmentationResult(
                mask=mask,
                class_name=class_name,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=confidence,
                warnings=warnings,
                source="yolo",
                is_ingredient=False,
            ))

        merge_warnings = check_mask_multiplicity(
            [r.mask for r in results], [r.class_name for r in results])
        if merge_warnings:
            for r in results:
                r.warnings.extend(merge_warnings)

        # ---- ingredient extras (one result per component) ----
        for ex in extras:
            mask = ex["mask"]
            warnings = list(ex.get("warnings", []))
            purity = ex.get("semantic_purity")
            warnings.append(
                f"Thành phần FoodSAM ({ex.get('mapping_type', '')} map, "
                f"semantic_purity={purity if purity is not None else 'n/a'})")
            mask_eroded = erode_boundary(mask, self.erode_kernel)
            if mask_eroded.any():
                mask = mask_eroded
            x1, y1, x2, y2 = ex["bbox"]
            results.append(SegmentationResult(
                mask=mask,
                class_name=ex["class_name"],
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(purity) if purity is not None else 0.5,
                warnings=warnings,
                source="foodsam_ingredient",
                is_ingredient=True,
                component_id=ex.get("component_id", ""),
                semantic_purity=purity,
                mapping_type=ex.get("mapping_type", ""),
            ))

        return results

    def _run_bridge(self, image_rgb, detections):
        """Write image+detections+config to a temp dir, run foodsam_infer.py in
        the FoodSAM env, load dish masks + extras + suppression audit back.
        Returns (dish_masks, dish_infos, extras, suppressed, bridge_fallback)."""
        dish_masks = [None] * len(detections)
        dish_infos = [{"warnings": []} for _ in detections]
        extras: List[dict] = []
        suppressed: List[dict] = []
        bridge_fallback = False
        try:
            with tempfile.TemporaryDirectory(prefix="foodsam_") as tmp:
                img_path = os.path.join(tmp, "input.png")
                dets_path = os.path.join(tmp, "detections.json")
                map_path = os.path.join(tmp, "ingredient_map.json")
                suspect_path = os.path.join(tmp, "suspect_classes.json")
                out_dir = os.path.join(tmp, "out")
                cv2.imwrite(img_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                with open(dets_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [{"bbox": [int(d["bbox"][0]), int(d["bbox"][1]),
                                   int(d["bbox"][2]), int(d["bbox"][3])],
                          "class_name": d["class_name"],
                          "confidence": float(d.get("confidence", 0.0))}
                         for d in detections],
                        f, ensure_ascii=False)
                with open(map_path, "w", encoding="utf-8") as f:
                    json.dump(self.ingredient_map, f, ensure_ascii=False)
                with open(suspect_path, "w", encoding="utf-8") as f:
                    json.dump(self.suspect_classes, f, ensure_ascii=False)

                ia = self.ingredient_args
                cmd = [
                    self.env_python, self.infer_script,
                    img_path, dets_path, out_dir,
                    "--pps", str(self.points_per_side),
                    "--ingredient-mode", self.ingredient_mode,
                    "--ingredient-map", map_path,
                    "--suspect-classes", suspect_path,
                    "--overlap-breakdown", str(ia["overlap_breakdown"]),
                    "--min-area-ratio", str(ia["min_area_ratio"]),
                    "--large-area-flag", str(ia["large_area_flag"]),
                    "--max-area-ratio", str(ia["max_area_ratio"]),
                    "--min-purity", str(ia["min_purity"]),
                    "--owner-tie-margin", str(ia["owner_tie_margin"]),
                    "--promote-min-ratio", str(ia["promote_min_ratio"]),
                    "--mixed-min-nonstaple", str(ia["mixed_min_nonstaple"]),
                    "--mixed-group-min-ratio", str(ia["mixed_group_min_ratio"]),
                    "--mixed-min-bbox-ratio", str(ia["mixed_min_bbox_ratio"]),
                ]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=self.timeout_s, cwd=os.path.dirname(self.infer_script))
                if proc.returncode != 0:
                    logger.warning(f"foodsam_infer failed:\n{proc.stderr[-2000:]}")
                    self.last_error = f"exit {proc.returncode}: {proc.stderr[-300:]}"
                    return dish_masks, dish_infos, extras, suppressed, True

                with open(os.path.join(out_dir, "result.json"), encoding="utf-8") as f:
                    payload = json.load(f)
                npz = np.load(os.path.join(out_dir, "masks.npz"))["masks"]
                extras_npz = np.load(os.path.join(out_dir, "extras.npz"))["extras"]

                suppressed = list(payload.get("suppressed", []))
                suppressed_idx = {s.get("det_index") for s in suppressed}
                for i, det_info in enumerate(payload["detections"]):
                    if i in suppressed_idx:
                        continue  # suppressed dish — audit record only, no mask
                    if det_info.get("bbox_coverage", 0) > 0 and i < npz.shape[0]:
                        dish_masks[i] = npz[i]
                        dish_infos[i] = {
                            "warnings": det_info.get("warnings", []),
                            "bbox_coverage": det_info.get("bbox_coverage"),
                            "foodsam_labels": det_info.get("foodsam_labels"),
                            "ingredient_breakdown":
                                det_info.get("ingredient_breakdown", []),
                        }
                for j, ex in enumerate(payload.get("extras", [])):
                    if j < extras_npz.shape[0] and extras_npz[j].any():
                        ex = dict(ex)
                        ex["mask"] = extras_npz[j]
                        extras.append(ex)
                self.last_info = {
                    k: payload.get(k) for k in
                    ("timing_s", "vram_peak_mib", "n_sam_masks",
                     "ingredient_extraction")}
                self.last_info["suppressed_dishes"] = suppressed
                logger.info(
                    f"FoodSAM bridge ok: {payload.get('n_sam_masks')} masks, "
                    f"{len(extras)} extras, {len(suppressed)} suppressed, "
                    f"{payload.get('timing_s')}")
        except subprocess.TimeoutExpired:
            self.last_error = f"timeout after {self.timeout_s}s"
            logger.warning(f"FoodSAM bridge timeout after {self.timeout_s}s")
            bridge_fallback = True
        except Exception as e:  # noqa: BLE001 — bridge must never kill the pipeline
            self.last_error = str(e)
            logger.warning(f"FoodSAM bridge failed: {e}")
            bridge_fallback = True
        return dish_masks, dish_infos, extras, suppressed, bridge_fallback

    def visualize_masks(self, image_rgb: np.ndarray,
                        seg_results: List[SegmentationResult],
                        alpha: float = 0.45) -> np.ndarray:
        """Overlay colored masks on the original image (SAM2Segmenter parity)."""
        overlay = image_rgb.copy()
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 50), (50, 200, 150), (150, 50, 200),
        ]
        for i, seg in enumerate(seg_results):
            color = colors[i % len(colors)]
            mask_3ch = np.stack([seg.mask] * 3, axis=-1)
            color_mask = np.zeros_like(overlay)
            color_mask[:] = color
            overlay = np.where(
                mask_3ch,
                (overlay * (1 - alpha) + color_mask * alpha).astype(np.uint8),
                overlay)
            x1, y1, x2, y2 = seg.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{seg.class_name} ({seg.confidence:.0%})"
            cv2.putText(overlay, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return overlay
