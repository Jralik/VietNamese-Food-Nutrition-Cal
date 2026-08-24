"""
Volume → Mass → Nutrition Estimation Module.

Takes a depth map + food mask + scale → computes:
  1. Volume (cm³) via column integration
  2. Mass (g) via density database lookup
  3. Nutrition (kcal, protein, fat, ...) via per-100g scaling

Propagates uncertainty from density variance into mass and nutrition estimates.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NutritionEstimate:
    """Full estimation result for a single food item."""
    class_name: str
    
    # Volume
    volume_cm3: float
    volume_method: str              # "column_integration" | "bbox_approximation"
    
    # Mass
    mass_g: float
    mass_std_g: float               # uncertainty from density variance
    density_g_per_cm3: float
    
    # Nutrition (scaled to estimated mass)
    nutrition: Dict[str, float]     # Calories, Protein, Fat, Carbs, Saturates, Sugar, Salt
    nutrition_std: Dict[str, float] # uncertainty propagated from mass_std
    
    # Metadata
    estimation_method: str          # "aruco_calibrated" | "plate_heuristic" | "bbox_fallback"
    confidence: str                 # "high" | "medium" | "low"
    confidence_note: str
    warnings: list = field(default_factory=list)
    
    # Per-100g reference values (for comparison)
    nutrition_per_100g: Optional[Dict[str, float]] = None


def fit_plane_to_boundary(
    depth_map_mm: np.ndarray,
    food_mask: np.ndarray,
    border_width: int = 20,
) -> np.ndarray:
    """Fit a 3D reference plane Z = Ax + By + C to the table/plate surface around the food mask.
    
    Uses a separated outer ring buffer and an 80th-percentile offset to guarantee that the
    reference plane represents the supporting surface beneath the food (not slicing through it).
    """
    import cv2
    h, w = depth_map_mm.shape[:2]
    mask_u8 = (food_mask > 0).astype(np.uint8) * 255
    kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1))
    
    dil_in = cv2.dilate(mask_u8, kernel_in)
    dil_out = cv2.dilate(mask_u8, kernel_out)
    ring = (dil_out > 0) & (dil_in == 0)

    ys, xs = np.where(ring)
    zs = depth_map_mm[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    if len(zs) < 12:
        masked_d = depth_map_mm[food_mask.astype(bool)]
        valid_m = masked_d[masked_d > 0]
        median_z = float(np.median(valid_m)) if len(valid_m) > 0 else 0.0
        return np.full((h, w), median_z, dtype=np.float32)

    # Least squares plane fit: Z = A*x + B*y + C
    A_mat = np.column_stack([xs.astype(np.float32), ys.astype(np.float32), np.ones_like(xs, dtype=np.float32)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, zs.astype(np.float32), rcond=None)
    A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # Offset plane to supporting surface level (80th percentile of residuals)
    ring_res = zs - (A * xs + B * ys + C)
    plane_offset = float(np.percentile(ring_res, 80))

    # Memory-safe float32 plane generation
    row_y = np.arange(h, dtype=np.float32)[:, None] * np.float32(B) + np.float32(C + plane_offset)
    col_x = np.arange(w, dtype=np.float32)[None, :] * np.float32(A)
    plane_z = row_y + col_x
    return plane_z.astype(np.float32)


def compute_volume_bowl_container(
    depth_map_mm: Optional[np.ndarray],
    food_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mm_per_pixel: float,
    dish_type: str = "soup_bowl",
    K: Optional[np.ndarray] = None,
    scale_source: str = "unknown",
) -> Tuple[float, List[str]]:
    """Compute 3D volume for concave bowl containers (soup, noodles, porridge, rice).
    
    Models the bowl as a 3D truncated circular frustum/paraboloid with depth downwards.
    Measures top surface headroom Delta_Z = Z_food - Z_rim to calculate exact liquid/solid fill level.
    """
    warnings = []
    from pipeline_config import (
        PLATE_DIAMETER_HEURISTICS, BOWL_DEPTH_RATIO,
        BOWL_BASE_RATIO, BOWL_DEFAULT_FILL_RATIO,
    )

    x1, y1, x2, y2 = bbox
    bw_px = max(10, x2 - x1)
    bh_px = max(10, y2 - y1)
    
    # 1. Bowl Diameter and Dimensions (mm)
    expected_dia_mm = PLATE_DIAMETER_HEURISTICS.get(dish_type, 180.0)
    
    # If ArUco or calibrated scale is available, compute physical diameter from bbox
    if scale_source == "aruco_marker":
        d_major_px = max(bw_px, bh_px)
        D_rim_mm = float(d_major_px * mm_per_pixel)
        # Sanity bound: bowl diameter typically 100 - 280 mm
        D_rim_mm = float(np.clip(D_rim_mm, 100.0, 280.0))
    else:
        D_rim_mm = float(expected_dia_mm)

    R_rim_mm = D_rim_mm / 2.0
    R_base_mm = R_rim_mm * BOWL_BASE_RATIO
    H_bowl_mm = D_rim_mm * BOWL_DEPTH_RATIO

    # 2. Depth map analysis for fill level
    has_valid_depth = False
    delta_z = 0.0
    above_rim_volume_cm3 = 0.0

    if depth_map_mm is not None and food_mask.any():
        mask_bool = food_mask.astype(bool)
        masked_d = depth_map_mm[mask_bool]
        valid_d = masked_d[masked_d > 0]

        if len(valid_d) > 20:
            plane_z = fit_plane_to_boundary(depth_map_mm, food_mask)
            plane_d = plane_z[mask_bool]
            
            z_rim = float(np.median(plane_d))
            z_food = float(np.median(valid_d))
            delta_z = z_food - z_rim  # > 0 means soup is recessed below rim
            has_valid_depth = True

            # If food mounds above the rim (e.g. noodles/meat piled high)
            raw_heights_above_rim = plane_d - masked_d
            pos_mound = raw_heights_above_rim[raw_heights_above_rim > 0]
            if len(pos_mound) > 0.05 * len(raw_heights_above_rim):
                # Calculate per-pixel area
                if scale_source == "aruco_marker" and K is not None and K[0, 0] > 0 and K[1, 1] > 0:
                    px_area = (masked_d[raw_heights_above_rim > 0] ** 2) / (float(K[0, 0]) * float(K[1, 1]))
                else:
                    px_area = mm_per_pixel ** 2
                above_rim_volume_cm3 = float(np.sum(pos_mound * px_area) / 1000.0)

    # 3. Calculate bowl liquid/food fill height
    if has_valid_depth:
        if delta_z > 0:
            # Soup surface is recessed below the rim
            # Headroom is physically between 4mm and 20mm (at most 0.25 * H_bowl)
            headroom_mm = float(np.clip(delta_z, 4.0, 0.25 * H_bowl_mm))
            h_fill = H_bowl_mm - headroom_mm
        else:
            # Soup is filled to rim (or mounded above)
            h_fill = H_bowl_mm
    else:
        # Default typical fill level (~82% of bowl capacity)
        h_fill = H_bowl_mm * BOWL_DEFAULT_FILL_RATIO

    fill_fraction = h_fill / H_bowl_mm
    R_fill_mm = R_base_mm + (R_rim_mm - R_base_mm) * fill_fraction

    # 4. Truncated cone (frustum) volume: V = (pi * h / 3) * (R1^2 + R1*R2 + R2^2)
    frustum_mm3 = (np.pi * h_fill / 3.0) * (R_fill_mm**2 + R_fill_mm * R_base_mm + R_base_mm**2)
    bowl_volume_cm3 = float(frustum_mm3 / 1000.0)

    total_volume_cm3 = bowl_volume_cm3 + above_rim_volume_cm3

    # Sanity bounds for bowls: single serving soup bowl 200 - 1200 cm³
    total_volume_cm3 = float(np.clip(total_volume_cm3, 150.0, 1200.0))

    return total_volume_cm3, warnings


def compute_volume_small_ingredient(
    bbox: Tuple[int, int, int, int],
    mm_per_pixel: float,
    class_name: str,
) -> Tuple[float, List[str]]:
    """Compute 3D volume for small ingredients/garnishes (lime, chili, egg, pickles).
    
    Models object as a 3D tri-axial spheroid / wedge:
      V = (4/3) * pi * (w/2) * (h/2) * (t/2) * k_shape
    Inherits global scale and bounds physical maximums.
    """
    warnings = []
    x1, y1, x2, y2 = bbox
    w_mm = max(2.0, (x2 - x1) * mm_per_pixel)
    h_mm = max(2.0, (y2 - y1) * mm_per_pixel)

    # Thickness and shape factor
    if "Chanh" in class_name:
        # Lime slice / wedge
        t_mm = min(w_mm, h_mm) * 0.50
        k_shape = 0.55
        max_vol_cm3 = 25.0
    elif "Trung" in class_name:
        # Whole or half egg
        t_mm = min(w_mm, h_mm) * 0.70
        k_shape = 0.75
        max_vol_cm3 = 60.0
    elif "Ot" in class_name:
        # Chili / pepper piece
        t_mm = min(w_mm, h_mm) * 0.40
        k_shape = 0.50
        max_vol_cm3 = 15.0
    elif "Ca chua" in class_name or "Dua leo" in class_name:
        # Tomato or cucumber slice
        t_mm = min(15.0, min(w_mm, h_mm) * 0.35)
        k_shape = 0.65
        max_vol_cm3 = 35.0
    else:
        # Generic small side garnish
        t_mm = min(w_mm, h_mm) * 0.50
        k_shape = 0.60
        max_vol_cm3 = 30.0

    vol_mm3 = (4.0 / 3.0) * np.pi * (w_mm / 2.0) * (h_mm / 2.0) * (t_mm / 2.0) * k_shape
    vol_cm3 = float(vol_mm3 / 1000.0)
    vol_cm3 = float(np.clip(vol_cm3, 1.0, max_vol_cm3))

    return vol_cm3, warnings


def compute_volume_side_vegetables(
    depth_map_mm: Optional[np.ndarray],
    food_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mm_per_pixel: float,
) -> Tuple[float, List[str]]:
    """Compute volume for leafy vegetable side plates (Rau, Salad).
    
    Models vegetable mound with loose structure and depth height integration.
    """
    warnings = []
    x1, y1, x2, y2 = bbox
    mask_pixels = food_mask.sum()
    if mask_pixels == 0:
        mask_pixels = max(100, (x2 - x1) * (y2 - y1) * 0.70)

    area_mm2 = mask_pixels * (mm_per_pixel ** 2)

    # Height estimation
    mean_h_mm = 20.0
    if depth_map_mm is not None and food_mask.any():
        mask_bool = food_mask.astype(bool)
        plane_z = fit_plane_to_boundary(depth_map_mm, food_mask)
        raw_h = plane_z[mask_bool] - depth_map_mm[mask_bool]
        pos_h = raw_h[raw_h > 0]
        if len(pos_h) > 50:
            mean_h_mm = float(np.median(pos_h))
    
    # Clip vegetable pile height to realistic range (8 - 22 mm)
    mean_h_mm = float(np.clip(mean_h_mm, 8.0, 22.0))
    vol_cm3 = float((area_mm2 * mean_h_mm) / 1000.0)
    vol_cm3 = float(np.clip(vol_cm3, 20.0, 250.0))

    return vol_cm3, warnings


def compute_volume_column_integration(
    depth_map_mm: np.ndarray,
    food_mask: np.ndarray,
    mm_per_pixel: float,
    K: Optional[np.ndarray] = None,
    homography: Optional[np.ndarray] = None,
    scale_source: str = "unknown",
    reference_depth_mm: Optional[float] = None,
    max_height_mm: float = 40.0,
) -> float:
    """Compute food volume via column integration with 3D table plane fitting for flat plates.
    
    For each masked pixel, the food height is the difference between
    the plate surface plane and the food surface depth: h(x,y) = Z_plane(x,y) - Z(x,y).
    """
    if food_mask.sum() == 0:
        return 0.0

    mask_bool = food_mask.astype(bool)
    masked_depths = depth_map_mm[mask_bool]
    valid_depths = masked_depths[masked_depths > 0]
    if len(valid_depths) == 0:
        return 0.0

    # 1. Plane fitting for plate reference surface
    if reference_depth_mm is not None:
        plane_depths = np.full(masked_depths.shape, reference_depth_mm, dtype=np.float32)
    else:
        plane_z = fit_plane_to_boundary(depth_map_mm, food_mask)
        plane_depths = plane_z[mask_bool]

    # 2. Food height above plate = Plate plane depth - Food surface depth
    raw_heights = plane_depths - masked_depths
    food_heights_mm = np.clip(raw_heights, 2.0, max_height_mm)

    # If almost no positive heights (planar or tilt issue), use default mound height
    if (food_heights_mm > 0).mean() < 0.08:
        food_heights_mm = np.full_like(food_heights_mm, min(18.0, max_height_mm * 0.5))

    # 3. Per-pixel area computation
    # mm_per_pixel is directly calibrated on the table plane (from ArUco marker or dish heuristic)
    if mm_per_pixel > 0:
        median_z = float(np.median(valid_depths))
        if median_z > 0:
            # Perspective-aware pixel area relative to object median distance
            pixel_area_mm2 = (mm_per_pixel * (masked_depths / median_z)) ** 2
        else:
            pixel_area_mm2 = mm_per_pixel ** 2
    elif K is not None and K[0, 0] > 0 and K[1, 1] > 0:
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        pixel_area_mm2 = (masked_depths ** 2) / (fx * fy)
    else:
        pixel_area_mm2 = 0.25 ** 2

    volume_mm3 = float(np.sum(food_heights_mm * pixel_area_mm2))
    volume_cm3 = volume_mm3 / 1000.0

    return volume_cm3


def compute_volume_bbox_approximation(
    bbox: Tuple[int, int, int, int],
    mm_per_pixel: float,
    estimated_height_mm: float = 25.0,
    fill_factor: float = 0.55,
) -> float:
    """Rough volume estimate from bbox dimensions when depth is unavailable."""
    x1, y1, x2, y2 = bbox
    width_mm = (x2 - x1) * mm_per_pixel
    height_mm = (y2 - y1) * mm_per_pixel

    a = width_mm / 2.0
    b = height_mm / 2.0
    c = estimated_height_mm / 2.0

    volume_mm3 = (4.0 / 3.0) * np.pi * a * b * c * fill_factor
    volume_cm3 = volume_mm3 / 1000.0

    return volume_cm3


class VolumeNutritionEstimator:
    """Estimate volume, mass, and nutrition from depth + mask + scale.
    
    Integrates density_db.py for density → mass → nutrition conversion.
    """

    def __init__(self):
        from density_db import get_density, get_nutrition_per_100g, estimate_nutrition
        self._get_density = get_density
        self._get_nutrition_per_100g = get_nutrition_per_100g
        self._estimate_nutrition = estimate_nutrition

    def estimate(
        self,
        depth_map_mm: Optional[np.ndarray],
        food_mask: np.ndarray,
        class_name: str,
        mm_per_pixel: float,
        scale_source: str = "unknown",
        scale_confidence: str = "low",
        bbox: Optional[Tuple[int, int, int, int]] = None,
        K: Optional[np.ndarray] = None,
        homography: Optional[np.ndarray] = None,
        max_height_mm: Optional[float] = None,
    ) -> NutritionEstimate:
        """Full estimation: depth + mask → volume → mass → nutrition.
        
        Dispatches to container-aware estimation based on dish category.
        """
        warnings = []
        from pipeline_config import CLASS_TO_DISH_TYPE

        dish_type = CLASS_TO_DISH_TYPE.get(class_name, "default")

        # ── Non-Food Handling ──
        if dish_type == "non_food":
            return NutritionEstimate(
                class_name=class_name,
                volume_cm3=0.0,
                volume_method="non_food",
                mass_g=0.0,
                mass_std_g=0.0,
                density_g_per_cm3=0.0,
                nutrition={"Calories": 0.0, "Protein": 0.0, "Fat": 0.0, "Carbs": 0.0, "Saturates": 0.0, "Sugar": 0.0, "Salt": 0.0},
                nutrition_std={"Calories": 0.0, "Protein": 0.0, "Fat": 0.0, "Carbs": 0.0, "Saturates": 0.0, "Sugar": 0.0, "Salt": 0.0},
                estimation_method="non_food",
                confidence="high",
                confidence_note="Đối tượng không phải thực phẩm",
                warnings=[],
                nutrition_per_100g=None,
            )

        # ── Ensure mask and depth map are aligned ──
        if depth_map_mm is not None and food_mask.any() and depth_map_mm.shape[:2] != food_mask.shape[:2]:
            import cv2
            food_mask = cv2.resize(
                food_mask.astype(np.uint8),
                (depth_map_mm.shape[1], depth_map_mm.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        has_valid_depth = (
            depth_map_mm is not None
            and food_mask.any()
            and (depth_map_mm[food_mask.astype(bool)] > 0).any()
        )

        # ── Local Plate Scale for Container Dishes (Heuristic Mode) ──
        # When no ArUco marker is present, each plate/bowl has its own local perspective scale
        from pipeline_config import CONTAINER_DISH_TYPES, PLATE_DIAMETER_HEURISTICS
        if scale_source == "plate_heuristic" and dish_type in CONTAINER_DISH_TYPES and bbox is not None:
            expected_dia = PLATE_DIAMETER_HEURISTICS.get(dish_type, 220.0)
            bw = max(10, bbox[2] - bbox[0])
            bh = max(10, bbox[3] - bbox[1])
            max_dim = max(bw, bh)
            if max_dim > 20:
                mm_per_pixel = expected_dia / float(max_dim)

        # ── Category-Aware Volume Estimation ──
        if dish_type in {"soup_bowl", "large_bowl", "rice_bowl"} and bbox is not None:
            # 1. 3D Bowl Container Model
            volume_cm3, bowl_warns = compute_volume_bowl_container(
                depth_map_mm=depth_map_mm,
                food_mask=food_mask,
                bbox=bbox,
                mm_per_pixel=mm_per_pixel,
                dish_type=dish_type,
                K=K,
                scale_source=scale_source,
            )
            volume_method = "bowl_container"
            warnings.extend(bowl_warns)

        elif dish_type == "ingredient_small" and bbox is not None:
            # 2. 3D Small Ingredient / Spheroid Model
            volume_cm3, ingr_warns = compute_volume_small_ingredient(
                bbox=bbox,
                mm_per_pixel=mm_per_pixel,
                class_name=class_name,
            )
            volume_method = "spheroid_model"
            warnings.extend(ingr_warns)

        elif dish_type == "side_vegetables" and bbox is not None:
            # 3. Side Leafy Vegetable Mound Model
            volume_cm3, veg_warns = compute_volume_side_vegetables(
                depth_map_mm=depth_map_mm,
                food_mask=food_mask,
                bbox=bbox,
                mm_per_pixel=mm_per_pixel,
            )
            volume_method = "vegetable_mound"
            warnings.extend(veg_warns)

        elif has_valid_depth:
            # 4. Standard Plate Column Integration
            if max_height_mm is None:
                height_limits = {
                    "flat_plate":  26.0,
                    "small_plate": 20.0,
                    "default":     30.0,
                }
                if "Banh mi" in class_name or "Hamburger" in class_name:
                    max_height_mm = 35.0
                else:
                    max_height_mm = height_limits.get(dish_type, 26.0)

            volume_cm3 = compute_volume_column_integration(
                depth_map_mm=depth_map_mm,
                food_mask=food_mask,
                mm_per_pixel=mm_per_pixel,
                K=K,
                homography=homography,
                scale_source=scale_source,
                max_height_mm=max_height_mm,
            )
            volume_method = "column_integration"

        elif bbox is not None:
            # 5. Bbox fallback
            volume_cm3 = compute_volume_bbox_approximation(bbox, mm_per_pixel)
            volume_method = "bbox_approximation"
            warnings.append("Depth data unavailable — volume from bbox approximation")
        else:
            volume_cm3 = 0.0
            volume_method = "unavailable"
            warnings.append("Neither depth data nor bbox available — volume = 0")

        # Sanity clamp
        if volume_cm3 < 0.5 and volume_method != "unavailable":
            warnings.append(f"Volume suspiciously small ({volume_cm3:.2f} cm³)")
        if volume_cm3 > 2500:
            warnings.append(f"Volume suspiciously large ({volume_cm3:.1f} cm³) — clamped to 2500")
            volume_cm3 = min(volume_cm3, 2500.0)

        # ── Density → Mass ──
        try:
            density_mean, density_std = self._get_density(class_name)
            mass_g = volume_cm3 * density_mean
            mass_std_g = volume_cm3 * density_std
        except KeyError as e:
            logger.warning(str(e))
            density_mean = 0.8
            density_std = 0.2
            mass_g = volume_cm3 * density_mean
            mass_std_g = volume_cm3 * density_std
            warnings.append(f"No density data for '{class_name}' — using generic 0.8±0.2 g/cm³")

        # ── Nutrition ──
        try:
            nutrition = self._estimate_nutrition(mass_g, class_name)
            nutrition_per_100g = self._get_nutrition_per_100g(class_name)
        except KeyError as e:
            logger.warning(str(e))
            nutrition = {}
            nutrition_per_100g = None
            warnings.append(f"No nutrition data for '{class_name}'")

        # Propagate mass uncertainty into nutrition
        nutrition_std = {}
        if nutrition and mass_g > 0:
            uncertainty_ratio = mass_std_g / mass_g
            nutrition_std = {k: round(v * uncertainty_ratio, 1) for k, v in nutrition.items()}

        # ── Determine overall estimation method & confidence ──
        if scale_source == "aruco_marker":
            estimation_method = "aruco_calibrated"
            confidence = "high"
        elif volume_method in {"bowl_container", "column_integration", "vegetable_mound", "spheroid_model"}:
            estimation_method = "plate_heuristic"
            confidence = "medium"
        else:
            estimation_method = "bbox_fallback"
            confidence = "low"

        # ── Confidence note ──
        confidence_note = self._build_confidence_note(
            estimation_method, scale_source, volume_method, warnings
        )

        return NutritionEstimate(
            class_name=class_name,
            volume_cm3=round(volume_cm3, 2),
            volume_method=volume_method,
            mass_g=round(mass_g, 1),
            mass_std_g=round(mass_std_g, 1),
            density_g_per_cm3=density_mean,
            nutrition=nutrition,
            nutrition_std=nutrition_std,
            estimation_method=estimation_method,
            confidence=confidence,
            confidence_note=confidence_note,
            warnings=warnings,
            nutrition_per_100g=nutrition_per_100g,
        )

    def _build_confidence_note(
        self,
        estimation_method: str,
        scale_source: str,
        volume_method: str,
        warnings: list,
    ) -> str:
        """Build a human-readable confidence note for the UI."""
        if estimation_method == "aruco_calibrated":
            note = (
                "Ước lượng chính xác cao — sử dụng ArUco marker và depth map. "
                "Sai số dự kiến: ~15-25%."
            )
        elif estimation_method == "plate_heuristic":
            if volume_method == "bowl_container":
                note = (
                    "Ước lượng bát 3D chuẩn hóa — sử dụng kích thước bát và đo mức đầy từ depth map. "
                    "Sai số dự kiến: ~20-35%."
                )
            else:
                note = (
                    "Ước lượng gần đúng — sử dụng kích thước đĩa/bát ước lượng. "
                    "Sai số dự kiến: ~25-40%. Để chính xác hơn, đặt ArUco marker cạnh đĩa."
                )
        else:
            note = (
                "Ước lượng sơ bộ — không có depth map hoặc reference scale. "
                "Sai số có thể lớn (>50%). Kết quả chỉ mang tính tham khảo."
            )

        if warnings:
            note += f" ({len(warnings)} cảnh báo)"

        return note
