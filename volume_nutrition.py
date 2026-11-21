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

# Mean-height saturation cap, ADOPTED after the Phase-C experiment: when
# the depth model exaggerates relief, per-pixel heights saturate at
# max_height over nearly the whole mask and the integrated volume inflates
# (measured on ground truth: one 200 g loaf estimated 167/363/356 g across
# three views). With cap = 0.65 x max_height the same loaf reads
# 167/239/338 g (group CV 0.40 -> 0.23, MAPE 59% -> 30%) while the banh-mi
# (225 g) and bun-bo-hue (750 cm3) groups are unchanged. The env override
# VOLUME_SATURATION_CAP_RATIO remains for experiments.
SATURATION_CAP_RATIO = 0.65


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

    # Micronutrient convenience properties (in mg)
    @property
    def sodium_mg(self) -> float:
        return float(self.nutrition.get("Sodium", 0.0))

    @property
    def calcium_mg(self) -> float:
        return float(self.nutrition.get("Calcium", 0.0))

    @property
    def iron_mg(self) -> float:
        return float(self.nutrition.get("Iron", 0.0))

    @property
    def zinc_mg(self) -> float:
        return float(self.nutrition.get("Zinc", 0.0))

    @property
    def cholesterol_mg(self) -> float:
        return float(self.nutrition.get("Cholesterol", 0.0))


def relief_scale_factor(
    plane_z_median_mm: float,
    mm_per_pixel: float,
    K: Optional[np.ndarray],
) -> float:
    """Scale factor that re-anchors depth *relief* to the lateral mm/pixel scale.

    Monocular metric-depth models carry a multiplicative bias b (often 10-30%):
    Z_measured = b * Z_true. The lateral scale (mm/pixel from an ArUco marker
    or a plate of known diameter) is independent of that bias, so the true
    focal length f satisfies mm_per_pixel = Z_true_plane / f. Hence
      alpha = mm_per_pixel * f / Z_measured_plane = 1/b
    and corrected relief = measured relief * alpha. Alpha = 1 exactly when the
    metric depth is unbiased, so the correction is safe to apply uniformly.
    """
    if K is None or mm_per_pixel <= 0 or plane_z_median_mm <= 0:
        return 1.0
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 0 or fy <= 0:
        return 1.0
    f = (fx * fy) ** 0.5
    alpha = (mm_per_pixel * f) / plane_z_median_mm
    # Sanity: a wildly out-of-range factor means the plane/scale inputs are
    # inconsistent — prefer the raw metric depth over a bogus correction.
    return float(np.clip(alpha, 0.5, 2.0))


def fit_plane_to_boundary(
    depth_map_mm: np.ndarray,
    food_mask: np.ndarray,
    border_width: int = 10,
    stabilize_step: bool = True,
    alpha: float = 0.79,
) -> np.ndarray:
    """Fit a 3D reference plane Z = Ax + By + C to the table/plate surface around the food mask.

    Uses a separated outer ring buffer; the plane is offset by the MEDIAN ring
    residual so it represents the supporting surface beneath the food.

    When `stabilize_step=True` (A5), outer-ring boundary steps are checked against
     adjacent inner-mask depths. If the 85th-percentile step exceeds 40.0 mm
    (indicating a background occlusion shadow drop-off behind a tabletop loaf/plate),
    distant background steps are clamped locally while keeping all 360-degree
    (x, y) coordinates so the least-squares plane never extrapolates wildly.
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

    # Subsample dense rings uniformly around the 360-degree perimeter to keep
    # memory < 5 MiB and speed up NN step-stabilization on high-res masks.
    if len(zs) > 4000:
        r_step = max(1, len(zs) // 4000)
        xs, ys, zs = xs[::r_step], ys[::r_step], zs[::r_step]

    if stabilize_step:
        alpha_eff = max(0.5, float(alpha)) if alpha is not None else 0.79
        eroded = cv2.erode(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        inner_edge = (mask_u8 > 0) & (eroded == 0)
        iy, ix = np.where(inner_edge)
        iz = depth_map_mm[iy, ix]
        iv = iz > 0
        ix, iy, iz = ix[iv], iy[iv], iz[iv]

        if len(iz) >= 12:
            step = max(1, len(ix) // 360)
            ix_s, iy_s, iz_s = ix[::step], iy[::step], iz[::step]
            dx = xs[:, None].astype(np.float32) - ix_s[None, :].astype(np.float32)
            dy = ys[:, None].astype(np.float32) - iy_s[None, :].astype(np.float32)
            dist2 = dx * dx + dy * dy
            k_nn = min(5, len(iz_s))
            nn_idx = np.argpartition(dist2, k_nn - 1, axis=1)[:, :k_nn]
            local_mask_z = np.median(iz_s[nn_idx], axis=1).astype(np.float32)

            steps_mm = (zs.astype(np.float32) - local_mask_z) * alpha_eff
            if float(np.percentile(steps_mm, 85)) > 40.0:
                pos_steps = steps_mm[steps_mm > 0]
                cap_mm = (
                    min(30.0, max(16.0, float(np.percentile(pos_steps, 40))))
                    if len(pos_steps) > 10
                    else 24.0
                )
                max_step_raw = cap_mm / alpha_eff
                min_step_raw = -8.0 / alpha_eff
                zs = np.clip(
                    zs.astype(np.float32),
                    local_mask_z + min_step_raw,
                    local_mask_z + max_step_raw,
                )

    xf, yf, zf = xs.astype(np.float32), ys.astype(np.float32), zs.astype(np.float32)
    for _ in range(2):
        A_mat = np.column_stack([xf, yf, np.ones_like(xf)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, zf, rcond=None)
        A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        res = zf - (A * xf + B * yf + C)
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med))) * 1.4826
        keep = np.abs(res - med) <= max(2.5 * mad, 10.0)
        if keep.all() or keep.sum() < 12:
            break
        xf, yf, zf = xf[keep], yf[keep], zf[keep]

    ring_res = zf - (A * xf + B * yf + C)
    plane_offset = float(np.percentile(ring_res, 50))

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
    class_name: str = "",
    K: Optional[np.ndarray] = None,
    scale_source: str = "unknown",
    relief_scale: Optional[float] = None,
    stabilize_plane: bool = True,
) -> Tuple[float, List[str]]:
    """Compute 3D volume for concave bowl containers (soup, noodles, porridge, rice).

    Models the bowl with three distinct physical diameters (A1):
      - D_food: food/liquid top surface diameter measured from the segmentation mask
      - D_rim : outer bowl rim diameter related via frustum fill fraction
                D_food = D_rim * (beta + (1 - beta) * f)
      - D_base: bowl base diameter = beta * D_rim
    Replaces the arbitrary 150 cm3 floor with physical geometric validity checks (A2).
    """
    import cv2 as _cv2
    warnings = []
    from pipeline_config import (
        PLATE_DIAMETER_HEURISTICS, BOWL_DEPTH_RATIO,
        BOWL_BASE_RATIO, BOWL_DEFAULT_FILL_RATIO,
        BOWL_NOISE_FILL_RATIO, BOWL_DELTA_Z_NOISE_MM,
        BOWL_HEADROOM_MAX_RATIO, BOWL_DIAMETER_TOLERANCE,
    )

    x1, y1, x2, y2 = bbox
    bw_px = max(10, x2 - x1)
    bh_px = max(10, y2 - y1)

    expected_dia_mm = PLATE_DIAMETER_HEURISTICS.get(dish_type, 180.0)
    d_major_px = max(bw_px, bh_px)
    D_food_raw_mm = float(d_major_px * mm_per_pixel)

    aspect_ratio = 1.0
    if food_mask is not None and food_mask.any():
        try:
            cnts = _cv2.findContours(
                food_mask.astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_NONE
            )[0]
            if cnts:
                cnt = max(cnts, key=_cv2.contourArea)
                if len(cnt) >= 5:
                    (_ex, _ey), (ew, eh), _ea = _cv2.fitEllipse(cnt)
                    if max(ew, eh) > 0:
                        aspect_ratio = float(min(ew, eh) / max(ew, eh))
        except (_cv2.error, ValueError):
            pass

    is_sticky_rice_plate = ("Xoi" in class_name)
    if scale_source in ("aruco_marker", "checkerboard") and not is_sticky_rice_plate:
        D_food_mm = D_food_raw_mm
        f_nom = BOWL_DEFAULT_FILL_RATIO
        rim_from_food_factor = 1.0 / (BOWL_BASE_RATIO + (1.0 - BOWL_BASE_RATIO) * f_nom)
        D_rim_est = D_food_mm * rim_from_food_factor

        if dish_type == "rice_bowl":
            min_rim_mm, max_rim_mm = 75.0, 145.0
        elif dish_type == "soup_bowl":
            min_rim_mm, max_rim_mm = 95.0, 198.0
        else:
            min_rim_mm, max_rim_mm = 110.0, 240.0

        if aspect_ratio < 0.75 and D_food_mm >= 0.80 * expected_dia_mm:
            D_rim_est = max(D_rim_est, (1.0 + BOWL_DIAMETER_TOLERANCE) * expected_dia_mm)

        D_rim_mm = float(np.clip(D_rim_est, min_rim_mm, max_rim_mm))
    elif scale_source in ("aruco_marker", "checkerboard") and is_sticky_rice_plate:
        D_rim_mm = float(np.clip(
            D_food_raw_mm,
            (1.0 - BOWL_DIAMETER_TOLERANCE) * expected_dia_mm,
            (1.0 + BOWL_DIAMETER_TOLERANCE) * expected_dia_mm,
        ))
        if aspect_ratio < 0.75:
            D_rim_mm = max(D_rim_mm, (1.0 + BOWL_DIAMETER_TOLERANCE) * expected_dia_mm)
    else:
        D_rim_mm = float(expected_dia_mm)

    R_rim_mm = D_rim_mm / 2.0
    R_base_mm = R_rim_mm * BOWL_BASE_RATIO
    H_bowl_mm = D_rim_mm * BOWL_DEPTH_RATIO

    has_valid_depth = False
    delta_z = 0.0
    above_rim_volume_cm3 = 0.0

    if depth_map_mm is not None and food_mask.any():
        mask_bool = food_mask.astype(bool)
        masked_d = depth_map_mm[mask_bool]
        valid_d = masked_d[masked_d > 0]

        if len(valid_d) > 20:
            plane_z = fit_plane_to_boundary(
                depth_map_mm, food_mask,
                stabilize_step=stabilize_plane,
                alpha=relief_scale or 0.79,
            )
            plane_d = plane_z[mask_bool]

            z_rim = float(np.median(plane_d))
            z_food = float(np.median(valid_d))
            alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_rim, mm_per_pixel, K)
            z_rim_true = z_rim * alpha
            z_food_true = z_food * alpha
            delta_z = z_food_true - z_rim_true
            has_valid_depth = True
            delta_z_resolved = delta_z > BOWL_DELTA_Z_NOISE_MM

            if delta_z_resolved:
                r_px = int(np.sqrt(mask_bool.sum() / np.pi))
                erode_px = max(3, int(0.04 * r_px))
                eroded = _cv2.erode(
                    mask_bool.astype(np.uint8),
                    _cv2.getStructuringElement(
                        _cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)),
                ).astype(bool)
                interior_d = depth_map_mm[eroded]
                interior_plane = plane_z[eroded]
                raw_heights_above_rim = interior_plane - interior_d
                mound_cap = BOWL_HEADROOM_MAX_RATIO * H_bowl_mm
                clipped_heights = np.minimum(raw_heights_above_rim, mound_cap)
                pos_mound = clipped_heights[clipped_heights > 0]
                if len(pos_mound) > 0.05 * len(raw_heights_above_rim):
                    interior_masked = interior_d[raw_heights_above_rim > 0]
                    if scale_source in ("aruco_marker", "checkerboard") and K is not None and K[0, 0] > 0 and K[1, 1] > 0:
                        px_area = (interior_masked ** 2) / (float(K[0, 0]) * float(K[1, 1]))
                    else:
                        px_area = mm_per_pixel ** 2
                    above_rim_volume_cm3 = float(np.sum(pos_mound * alpha * px_area) / 1000.0)

    if has_valid_depth:
        if delta_z > BOWL_DELTA_Z_NOISE_MM:
            headroom_mm = float(np.clip(
                delta_z, BOWL_DELTA_Z_NOISE_MM,
                BOWL_HEADROOM_MAX_RATIO * H_bowl_mm))
            h_fill = H_bowl_mm - headroom_mm
        else:
            h_fill = H_bowl_mm * BOWL_NOISE_FILL_RATIO
    else:
        h_fill = H_bowl_mm * BOWL_DEFAULT_FILL_RATIO

    fill_fraction = h_fill / max(1e-6, H_bowl_mm)
    R_fill_mm = R_base_mm + (R_rim_mm - R_base_mm) * fill_fraction

    frustum_mm3 = (np.pi * h_fill / 3.0) * (R_fill_mm**2 + R_fill_mm * R_base_mm + R_base_mm**2)
    bowl_volume_cm3 = float(frustum_mm3 / 1000.0)
    total_volume_cm3 = bowl_volume_cm3 + above_rim_volume_cm3

    # A2: Geometric validity check instead of arbitrary 150 cm3 floor
    v_brim_cm3 = float(
        (np.pi * H_bowl_mm / 3.0)
        * (R_rim_mm**2 + R_rim_mm * R_base_mm + R_base_mm**2)
        / 1000.0
    )
    geom_valid = (
        np.isfinite(total_volume_cm3)
        and total_volume_cm3 > 0.0
        and 0.0 < R_base_mm < R_fill_mm <= R_rim_mm * 1.05
        and 0.20 <= fill_fraction <= 1.05
    )
    if not geom_valid:
        h_def = H_bowl_mm * BOWL_DEFAULT_FILL_RATIO
        r_def = R_base_mm + (R_rim_mm - R_base_mm) * BOWL_DEFAULT_FILL_RATIO
        total_volume_cm3 = float(
            (np.pi * h_def / 3.0) * (r_def**2 + r_def * R_base_mm + R_base_mm**2) / 1000.0
        )
        warnings.append("Bowl geometry out of physical bounds — reverted to default fill ratio")
    else:
        min_phys_cm3 = max(15.0, 0.20 * v_brim_cm3)
        max_phys_cm3 = min(1500.0, 1.15 * v_brim_cm3)
        total_volume_cm3 = float(np.clip(total_volume_cm3, min_phys_cm3, max_phys_cm3))

    return total_volume_cm3, warnings


def compute_volume_small_ingredient(
    bbox: Tuple[int, int, int, int],
    mm_per_pixel: float,
    class_name: str,
    depth_map_mm: Optional[np.ndarray] = None,
    food_mask: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    relief_scale: Optional[float] = None,
) -> Tuple[float, List[str]]:
    """Compute 3D volume for small ingredients/garnishes (lime, chili, egg, pickles).

    For `Trung` (A3), uses multi-criteria morphological separation (`food_mask_area <= 38 cm2`,
    `d_maj <= 82 mm`, `Solidity >= 0.82`) to separate single boiled eggs (3D spheroid model)
    from multi-egg pan/skillet omelettes (`BanhMi_Trung`, routed to shallow skillet column integration).
    """
    import cv2 as _cv2
    warnings = []
    x1, y1, x2, y2 = bbox
    w_mm = max(2.0, (x2 - x1) * mm_per_pixel)
    h_mm = max(2.0, (y2 - y1) * mm_per_pixel)
    d_maj_mm = max(w_mm, h_mm)
    d_min_mm = min(w_mm, h_mm)

    if "Trung" in class_name and food_mask is not None and food_mask.any():
        area_cm2 = float(food_mask.sum() * (mm_per_pixel ** 2) / 100.0)
        solidity = 1.0
        try:
            cnts = _cv2.findContours(
                food_mask.astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
            )[0]
            if cnts:
                cnt = max(cnts, key=_cv2.contourArea)
                hull = _cv2.convexHull(cnt)
                hull_area = float(_cv2.contourArea(hull))
                if hull_area > 0:
                    solidity = float(_cv2.contourArea(cnt)) / hull_area
        except (_cv2.error, ValueError):
            pass

        is_single_boiled_egg = (area_cm2 <= 38.0 and d_maj_mm <= 82.0 and solidity >= 0.82)
        if is_single_boiled_egg:
            eff_maj = float(np.clip(d_maj_mm, 38.0, 68.0))
            eff_min = float(np.clip(d_min_mm, 32.0, 52.0))
            vol_mm3 = (4.0 / 3.0) * np.pi * (eff_maj / 2.0) * ((eff_min / 2.0) ** 2) * 0.92
            return float(np.clip(vol_mm3 / 1000.0, 20.0, 75.0)), warnings
        elif depth_map_mm is not None:
            vol_cm3 = compute_volume_column_integration(
                depth_map_mm=depth_map_mm,
                food_mask=food_mask,
                mm_per_pixel=mm_per_pixel,
                K=K,
                max_height_mm=16.0,
                relief_scale=relief_scale,
                min_mean_height_mm=8.5,
                mean_height_cap=10.5,
            )
            return vol_cm3, warnings

    # Thickness and shape factor priors (used when depth relief is unavailable)
    if "Chanh" in class_name:
        t_mm = d_min_mm * 0.50
        k_shape = 0.55
        max_vol_cm3 = 25.0
    elif "Trung" in class_name:
        t_mm = d_min_mm * 0.95
        k_shape = 0.90
        max_vol_cm3 = 80.0
    elif "Ot" in class_name:
        t_mm = d_min_mm * 0.40
        k_shape = 0.50
        max_vol_cm3 = 15.0
    elif "Ca chua" in class_name or "Dua leo" in class_name:
        t_mm = min(15.0, d_min_mm * 0.35)
        k_shape = 0.65
        max_vol_cm3 = 35.0
    else:
        t_mm = d_min_mm * 0.50
        k_shape = 0.60
        max_vol_cm3 = 30.0

    if (
        depth_map_mm is not None
        and food_mask is not None
        and food_mask.any()
    ):
        mask_bool = food_mask.astype(bool)
        masked_d = depth_map_mm[mask_bool]
        valid_d = masked_d[masked_d > 0]
        if len(valid_d) > 20:
            plane_z = fit_plane_to_boundary(depth_map_mm, food_mask, alpha=relief_scale or 0.79)
            plane_d = plane_z[mask_bool]
            z_plane_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
            alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_plane_med, mm_per_pixel, K)
            relief = (plane_d - masked_d) * alpha
            pos_relief = relief[relief > 0]
            if len(pos_relief) > 20:
                t_depth_mm = float(np.percentile(pos_relief, 90)) * 1.05
                t_depth_mm = float(np.clip(t_depth_mm, 0.25 * d_min_mm, 1.1 * d_min_mm))
                t_mm = t_depth_mm
                k_shape = 1.00

    vol_mm3 = (4.0 / 3.0) * np.pi * (w_mm / 2.0) * (h_mm / 2.0) * (t_mm / 2.0) * k_shape
    vol_cm3 = float(vol_mm3 / 1000.0)
    vol_cm3 = float(np.clip(vol_cm3, 1.0, max_vol_cm3))

    return vol_cm3, warnings


def compute_volume_side_vegetables(
    depth_map_mm: Optional[np.ndarray],
    food_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mm_per_pixel: float,
    K: Optional[np.ndarray] = None,
    relief_scale: Optional[float] = None,
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

    # Height estimation (C4: require >=35% positive pixels before overriding default
    # mound height so recessed herb bowls like BunMam/BunRieu2 don't collapse to 10 mm;
    # cap fluffy leaf pile height at 28 mm to account for air gaps)
    mean_h_mm = 22.0
    if depth_map_mm is not None and food_mask.any():
        mask_bool = food_mask.astype(bool)
        plane_z = fit_plane_to_boundary(depth_map_mm, food_mask, alpha=relief_scale or 0.79)
        plane_d = plane_z[mask_bool]
        z_plane_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
        alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_plane_med, mm_per_pixel, K)
        raw_h = (plane_z[mask_bool] - depth_map_mm[mask_bool]) * alpha
        if raw_h.size > 50 and float((raw_h > 0).mean()) > 0.35:
            pos_h = np.clip(raw_h, 0.0, 45.0)
            mean_h_mm = float(pos_h.mean())

    mean_h_mm = float(np.clip(mean_h_mm, 14.0, 28.0))
    vol_cm3 = float((area_mm2 * mean_h_mm) / 1000.0)
    vol_cm3 = float(np.clip(vol_cm3, 20.0, 450.0))

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
    relief_scale: Optional[float] = None,
    min_mean_height_mm: float = 0.0,
    mean_height_cap: Optional[float] = None,
    enable_recess_shift: bool = True,
    recess_offset_mult: float = 2.0,
    recess_max_shift_mm: float = 18.0,
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
        plane_z = fit_plane_to_boundary(depth_map_mm, food_mask, alpha=relief_scale or 0.79)
        plane_depths = plane_z[mask_bool]

    # 1.5 Re-anchor relief to the lateral scale: removes the multiplicative
    # bias of the metric depth model (see relief_scale_factor).
    valid_planes = plane_depths[plane_depths > 0]
    z_plane_med = float(np.median(valid_planes)) if len(valid_planes) > 0 else 0.0
    alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_plane_med, mm_per_pixel, K)

    # 2. Food height above plate = Plate plane depth - Food surface depth
    # A4: Check for concave/recessed plates BEFORE clipping to 2.0 mm (fixing the dead-code order).
    from pipeline_config import FOOD_HEIGHT_OFFSET_MM
    raw_heights = (plane_depths - masked_depths) * alpha
    med_raw_h = float(np.median(raw_heights))
    p85_raw_h = float(np.percentile(raw_heights, 85))
    neg_or_flat_frac = float((raw_heights <= 2.5).mean())
    area_cm2 = float(food_mask.sum() * (mm_per_pixel ** 2) / 100.0) if mm_per_pixel > 0 else 150.0

    if enable_recess_shift and area_cm2 <= 225.0 and (
        (med_raw_h <= 4.0 and p85_raw_h <= 16.0)
        or (med_raw_h <= 11.5 and p85_raw_h <= 27.0 and neg_or_flat_frac >= 0.22)
    ):
        recess_shift_mm = max(0.0, float(np.percentile(-raw_heights, 85)))
        recess_shift_mm = min(recess_shift_mm, recess_max_shift_mm) + recess_offset_mult * FOOD_HEIGHT_OFFSET_MM
        food_heights_mm = np.clip(raw_heights + recess_shift_mm, 2.0, max_height_mm)
    else:
        food_heights_mm = np.clip(raw_heights, 2.0, max_height_mm)

    # Loose-grain dishes (com tam, rice plates) systematically underestimate
    # depth relief (gaps between grains + mask erosion eat the pile edges).
    # Lift proportionally to a dish-type floor while keeping the relief shape.
    if min_mean_height_mm > 0 and food_heights_mm.size > 0:
        mean_h = float(food_heights_mm.mean())
        if 0 < mean_h < min_mean_height_mm:
            food_heights_mm = np.clip(
                food_heights_mm * (min_mean_height_mm / mean_h),
                2.0, max_height_mm)

    # Saturation cap (A5): triggers when depth plateau saturates the height cap
    if mean_height_cap is not None and food_heights_mm.size > 0:
        sat_band = max_height_mm - 1.5
        sat_fraction = float((food_heights_mm >= sat_band).mean())
        high_plateau_frac = float((food_heights_mm >= 0.78 * max_height_mm).mean())
        if sat_fraction > 0.35 or high_plateau_frac > 0.65:
            mean_h = float(food_heights_mm.mean())
            if mean_h > mean_height_cap > 0:
                food_heights_mm = np.clip(
                    food_heights_mm * (mean_height_cap / mean_h),
                    2.0, max_height_mm)

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
        from density_db import (
            get_density_with_source,
            get_nutrition_per_100g_with_source,
            estimate_nutrition,
        )
        self._get_density_with_source = get_density_with_source
        self._get_nutrition_per_100g_with_source = get_nutrition_per_100g_with_source
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
        depth_anchored: bool = False,
        z_marker_mm: Optional[float] = None,
        scene_classes: Optional[List[str]] = None,
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
                nutrition={"Calories": 0.0, "Protein": 0.0, "Fat": 0.0, "Carbs": 0.0, "Saturates": 0.0, "Sugar": 0.0, "Salt": 0.0,
                           "Sodium": 0.0, "Calcium": 0.0, "Iron": 0.0, "Zinc": 0.0, "Cholesterol": 0.0},
                nutrition_std={"Calories": 0.0, "Protein": 0.0, "Fat": 0.0, "Carbs": 0.0, "Saturates": 0.0, "Sugar": 0.0, "Salt": 0.0,
                               "Sodium": 0.0, "Calcium": 0.0, "Iron": 0.0, "Zinc": 0.0, "Cholesterol": 0.0},
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

        # When the pipeline has already re-anchored the metric depth to a
        # board/marker tvec, the relief is metrically scaled — skip the alpha
        # correction (with a heuristic focal length it would only inject error).
        # Metric depth models additionally EXAGGERATE relief on out-of-
        # distribution close-ups. Empirical relief factor calibrated on the
        # banh-mi ground truth (measured ~500 cm3 / 225 g, two shots, robust
        # plane fit): 0.79. Applied only to anchored scenes; heuristic-scale
        # scenes go through the alpha correction instead.
        relief_scale = 0.79 if depth_anchored else None

        # ── C6: Perspective depth-ratio scale correction (Z_food / Z_marker) ──
        # Corrects pinhole foreshortening when the ArUco marker sits closer or
        # farther from the camera than the food item (e.g. oblique table shots).
        if (
            scale_source == "aruco_marker"
            and z_marker_mm is not None
            and z_marker_mm > 0
            and has_valid_depth
        ):
            fd = depth_map_mm[food_mask.astype(bool)]
            vfd = fd[fd > 0]
            if len(vfd) > 20:
                z_food = float(np.median(vfd))
                alpha_z = 0.79 if depth_anchored else 0.85
                persp_ratio = float(np.clip(1.0 + alpha_z * (z_food - z_marker_mm) / z_marker_mm, 0.72, 1.35))
                mm_per_pixel = mm_per_pixel * persp_ratio

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

        # ── Container-in-mask guard ──
        # A real food silhouette never fills its bounding box completely
        # (there is always background between the food edge and the box).
        # When the mask covers >92% of the bbox, SAM2 segmented the whole
        # bowl/plate together with the food — integrating that mask produces
        # absurd volumes (measured: 2099 cm3 of "vermicelli" in a 11 cm bowl).
        # Fall back to a conservative bbox-based estimate instead.
        container_in_mask = False
        if bbox is not None and food_mask.any():
            bbox_area_px = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            container_in_mask = food_mask.sum() / bbox_area_px > 0.92
        if container_in_mask:
            warnings.append(
                "Segmentation phủ gần trọn bbox (mask có thể gồm cả bát/đĩa) — "
                "dùng ước lượng bảo thủ từ bbox"
            )
            volume_cm3 = compute_volume_bbox_approximation(bbox, mm_per_pixel)
            volume_method = "bbox_approximation"
            volume_cm3 = float(np.clip(volume_cm3, 20.0, 1500.0))
            mass_g, mass_std_density, density_mean = self._mass_from_volume(volume_cm3, class_name, warnings)
            nutrition, nutrition_per_100g = self._nutrition_from_mass(mass_g, class_name, warnings)
            mass_std_g, nutrition_std = self._propagate_uncertainty(
                nutrition, mass_g, mass_std_density, scale_source, volume_method)
            confidence_note = self._build_confidence_note(
                "bbox_fallback", scale_source, volume_method, warnings)
            return NutritionEstimate(
                class_name=class_name,
                volume_cm3=round(volume_cm3, 2),
                volume_method=volume_method,
                mass_g=round(mass_g, 1),
                mass_std_g=round(mass_std_g, 1),
                density_g_per_cm3=density_mean,
                nutrition=nutrition,
                nutrition_std=nutrition_std,
                estimation_method="bbox_fallback",
                confidence="low",
                confidence_note=confidence_note,
                warnings=warnings,
                nutrition_per_100g=nutrition_per_100g,
            )

        # ── Category-Aware Volume Estimation ──
        if dish_type in {"soup_bowl", "large_bowl", "rice_bowl"} and bbox is not None:
            # 1. 3D Bowl Container Model
            volume_cm3, bowl_warns = compute_volume_bowl_container(
                depth_map_mm=depth_map_mm,
                food_mask=food_mask,
                bbox=bbox,
                mm_per_pixel=mm_per_pixel,
                dish_type=dish_type,
                class_name=class_name,
                K=K,
                scale_source=scale_source,
                relief_scale=relief_scale,
            )
            volume_method = "bowl_container"
            warnings.extend(bowl_warns)

        elif dish_type == "ingredient_small" and bbox is not None:
            # 2. 3D Small Ingredient / Spheroid Model (C3: whole/bowl Ca chua uses depth integration)
            area_cm2 = float(food_mask.sum() * (mm_per_pixel ** 2) / 100.0) if food_mask.any() else 0.0
            if "Ca chua" in class_name and has_valid_depth and area_cm2 > 20.0:
                volume_cm3 = compute_volume_column_integration(
                    depth_map_mm=depth_map_mm,
                    food_mask=food_mask,
                    mm_per_pixel=mm_per_pixel,
                    K=K,
                    homography=homography,
                    scale_source=scale_source,
                    max_height_mm=55.0,
                    relief_scale=relief_scale,
                    min_mean_height_mm=0.0,
                    mean_height_cap=(SATURATION_CAP_RATIO * 55.0 if SATURATION_CAP_RATIO else None),
                )
                volume_method = "column_integration"
            else:
                volume_cm3, ingr_warns = compute_volume_small_ingredient(
                    bbox=bbox,
                    mm_per_pixel=mm_per_pixel,
                    class_name=class_name,
                    depth_map_mm=depth_map_mm if has_valid_depth else None,
                    food_mask=food_mask if has_valid_depth else None,
                    K=K,
                    relief_scale=relief_scale,
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
                K=K,
                relief_scale=relief_scale,
            )
            volume_method = "vegetable_mound"
            warnings.extend(veg_warns)

        elif has_valid_depth:
            # 4. Standard Plate Column Integration (B2 + C2 + C5)
            _loose_grain_keys = ("Com", "Banh cuon", "Nui", "Bun", "Banh beo", "Banh trang tron", "Xoi")
            _thin_sliced_meats = ("Thit nuong", "Thit heo", "Thit bo", "Ca (Fish)")
            is_loose_grain = any(k in class_name for k in _loose_grain_keys)
            is_banh_trang = class_name.startswith("Banh trang (")
            is_banh_xeo = class_name.startswith("Banh xeo")
            is_thin_meat = any(class_name.startswith(k) for k in _thin_sliced_meats)

            min_mean_h = 0.0
            mean_h_cap = None
            if max_height_mm is None:
                if is_banh_trang:
                    max_height_mm = 6.0
                    mean_h_cap = 4.0
                elif is_banh_xeo:
                    max_height_mm = 16.0
                    mean_h_cap = 8.5
                elif "Banh mi" in class_name or "Hamburger" in class_name:
                    max_height_mm = 55.0
                    mean_h_cap = SATURATION_CAP_RATIO * max_height_mm if SATURATION_CAP_RATIO else None
                elif "Kho qua thit" in class_name:
                    max_height_mm = 35.0
                    mean_h_cap = SATURATION_CAP_RATIO * max_height_mm if SATURATION_CAP_RATIO else None
                elif dish_type == "small_plate":
                    max_height_mm = 24.0
                    mean_h_cap = SATURATION_CAP_RATIO * max_height_mm if SATURATION_CAP_RATIO else None
                else:
                    max_height_mm = 45.0
                    min_mean_h = 15.0 if is_loose_grain else 0.0
                    mean_h_cap = SATURATION_CAP_RATIO * max_height_mm if SATURATION_CAP_RATIO else None
            else:
                mean_h_cap = SATURATION_CAP_RATIO * max_height_mm if SATURATION_CAP_RATIO else None

            volume_cm3 = compute_volume_column_integration(
                depth_map_mm=depth_map_mm,
                food_mask=food_mask,
                mm_per_pixel=mm_per_pixel,
                K=K,
                homography=homography,
                scale_source=scale_source,
                max_height_mm=max_height_mm,
                relief_scale=relief_scale,
                min_mean_height_mm=min_mean_h,
                mean_height_cap=mean_h_cap,
                enable_recess_shift=(not is_banh_trang),
                recess_offset_mult=(1.0 if is_thin_meat else 2.0),
                recess_max_shift_mm=(10.0 if is_thin_meat else 18.0),
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

        # ── Density → Mass → Nutrition (provenance-aware helpers) ──
        mass_g, mass_std_density, density_mean = self._mass_from_volume(
            volume_cm3, class_name, warnings)

        # C1: Co-occurrence physical density for plain dipping baguette (bánh mì không,
        # USDA white baguette density ~0.20 g/cm3 vs 0.45 g/cm3 stuffed meat sandwich)
        if "Banh mi" in class_name and scene_classes:
            _dip_partners = ("Bo kho", "Trung", "Ca ri", "Pate", "Xiu mai")
            if any(
                any(k in other for k in _dip_partners)
                for other in scene_classes
                if other != class_name
            ):
                density_mean = 0.20
                mass_g = volume_cm3 * density_mean
                mass_std_density = volume_cm3 * 0.04

        nutrition, nutrition_per_100g = self._nutrition_from_mass(
            mass_g, class_name, warnings)
        mass_std_g, nutrition_std = self._propagate_uncertainty(
            nutrition, mass_g, mass_std_density, scale_source, volume_method)

        # ── Determine overall estimation method & confidence ──
        if scale_source == "aruco_marker":
            estimation_method = "aruco_calibrated"
            confidence = "high"
        elif scale_source == "checkerboard":
            estimation_method = "board_calibrated"
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

    def _mass_from_volume(self, volume_cm3: float, class_name: str, warnings: list):
        """Density lookup → mass, flagging fallback data sources."""
        density_mean, density_std, density_source = self._get_density_with_source(class_name)
        mass_g = volume_cm3 * density_mean
        mass_std_g = volume_cm3 * density_std
        if density_source == "category_estimate":
            warnings.append(
                f"No class-specific density for '{class_name}' — "
                f"using dish-type category estimate {density_mean:.2f}±{density_std:.2f} g/cm³"
            )
        elif density_source == "generic_estimate":
            warnings.append(
                f"No density data for '{class_name}' — using generic {density_mean:.2f}±{density_std:.2f} g/cm³"
            )
        return mass_g, mass_std_g, density_mean

    def _nutrition_from_mass(self, mass_g: float, class_name: str, warnings: list):
        """Per-100g scaling → nutrition, flagging fallback data sources."""
        nutrition = self._estimate_nutrition(mass_g, class_name)
        nutrition_per_100g, nutrition_source = self._get_nutrition_per_100g_with_source(class_name)
        if nutrition_source == "category_estimate":
            warnings.append(
                f"No class-specific nutrition for '{class_name}' — using dish-type category values"
            )
        elif nutrition_source == "generic_estimate":
            warnings.append(f"No nutrition data for '{class_name}' — using generic per-100g values")
        return nutrition, nutrition_per_100g

    @staticmethod
    def _propagate_uncertainty(nutrition: Dict[str, float], mass_g: float,
                               mass_std_density: float, scale_source: str,
                               volume_method: str):
        """Density variance ⊕ volume/scale error → mass & nutrition std.

        Relative volume error is dominated by the scale source (assumed
        plate diameter, marker calibration) and by the volume method.
        """
        # ASSUMED, NOT MEASURED (2026-09-28): các hệ số dưới đây là giả định kỹ thuật,
        # không phải độ lệch chuẩn đo được. Frozen benchmark outputs phụ thuộc giá trị này —
        # KHÔNG đổi nếu không chạy post-freeze evaluation riêng (results/frozen_numbers.md §0).
        scale_rel_uncertainty = {
            "aruco_marker": 0.10,      # calibrated marker (black square = 30mm)
            "checkerboard": 0.10,      # 30mm-square calibration board
            "plate_heuristic": 0.18,   # assumed plate/bowl diameter
            "exif": 0.25,              # focal length from EXIF heuristic (vestigial — chưa từng được gán)
        }
        volume_rel = scale_rel_uncertainty.get(scale_source, 0.30)
        if volume_method == "bbox_approximation":
            volume_rel = max(volume_rel, 0.35)
        density_rel = (mass_std_density / mass_g) if mass_g > 0 else 0.0
        total_rel = (density_rel**2 + volume_rel**2) ** 0.5
        mass_std_g = mass_g * total_rel if mass_g > 0 else 0.0
        nutrition_std = {}
        if nutrition and mass_g > 0:
            nutrition_std = {
                k: round(v * total_rel, 2) if k in ("Iron", "Zinc") else round(v * total_rel, 1)
                for k, v in nutrition.items()
            }
        return mass_std_g, nutrition_std

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
                "Ước lượng có hiệu chuẩn reference (ArUco marker + depth map + phối cảnh Z_food/Z_marker). "
                "Sai số trên benchmark mở rộng 418 ảnh / 55 lớp (MobileSAM Config C): MAPE toàn tập 30.7% "
                "(Full-Meal 28.8%, các món tô nước ~3-15%, End-to-End matched 35.9%). "
                "Xem results/benchmark_v2_418_report.md."
            )
        elif estimation_method == "plate_heuristic":
            if volume_method == "bowl_container":
                note = (
                    "Ước lượng bát 3D chuẩn hóa — sử dụng kích thước bát và đo mức đầy từ depth map. "
                    "Tham chiếu benchmark 418 ảnh / 55 lớp: MAPE 30.7% khi có marker (results/benchmark_v2_418_report.md)."
                )
            else:
                note = (
                    "Ước lượng gần đúng — sử dụng kích thước đĩa/bát ước lượng, sai số cao hơn "
                    "cấu hình có marker (tham chiếu MAPE 30.7% trên 418 ảnh, results/benchmark_v2_418_report.md). "
                    "Để chính xác hơn, đặt ArUco marker cạnh đĩa."
                )
        else:
            note = (
                "Ước lượng sơ bộ — không có depth map hoặc reference scale. "
                "Cấu hình này CHƯA được kiểm chứng bằng benchmark; "
                "kết quả chỉ mang tính tham khảo, không dùng cho mục đích dinh dưỡng."
            )

        if warnings:
            note += f" ({len(warnings)} cảnh báo)"

        return note
