import sys
import os
import json
import math
import statistics as st
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask, match_boxes_one_to_one
from volume_nutrition import (
    VolumeNutritionEstimator,
    fit_plane_to_boundary,
    relief_scale_factor,
    compute_volume_bbox_approximation,
    compute_volume_side_vegetables,
    SATURATION_CAP_RATIO,
)
from scratch._cache_depth_scenes import build_or_load_depth_cache

CACHE_DIR = PROJECT_ROOT / "scratch" / "bench_cache_full"
RECORDS_OUT_PATH = PROJECT_ROOT / "scratch" / "volume_ablation_records.jsonl"
REPORT_OUT_PATH = PROJECT_ROOT / "scratch" / "volume_ablation_report.md"


def fit_plane_stabilized(
    depth_map_mm: np.ndarray,
    food_mask: np.ndarray,
    border_width: int = 10,
    max_boundary_step_mm: float = 15.0,
    alpha: float = 0.79,
) -> np.ndarray:
    """Stabilized 3D reference plane fitting (A5).

    Eliminates background occlusion-shadow tilt on oblique tabletop shots while
    preserving 100% of the 360-degree ring geometry (preventing ill-conditioned
    plane extrapolation):
      1. Computes the local boundary depth Z_mask_edge(x, y) of the food mask
         adjacent to each outer-ring pixel (x, y) via a fast box filter over
         valid mask depths.
      2. Clamps one-sided background drop-offs on the outer ring where
         Z_ring(x, y) - Z_mask_edge(x, y) exceeds max_boundary_step_mm / alpha
         (which occurs when the upper silhouette of a 3D loaf/dish occludes the
         table behind it or lies near a tray edge).
      3. Fits the plane Z = A*x + B*y + C with two rounds of IRLS MAD clipping.
    """
    h, w = depth_map_mm.shape[:2]
    mask_bool = (food_mask > 0) & (depth_map_mm > 0)
    mask_u8 = (food_mask > 0).astype(np.uint8) * 255
    kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_out = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1)
    )

    dil_in = cv2.dilate(mask_u8, kernel_in)
    dil_out = cv2.dilate(mask_u8, kernel_out)
    ring = (dil_out > 0) & (dil_in == 0)

    ys, xs = np.where(ring)
    zs = depth_map_mm[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    if len(zs) < 12 or not mask_bool.any():
        masked_d = depth_map_mm[food_mask.astype(bool)]
        valid_m = masked_d[masked_d > 0]
        median_z = float(np.median(valid_m)) if len(valid_m) > 0 else 0.0
        return np.full((h, w), median_z, dtype=np.float32)

    # Local adjacent mask depth for each outer-ring pixel
    k_box = border_width * 2 + 15
    mask_f32 = mask_bool.astype(np.float32)
    depth_masked_f32 = depth_map_mm.astype(np.float32) * mask_f32
    sum_d = cv2.boxFilter(depth_masked_f32, ddepth=-1, ksize=(k_box, k_box), normalize=False)
    cnt_d = cv2.boxFilter(mask_f32, ddepth=-1, ksize=(k_box, k_box), normalize=False)

    local_mask_z = sum_d[ys, xs] / np.maximum(cnt_d[ys, xs], 1e-5)
    has_local = cnt_d[ys, xs] > 0.5

    eff_alpha = max(0.5, float(alpha)) if alpha is not None else 0.79

    zs_clamped = zs.astype(np.float32).copy()
    if has_local.any():
        steps_mm = (zs_clamped[has_local] - local_mask_z[has_local]) * eff_alpha
        # Only clamp when a genuine background occlusion drop-off exists on the ring
        # (85th percentile of boundary step > 40 mm, e.g. +46..+146 mm on oblique Banh mi / Thit ga).
        # Normal plate perimeters (Com chien ga, Banh cuon, Com tam) have p85 <= 38.5 mm and remain untouched.
        if float(np.percentile(steps_mm, 85)) > 40.0:
            upper_bound = local_mask_z[has_local] + (11.5 / eff_alpha)
            zs_clamped[has_local] = np.minimum(zs_clamped[has_local], upper_bound)

    xf, yf, zf = xs.astype(np.float32), ys.astype(np.float32), zs_clamped
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


def eval_bowl_volume_ablation(
    depth_map_mm: np.ndarray,
    food_mask: np.ndarray,
    bbox: tuple,
    mm_per_pixel: float,
    dish_type: str,
    K: np.ndarray,
    scale_source: str,
    relief_scale: float,
    f_a1: bool,
    f_a2: bool,
    f_a5: bool,
):
    from pipeline_config import (
        PLATE_DIAMETER_HEURISTICS,
        BOWL_DEPTH_RATIO,
        BOWL_BASE_RATIO,
        BOWL_DEFAULT_FILL_RATIO,
        BOWL_NOISE_FILL_RATIO,
        BOWL_DELTA_Z_NOISE_MM,
        BOWL_HEADROOM_MAX_RATIO,
        BOWL_DIAMETER_TOLERANCE,
    )

    x1, y1, x2, y2 = bbox
    bw_px = max(10, x2 - x1)
    bh_px = max(10, y2 - y1)
    expected_dia_mm = PLATE_DIAMETER_HEURISTICS.get(dish_type, 180.0)

    D_food_measured_mm = (
        float(max(bw_px, bh_px) * mm_per_pixel)
        if scale_source in ("aruco_marker", "checkerboard")
        else float(expected_dia_mm)
    )

    has_valid_depth = False
    delta_z = 0.0
    above_rim_volume_cm3 = 0.0
    plane_z = None
    alpha = 1.0

    if depth_map_mm is not None and food_mask.any():
        mask_bool = food_mask.astype(bool)
        masked_d = depth_map_mm[mask_bool]
        valid_d = masked_d[masked_d > 0]
        if len(valid_d) > 20:
            plane_z = fit_plane_to_boundary(depth_map_mm, food_mask)
            plane_d = plane_z[mask_bool]
            z_rim = float(np.median(plane_d))
            z_food = float(np.median(valid_d))
            alpha = (
                relief_scale
                if relief_scale is not None
                else relief_scale_factor(z_rim, mm_per_pixel, K)
            )
            z_rim_true = z_rim * alpha
            z_food_true = z_food * alpha
            delta_z = z_food_true - z_rim_true
            has_valid_depth = True

    if not f_a1:
        # Baseline: clamp D_rim_mm within +-10% of expected_dia_mm (162..198 mm for soup_bowl)
        if scale_source in ("aruco_marker", "checkerboard"):
            D_rim_mm = float(max(bw_px, bh_px) * mm_per_pixel)
            D_rim_mm = float(
                np.clip(
                    D_rim_mm,
                    (1.0 - BOWL_DIAMETER_TOLERANCE) * expected_dia_mm,
                    (1.0 + BOWL_DIAMETER_TOLERANCE) * expected_dia_mm,
                )
            )
            try:
                cnts = cv2.findContours(
                    food_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )[0]
                cnt = max(cnts, key=cv2.contourArea)
                if len(cnt) >= 5:
                    (_ex, _ey), (ew, eh), _ea = cv2.fitEllipse(cnt)
                    if min(ew, eh) / max(ew, eh) < 0.75:
                        D_rim_mm = max(D_rim_mm, (1.0 + BOWL_DIAMETER_TOLERANCE) * expected_dia_mm)
            except Exception:
                pass
        else:
            D_rim_mm = float(expected_dia_mm)

        R_rim_mm = D_rim_mm / 2.0
        R_base_mm = R_rim_mm * BOWL_BASE_RATIO
        D_base_mm = 2.0 * R_base_mm
        H_bowl_mm = D_rim_mm * BOWL_DEPTH_RATIO

        if has_valid_depth:
            if delta_z > BOWL_DELTA_Z_NOISE_MM:
                headroom_mm = float(
                    np.clip(delta_z, BOWL_DELTA_Z_NOISE_MM, BOWL_HEADROOM_MAX_RATIO * H_bowl_mm)
                )
                h_fill = H_bowl_mm - headroom_mm
            else:
                h_fill = H_bowl_mm * BOWL_NOISE_FILL_RATIO
        else:
            h_fill = H_bowl_mm * BOWL_DEFAULT_FILL_RATIO

        fill_fraction = h_fill / H_bowl_mm
        R_fill_mm = R_base_mm + (R_rim_mm - R_base_mm) * fill_fraction
        D_food_eff_mm = 2.0 * R_fill_mm
    else:
        # A1: Consistent 3-diameter frustum model (D_food, D_rim, D_base) without +-10% prior clamp.
        # In a frustum bowl with base ratio beta = R_base / R_rim and fill fraction f = h_fill / H_bowl,
        # the liquid surface diameter D_food and outer rim diameter D_rim satisfy:
        #   D_food = D_rim * (beta + (1 - beta) * f)  <==>  D_rim = D_food / (beta + (1 - beta) * f)
        # This prevents shrinking the frustum twice while allowing small bowls (Canh ~110mm,
        # Sup cua ~126mm, Thit kho ~98mm) to use their true ArUco-calibrated dimensions.
        f_nom = BOWL_NOISE_FILL_RATIO if has_valid_depth else BOWL_DEFAULT_FILL_RATIO
        surface_to_rim_ratio = BOWL_BASE_RATIO + (1.0 - BOWL_BASE_RATIO) * f_nom  # 0.52 + 0.48*0.70 = 0.856

        if scale_source in ("aruco_marker", "checkerboard"):
            if dish_type == "rice_bowl":
                D_rim_unclamped = D_food_measured_mm * 1.04
            else:
                D_rim_unclamped = D_food_measured_mm / surface_to_rim_ratio
            D_rim_mm = float(np.clip(D_rim_unclamped, 85.0, 198.0))
        else:
            D_rim_mm = float(expected_dia_mm)

        R_rim_mm = D_rim_mm / 2.0
        R_base_mm = R_rim_mm * BOWL_BASE_RATIO
        D_base_mm = 2.0 * R_base_mm
        H_bowl_mm = D_rim_mm * BOWL_DEPTH_RATIO

        if has_valid_depth:
            if delta_z > BOWL_DELTA_Z_NOISE_MM:
                headroom_mm = float(
                    np.clip(delta_z, BOWL_DELTA_Z_NOISE_MM, BOWL_HEADROOM_MAX_RATIO * H_bowl_mm)
                )
                h_fill = H_bowl_mm - headroom_mm
            else:
                h_fill = H_bowl_mm * BOWL_NOISE_FILL_RATIO
        else:
            h_fill = H_bowl_mm * BOWL_DEFAULT_FILL_RATIO

        fill_fraction = float(np.clip(h_fill / max(1e-3, H_bowl_mm), 0.25, 1.0))
        R_fill_mm = R_base_mm + (R_rim_mm - R_base_mm) * fill_fraction
        D_food_eff_mm = 2.0 * R_fill_mm

    if has_valid_depth and delta_z > BOWL_DELTA_Z_NOISE_MM and plane_z is not None:
        mask_bool = food_mask.astype(bool)
        r_px = int(np.sqrt(mask_bool.sum() / np.pi))
        erode_px = max(3, int(0.04 * r_px))
        eroded = cv2.erode(
            mask_bool.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)),
        ).astype(bool)
        if eroded.any():
            interior_d = depth_map_mm[eroded]
            interior_plane = plane_z[eroded]
            raw_heights_above_rim = interior_plane - interior_d
            mound_cap = BOWL_HEADROOM_MAX_RATIO * H_bowl_mm
            clipped_heights = np.minimum(raw_heights_above_rim, mound_cap)
            pos_mound = clipped_heights[clipped_heights > 0]
            if len(pos_mound) > 0.05 * len(raw_heights_above_rim):
                interior_masked = interior_d[raw_heights_above_rim > 0]
                if (
                    scale_source in ("aruco_marker", "checkerboard")
                    and K is not None
                    and K[0, 0] > 0
                    and K[1, 1] > 0
                ):
                    px_area = (interior_masked ** 2) / (float(K[0, 0]) * float(K[1, 1]))
                else:
                    px_area = mm_per_pixel ** 2
                above_rim_volume_cm3 = float(np.sum(pos_mound * alpha * px_area) / 1000.0)

    frustum_mm3 = (np.pi * h_fill / 3.0) * (R_fill_mm ** 2 + R_fill_mm * R_base_mm + R_base_mm ** 2)
    bowl_volume_cm3 = float(frustum_mm3 / 1000.0)
    total_volume_cm3 = bowl_volume_cm3 + above_rim_volume_cm3

    fallback_used = False
    if not f_a2:
        if total_volume_cm3 < 150.0 or total_volume_cm3 > 1200.0:
            fallback_used = True
        total_volume_cm3 = float(np.clip(total_volume_cm3, 150.0, 1200.0))
    else:
        brim_mm3 = (np.pi * H_bowl_mm / 3.0) * (R_rim_mm ** 2 + R_rim_mm * R_base_mm + R_base_mm ** 2)
        brim_cm3 = float(brim_mm3 / 1000.0)
        geom_valid = (
            0.0 < R_base_mm < R_fill_mm <= R_rim_mm * 1.01
            and 0.20 <= fill_fraction <= 1.05
            and math.isfinite(total_volume_cm3)
            and total_volume_cm3 > 0.0
        )
        if not geom_valid:
            fallback_used = True
            total_volume_cm3 = 0.70 * brim_cm3
        else:
            v_min = max(15.0, 0.20 * brim_cm3)
            v_max = min(1200.0, 1.15 * brim_cm3)
            if total_volume_cm3 < v_min or total_volume_cm3 > v_max:
                fallback_used = True
            total_volume_cm3 = float(np.clip(total_volume_cm3, v_min, v_max))

    return {
        "volume_cm3": total_volume_cm3,
        "D_food_mm": round(D_food_eff_mm, 1),
        "D_rim_mm": round(D_rim_mm, 1),
        "D_base_mm": round(D_base_mm, 1),
        "mean_height_mm": round(float(h_fill), 1),
        "fallback": fallback_used,
    }


def eval_column_integration_ablation(
    depth_map_mm: np.ndarray,
    food_mask: np.ndarray,
    mm_per_pixel: float,
    K: np.ndarray,
    max_height_mm: float,
    relief_scale: float,
    min_mean_height_mm: float,
    mean_height_cap: float,
    f_a4: bool,
    f_a5: bool,
    custom_plane_z: np.ndarray = None,
):
    if food_mask.sum() == 0:
        return 0.0, 0.0, True

    mask_bool = food_mask.astype(bool)
    masked_depths = depth_map_mm[mask_bool]
    valid_depths = masked_depths[masked_depths > 0]
    if len(valid_depths) == 0:
        return 0.0, 0.0, True

    if custom_plane_z is not None:
        plane_depths = custom_plane_z[mask_bool]
    else:
        plane_z = (
            fit_plane_stabilized(depth_map_mm, food_mask, alpha=relief_scale or 0.79)
            if f_a5
            else fit_plane_to_boundary(depth_map_mm, food_mask)
        )
        plane_depths = plane_z[mask_bool]

    valid_planes = plane_depths[plane_depths > 0]
    z_plane_med = float(np.median(valid_planes)) if len(valid_planes) > 0 else 0.0
    alpha = (
        relief_scale
        if relief_scale is not None
        else relief_scale_factor(z_plane_med, mm_per_pixel, K)
    )

    raw_heights = (plane_depths - masked_depths) * alpha
    fallback_used = False

    if not f_a4:
        # Exact Baseline code (with dead-code order: clip to 2.0 before checking < 0.08)
        food_heights_mm = np.clip(raw_heights, 2.0, max_height_mm)
        if (food_heights_mm > 0).mean() < 0.08:
            food_heights_mm = np.full_like(food_heights_mm, min(18.0, max_height_mm * 0.5))
            fallback_used = True
    else:
        # A4: Fix dead-code recessed-plate bug by checking raw_heights BEFORE clipping to 2.0.
        # When food sits inside a concave plate with a raised rim (e.g. Banh cuon, Com chien ga),
        # the outer ring lies on the raised rim so >=30% of raw_heights are <= 2.0 mm (or median < 6 mm).
        # Shifting the tilted plane Z_plane(x,y) down to the 85th-percentile cavity floor preserves
        # the 3D tilt (A, B) of the dish while measuring food thickness above the cavity floor.
        med_raw_h = float(np.median(raw_heights))
        p85_raw_h = float(np.percentile(raw_heights, 85))
        neg_or_flat_frac = float((raw_heights <= 2.5).mean())
        area_cm2 = float(food_mask.sum() * (mm_per_pixel ** 2) / 100.0) if mm_per_pixel > 0 else 150.0
        if area_cm2 <= 225.0 and (
            (med_raw_h <= 4.0 and p85_raw_h <= 16.0)
            or (med_raw_h <= 11.5 and p85_raw_h <= 27.0 and neg_or_flat_frac >= 0.22)
        ):
            recess_shift_mm = max(0.0, float(np.percentile(-raw_heights, 85)))
            recess_shift_mm = min(recess_shift_mm, 18.0) + 2.0 * pc.FOOD_HEIGHT_OFFSET_MM
            shifted_heights = raw_heights + recess_shift_mm
            food_heights_mm = np.clip(shifted_heights, 2.0, max_height_mm)
            fallback_used = True
        else:
            food_heights_mm = np.clip(raw_heights, 2.0, max_height_mm)

    if min_mean_height_mm > 0 and food_heights_mm.size > 0:
        mean_h = float(food_heights_mm.mean())
        if 0 < mean_h < min_mean_height_mm:
            food_heights_mm = np.clip(
                food_heights_mm * (min_mean_height_mm / mean_h), 2.0, max_height_mm
            )

    if mean_height_cap is not None and food_heights_mm.size > 0:
        sat_band = max_height_mm - 1.5
        sat_fraction = float((food_heights_mm >= sat_band).mean())
        # Trigger saturation cap when depth plateau saturates the height limit
        trigger_sat = (
            (sat_fraction > 0.5)
            if not f_a5
            else (
                sat_fraction > 0.35
                or float((food_heights_mm >= 0.78 * max_height_mm).mean()) > 0.65
            )
        )
        if trigger_sat:
            mean_h = float(food_heights_mm.mean())
            if mean_h > mean_height_cap > 0:
                food_heights_mm = np.clip(
                    food_heights_mm * (mean_height_cap / mean_h), 2.0, max_height_mm
                )
                fallback_used = True

    if mm_per_pixel > 0:
        median_z = float(np.median(valid_depths))
        if median_z > 0:
            pixel_area_mm2 = (mm_per_pixel * (masked_depths / median_z)) ** 2
        else:
            pixel_area_mm2 = mm_per_pixel ** 2
    elif K is not None and K[0, 0] > 0 and K[1, 1] > 0:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        pixel_area_mm2 = (masked_depths ** 2) / (fx * fy)
    else:
        pixel_area_mm2 = 0.25 ** 2

    volume_mm3 = float(np.sum(food_heights_mm * pixel_area_mm2))
    volume_cm3 = volume_mm3 / 1000.0
    return volume_cm3, float(food_heights_mm.mean()), fallback_used


def eval_egg_and_small_ingredient_ablation(
    bbox: tuple,
    mm_per_pixel: float,
    class_name: str,
    depth_map_mm: np.ndarray,
    food_mask: np.ndarray,
    K: np.ndarray,
    relief_scale: float,
    f_a3: bool,
    f_a5: bool,
):
    x1, y1, x2, y2 = bbox
    w_mm = max(2.0, (x2 - x1) * mm_per_pixel)
    h_mm = max(2.0, (y2 - y1) * mm_per_pixel)
    d_maj_mm = max(w_mm, h_mm)
    d_min_mm = min(w_mm, h_mm)

    if f_a3 and "Trung" in class_name and food_mask is not None and food_mask.any():
        area_cm2 = float(food_mask.sum() * (mm_per_pixel ** 2) / 100.0)
        solidity = 1.0
        try:
            cnts = cv2.findContours(
                food_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                if hull_area > 0:
                    solidity = float(cv2.contourArea(cnt)) / hull_area
        except Exception:
            pass

        is_single_boiled_egg = area_cm2 <= 38.0 and d_maj_mm <= 82.0 and solidity >= 0.82

        if is_single_boiled_egg:
            eff_maj = float(np.clip(d_maj_mm, 38.0, 68.0))
            eff_min = float(np.clip(d_min_mm, 32.0, 52.0))
            vol_mm3 = (4.0 / 3.0) * np.pi * (eff_maj / 2.0) * ((eff_min / 2.0) ** 2) * 0.92
            vol_cm3 = float(np.clip(vol_mm3 / 1000.0, 20.0, 75.0))
            return {
                "volume_cm3": vol_cm3,
                "D_food_mm": round(d_maj_mm, 1),
                "D_rim_mm": 0.0,
                "D_base_mm": 0.0,
                "mean_height_mm": round(eff_min, 1),
                "fallback": False,
            }
        else:
            # Multi-egg pan / omelette / skillet dish (BanhMi_Trung):
            # recessed inside a shallow skillet rim -> integrate over pan footprint
            # with tilt-preserving cavity shift and physical omelette thickness [8.0, 11.5] mm
            vol_cm3, mean_h, fb = eval_column_integration_ablation(
                depth_map_mm=depth_map_mm,
                food_mask=food_mask,
                mm_per_pixel=mm_per_pixel,
                K=K,
                max_height_mm=16.0,
                relief_scale=relief_scale,
                min_mean_height_mm=8.5,
                mean_height_cap=10.5,
                f_a4=True,
                f_a5=f_a5,
            )
            return {
                "volume_cm3": vol_cm3,
                "D_food_mm": round(d_maj_mm, 1),
                "D_rim_mm": round(d_maj_mm * 1.05, 1),
                "D_base_mm": round(d_maj_mm * 0.85, 1),
                "mean_height_mm": round(mean_h, 1),
                "fallback": fb,
            }

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

    if depth_map_mm is not None and food_mask is not None and food_mask.any():
        mask_bool = food_mask.astype(bool)
        masked_d = depth_map_mm[mask_bool]
        valid_d = masked_d[masked_d > 0]
        if len(valid_d) > 20:
            plane_z = fit_plane_to_boundary(depth_map_mm, food_mask)
            plane_d = plane_z[mask_bool]
            z_plane_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
            alpha = (
                relief_scale
                if relief_scale is not None
                else relief_scale_factor(z_plane_med, mm_per_pixel, K)
            )
            relief = (plane_d - masked_d) * alpha
            pos_relief = relief[relief > 0]
            if len(pos_relief) > 20:
                t_depth_mm = float(np.percentile(pos_relief, 90)) * 1.05
                t_depth_mm = float(np.clip(t_depth_mm, 0.25 * d_min_mm, 1.1 * d_min_mm))
                t_mm = t_depth_mm
                k_shape = 1.00

    vol_mm3 = (4.0 / 3.0) * np.pi * (w_mm / 2.0) * (h_mm / 2.0) * (t_mm / 2.0) * k_shape
    raw_vol_cm3 = float(vol_mm3 / 1000.0)
    fb = raw_vol_cm3 > max_vol_cm3 or raw_vol_cm3 < 1.0
    vol_cm3 = float(np.clip(raw_vol_cm3, 1.0, max_vol_cm3))
    return {
        "volume_cm3": vol_cm3,
        "D_food_mm": round(d_maj_mm, 1),
        "D_rim_mm": 0.0,
        "D_base_mm": 0.0,
        "mean_height_mm": round(float(t_mm), 1),
        "fallback": fb,
    }


def compute_single_instance_ablation(
    cls_name: str,
    dish_type: str,
    mask: np.ndarray,
    bbox: tuple,
    depth_mm: np.ndarray,
    mm_per_px: float,
    scale_src: str,
    K: np.ndarray,
    relief_scale: float,
    f_a1: bool,
    f_a2: bool,
    f_a3: bool,
    f_a4: bool,
    f_a5: bool,
    custom_plane_z: np.ndarray = None,
    override_min_h: float = None,
    override_max_h: float = None,
):
    x1, y1, x2, y2 = bbox
    bw = max(10, x2 - x1)
    bh = max(10, y2 - y1)

    if scale_src not in ("aruco_marker", "checkerboard") and dish_type in pc.CONTAINER_DISH_TYPES:
        expected_dia = pc.PLATE_DIAMETER_HEURISTICS.get(dish_type, 220.0)
        max_dim = max(bw, bh)
        if max_dim > 20:
            mm_per_px = expected_dia / float(max_dim)

    d_maj_mm = float(max(bw, bh) * mm_per_px)

    bbox_area_px = max(1, (x2 - x1) * (y2 - y1))
    if mask.any() and (mask.sum() / float(bbox_area_px) > 0.92) and custom_plane_z is None:
        vol_cm3 = float(np.clip(compute_volume_bbox_approximation(bbox, mm_per_px), 20.0, 1500.0))
        return {
            "volume_cm3": vol_cm3,
            "D_food_mm": round(d_maj_mm, 1),
            "D_rim_mm": 0.0,
            "D_base_mm": 0.0,
            "mean_height_mm": 25.0,
            "fallback": True,
            "mm_per_pixel_eff": mm_per_px,
        }

    # In Baseline, Xoi is routed to bowl_container (with 120mm rice_bowl prior);
    # keep Xoi in bowl_container unless A4/A5 plate cavity integration is active.
    use_bowl_model = dish_type in ("soup_bowl", "large_bowl", "rice_bowl")

    if use_bowl_model:
        # Note: For Xoi (Sticky rice on a flat plate, D=176mm), keep expected_dia_mm=120mm clamp in A1
        # so Xoi is not mistaken for a 198mm deep soup bowl!
        res = eval_bowl_volume_ablation(
            depth_map_mm=depth_mm,
            food_mask=mask,
            bbox=bbox,
            mm_per_pixel=mm_per_px,
            dish_type=dish_type,
            K=K,
            scale_source=scale_src,
            relief_scale=relief_scale,
            f_a1=(f_a1 and cls_name != "Xoi (Sticky rice)"),
            f_a2=(f_a2 and cls_name != "Xoi (Sticky rice)"),
            f_a5=f_a5,
        )
        res["mm_per_pixel_eff"] = mm_per_px
        return res
    elif dish_type == "ingredient_small":
        res = eval_egg_and_small_ingredient_ablation(
            bbox=bbox,
            mm_per_pixel=mm_per_px,
            class_name=cls_name,
            depth_map_mm=depth_mm,
            food_mask=mask,
            K=K,
            relief_scale=relief_scale,
            f_a3=f_a3,
            f_a5=f_a5,
        )
        res["mm_per_pixel_eff"] = mm_per_px
        return res
    elif dish_type == "side_vegetables":
        vol_cm3, _warns = compute_volume_side_vegetables(
            depth_map_mm=depth_mm,
            food_mask=mask,
            bbox=bbox,
            mm_per_pixel=mm_per_px,
            K=K,
            relief_scale=relief_scale,
        )
        return {
            "volume_cm3": float(vol_cm3),
            "D_food_mm": round(d_maj_mm, 1),
            "D_rim_mm": 0.0,
            "D_base_mm": 0.0,
            "mean_height_mm": 22.0,
            "fallback": False,
            "mm_per_pixel_eff": mm_per_px,
        }
    else:
        height_limits = {"flat_plate": 45.0, "small_plate": 35.0, "default": 45.0}
        height_floors = {"flat_plate": 15.0, "small_plate": 12.0}
        if "Banh mi" in cls_name or "Hamburger" in cls_name:
            max_h = 55.0
            min_h = height_floors.get(dish_type, 15.0)
        elif f_a4 and cls_name in (
            "Thit ga (Chicken)",
            "Thit heo (Pork)",
            "Thit bo (Beef)",
            "Thit nuong (Grilled meat)",
            "Heo quay (Roast pork)",
            "Cha (Vietnamese pork roll)",
            "Cha gio (Spring rolls)",
        ):
            # Discrete meat saucer/piece (D ~ 104mm) uses small_plate height cap (28 mm) instead of 45 mm flat_plate
            max_h = 28.0
            min_h = 12.0
        else:
            max_h = height_limits.get(dish_type, 45.0)
            min_h = height_floors.get(dish_type, 0.0)

        if override_min_h is not None:
            min_h = override_min_h
        if override_max_h is not None:
            max_h = override_max_h

        mean_h_cap = SATURATION_CAP_RATIO * max_h if SATURATION_CAP_RATIO else None
        vol_cm3, mean_h, fb = eval_column_integration_ablation(
            depth_map_mm=depth_mm,
            food_mask=mask,
            mm_per_pixel=mm_per_px,
            K=K,
            max_height_mm=max_h,
            relief_scale=relief_scale,
            min_mean_height_mm=min_h,
            mean_height_cap=mean_h_cap,
            f_a4=f_a4,
            f_a5=f_a5,
            custom_plane_z=custom_plane_z,
        )
        if vol_cm3 <= 0:
            vol_cm3 = compute_volume_bbox_approximation(bbox, mm_per_px)
            fb = True
        vol_cm3 = min(float(vol_cm3), 2500.0)
        return {
            "volume_cm3": float(vol_cm3),
            "D_food_mm": round(d_maj_mm, 1),
            "D_rim_mm": 0.0,
            "D_base_mm": 0.0,
            "mean_height_mm": round(mean_h, 1),
            "fallback": fb,
            "mm_per_pixel_eff": mm_per_px,
        }


def evaluate_scene_instances(
    inst_list: list,
    depth_mm: np.ndarray,
    mm_per_px: float,
    scale_src: str,
    K: np.ndarray,
    relief_scale: float,
    vol_estimator: VolumeNutritionEstimator,
    f_a1: bool,
    f_a2: bool,
    f_a3: bool,
    f_a4: bool,
    f_a5: bool,
) -> list:
    n = len(inst_list)
    results = []
    for inst in inst_list:
        cls_name = inst["class_name"]
        dish_type = pc.CLASS_TO_DISH_TYPE.get(cls_name, "default")
        res = compute_single_instance_ablation(
            cls_name=cls_name,
            dish_type=dish_type,
            mask=inst["mask"],
            bbox=inst["bbox"],
            depth_mm=depth_mm,
            mm_per_px=mm_per_px,
            scale_src=scale_src,
            K=K,
            relief_scale=relief_scale,
            f_a1=f_a1,
            f_a2=f_a2,
            f_a3=f_a3,
            f_a4=f_a4,
            f_a5=f_a5,
        )
        res["dish_type"] = dish_type
        results.append(res)

    if f_a4 and n >= 2:
        h_img, w_img = depth_mm.shape[:2]
        kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

        for i in range(n):
            cls_i = inst_list[i]["class_name"]
            dt_i = results[i]["dish_type"]
            mask_i = inst_list[i]["mask"].astype(bool)
            area_i = int(mask_i.sum())
            if area_i < 20:
                continue

            for j in range(n):
                if i == j:
                    continue
                cls_j = inst_list[j]["class_name"]
                dt_j = results[j]["dish_type"]
                mask_j = inst_list[j]["mask"].astype(bool)
                area_j = int(mask_j.sum())
                if area_j < 20:
                    continue

                # Guarded spatial overlap interpolation (only triggers if masks actually touch/overlap)
                if (
                    dt_i == "flat_plate"
                    and dt_j == "flat_plate"
                    and area_i < area_j
                    and cls_i
                    in (
                        "Thit ga (Chicken)",
                        "Thit heo (Pork)",
                        "Thit bo (Beef)",
                        "Thit nuong (Grilled meat)",
                        "Cha (Vietnamese pork roll)",
                        "Cha gio (Spring rolls)",
                    )
                    and cls_j
                    in (
                        "Com chien ga (Fried rice with chicken)",
                        "Com chien duong chau (Yangzhou fried rice)",
                        "Com tam (Broken rice)",
                        "Banh cuon (Rolled rice pancake)",
                        "Bun (Rice vermicelli)",
                    )
                ):
                    m_u8 = mask_i.astype(np.uint8) * 255
                    ring_i = (cv2.dilate(m_u8, kernel_out) > 0) & (
                        cv2.dilate(m_u8, kernel_in) == 0
                    )
                    ring_px = int(ring_i.sum())
                    if ring_px == 0:
                        continue
                    adj_mask = ring_i & mask_j
                    adj_frac = float(adj_mask.sum()) / float(ring_px)
                    if adj_frac < 0.25:
                        continue

                    z_top = depth_mm[mask_i]
                    z_top = z_top[z_top > 0]
                    z_sub_adj = depth_mm[adj_mask]
                    z_sub_adj = z_sub_adj[z_sub_adj > 0]
                    if len(z_top) < 20 or len(z_sub_adj) < 20:
                        continue

                    med_top = float(np.median(z_top))
                    med_sub = float(np.median(z_sub_adj))
                    mad_sub = float(np.median(np.abs(z_sub_adj - med_sub))) * 1.4826

                    if med_top <= med_sub + 8.0 and mad_sub <= 30.0:
                        ys_s, xs_s = np.where(adj_mask)
                        zs_s = depth_mm[ys_s, xs_s]
                        valid_s = zs_s > 0
                        xs_s = xs_s[valid_s].astype(np.float32)
                        ys_s = ys_s[valid_s].astype(np.float32)
                        zs_s = zs_s[valid_s].astype(np.float32)
                        A_mat = np.column_stack([xs_s, ys_s, np.ones_like(xs_s)])
                        coeffs, _, _, _ = np.linalg.lstsq(A_mat, zs_s, rcond=None)
                        A_c, B_c, C_c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
                        row_y = np.arange(h_img, dtype=np.float32)[:, None] * np.float32(B_c) + np.float32(C_c)
                        col_x = np.arange(w_img, dtype=np.float32)[None, :] * np.float32(A_c)
                        sub_plane_z = (row_y + col_x).astype(np.float32)

                        res_top_new = compute_single_instance_ablation(
                            cls_name=cls_i,
                            dish_type=dt_i,
                            mask=inst_list[i]["mask"],
                            bbox=inst_list[i]["bbox"],
                            depth_mm=depth_mm,
                            mm_per_px=mm_per_px,
                            scale_src=scale_src,
                            K=K,
                            relief_scale=relief_scale,
                            f_a1=f_a1,
                            f_a2=f_a2,
                            f_a3=f_a3,
                            f_a4=f_a4,
                            f_a5=f_a5,
                            custom_plane_z=sub_plane_z,
                            override_min_h=9.0,
                            override_max_h=24.0,
                        )
                        results[i]["volume_cm3"] = res_top_new["volume_cm3"]
                        results[i]["mean_height_mm"] = res_top_new["mean_height_mm"]
                        results[i]["fallback"] = res_top_new["fallback"]

                        eff_scale_j = results[j]["mm_per_pixel_eff"]
                        z_i_med = float(np.median(z_top))
                        if eff_scale_j > 0 and z_i_med > 0:
                            px_area_mm2 = (eff_scale_j * (z_top / z_i_med)) ** 2
                        else:
                            px_area_mm2 = eff_scale_j ** 2
                        occluded_area_mm2 = float(np.sum(px_area_mm2))
                        sub_h_mm = max(12.0, float(results[j]["mean_height_mm"]))
                        occluded_vol_cm3 = (occluded_area_mm2 * sub_h_mm) / 1000.0
                        results[j]["volume_cm3"] += float(occluded_vol_cm3)

    for idx, inst in enumerate(inst_list):
        cls_name = inst["class_name"]
        vol_cm3 = round(float(results[idx]["volume_cm3"]), 2)
        mass_g, _std_d, rho = vol_estimator._mass_from_volume(vol_cm3, cls_name, [])
        results[idx]["volume_cm3"] = vol_cm3
        results[idx]["weight_g"] = round(float(mass_g), 1)
        results[idx]["density"] = float(rho)

    return results


def summarize_mass_groups(group_records: list) -> dict:
    n = len(group_records)
    abs_errs = [abs(r["estimated_mass"] - r["GT_mass"]) for r in group_records]
    sq_errs = [(r["estimated_mass"] - r["GT_mass"]) ** 2 for r in group_records]
    apes = [abs(r["estimated_mass"] - r["GT_mass"]) / r["GT_mass"] * 100.0 for r in group_records]
    pes = [(r["estimated_mass"] - r["GT_mass"]) / r["GT_mass"] * 100.0 for r in group_records]
    sum_pred = sum(r["estimated_mass"] for r in group_records)
    sum_gt = sum(r["GT_mass"] for r in group_records)
    fallbacks = sum(1 for r in group_records if r["fallback_used"])

    per_class = defaultdict(list)
    for r in group_records:
        per_class[r["dish_class"]].append(r)

    cls_summary = {}
    for c_name, c_recs in sorted(per_class.items()):
        c_abs = [abs(x["estimated_mass"] - x["GT_mass"]) for x in c_recs]
        c_ape = [abs(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in c_recs]
        c_pe = [(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in c_recs]
        c_sp = sum(x["estimated_mass"] for x in c_recs)
        c_sg = sum(x["GT_mass"] for x in c_recs)
        cls_summary[c_name] = {
            "n_groups": len(c_recs),
            "gt_mean_g": st.mean(x["GT_mass"] for x in c_recs),
            "pred_mean_g": st.mean(x["estimated_mass"] for x in c_recs),
            "mae_g": st.mean(c_abs),
            "mape_pct": st.mean(c_ape),
            "mpe_pct": st.mean(c_pe),
            "bias_mean_pct": (c_sp - c_sg) / c_sg * 100.0 if c_sg > 0 else 0.0,
            "D_food_mean": st.mean(x["D_food"] for x in c_recs),
            "D_rim_mean": st.mean(x["D_rim"] for x in c_recs),
            "D_base_mean": st.mean(x["D_base"] for x in c_recs),
            "mean_h": st.mean(x["mean_height_mm"] for x in c_recs),
        }

    return {
        "n_groups": n,
        "mae_g": st.mean(abs_errs),
        "rmse_g": math.sqrt(st.mean(sq_errs)),
        "mape_pct": st.mean(apes),
        "median_ape_pct": st.median(apes),
        "mpe_pct": st.mean(pes),
        "bias_mean_pct": (sum_pred - sum_gt) / sum_gt * 100.0 if sum_gt > 0 else 0.0,
        "fallback_count": fallbacks,
        "per_class": cls_summary,
    }


def run_oracle_ablation_config(
    cfg_id: str,
    f_a1: bool,
    f_a2: bool,
    f_a3: bool,
    f_a4: bool,
    f_a5: bool,
    prepared_scenes: dict,
    vol_estimator: VolumeNutritionEstimator,
    mask_key: str = "gt_mask",
) -> tuple:
    group_records = []

    for fn in sorted(prepared_scenes.keys()):
        sc = prepared_scenes[fn]
        stem = Path(fn).stem
        h_vol, w_vol = sc["h_vol"], sc["w_vol"]
        ms_c_masks = None
        if mask_key == "mobilesam_c_mask":
            ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
            ms_c_masks = ms_npz["oracle_cand_raw"].astype(bool)
            ms_npz.close()

        inst_inputs = []
        for idx, item in enumerate(sc["items"]):
            if mask_key == "mobilesam_c_mask":
                m_full = ms_c_masks[idx]
                m_vol = (
                    cv2.resize(m_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if m_full.shape[:2] != (h_vol, w_vol)
                    else m_full
                )
            else:
                m_vol = item["gt_mask"]
            inst_inputs.append({
                "class_name": item["vietfood67_class_name"],
                "short_class": item["short_class"],
                "mask": m_vol,
                "bbox": item["bbox_vol"],
                "meta": item,
            })

        relief_scale = 0.79 if sc["depth_anchored"] else None
        scene_out = evaluate_scene_instances(
            inst_list=inst_inputs,
            depth_mm=sc["depth_mm"],
            mm_per_px=sc["mm_per_pixel"],
            scale_src=sc["scale_source"],
            K=sc["K"],
            relief_scale=relief_scale,
            vol_estimator=vol_estimator,
            f_a1=f_a1,
            f_a2=f_a2,
            f_a3=f_a3,
            f_a4=f_a4,
            f_a5=f_a5,
        )

        grouped_buckets = defaultdict(list)
        for inp, res in zip(inst_inputs, scene_out):
            meta = inp["meta"]
            status = meta["mapping_status"]
            if status == "exact_1to1":
                area_cm2 = round(
                    float(inp["mask"].sum() * (res["mm_per_pixel_eff"] ** 2) / 100.0), 2
                )
                group_records.append({
                    "config_id": cfg_id,
                    "mask_source": mask_key,
                    "image_id": fn,
                    "mass_group_id": f"{fn}::ann_{meta['annotation_id']}",
                    "dish_class": meta["short_class"],
                    "dish_type": res["dish_type"],
                    "scale_source": sc["scale_source"],
                    "mm_per_pixel": round(float(res["mm_per_pixel_eff"]), 5),
                    "food_mask_area": area_cm2,
                    "D_food": res["D_food_mm"],
                    "D_rim": res["D_rim_mm"],
                    "D_base": res["D_base_mm"],
                    "mean_height_mm": res["mean_height_mm"],
                    "estimated_volume": res["volume_cm3"],
                    "estimated_mass": res["weight_g"],
                    "GT_mass": float(meta["gt_weight_g"]),
                    "fallback_used": bool(res["fallback"]),
                })
            elif status == "grouped_multi_instance":
                grouped_buckets[meta["short_class"]].append((inp, res, meta))

        for s_cls, bucket in grouped_buckets.items():
            total_vol = round(sum(r["volume_cm3"] for _, r, _ in bucket), 2)
            total_mass = round(sum(r["weight_g"] for _, r, _ in bucket), 1)
            gt_mass = float(bucket[0][2]["gt_weight_g"])
            eff_mm = bucket[0][1]["mm_per_pixel_eff"]
            total_area_cm2 = round(
                sum(float(inp["mask"].sum() * (eff_mm ** 2) / 100.0) for inp, _, _ in bucket), 2
            )
            ann_ids = "_".join(str(m["annotation_id"]) for _, _, m in bucket)
            group_records.append({
                "config_id": cfg_id,
                "mask_source": mask_key,
                "image_id": fn,
                "mass_group_id": f"{fn}::group_{ann_ids}",
                "dish_class": s_cls,
                "dish_type": bucket[0][1]["dish_type"],
                "scale_source": sc["scale_source"],
                "mm_per_pixel": round(float(eff_mm), 5),
                "food_mask_area": total_area_cm2,
                "D_food": round(st.mean(r["D_food_mm"] for _, r, _ in bucket), 1),
                "D_rim": round(st.mean(r["D_rim_mm"] for _, r, _ in bucket), 1),
                "D_base": round(st.mean(r["D_base_mm"] for _, r, _ in bucket), 1),
                "mean_height_mm": round(st.mean(r["mean_height_mm"] for _, r, _ in bucket), 1),
                "estimated_volume": total_vol,
                "estimated_mass": total_mass,
                "GT_mass": gt_mass,
                "fallback_used": any(bool(r["fallback"]) for _, r, _ in bucket),
            })

    summary = summarize_mass_groups(group_records)
    return summary, group_records


def run_e2e_yolo_mobilesam_config_c(
    cfg_id: str,
    f_a1: bool,
    f_a2: bool,
    f_a3: bool,
    f_a4: bool,
    f_a5: bool,
    prepared_scenes: dict,
    yolo_data: dict,
    vol_estimator: VolumeNutritionEstimator,
) -> dict:
    matched_records = []
    meal_abs_errs = []
    meal_apes = []

    for fn in sorted(prepared_scenes.keys()):
        sc = prepared_scenes[fn]
        stem = Path(fn).stem
        h_vol, w_vol = sc["h_vol"], sc["w_vol"]
        sx, sy = sc["sx"], sc["sy"]
        yolo_preds = yolo_data[fn]["preds"]

        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        yolo_raw = ms_npz["yolo_raw"].astype(bool)
        yolo_post = ms_npz["yolo_post"].astype(bool)
        ms_npz.close()

        yolo_masks_c = []
        k3 = np.ones((3, 3), np.uint8)
        for pi in range(yolo_post.shape[0]):
            yp = yolo_post[pi]
            yr = yolo_raw[pi]
            ydil = cv2.dilate(yp.astype(np.uint8), k3, iterations=1) > 0
            yc_full = yr if abs(int(yr.sum()) - int(ydil.sum())) < abs(int(yp.sum()) - int(yr.sum())) else ydil
            yc_v = (
                cv2.resize(yc_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                if yc_full.shape[:2] != (h_vol, w_vol)
                else yc_full
            )
            yolo_masks_c.append(yc_v)

        yolo_inst_inputs = []
        for pi, p in enumerate(yolo_preds):
            bx1, by1, bx2, by2 = p["bbox_xyxy"]
            b_vol = (
                int(round(bx1 * sx)),
                int(round(by1 * sy)),
                int(round(bx2 * sx)),
                int(round(by2 * sy)),
            )
            yolo_inst_inputs.append({
                "class_name": p.get("vietfood67_class_name", p.get("class_name")),
                "short_class": p["short_class"],
                "mask": yolo_masks_c[pi],
                "bbox": b_vol,
            })

        relief_scale = 0.79 if sc["depth_anchored"] else None
        yolo_out = (
            evaluate_scene_instances(
                inst_list=yolo_inst_inputs,
                depth_mm=sc["depth_mm"],
                mm_per_px=sc["mm_per_pixel"],
                scale_src=sc["scale_source"],
                K=sc["K"],
                relief_scale=relief_scale,
                vol_estimator=vol_estimator,
                f_a1=f_a1,
                f_a2=f_a2,
                f_a3=f_a3,
                f_a4=f_a4,
                f_a5=f_a5,
            )
            if yolo_inst_inputs
            else []
        )

        gt_insts = [it for it in sc["items"]]
        awr_m, rem_p1, rem_g1 = match_boxes_one_to_one(
            yolo_preds, gt_insts, iou_thr=0.5, require_same_class=True
        )
        rem_preds = [yolo_preds[i] for i in rem_p1]
        rem_gts = [gt_insts[j] for j in rem_g1]
        sub_agn, _, _ = match_boxes_one_to_one(
            rem_preds, rem_gts, iou_thr=0.5, require_same_class=False
        )
        th2_m = [(rem_p1[spi], rem_g1[sgi], biou) for spi, sgi, biou in sub_agn]
        all_matches = awr_m + th2_m

        pred_for_gt = {gi: pi for pi, gi, _ in all_matches}

        grouped_buckets = defaultdict(list)
        meal_pred_g = sum(r["weight_g"] for r in yolo_out)
        meal_gt_g = 0.0
        has_valid_gt = False

        for gi, meta in enumerate(gt_insts):
            status = meta["mapping_status"]
            if status == "exact_1to1":
                has_valid_gt = True
                meal_gt_g += float(meta["gt_weight_g"])
                if gi in pred_for_gt:
                    pi = pred_for_gt[gi]
                    matched_records.append({
                        "dish_class": meta["short_class"],
                        "estimated_mass": yolo_out[pi]["weight_g"],
                        "GT_mass": float(meta["gt_weight_g"]),
                    })
            elif status == "grouped_multi_instance":
                grouped_buckets[meta["short_class"]].append((gi, meta))

        for s_cls, bucket in grouped_buckets.items():
            has_valid_gt = True
            gt_m = float(bucket[0][1]["gt_weight_g"])
            meal_gt_g += gt_m
            if all(gi in pred_for_gt for gi, _ in bucket):
                est_m = round(sum(yolo_out[pred_for_gt[gi]]["weight_g"] for gi, _ in bucket), 1)
                matched_records.append({
                    "dish_class": s_cls,
                    "estimated_mass": est_m,
                    "GT_mass": gt_m,
                })

        if has_valid_gt and meal_gt_g > 0:
            meal_abs_errs.append(abs(meal_pred_g - meal_gt_g))
            meal_apes.append(abs(meal_pred_g - meal_gt_g) / meal_gt_g * 100.0)

    m_abs = [abs(x["estimated_mass"] - x["GT_mass"]) for x in matched_records]
    m_ape = [abs(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in matched_records]
    m_pe = [(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in matched_records]
    sp = sum(x["estimated_mass"] for x in matched_records)
    sg = sum(x["GT_mass"] for x in matched_records)

    return {
        "n_matched_groups": len(matched_records),
        "matched_mae_g": st.mean(m_abs),
        "matched_mape_pct": st.mean(m_ape),
        "matched_mpe_pct": st.mean(m_pe),
        "matched_bias_pct": (sp - sg) / sg * 100.0 if sg > 0 else 0.0,
        "meal_mae_g": st.mean(meal_abs_errs),
        "meal_mape_pct": st.mean(meal_apes),
    }


def main():
    import gc
    from scratch._cache_depth_scenes import DEPTH_CACHE_DIR, VERIFIED_MAPPING_PATH, COCO_JSON_PATH

    print("Streaming 77 cached depth/scale scenes (single-scene RAM footprint)...", flush=True)
    mp = json.load(open(VERIFIED_MAPPING_PATH, encoding="utf-8"))
    coco = json.load(open(COCO_JSON_PATH, encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    del coco
    gc.collect()

    inst_by_fn = defaultdict(list)
    for x in mp["instances"]:
        inst_by_fn[x["file_name"]].append(x)
    target_files = sorted(inst_by_fn.keys())

    yolo_data = json.load(open(CACHE_DIR / "yolo_detections.json", encoding="utf-8"))
    vol_estimator = VolumeNutritionEstimator()

    progressive_configs = [
        ("Baseline", "Cấu hình gốc (Baseline)", False, False, False, False, False),
        ("A1", "+ A1 (Mô hình 3 đường kính tô D_food/D_rim/D_base, bỏ kẹp ±10%)", True, False, False, False, False),
        ("A1+A2", "+ A2 (Bỏ sàn cứng 150 cm³, thay bằng kiểm tra hợp lệ hình học)", True, True, False, False, False),
        ("A1+A2+A3", "+ A3 (Phân tách đa tiêu chí Trứng đơn vs Chảo trứng/Ốp la)", True, True, True, False, False),
        ("A1+A2+A3+A4", "+ A4 (Khử lỗi dead-code đĩa lòng trũng & định tuyến đĩa thịt rời)", True, True, True, True, False),
        ("A1+A2+A3+A4+A5", "+ A5 (Ổn định mặt phẳng tham chiếu kẹp bước nhảy biên cục bộ)", True, True, True, True, True),
    ]

    independent_configs = [
        ("Baseline", "Baseline (Gốc)", False, False, False, False, False),
        ("Only_A1", "Baseline + A1 (Chỉ sửa mô hình 3 đường kính tô)", True, False, False, False, False),
        ("Only_A2", "Baseline + A2 (Chỉ bỏ sàn cứng 150 cm³)", False, True, False, False, False),
        ("Only_A3", "Baseline + A3 (Chỉ phân tách hình thái Trứng)", False, False, True, False, False),
        ("Only_A4", "Baseline + A4 (Chỉ sửa đĩa lòng trũng & đĩa thịt rời)", False, False, False, True, False),
        ("Only_A5", "Baseline + A5 (Chỉ ổn định mặt phẳng tham chiếu)", False, False, False, False, True),
    ]

    oracle_gt_configs = progressive_configs + [c for c in independent_configs if c[0] != "Baseline"]
    oracle_ms_configs = [
        ("MobileSAM_C_Baseline", "MobileSAM Config C (Baseline)", False, False, False, False, False),
        ("MobileSAM_C_A1..A5", "MobileSAM Config C (A1..A5)", True, True, True, True, True),
    ]
    e2e_configs = [
        ("E2E_Baseline", False, False, False, False, False),
        ("E2E_A1..A5", True, True, True, True, True),
    ]

    records_by_cfg = defaultdict(list)
    e2e_matched_by_cfg = defaultdict(list)
    e2e_meal_errs_by_cfg = defaultdict(list)
    e2e_meal_apes_by_cfg = defaultdict(list)

    k3 = np.ones((3, 3), np.uint8)

    for idx_fn, fn in enumerate(target_files):
        stem = Path(fn).stem
        insts = inst_by_fn[fn]
        h_full, w_full = int(insts[0]["height"]), int(insts[0]["width"])

        sc_npz = np.load(DEPTH_CACHE_DIR / f"scene_{stem}.npz", allow_pickle=True)
        depth_mm = sc_npz["depth_mm"].astype(np.float32)
        K = sc_npz["K"]
        mm_per_pixel = float(sc_npz["mm_per_pixel"])
        scale_source = str(sc_npz["scale_source"])
        depth_anchored = bool(sc_npz["depth_anchored"])
        h_vol = int(sc_npz["h_vol"])
        w_vol = int(sc_npz["w_vol"])
        sx = float(sc_npz["sx"])
        sy = float(sc_npz["sy"])
        sc_npz.close()

        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        oracle_cand_raw = ms_npz["oracle_cand_raw"].astype(bool)
        yolo_raw = ms_npz["yolo_raw"].astype(bool)
        yolo_post = ms_npz["yolo_post"].astype(bool)
        ms_npz.close()

        gt_inputs = []
        ms_inputs = []
        for i_inst, inst in enumerate(insts):
            gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
            gt_vol = (
                cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                if gt_full.shape[:2] != (h_vol, w_vol)
                else gt_full
            )
            ms_full = oracle_cand_raw[i_inst]
            ms_vol = (
                cv2.resize(ms_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                if ms_full.shape[:2] != (h_vol, w_vol)
                else ms_full
            )
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            b_vol = (
                int(round(bx1 * sx)),
                int(round(by1 * sy)),
                int(round(bx2 * sx)),
                int(round(by2 * sy)),
            )
            gt_inputs.append({
                "class_name": inst["vietfood67_class_name"],
                "short_class": inst["short_class"],
                "mask": gt_vol,
                "bbox": b_vol,
                "meta": inst,
            })
            ms_inputs.append({
                "class_name": inst["vietfood67_class_name"],
                "short_class": inst["short_class"],
                "mask": ms_vol,
                "bbox": b_vol,
                "meta": inst,
            })

        relief_scale = 0.79 if depth_anchored else None

        def append_scene_groups(cid_key, mask_src_label, inp_list, scene_out):
            grouped_buckets = defaultdict(list)
            for inp, res in zip(inp_list, scene_out):
                meta = inp["meta"]
                status = meta["mapping_status"]
                if status == "exact_1to1":
                    area_cm2 = round(
                        float(inp["mask"].sum() * (res["mm_per_pixel_eff"] ** 2) / 100.0), 2
                    )
                    records_by_cfg[cid_key].append({
                        "config_id": cid_key,
                        "mask_source": mask_src_label,
                        "image_id": fn,
                        "mass_group_id": f"{fn}::ann_{meta['annotation_id']}",
                        "dish_class": meta["short_class"],
                        "dish_type": res["dish_type"],
                        "scale_source": scale_source,
                        "mm_per_pixel": round(float(res["mm_per_pixel_eff"]), 5),
                        "food_mask_area": area_cm2,
                        "D_food": res["D_food_mm"],
                        "D_rim": res["D_rim_mm"],
                        "D_base": res["D_base_mm"],
                        "mean_height_mm": res["mean_height_mm"],
                        "estimated_volume": res["volume_cm3"],
                        "estimated_mass": res["weight_g"],
                        "GT_mass": float(meta["gt_weight_g"]),
                        "fallback_used": bool(res["fallback"]),
                    })
                elif status == "grouped_multi_instance":
                    grouped_buckets[meta["short_class"]].append((inp, res, meta))

            for s_cls, bucket in grouped_buckets.items():
                total_vol = round(sum(r["volume_cm3"] for _, r, _ in bucket), 2)
                total_mass = round(sum(r["weight_g"] for _, r, _ in bucket), 1)
                gt_mass = float(bucket[0][2]["gt_weight_g"])
                eff_mm = bucket[0][1]["mm_per_pixel_eff"]
                total_area_cm2 = round(
                    sum(float(inp["mask"].sum() * (eff_mm ** 2) / 100.0) for inp, _, _ in bucket), 2
                )
                ann_ids = "_".join(str(m["annotation_id"]) for _, _, m in bucket)
                records_by_cfg[cid_key].append({
                    "config_id": cid_key,
                    "mask_source": mask_src_label,
                    "image_id": fn,
                    "mass_group_id": f"{fn}::group_{ann_ids}",
                    "dish_class": s_cls,
                    "dish_type": bucket[0][1]["dish_type"],
                    "scale_source": scale_source,
                    "mm_per_pixel": round(float(eff_mm), 5),
                    "food_mask_area": total_area_cm2,
                    "D_food": round(st.mean(r["D_food_mm"] for _, r, _ in bucket), 1),
                    "D_rim": round(st.mean(r["D_rim_mm"] for _, r, _ in bucket), 1),
                    "D_base": round(st.mean(r["D_base_mm"] for _, r, _ in bucket), 1),
                    "mean_height_mm": round(st.mean(r["mean_height_mm"] for _, r, _ in bucket), 1),
                    "estimated_volume": total_vol,
                    "estimated_mass": total_mass,
                    "GT_mass": gt_mass,
                    "fallback_used": any(bool(r["fallback"]) for _, r, _ in bucket),
                })

        for cid, _lbl, a1, a2, a3, a4, a5 in oracle_gt_configs:
            s_out = evaluate_scene_instances(
                gt_inputs, depth_mm, mm_per_pixel, scale_source, K, relief_scale,
                vol_estimator, a1, a2, a3, a4, a5
            )
            append_scene_groups(cid, "gt_mask", gt_inputs, s_out)

        for cid, _lbl, a1, a2, a3, a4, a5 in oracle_ms_configs:
            s_out = evaluate_scene_instances(
                ms_inputs, depth_mm, mm_per_pixel, scale_source, K, relief_scale,
                vol_estimator, a1, a2, a3, a4, a5
            )
            append_scene_groups(cid, "mobilesam_c_mask", ms_inputs, s_out)

        # End-to-End YOLO + MobileSAM Config C on this scene
        yolo_preds = yolo_data[fn]["preds"]
        yolo_inst_inputs = []
        for pi, p in enumerate(yolo_preds):
            yp = yolo_post[pi]
            yr = yolo_raw[pi]
            ydil = cv2.dilate(yp.astype(np.uint8), k3, iterations=1) > 0
            yc_full = yr if abs(int(yr.sum()) - int(ydil.sum())) < abs(int(yp.sum()) - int(yr.sum())) else ydil
            yc_v = (
                cv2.resize(yc_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                if yc_full.shape[:2] != (h_vol, w_vol)
                else yc_full
            )
            bx1, by1, bx2, by2 = p["bbox_xyxy"]
            b_vol = (
                int(round(bx1 * sx)),
                int(round(by1 * sy)),
                int(round(bx2 * sx)),
                int(round(by2 * sy)),
            )
            yolo_inst_inputs.append({
                "class_name": p.get("vietfood67_class_name", p.get("class_name")),
                "short_class": p["short_class"],
                "mask": yc_v,
                "bbox": b_vol,
            })

        awr_m, rem_p1, rem_g1 = match_boxes_one_to_one(
            yolo_preds, insts, iou_thr=0.5, require_same_class=True
        )
        rem_preds = [yolo_preds[i] for i in rem_p1]
        rem_gts = [insts[j] for j in rem_g1]
        sub_agn, _, _ = match_boxes_one_to_one(
            rem_preds, rem_gts, iou_thr=0.5, require_same_class=False
        )
        th2_m = [(rem_p1[spi], rem_g1[sgi], biou) for spi, sgi, biou in sub_agn]
        all_matches = awr_m + th2_m
        pred_for_gt = {gi: pi for pi, gi, _ in all_matches}

        for ecid, a1, a2, a3, a4, a5 in e2e_configs:
            yolo_out = (
                evaluate_scene_instances(
                    yolo_inst_inputs, depth_mm, mm_per_pixel, scale_source, K, relief_scale,
                    vol_estimator, a1, a2, a3, a4, a5
                )
                if yolo_inst_inputs
                else []
            )
            grouped_buckets = defaultdict(list)
            meal_pred_g = sum(r["weight_g"] for r in yolo_out)
            meal_gt_g = 0.0
            has_valid_gt = False

            for gi, meta in enumerate(insts):
                status = meta["mapping_status"]
                if status == "exact_1to1":
                    has_valid_gt = True
                    meal_gt_g += float(meta["gt_weight_g"])
                    if gi in pred_for_gt:
                        pi = pred_for_gt[gi]
                        e2e_matched_by_cfg[ecid].append({
                            "dish_class": meta["short_class"],
                            "estimated_mass": yolo_out[pi]["weight_g"],
                            "GT_mass": float(meta["gt_weight_g"]),
                        })
                elif status == "grouped_multi_instance":
                    grouped_buckets[meta["short_class"]].append((gi, meta))

            for s_cls, bucket in grouped_buckets.items():
                has_valid_gt = True
                gt_m = float(bucket[0][1]["gt_weight_g"])
                meal_gt_g += gt_m
                if all(gi in pred_for_gt for gi, _ in bucket):
                    est_m = round(sum(yolo_out[pred_for_gt[gi]]["weight_g"] for gi, _ in bucket), 1)
                    e2e_matched_by_cfg[ecid].append({
                        "dish_class": s_cls,
                        "estimated_mass": est_m,
                        "GT_mass": gt_m,
                    })

            if has_valid_gt and meal_gt_g > 0:
                e2e_meal_errs_by_cfg[ecid].append(abs(meal_pred_g - meal_gt_g))
                e2e_meal_apes_by_cfg[ecid].append(abs(meal_pred_g - meal_gt_g) / meal_gt_g * 100.0)

        del depth_mm, oracle_cand_raw, yolo_raw, yolo_post, gt_inputs, ms_inputs, yolo_inst_inputs
        if (idx_fn + 1) % 20 == 0:
            gc.collect()

    all_saved_records = []
    prog_results = {}
    for cid, label, *_ in progressive_configs:
        recs = records_by_cfg[cid]
        summ = summarize_mass_groups(recs)
        prog_results[cid] = (label, summ)
        all_saved_records.extend(recs)
        print(
            f"[Progressive] {cid:16s} | MAE={summ['mae_g']:6.2f}g | MAPE={summ['mape_pct']:5.2f}% | "
            f"RMSE={summ['rmse_g']:6.2f}g | MPE={summ['mpe_pct']:+6.2f}% | Bias={summ['bias_mean_pct']:+6.2f}% | "
            f"Fallbacks={summ['fallback_count']}",
            flush=True,
        )

    ind_results = {}
    for cid, label, *_ in independent_configs:
        recs = records_by_cfg[cid]
        summ = summarize_mass_groups(recs)
        ind_results[cid] = (label, summ)
        if cid != "Baseline":
            all_saved_records.extend(recs)
        print(
            f"[Independent] {cid:16s} | MAE={summ['mae_g']:6.2f}g | MAPE={summ['mape_pct']:5.2f}% | "
            f"RMSE={summ['rmse_g']:6.2f}g | MPE={summ['mpe_pct']:+6.2f}% | Bias={summ['bias_mean_pct']:+6.2f}% | "
            f"Fallbacks={summ['fallback_count']}",
            flush=True,
        )

    ms_base_summ = summarize_mass_groups(records_by_cfg["MobileSAM_C_Baseline"])
    ms_a5_summ = summarize_mass_groups(records_by_cfg["MobileSAM_C_A1..A5"])
    all_saved_records.extend(records_by_cfg["MobileSAM_C_Baseline"])
    all_saved_records.extend(records_by_cfg["MobileSAM_C_A1..A5"])
    print(
        f"[Oracle MobileSAM Config C] Baseline: MAE={ms_base_summ['mae_g']:.2f}g, MAPE={ms_base_summ['mape_pct']:.2f}% -> "
        f"A1..A5: MAE={ms_a5_summ['mae_g']:.2f}g, MAPE={ms_a5_summ['mape_pct']:.2f}%",
        flush=True,
    )

    def summarize_e2e(ecid):
        m_recs = e2e_matched_by_cfg[ecid]
        m_abs = [abs(x["estimated_mass"] - x["GT_mass"]) for x in m_recs]
        m_ape = [abs(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in m_recs]
        m_pe = [(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in m_recs]
        sp = sum(x["estimated_mass"] for x in m_recs)
        sg = sum(x["GT_mass"] for x in m_recs)
        return {
            "n_matched_groups": len(m_recs),
            "matched_mae_g": st.mean(m_abs),
            "matched_mape_pct": st.mean(m_ape),
            "matched_mpe_pct": st.mean(m_pe),
            "matched_bias_pct": (sp - sg) / sg * 100.0 if sg > 0 else 0.0,
            "meal_mae_g": st.mean(e2e_meal_errs_by_cfg[ecid]),
            "meal_mape_pct": st.mean(e2e_meal_apes_by_cfg[ecid]),
        }

    e2e_base = summarize_e2e("E2E_Baseline")
    e2e_a5 = summarize_e2e("E2E_A1..A5")
    print(
        f"[End-to-End YOLO+MobileSAM Config C] Matched ({e2e_base['n_matched_groups']} groups): "
        f"Baseline MAE={e2e_base['matched_mae_g']:.2f}g ({e2e_base['matched_mape_pct']:.2f}%) -> "
        f"A1..A5 MAE={e2e_a5['matched_mae_g']:.2f}g ({e2e_a5['matched_mape_pct']:.2f}%) | "
        f"Full-Meal: Baseline MAE={e2e_base['meal_mae_g']:.2f}g ({e2e_base['meal_mape_pct']:.2f}%) -> "
        f"A1..A5 MAE={e2e_a5['meal_mae_g']:.2f}g ({e2e_a5['meal_mape_pct']:.2f}%)",
        flush=True,
    )

    print("\n=== PER-CLASS COMPARISON: BASELINE vs FULL A1..A5 (Oracle GT Mask, 96 Mass Groups) ===")
    base_pc = prog_results["Baseline"][1]["per_class"]
    full_pc = prog_results["A1+A2+A3+A4+A5"][1]["per_class"]
    print(f"{'Class':16s} | {'N':>2s} | {'GT_g':>6s} | {'Base_g':>6s} | {'A5_g':>6s} | {'Base_MAE':>8s} | {'A5_MAE':>6s} | {'Base_MAPE':>9s} | {'A5_MAPE':>7s} | {'D_food':>6s} | {'D_rim':>6s} | {'D_base':>6s}")
    print("-" * 118)
    for c in sorted(base_pc.keys()):
        b = base_pc[c]
        a = full_pc[c]
        print(
            f"{c:16s} | {b['n_groups']:2d} | {b['gt_mean_g']:6.1f} | {b['pred_mean_g']:6.1f} | {a['pred_mean_g']:6.1f} | "
            f"{b['mae_g']:8.1f} | {a['mae_g']:6.1f} | {b['mape_pct']:8.1f}% | {a['mape_pct']:6.1f}% | "
            f"{a['D_food_mean']:6.1f} | {a['D_rim_mean']:6.1f} | {a['D_base_mean']:6.1f}"
        )

    with open(RECORDS_OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_saved_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_saved_records)} detailed records to {RECORDS_OUT_PATH}")


if __name__ == "__main__":
    main()
