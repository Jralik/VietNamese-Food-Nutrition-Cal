# -*- coding: utf-8 -*-
"""run_aruco_ablation.py

5-Configuration ArUco & Depth-Anchoring Ablation Study on the 77-image /
96-mass-group benchmark (`Oracle GT Polygon Mask` + `Full A1..A5`).

Configurations (single-variable step progression):
  1. Config_1_GT_Assisted_Plate_Heuristic
       - scale_src                     = "plate_heuristic" | "image_heuristic" (77/77 scenes)
       - mm_per_pixel                  = heuristic_mm_per_pixel (with container override
                                         expected_dia / max(bw, bh) from Oracle GT bbox/class)
       - depth_mm                      = raw_depth_mm (alpha_anchor = 1.0)
       - use_depth_intrinsics_for_area = False (px_area = mm_per_pixel ** 2)
       - relief_scale                  = 1.0 (fixed float; relief_scale_factor bypassed)
       * Note on Config 1 Data Source: Because all 5 configurations are evaluated in
         the `Oracle GT Polygon Mask` setting (where instance masks, bounding boxes,
         and food classes come from `gt_mapping_verified.json`), Config 1 uses those
         same Oracle GT bounding boxes and classes for container/plate heuristic scale
         recovery (`scaled_dets` and `expected_dia / max(bw, bh)`). It is therefore an
         upper-bound **GT-Assisted Plate Heuristic** that isolates the scale/anchoring
         mechanism without confounding from YOLO detection misses or misclassifications.
  2. Config_2_ArUco_XY_Only
       - scale_src                     = aruco_scale_source (76 ArUco + 1 fallback)
       - mm_per_pixel                  = aruco_mm_per_pixel
       - depth_mm                      = raw_depth_mm (alpha_anchor = 1.0)
       - use_depth_intrinsics_for_area = False (px_area = mm_per_pixel ** 2)
       - relief_scale                  = 1.0 (fixed float; relief_scale_factor bypassed)
  3. Config_3a_ArUco_XY_Depth_Anchor
       - scale_src                     = aruco_scale_source (76 ArUco + 1 fallback)
       - mm_per_pixel                  = aruco_mm_per_pixel
       - depth_mm                      = anchored_depth_mm (raw_depth_mm * scale_anchor_poly)
       - use_depth_intrinsics_for_area = False (px_area = mm_per_pixel ** 2)
       - relief_scale                  = 1.0 (fixed float; relief_scale_factor bypassed)
  4. Config_3a_Intrinsics
       - scale_src                     = aruco_scale_source (76 ArUco + 1 fallback)
       - mm_per_pixel                  = aruco_mm_per_pixel
       - depth_mm                      = anchored_depth_mm (raw_depth_mm * scale_anchor_poly)
       - use_depth_intrinsics_for_area = True (px_area = (Z_i ** 2) / (fx * fy) on bowl mound)
       - relief_scale                  = 1.0 (fixed float; relief_scale_factor bypassed)
  5. Config_3b_Full_ArUco (Current A1..A5 Production Baseline)
       - scale_src                     = aruco_scale_source (76 ArUco + 1 fallback)
       - mm_per_pixel                  = aruco_mm_per_pixel
       - depth_mm                      = anchored_depth_mm (raw_depth_mm * scale_anchor_poly)
       - use_depth_intrinsics_for_area = True (px_area = (Z_i ** 2) / (fx * fy) on bowl mound)
       - relief_scale                  = 0.79 on 76 anchored scenes, 1.0 on 1 fallback scene

Outputs (without modifying any existing files):
  - scratch/aruco_ablation_records.jsonl
  - scratch/aruco_ablation_report.md
"""

import sys
import os
import json
import math
import statistics as st
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask
from volume_nutrition import (
    VolumeNutritionEstimator,
    fit_plane_to_boundary,
    relief_scale_factor,
    compute_volume_bbox_approximation,
    compute_volume_side_vegetables,
    SATURATION_CAP_RATIO,
)
from scratch.run_volume_ablation import (
    fit_plane_stabilized,
    eval_bowl_volume_ablation as eval_bowl_volume_ablation_orig,
    eval_column_integration_ablation,
    eval_egg_and_small_ingredient_ablation,
    summarize_mass_groups,
)
from scratch._build_aruco_ablation_cache import (
    COCO_JSON_PATH,
    VERIFIED_MAPPING_PATH,
    ARUCO_ABLATION_CACHE_DIR,
    build_or_verify_aruco_ablation_cache,
)

REF_RECORDS_PATH = PROJECT_ROOT / "scratch" / "volume_ablation_records.jsonl"
OUT_RECORDS_PATH = PROJECT_ROOT / "scratch" / "aruco_ablation_records.jsonl"
OUT_REPORT_PATH = PROJECT_ROOT / "scratch" / "aruco_ablation_report.md"


def eval_bowl_volume_aruco_ablation(
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
    use_depth_intrinsics_for_area: bool = True,
) -> dict:
    """Identical to `eval_bowl_volume_ablation` in `scratch/run_volume_ablation.py`
    (lines 119-325), except for the explicit boolean flag `use_depth_intrinsics_for_area`
    controlling the per-pixel projected area formula for `above_rim_volume_cm3`.

    Unit & Per-Pixel Array Specification for `above_rim_volume_cm3`:
      - `interior_d = depth_map_mm[eroded]`: 1D float32 array of per-pixel depths (mm)
        inside the eroded bowl mask.
      - `pos_mound = clipped_heights[clipped_heights > 0]`: 1D array of positive
        heights above rim plane (mm) for pixels where `raw_heights_above_rim > 0`.
      - `interior_masked = interior_d[raw_heights_above_rim > 0]`: 1D array of
        individual per-pixel camera-axis depths Z_i (mm), aligned 1-to-1 with `pos_mound`.
      - When `use_depth_intrinsics_for_area=True` (and ArUco/checkerboard + valid K):
          px_area_i = (Z_i ** 2) / (f_x * f_y)   [units: mm^2 / px^2 * px^2 = mm^2]
        which computes the pinhole projected area on the plane perpendicular to the
        camera optical axis at each pixel's individual depth Z_i.
      - When `use_depth_intrinsics_for_area=False`:
          px_area = mm_per_pixel ** 2            [units: mm^2]
        using the lateral scale `mm_per_pixel` directly without referencing depth Z_i.
    """
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
    used_intrinsics_area_branch = False

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
        f_nom = BOWL_NOISE_FILL_RATIO if has_valid_depth else BOWL_DEFAULT_FILL_RATIO
        surface_to_rim_ratio = BOWL_BASE_RATIO + (1.0 - BOWL_BASE_RATIO) * f_nom

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
                    use_depth_intrinsics_for_area
                    and scale_source in ("aruco_marker", "checkerboard")
                    and K is not None
                    and K[0, 0] > 0
                    and K[1, 1] > 0
                ):
                    px_area = (interior_masked ** 2) / (float(K[0, 0]) * float(K[1, 1]))
                    used_intrinsics_area_branch = True
                else:
                    px_area = mm_per_pixel ** 2
                    used_intrinsics_area_branch = False
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
        "above_rim_volume_cm3": round(float(above_rim_volume_cm3), 3),
        "used_intrinsics_area_branch": used_intrinsics_area_branch,
    }


def compute_single_instance_aruco_ablation(
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
    use_depth_intrinsics_for_area: bool = True,
    custom_plane_z: np.ndarray = None,
    override_min_h: float = None,
    override_max_h: float = None,
    verify_bowl_equiv: bool = False,
    bowl_check_counter: dict = None,
    inst_meta: dict = None,
) -> dict:
    """Identical to `compute_single_instance_ablation` in `scratch/run_volume_ablation.py`,
    forwarding `use_depth_intrinsics_for_area` to `eval_bowl_volume_aruco_ablation`
    and executing complete Gate 3 equivalence and branch-safety checks when `verify_bowl_equiv=True`.
    """
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

    use_bowl_model = dish_type in ("soup_bowl", "large_bowl", "rice_bowl")

    if use_bowl_model:
        eff_a1 = f_a1 and cls_name != "Xoi (Sticky rice)"
        eff_a2 = f_a2 and cls_name != "Xoi (Sticky rice)"
        res = eval_bowl_volume_aruco_ablation(
            depth_map_mm=depth_mm,
            food_mask=mask,
            bbox=bbox,
            mm_per_pixel=mm_per_px,
            dish_type=dish_type,
            K=K,
            scale_source=scale_src,
            relief_scale=relief_scale,
            f_a1=eff_a1,
            f_a2=eff_a2,
            f_a5=f_a5,
            use_depth_intrinsics_for_area=use_depth_intrinsics_for_area,
        )

        # Gate 3 complete verification on all bowl instances
        if verify_bowl_equiv:
            assert use_depth_intrinsics_for_area is True
            # (1) Numerical equivalence within tolerance (rel_tol=1e-9, abs_tol=1e-9)
            #     against original eval_bowl_volume_ablation when True
            res_orig = eval_bowl_volume_ablation_orig(
                depth_map_mm=depth_mm,
                food_mask=mask,
                bbox=bbox,
                mm_per_pixel=mm_per_px,
                dish_type=dish_type,
                K=K,
                scale_source=scale_src,
                relief_scale=relief_scale,
                f_a1=eff_a1,
                f_a2=eff_a2,
                f_a5=f_a5,
            )
            for k_orig in ("volume_cm3", "D_food_mm", "D_rim_mm", "D_base_mm", "mean_height_mm", "fallback"):
                if isinstance(res_orig[k_orig], float):
                    assert math.isclose(res[k_orig], res_orig[k_orig], rel_tol=1e-9, abs_tol=1e-9), (
                        f"Gate 3 bowl mismatch on key {k_orig}: {res[k_orig]} != {res_orig[k_orig]}"
                    )
                else:
                    assert res[k_orig] == res_orig[k_orig], (
                        f"Gate 3 bowl mismatch on key {k_orig}: {res[k_orig]} != {res_orig[k_orig]}"
                    )

            # (2) use_depth_intrinsics_for_area=False -> must use mm_per_pixel ** 2
            res_false = eval_bowl_volume_aruco_ablation(
                depth_map_mm=depth_mm,
                food_mask=mask,
                bbox=bbox,
                mm_per_pixel=mm_per_px,
                dish_type=dish_type,
                K=K,
                scale_source=scale_src,
                relief_scale=relief_scale,
                f_a1=eff_a1,
                f_a2=eff_a2,
                f_a5=f_a5,
                use_depth_intrinsics_for_area=False,
            )
            assert res_false["used_intrinsics_area_branch"] is False

            # (3) K=None -> must safely fall back to mm_per_pixel ** 2 without exception
            res_k_none = eval_bowl_volume_aruco_ablation(
                depth_map_mm=depth_mm,
                food_mask=mask,
                bbox=bbox,
                mm_per_pixel=mm_per_px,
                dish_type=dish_type,
                K=None,
                scale_source=scale_src,
                relief_scale=relief_scale,
                f_a1=eff_a1,
                f_a2=eff_a2,
                f_a5=f_a5,
                use_depth_intrinsics_for_area=True,
            )
            assert res_k_none["used_intrinsics_area_branch"] is False

            # (4) Invalid camera intrinsics (fx <= 0 or fy <= 0) -> must fall back safely
            K_bad_fx = np.array([[0.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            K_bad_fy = np.array([[1000.0, 0.0, 500.0], [0.0, -5.0, 500.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            res_bad_fx = eval_bowl_volume_aruco_ablation(
                depth_map_mm=depth_mm,
                food_mask=mask,
                bbox=bbox,
                mm_per_pixel=mm_per_px,
                dish_type=dish_type,
                K=K_bad_fx,
                scale_source=scale_src,
                relief_scale=relief_scale,
                f_a1=eff_a1,
                f_a2=eff_a2,
                f_a5=f_a5,
                use_depth_intrinsics_for_area=True,
            )
            res_bad_fy = eval_bowl_volume_aruco_ablation(
                depth_map_mm=depth_mm,
                food_mask=mask,
                bbox=bbox,
                mm_per_pixel=mm_per_px,
                dish_type=dish_type,
                K=K_bad_fy,
                scale_source=scale_src,
                relief_scale=relief_scale,
                f_a1=eff_a1,
                f_a2=eff_a2,
                f_a5=f_a5,
                use_depth_intrinsics_for_area=True,
            )
            assert res_bad_fx["used_intrinsics_area_branch"] is False
            assert res_bad_fy["used_intrinsics_area_branch"] is False

            # (5) Non-ArUco scale_source ("plate_heuristic") -> must fall back to mm_per_pixel ** 2
            res_non_aruco = eval_bowl_volume_aruco_ablation(
                depth_map_mm=depth_mm,
                food_mask=mask,
                bbox=bbox,
                mm_per_pixel=mm_per_px,
                dish_type=dish_type,
                K=K,
                scale_source="plate_heuristic",
                relief_scale=relief_scale,
                f_a1=eff_a1,
                f_a2=eff_a2,
                f_a5=f_a5,
                use_depth_intrinsics_for_area=True,
            )
            assert res_non_aruco["used_intrinsics_area_branch"] is False

            if bowl_check_counter is not None:
                assert inst_meta is not None, (
                    "Gate 3: inst_meta is required when bowl_check_counter is active"
                )
                status = inst_meta.get("mapping_status")
                gt_w = inst_meta.get("gt_weight_g")
                if status in ("exact_1to1", "grouped_multi_instance") and gt_w is not None:
                    bucket_key = "valid"
                elif (
                    inst_meta.get("annotation_id") == 96
                    and inst_meta.get("file_name") == "com-suon1 (2).jpg"
                    and inst_meta.get("vietfood67_class_name") == "Canh (Soup)"
                    and dish_type == "soup_bowl"
                    and status == "excluded_unverified_line_mismatch"
                    and gt_w is None
                ):
                    bucket_key = "excluded"
                    bowl_check_counter["excluded"]["items"].append({
                        "annotation_id": inst_meta.get("annotation_id"),
                        "file_name": inst_meta.get("file_name"),
                        "class_name": inst_meta.get("vietfood67_class_name"),
                        "dish_type": dish_type,
                        "mapping_status": status,
                        "gt_weight_g": gt_w,
                    })
                else:
                    raise AssertionError(
                        f"Gate 3: Unexpected bowl instance metadata during equivalence check: {inst_meta}"
                    )

                for bk in (bucket_key, "total"):
                    bowl_check_counter[bk]["checked"] += 1
                    bowl_check_counter[bk]["by_type"][dish_type] = (
                        bowl_check_counter[bk]["by_type"].get(dish_type, 0) + 1
                    )
                    if res["above_rim_volume_cm3"] > 0:
                        bowl_check_counter[bk]["with_mound"] += 1

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


def evaluate_scene_instances_aruco(
    inst_list: list,
    depth_mm: np.ndarray,
    mm_per_px: float,
    scale_src: str,
    K: np.ndarray,
    relief_scale: float,
    vol_estimator: VolumeNutritionEstimator,
    use_depth_intrinsics_for_area: bool = True,
    verify_bowl_equiv: bool = False,
    bowl_check_counter: dict = None,
) -> list:
    """Evaluate all instances in a single scene with Full A1..A5 enabled."""
    assert relief_scale is not None, "relief_scale must be an explicit float (never None)!"
    f_a1 = f_a2 = f_a3 = f_a4 = f_a5 = True
    n = len(inst_list)
    results = []
    for inst in inst_list:
        cls_name = inst["class_name"]
        dish_type = pc.CLASS_TO_DISH_TYPE.get(cls_name, "default")
        res = compute_single_instance_aruco_ablation(
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
            use_depth_intrinsics_for_area=use_depth_intrinsics_for_area,
            verify_bowl_equiv=verify_bowl_equiv,
            bowl_check_counter=bowl_check_counter,
            inst_meta=inst.get("meta"),
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

                        res_top_new = compute_single_instance_aruco_ablation(
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
                            use_depth_intrinsics_for_area=use_depth_intrinsics_for_area,
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


def verify_gates_3_4_5(
    records_by_cfg: dict,
    summaries_by_cfg: dict,
    bowl_check_counter: dict,
):
    """Execute Gate 3 (bowl equivalence & branch safety), Gate 4 (per-group reproduction),
    and Gate 5 (unrounded & rounded aggregate reproduction).
    """
    # 1. Verify all 5 configs have the exact same 96 mass_group_ids, GT_mass, and density
    cfg_names = list(records_by_cfg.keys())
    base_ids = [r["mass_group_id"] for r in records_by_cfg[cfg_names[0]]]
    assert len(base_ids) == 96 and len(set(base_ids)) == 96, (
        f"Expected 96 unique mass groups, got {len(base_ids)}"
    )
    for cname in cfg_names[1:]:
        ids_c = [r["mass_group_id"] for r in records_by_cfg[cname]]
        assert ids_c == base_ids, f"mass_group_id ordering/set mismatch in {cname}"
        for r0, rc in zip(records_by_cfg[cfg_names[0]], records_by_cfg[cname]):
            assert math.isclose(r0["GT_mass"], rc["GT_mass"], rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(r0["density"], rc["density"], rel_tol=1e-9, abs_tol=1e-9)

    # 2. Verify Config 1 uses 100% heuristic scale_source (77/77 scenes -> 96/96 groups)
    for r1 in records_by_cfg["Config_1_GT_Assisted_Plate_Heuristic"]:
        assert r1["scale_source"] in ("plate_heuristic", "image_heuristic"), (
            f"Gate 2 violation in Config 1: {r1['mass_group_id']} has scale_source={r1['scale_source']}"
        )
        assert math.isclose(r1["scale_anchor"], 1.0, rel_tol=1e-9)
        assert math.isclose(r1["relief_scale"], 1.0, rel_tol=1e-9)

    # 3. Gate 3 check: bowl instances count & breakdown by dish_type,
    #    explicitly separated into valid evaluated instances vs excluded context instance:
    #      - Valid (in the 100 valid GT instances / 96 mass groups):
    #          soup_bowl  = Bun bo Hue (5) + Bun rieu (7) + Canh (2) + Pho (5) + Sup cua (6) = 25
    #          large_bowl = 0
    #          rice_bowl  = Thit kho (6) + Xoi (7) = 13
    #          total valid = 38
    #      - Excluded context instance (evaluated in scene geometry, filtered before JSONL/metrics):
    #          annotation_id=96, file_name='com-suon1 (2).jpg', Canh (Soup) / soup_bowl,
    #          mapping_status='excluded_unverified_line_mismatch', gt_weight_g=None -> 1 soup_bowl
    #      - Total processed across all 102 annotations = 26 soup_bowl + 0 large_bowl + 13 rice_bowl = 39
    c_val = bowl_check_counter["valid"]
    c_exc = bowl_check_counter["excluded"]
    c_tot = bowl_check_counter["total"]

    assert c_val["by_type"].get("soup_bowl", 0) == 25, (
        f"Gate 3 (valid): Expected 25 soup_bowl instances, got {c_val['by_type'].get('soup_bowl', 0)}"
    )
    assert c_val["by_type"].get("large_bowl", 0) == 0, (
        f"Gate 3 (valid): Expected 0 large_bowl instances, got {c_val['by_type'].get('large_bowl', 0)}"
    )
    assert c_val["by_type"].get("rice_bowl", 0) == 13, (
        f"Gate 3 (valid): Expected 13 rice_bowl instances, got {c_val['by_type'].get('rice_bowl', 0)}"
    )
    assert c_val["checked"] == 38, (
        f"Gate 3 (valid): Expected 38 valid bowl instances, got {c_val['checked']}"
    )

    assert c_exc["by_type"].get("soup_bowl", 0) == 1, (
        f"Gate 3 (excluded): Expected 1 soup_bowl instance, got {c_exc['by_type'].get('soup_bowl', 0)}"
    )
    assert c_exc["by_type"].get("large_bowl", 0) == 0, (
        f"Gate 3 (excluded): Expected 0 large_bowl instances, got {c_exc['by_type'].get('large_bowl', 0)}"
    )
    assert c_exc["by_type"].get("rice_bowl", 0) == 0, (
        f"Gate 3 (excluded): Expected 0 rice_bowl instances, got {c_exc['by_type'].get('rice_bowl', 0)}"
    )
    assert c_exc["checked"] == 1, (
        f"Gate 3 (excluded): Expected 1 excluded bowl instance, got {c_exc['checked']}"
    )
    assert c_exc["items"] == [{
        "annotation_id": 96,
        "file_name": "com-suon1 (2).jpg",
        "class_name": "Canh (Soup)",
        "dish_type": "soup_bowl",
        "mapping_status": "excluded_unverified_line_mismatch",
        "gt_weight_g": None,
    }], f"Gate 3 (excluded): Unexpected excluded item breakdown: {c_exc['items']}"

    assert c_tot["by_type"].get("soup_bowl", 0) == 26, (
        f"Gate 3 (total): Expected 26 soup_bowl instances, got {c_tot['by_type'].get('soup_bowl', 0)}"
    )
    assert c_tot["by_type"].get("large_bowl", 0) == 0, (
        f"Gate 3 (total): Expected 0 large_bowl instances, got {c_tot['by_type'].get('large_bowl', 0)}"
    )
    assert c_tot["by_type"].get("rice_bowl", 0) == 13, (
        f"Gate 3 (total): Expected 13 rice_bowl instances, got {c_tot['by_type'].get('rice_bowl', 0)}"
    )
    assert c_tot["checked"] == 39 and c_tot["checked"] == c_val["checked"] + c_exc["checked"], (
        f"Gate 3 (total): Expected 39 total bowl instances (38+1), got {c_tot['checked']}"
    )
    print(
        f"[Gate 3 PASS] Valid={c_val['checked']}/38 (soup_bowl={c_val['by_type']['soup_bowl']}, "
        f"large_bowl={c_val['by_type']['large_bowl']}, rice_bowl={c_val['by_type']['rice_bowl']}), "
        f"Excluded={c_exc['checked']}/1 (ann_96 'com-suon1 (2).jpg' Canh), "
        f"Total={c_tot['checked']}/39 matched original eval_bowl_volume_ablation numerically "
        f"within tol=1e-9 and passed all 5 branch-safety checks "
        f"({c_tot['with_mound']} with above-rim mound)."
    )

    # 4. Gate 4: Per-group reproduction against reference `A1+A2+A3+A4+A5` in volume_ablation_records.jsonl
    assert REF_RECORDS_PATH.exists(), f"Reference file missing: {REF_RECORDS_PATH}"
    ref_records = []
    with open(REF_RECORDS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj.get("config_id") == "A1+A2+A3+A4+A5" and obj.get("mask_source") == "gt_mask":
                    ref_records.append(obj)

    assert len(ref_records) == 96, f"Expected 96 reference records, got {len(ref_records)}"
    ref_by_id = {r["mass_group_id"]: r for r in ref_records}
    rec_3b = records_by_cfg["Config_3b_Full_ArUco"]
    assert set(ref_by_id.keys()) == set(r["mass_group_id"] for r in rec_3b), (
        "Gate 4: mass_group_id set mismatch against reference A1+A2+A3+A4+A5!"
    )

    for r_new in rec_3b:
        gid = r_new["mass_group_id"]
        r_ref = ref_by_id[gid]
        assert r_new["scale_source"] == r_ref["scale_source"], f"Gate 4 scale_source mismatch on {gid}"
        assert math.isclose(r_new["mm_per_pixel"], r_ref["mm_per_pixel"], rel_tol=1e-5, abs_tol=1e-5), (
            f"Gate 4 mm_per_pixel mismatch on {gid}: {r_new['mm_per_pixel']} vs {r_ref['mm_per_pixel']}"
        )
        assert math.isclose(r_new["estimated_volume"], r_ref["estimated_volume"], rel_tol=1e-5, abs_tol=1e-2), (
            f"Gate 4 volume mismatch on {gid}: {r_new['estimated_volume']} vs {r_ref['estimated_volume']}"
        )
        assert math.isclose(r_new["estimated_mass"], r_ref["estimated_mass"], rel_tol=1e-5, abs_tol=1e-1), (
            f"Gate 4 mass mismatch on {gid}: {r_new['estimated_mass']} vs {r_ref['estimated_mass']}"
        )
        assert r_new["fallback_used"] == r_ref["fallback_used"], (
            f"Gate 4 fallback_used mismatch on {gid}: {r_new['fallback_used']} vs {r_ref['fallback_used']}"
        )
        assert math.isclose(r_new["D_food"], r_ref["D_food"], abs_tol=1e-1), (
            f"Gate 4 D_food mismatch on {gid}: {r_new['D_food']} vs {r_ref['D_food']}"
        )
        assert math.isclose(r_new["D_rim"], r_ref["D_rim"], abs_tol=1e-1), (
            f"Gate 4 D_rim mismatch on {gid}: {r_new['D_rim']} vs {r_ref['D_rim']}"
        )
        assert math.isclose(r_new["D_base"], r_ref["D_base"], abs_tol=1e-1), (
            f"Gate 4 D_base mismatch on {gid}: {r_new['D_base']} vs {r_ref['D_base']}"
        )
        assert math.isclose(r_new["mean_height_mm"], r_ref["mean_height_mm"], abs_tol=1e-1), (
            f"Gate 4 mean_height_mm mismatch on {gid}: {r_new['mean_height_mm']} vs {r_ref['mean_height_mm']}"
        )
        # Verify relief_scale and scale_anchor provenance (must satisfy both
        # _cache_depth_scenes.py [0.25, 4.0] and food_volume_pipeline.py [0.30, 3.00])
        if r_new["scale_source"] == "aruco_marker":
            assert math.isclose(r_new["relief_scale"], 0.79, rel_tol=1e-9)
            assert 0.25 <= r_new["scale_anchor"] <= 4.0
            assert 0.30 <= r_new["scale_anchor"] <= 3.00
        else:
            assert math.isclose(r_new["relief_scale"], 1.0, rel_tol=1e-9)
            assert math.isclose(r_new["scale_anchor"], 1.0, rel_tol=1e-9)

    print("[Gate 4 PASS] All 96/96 mass groups in Config_3b_Full_ArUco matched reference A1+A2+A3+A4+A5 records.")

    # 5. Gate 5: Unrounded and rounded aggregate metrics verification
    ref_summary = summarize_mass_groups(ref_records)
    sum_3b = summaries_by_cfg["Config_3b_Full_ArUco"]
    for m_key in ("mae_g", "mape_pct", "median_ape_pct", "rmse_g", "mpe_pct", "bias_mean_pct"):
        assert math.isclose(sum_3b[m_key], ref_summary[m_key], rel_tol=1e-6, abs_tol=1e-5), (
            f"Gate 5 unrounded mismatch on {m_key}: {sum_3b[m_key]} vs {ref_summary[m_key]}"
        )
    assert sum_3b["fallback_count"] == ref_summary["fallback_count"] == 19
    assert round(sum_3b["mae_g"], 2) == 67.89
    assert round(sum_3b["mape_pct"], 2) == 26.65
    assert round(sum_3b["rmse_g"], 2) == 95.62
    assert round(sum_3b["mpe_pct"], 2) == 1.28
    assert round(sum_3b["bias_mean_pct"], 2) == -8.62
    print(
        f"[Gate 5 PASS] Unrounded & rounded aggregate metrics verified: "
        f"MAE={sum_3b['mae_g']:.6f}g (67.89), MAPE={sum_3b['mape_pct']:.6f}% (26.65%), "
        f"RMSE={sum_3b['rmse_g']:.6f}g (95.62), Fallback={sum_3b['fallback_count']}."
    )


def pct_quantiles(vals: list) -> dict:
    """Compute min, P10, P25, median (P50), mean, P75, P90, max for a numeric list."""
    arr = np.asarray([v for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if len(arr) == 0:
        return {k: float("nan") for k in ("n", "mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "max")}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def main():
    # Step 1: Build or verify isolated cache (runs Gate 1 & Gate 2)
    cached_scenes = build_or_verify_aruco_ablation_cache(force_rebuild=False)

    mp = json.load(open(VERIFIED_MAPPING_PATH, encoding="utf-8"))
    coco = json.load(open(COCO_JSON_PATH, encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    del coco

    inst_by_fn = defaultdict(list)
    for x in mp["instances"]:
        inst_by_fn[x["file_name"]].append(x)
    target_files = sorted(inst_by_fn.keys())

    vol_estimator = VolumeNutritionEstimator()

    # Define the 5 configurations
    configs_spec = [
        (
            "Config_1_GT_Assisted_Plate_Heuristic",
            "Config 1: GT-Assisted Plate Heuristic (Không ArUco, Raw Depth, relief=1.0, area=mm_px²)",
        ),
        (
            "Config_2_ArUco_XY_Only",
            "Config 2: ArUco XY Only (Tỷ lệ ngang ArUco, Raw Depth, relief=1.0, area=mm_px²)",
        ),
        (
            "Config_3a_ArUco_XY_Depth_Anchor",
            "Config 3a: ArUco XY + Depth Anchor (Neo độ sâu tvec_z/d_pred, relief=1.0, area=mm_px²)",
        ),
        (
            "Config_3a_Intrinsics",
            "Config 3a-Intrinsics: Depth Anchor + Intrinsics Area (relief=1.0, area=Z²/(fx·fy))",
        ),
        (
            "Config_3b_Full_ArUco",
            "Config 3b: Full ArUco Hiện Hành (Depth Anchor + Intrinsics Area + relief=0.79)",
        ),
    ]

    records_by_cfg = defaultdict(list)
    bowl_check_counter = {
        "valid": {
            "checked": 0,
            "with_mound": 0,
            "by_type": {"soup_bowl": 0, "large_bowl": 0, "rice_bowl": 0},
        },
        "excluded": {
            "checked": 0,
            "with_mound": 0,
            "by_type": {"soup_bowl": 0, "large_bowl": 0, "rice_bowl": 0},
            "items": [],
        },
        "total": {
            "checked": 0,
            "with_mound": 0,
            "by_type": {"soup_bowl": 0, "large_bowl": 0, "rice_bowl": 0},
        },
    }

    for fn in target_files:
        sc = cached_scenes[fn]
        insts = inst_by_fn[fn]
        h_full, w_full = int(insts[0]["height"]), int(insts[0]["width"])
        h_vol, w_vol = int(sc["h_vol"]), int(sc["w_vol"])
        sx, sy = float(sc["sx"]), float(sc["sy"])
        K = sc["K"]

        gt_inputs = []
        for inst in insts:
            gt_full = coco_poly_to_mask(
                ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full
            )
            gt_vol = (
                cv2.resize(
                    gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                if gt_full.shape[:2] != (h_vol, w_vol)
                else gt_full
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

        is_anc = bool(sc["depth_anchored"])
        aruco_src = str(sc["aruco_scale_source"])
        aruco_mm = float(sc["aruco_mm_per_pixel"])
        heur_src = str(sc["heuristic_scale_source"])
        heur_mm = float(sc["heuristic_mm_per_pixel"])
        raw_d = sc["raw_depth_mm"]
        anc_d = sc["anchored_depth_mm"]
        sc_anchor_val = float(sc["scale_anchor_poly"]) if is_anc else 1.0

        scene_cfg_params = {
            "Config_1_GT_Assisted_Plate_Heuristic": {
                "depth_mm": raw_d,
                "mm_per_px": heur_mm,
                "scale_src": heur_src,
                "relief_scale": 1.0,
                "use_intr": False,
                "scale_anchor_used": 1.0,
                "verify_bowl": False,
            },
            "Config_2_ArUco_XY_Only": {
                "depth_mm": raw_d,
                "mm_per_px": aruco_mm,
                "scale_src": aruco_src,
                "relief_scale": 1.0,
                "use_intr": False,
                "scale_anchor_used": 1.0,
                "verify_bowl": False,
            },
            "Config_3a_ArUco_XY_Depth_Anchor": {
                "depth_mm": anc_d,
                "mm_per_px": aruco_mm,
                "scale_src": aruco_src,
                "relief_scale": 1.0,
                "use_intr": False,
                "scale_anchor_used": sc_anchor_val,
                "verify_bowl": False,
            },
            "Config_3a_Intrinsics": {
                "depth_mm": anc_d,
                "mm_per_px": aruco_mm,
                "scale_src": aruco_src,
                "relief_scale": 1.0,
                "use_intr": True,
                "scale_anchor_used": sc_anchor_val,
                "verify_bowl": False,
            },
            "Config_3b_Full_ArUco": {
                "depth_mm": anc_d,
                "mm_per_px": aruco_mm,
                "scale_src": aruco_src,
                "relief_scale": 0.79 if is_anc else 1.0,
                "use_intr": True,
                "scale_anchor_used": sc_anchor_val,
                "verify_bowl": True,
            },
        }

        for cid, _desc in configs_spec:
            p = scene_cfg_params[cid]
            s_out = evaluate_scene_instances_aruco(
                inst_list=gt_inputs,
                depth_mm=p["depth_mm"],
                mm_per_px=p["mm_per_px"],
                scale_src=p["scale_src"],
                K=K,
                relief_scale=p["relief_scale"],
                vol_estimator=vol_estimator,
                use_depth_intrinsics_for_area=p["use_intr"],
                verify_bowl_equiv=p["verify_bowl"],
                bowl_check_counter=bowl_check_counter if p["verify_bowl"] else None,
            )

            grouped_buckets = defaultdict(list)
            for inp, res in zip(gt_inputs, s_out):
                meta = inp["meta"]
                status = meta["mapping_status"]
                if status == "exact_1to1":
                    area_cm2 = round(
                        float(inp["mask"].sum() * (res["mm_per_pixel_eff"] ** 2) / 100.0), 2
                    )
                    records_by_cfg[cid].append({
                        "config_id": cid,
                        "mask_source": "gt_mask",
                        "image_id": fn,
                        "mass_group_id": f"{fn}::ann_{meta['annotation_id']}",
                        "dish_class": meta["short_class"],
                        "dish_type": res["dish_type"],
                        "scale_source": p["scale_src"],
                        "mm_per_pixel": round(float(res["mm_per_pixel_eff"]), 5),
                        "scale_anchor": round(float(p["scale_anchor_used"]), 5),
                        "relief_scale": float(p["relief_scale"]),
                        "use_depth_intrinsics_for_area": bool(p["use_intr"]),
                        "food_mask_area": area_cm2,
                        "D_food": res["D_food_mm"],
                        "D_rim": res["D_rim_mm"],
                        "D_base": res["D_base_mm"],
                        "mean_height_mm": res["mean_height_mm"],
                        "estimated_volume": res["volume_cm3"],
                        "estimated_mass": res["weight_g"],
                        "GT_mass": float(meta["gt_weight_g"]),
                        "density": res["density"],
                        "fallback_used": bool(res["fallback"]),
                    })
                elif status == "grouped_multi_instance":
                    grouped_buckets[meta["short_class"]].append((inp, res, meta))

            for s_cls, bucket in grouped_buckets.items():
                total_vol = round(sum(r["volume_cm3"] for _, r, _ in bucket), 2)
                total_mass = round(sum(r["weight_g"] for _, r, _ in bucket), 1)
                gt_mass = float(bucket[0][2]["gt_weight_g"])
                eff_mm = bucket[0][1]["mm_per_pixel_eff"]
                rho_val = bucket[0][1]["density"]
                total_area_cm2 = round(
                    sum(float(inp["mask"].sum() * (eff_mm ** 2) / 100.0) for inp, _, _ in bucket), 2
                )
                ann_ids = "_".join(str(m["annotation_id"]) for _, _, m in bucket)
                records_by_cfg[cid].append({
                    "config_id": cid,
                    "mask_source": "gt_mask",
                    "image_id": fn,
                    "mass_group_id": f"{fn}::group_{ann_ids}",
                    "dish_class": s_cls,
                    "dish_type": bucket[0][1]["dish_type"],
                    "scale_source": p["scale_src"],
                    "mm_per_pixel": round(float(eff_mm), 5),
                    "scale_anchor": round(float(p["scale_anchor_used"]), 5),
                    "relief_scale": float(p["relief_scale"]),
                    "use_depth_intrinsics_for_area": bool(p["use_intr"]),
                    "food_mask_area": total_area_cm2,
                    "D_food": round(st.mean(r["D_food_mm"] for _, r, _ in bucket), 1),
                    "D_rim": round(st.mean(r["D_rim_mm"] for _, r, _ in bucket), 1),
                    "D_base": round(st.mean(r["D_base_mm"] for _, r, _ in bucket), 1),
                    "mean_height_mm": round(st.mean(r["mean_height_mm"] for _, r, _ in bucket), 1),
                    "estimated_volume": total_vol,
                    "estimated_mass": total_mass,
                    "GT_mass": gt_mass,
                    "density": rho_val,
                    "fallback_used": any(bool(r["fallback"]) for _, r, _ in bucket),
                })

    summaries_by_cfg = {
        cid: summarize_mass_groups(records_by_cfg[cid]) for cid, _ in configs_spec
    }

    # Execute Gate 3, Gate 4, Gate 5 before writing any outputs
    verify_gates_3_4_5(records_by_cfg, summaries_by_cfg, bowl_check_counter)

    # Save JSONL records
    with open(OUT_RECORDS_PATH, "w", encoding="utf-8") as f_out:
        for cid, _ in configs_spec:
            for r in records_by_cfg[cid]:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {sum(len(v) for v in records_by_cfg.values())} records to {OUT_RECORDS_PATH}")

    # Generate Markdown Report
    generate_aruco_ablation_report(cached_scenes, configs_spec, records_by_cfg, summaries_by_cfg)


def generate_aruco_ablation_report(
    cached_scenes: dict,
    configs_spec: list,
    records_by_cfg: dict,
    summaries_by_cfg: dict,
):
    """Write comprehensive Markdown report to `scratch/aruco_ablation_report.md`."""
    # 1. Collect marker & heuristic diagnostics across 77 scenes
    heur_sub_counts = Counter(str(sc["heuristic_submethod"]) for sc in cached_scenes.values())
    aruco_src_counts = Counter(str(sc["aruco_scale_source"]) for sc in cached_scenes.values())

    anchored_scenes = [sc for sc in cached_scenes.values() if bool(sc["depth_anchored"])]
    n_center_valid = sum(
        1 for sc in anchored_scenes if int(sc["center_5x5_valid_px"]) >= 15 and math.isfinite(float(sc["scale_anchor_center"]))
    )
    n_center_in_poly = sum(1 for sc in anchored_scenes if bool(sc["center_in_polygon"]))

    tvec_z_q = pct_quantiles([float(sc["z_marker_tvec_z"]) for sc in anchored_scenes])
    euclid_q = pct_quantiles([float(sc["z_marker_euclid"]) for sc in anchored_scenes])
    d_poly_q = pct_quantiles([float(sc["d_pred_poly_median"]) for sc in anchored_scenes])
    d_cent_q = pct_quantiles([float(sc["d_pred_center_5x5"]) for sc in anchored_scenes])
    tilt_q = pct_quantiles([float(sc["marker_tilt_deg"]) for sc in anchored_scenes])
    std_raw_q = pct_quantiles([float(sc["marker_depth_std_raw_mm"]) for sc in anchored_scenes])
    std_plane_q = pct_quantiles([float(sc["marker_plane_std_mm"]) for sc in anchored_scenes])
    std_res_fit_mm_q = pct_quantiles([float(sc["marker_res_std_plane_fit_mm"]) for sc in anchored_scenes])
    std_res_fit_pct_q = pct_quantiles([float(sc["marker_res_std_plane_fit_pct"]) for sc in anchored_scenes])
    std_res_anc_poly_q = pct_quantiles([float(sc["marker_res_std_anc_poly_mm"]) for sc in anchored_scenes])
    alpha_poly_q = pct_quantiles([float(sc["scale_anchor_poly"]) for sc in anchored_scenes])
    alpha_cent_q = pct_quantiles([float(sc["scale_anchor_center"]) for sc in anchored_scenes])
    alpha_fit_q = pct_quantiles([float(sc["scale_anchor_plane_fit"]) for sc in anchored_scenes])

    delta_alpha_abs = [
        abs(float(sc["scale_anchor_poly"]) - float(sc["scale_anchor_center"]))
        for sc in anchored_scenes
        if math.isfinite(float(sc["scale_anchor_center"]))
    ]
    delta_alpha_rel_pct = [
        abs(float(sc["scale_anchor_poly"]) - float(sc["scale_anchor_center"]))
        / float(sc["scale_anchor_center"])
        * 100.0
        for sc in anchored_scenes
        if math.isfinite(float(sc["scale_anchor_center"])) and float(sc["scale_anchor_center"]) > 0
    ]
    delta_alpha_fit_abs = [
        abs(float(sc["scale_anchor_poly"]) - float(sc["scale_anchor_plane_fit"]))
        for sc in anchored_scenes
        if math.isfinite(float(sc["scale_anchor_plane_fit"]))
    ]
    d_alpha_abs_q = pct_quantiles(delta_alpha_abs)
    d_alpha_rel_q = pct_quantiles(delta_alpha_rel_pct)
    d_alpha_fit_q = pct_quantiles(delta_alpha_fit_abs)

    lines = [
        "# Báo Cáo Thực Nghiệm ArUco & Depth-Anchoring Ablation (5 Cấu Hình)",
        "",
        "> [!IMPORTANT]",
        "> **Phạm vi, Nguồn Dữ Liệu và Quy ước Học thuật:**",
        "> - Thực nghiệm này đánh giá đóng góp thành phần của mô-đun hiệu chỉnh tỷ lệ **ArUco 30 mm** và **Neo độ sâu (`solvePnP` Depth Anchoring)** trên cùng **77 ảnh / 96 nhóm khối lượng (`Oracle GT Polygon Mask`)** đã sử dụng để phát triển chuỗi mô hình hình học `A1–A5`.",
        "> - **Làm rõ nguồn dữ liệu của Config 1 (`Config_1_GT_Assisted_Plate_Heuristic`):** Vì toàn bộ bảng thực nghiệm `Oracle GT Polygon Mask` cố định mặt nạ phân vùng, khung giới hạn (`bbox_xyxy`) và nhãn lớp món ăn (`vietfood67_class_name`) từ `gt_mapping_verified.json` cho cả 5 cấu hình, cơ chế ước lượng tỷ lệ theo vật chứa ở Config 1 (`scaled_dets` trong `_try_dish_guided_scale` và công thức ghi đè `mm_per_px = expected_dia / max(bw, bh)` trên các lớp vật chứa) sử dụng chính khung giới hạn và nhãn lớp GT này. Do đó, Config 1 là **GT-Assisted Plate Heuristic** (cận trên của phương pháp heuristic khi biết chính xác vị trí và loại vật chứa, giúp cô lập thuần túy cơ chế tỷ lệ ngang/neo độ sâu mà không bị nhiễu bởi lỗi phát hiện của YOLO), **không phải** là heuristic hoàn toàn độc lập với ground truth.",
        "> - Do 77 ảnh này là **tập dữ liệu kiểm định thực nghiệm (validation/evaluation benchmark)** chứ không phải tập kiểm thử độc lập (held-out test set), kết quả dưới đây dùng để chứng minh **cơ chế vật lý và đóng góp của từng biến số điều khiển**, không dùng để tuyên bố khả năng tổng quát hóa ngoài mẫu.",
        "> - Toàn bộ 5 cổng kiểm định tự động (**Gate 1 → Gate 5**) đều đạt 100%: `Config_3b_Full_ArUco` tái lập tương đương về mặt số học trong dung sai kiểm định toàn bộ 96 nhóm khối lượng của báo cáo `A1–A5` (`MAE = 67.89 g`, `MAPE = 26.65%`, `RMSE = 95.62 g`, `MPE = +1.28%`, `Bias_mean = -8.62%`, `Fallback = 19`).",
        "",
        "---",
        "",
        "## 1. Kiểm Định Hình Học Marker ArUco & Phân Rã Nhiễu Mặt Phẳng (77 Ảnh)",
        "",
        "### 1.1. Thống kê nguồn tỷ lệ ngang trên 77 ảnh (Gate 1 & Gate 2)",
        f"- **Chế độ chuẩn (Ưu tiên ArUco — Config 2, 3a, 3a-Intrinsics, 3b):**",
        f"  - `scale_source == 'aruco_marker'`: **`{aruco_src_counts.get('aruco_marker', 0)}/77` ảnh** (`95/96` nhóm khối lượng, `depth_anchored == True`).",
        f"  - `scale_source == 'checkerboard'`: **`{aruco_src_counts.get('checkerboard', 0)}/77` ảnh** (`0/96` nhóm khối lượng).",
        f"  - `scale_source == 'plate_heuristic'` (fallback): **`{aruco_src_counts.get('plate_heuristic', 0)}/77` ảnh** (`Pho1 (5).jpg`, `1/96` nhóm khối lượng, `depth_anchored == False`).",
        f"- **Chế độ không dùng thẻ chuẩn (`Config_1_GT_Assisted_Plate_Heuristic` — Stateless Heuristic Recovery):** Đạt tỷ lệ hợp lệ `77/77` ảnh (`0` ảnh lỗi), trong đó:",
        f"  - Nhánh khớp elip theo bounding box vật chứa (`dish_guided_ellipse`): **`{heur_sub_counts.get('dish_guided_ellipse', 0)}/77` ảnh**.",
        f"  - Nhánh dò đường tròn Hough (`hough_circle`): **`{heur_sub_counts.get('hough_circle', 0)}/77` ảnh**.",
        f"  - Nhánh kích thước ảnh dự phòng (`image_dimension_fallback`): **`{heur_sub_counts.get('image_dimension_fallback', 0)}/77` ảnh**.",
        f"- **Kiểm định cửa sổ $5\\times 5$ tại tâm chiếu $(0,0,0)$ của Marker:** Trên `{len(anchored_scenes)}` ảnh có ArUco, **`{n_center_in_poly}/{len(anchored_scenes)}` ảnh** có tâm chiếu nằm trọn trong đa giác 4 đỉnh và **`{n_center_valid}/{len(anchored_scenes)}` ảnh** có đủ $\\ge 15/25$ pixel độ sâu hợp lệ trong cửa sổ $5\\times 5$.",
        "",
        "### 1.2. Phân phối khoảng cách, góc nghiêng và phân rã phần dư độ sâu tại Marker ($N = 76$ ảnh ArUco)",
        "",
        "> [!NOTE]",
        "> Để tách biệt độ nghiêng hình học 3D của mặt phẳng marker khỏi nhiễu bề mặt nội tại của mạng DA2 mà không bị phụ thuộc tuyến tính vào hệ số neo một điểm $\\alpha_{\\text{poly}} = t_z / d_{\\text{pred,poly}}$, bảng dưới đây báo cáo thêm hệ số khớp mặt phẳng chẩn đoán độc lập $\\alpha_{\\text{fit}} = \\text{median}(Z_{\\text{plane}}(u,v) / Z_{\\text{DA2,raw}}(u,v))$ và độ lệch chuẩn phần dư khớp mặt phẳng $\\sigma_{\\text{res,plane\\_fit}} = \\text{Std}(\\alpha_{\\text{fit}} Z_{\\text{DA2,raw}} - Z_{\\text{plane}})$ (chỉ dùng để chẩn đoán trong Mục 1.2, hoàn toàn không thay đổi $\\alpha_{\\text{poly}}$ của Config 3a–3b).",
        "",
        "| Đại lượng kiểm định | Đơn vị | Mean ± Std | Min | P10 | P25 | Median (P50) | P75 | P90 | Max |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| Độ sâu tâm marker theo trục quang học $t_z$ (`tvec_z`) | mm | `{tvec_z_q['mean']:.1f} ± {tvec_z_q['std']:.1f}` | `{tvec_z_q['min']:.1f}` | `{tvec_z_q['p10']:.1f}` | `{tvec_z_q['p25']:.1f}` | **`{tvec_z_q['p50']:.1f}`** | `{tvec_z_q['p75']:.1f}` | `{tvec_z_q['p90']:.1f}` | `{tvec_z_q['max']:.1f}` |",
        f"| Khoảng cách Euclidean $\\|\\mathbf{{t}}\\|_2$ | mm | `{euclid_q['mean']:.1f} ± {euclid_q['std']:.1f}` | `{euclid_q['min']:.1f}` | `{euclid_q['p10']:.1f}` | `{euclid_q['p25']:.1f}` | **`{euclid_q['p50']:.1f}`** | `{euclid_q['p75']:.1f}` | `{euclid_q['p90']:.1f}` | `{euclid_q['max']:.1f}` |",
        f"| Độ sâu DA2 thô trung vị đa giác $d_{{\\text{{pred,poly}}}}$ | mm | `{d_poly_q['mean']:.1f} ± {d_poly_q['std']:.1f}` | `{d_poly_q['min']:.1f}` | `{d_poly_q['p10']:.1f}` | `{d_poly_q['p25']:.1f}` | **`{d_poly_q['p50']:.1f}`** | `{d_poly_q['p75']:.1f}` | `{d_poly_q['p90']:.1f}` | `{d_poly_q['max']:.1f}` |",
        f"| Độ sâu DA2 thô cửa sổ $5\\times 5$ tâm chiếu $d_{{\\text{{pred,center}}}}$ | mm | `{d_cent_q['mean']:.1f} ± {d_cent_q['std']:.1f}` | `{d_cent_q['min']:.1f}` | `{d_cent_q['p10']:.1f}` | `{d_cent_q['p25']:.1f}` | **`{d_cent_q['p50']:.1f}`** | `{d_cent_q['p75']:.1f}` | `{d_cent_q['p90']:.1f}` | `{d_cent_q['max']:.1f}` |",
        f"| Góc nghiêng pháp tuyến marker $\\theta_{{\\text{{tilt}}}}$ | độ ($^\\circ$) | `{tilt_q['mean']:.1f} ± {tilt_q['std']:.1f}` | `{tilt_q['min']:.1f}` | `{tilt_q['p10']:.1f}` | `{tilt_q['p25']:.1f}` | **`{tilt_q['p50']:.1f}`** | `{tilt_q['p75']:.1f}` | `{tilt_q['p90']:.1f}` | `{tilt_q['max']:.1f}` |",
        f"| Độ lệch chuẩn thô trong marker $\\sigma_{{\\text{{marker,raw}}}}$ | mm (raw) | `{std_raw_q['mean']:.2f} ± {std_raw_q['std']:.2f}` | `{std_raw_q['min']:.2f}` | `{std_raw_q['p10']:.2f}` | `{std_raw_q['p25']:.2f}` | **`{std_raw_q['p50']:.2f}`** | `{std_raw_q['p75']:.2f}` | `{std_raw_q['p90']:.2f}` | `{std_raw_q['max']:.2f}` |",
        f"| Độ lệch chuẩn hình học do nghiêng $\\sigma_{{\\text{{plane}}}} = \\text{{Std}}(Z_{{\\text{{plane}}}})$ | mm (thực) | `{std_plane_q['mean']:.2f} ± {std_plane_q['std']:.2f}` | `{std_plane_q['min']:.2f}` | `{std_plane_q['p10']:.2f}` | `{std_plane_q['p25']:.2f}` | **`{std_plane_q['p50']:.2f}`** | `{std_plane_q['p75']:.2f}` | `{std_plane_q['p90']:.2f}` | `{std_plane_q['max']:.2f}` |",
        f"| Độ lệch chuẩn phần dư khớp mặt phẳng $\\sigma_{{\\text{{res,plane\\_fit}}}} = \\text{{Std}}(\\alpha_{{\\text{{fit}}}} Z_{{\\text{{DA2,raw}}}} - Z_{{\\text{{plane}}}})$ | mm (thực) | `{std_res_fit_mm_q['mean']:.2f} ± {std_res_fit_mm_q['std']:.2f}` | `{std_res_fit_mm_q['min']:.2f}` | `{std_res_fit_mm_q['p10']:.2f}` | `{std_res_fit_mm_q['p25']:.2f}` | **`{std_res_fit_mm_q['p50']:.2f}`** | `{std_res_fit_mm_q['p75']:.2f}` | `{std_res_fit_mm_q['p90']:.2f}` | `{std_res_fit_mm_q['max']:.2f}` |",
        f"| Tỷ lệ nhiễu không phẳng tương đối $\\sigma_{{\\text{{res,plane\\_fit}}}} / t_z$ | % | `{std_res_fit_pct_q['mean']:.3f}% ± {std_res_fit_pct_q['std']:.3f}%` | `{std_res_fit_pct_q['min']:.3f}%` | `{std_res_fit_pct_q['p10']:.3f}%` | `{std_res_fit_pct_q['p25']:.3f}%` | **`{std_res_fit_pct_q['p50']:.3f}%`** | `{std_res_fit_pct_q['p75']:.3f}%` | `{std_res_fit_pct_q['p90']:.3f}%` | `{std_res_fit_pct_q['max']:.3f}%` |",
        f"| Độ lệch chuẩn phần dư bản đồ neo $\\sigma_{{\\text{{res,anc\\_poly}}}} = \\text{{Std}}(\\alpha_{{\\text{{poly}}}} Z_{{\\text{{DA2,raw}}}} - Z_{{\\text{{plane}}}})$ | mm (thực) | `{std_res_anc_poly_q['mean']:.2f} ± {std_res_anc_poly_q['std']:.2f}` | `{std_res_anc_poly_q['min']:.2f}` | `{std_res_anc_poly_q['p10']:.2f}` | `{std_res_anc_poly_q['p25']:.2f}` | **`{std_res_anc_poly_q['p50']:.2f}`** | `{std_res_anc_poly_q['p75']:.2f}` | `{std_res_anc_poly_q['p90']:.2f}` | `{std_res_anc_poly_q['max']:.2f}` |",
        f"| Hệ số neo đa giác $\\alpha_{{\\text{{poly}}}} = t_z / d_{{\\text{{pred,poly}}}}$ | tỷ số | `{alpha_poly_q['mean']:.4f} ± {alpha_poly_q['std']:.4f}` | `{alpha_poly_q['min']:.4f}` | `{alpha_poly_q['p10']:.4f}` | `{alpha_poly_q['p25']:.4f}` | **`{alpha_poly_q['p50']:.4f}`** | `{alpha_poly_q['p75']:.4f}` | `{alpha_poly_q['p90']:.4f}` | `{alpha_poly_q['max']:.4f}` |",
        f"| Hệ số neo tâm chiếu $\\alpha_{{\\text{{center}}}} = t_z / d_{{\\text{{pred,center}}}}$ | tỷ số | `{alpha_cent_q['mean']:.4f} ± {alpha_cent_q['std']:.4f}` | `{alpha_cent_q['min']:.4f}` | `{alpha_cent_q['p10']:.4f}` | `{alpha_cent_q['p25']:.4f}` | **`{alpha_cent_q['p50']:.4f}`** | `{alpha_cent_q['p75']:.4f}` | `{alpha_cent_q['p90']:.4f}` | `{alpha_cent_q['max']:.4f}` |",
        f"| Hệ số khớp mặt phẳng chẩn đoán $\\alpha_{{\\text{{fit}}}} = \\text{{median}}(Z_{{\\text{{plane}}}} / Z_{{\\text{{DA2,raw}}}})$ | tỷ số | `{alpha_fit_q['mean']:.4f} ± {alpha_fit_q['std']:.4f}` | `{alpha_fit_q['min']:.4f}` | `{alpha_fit_q['p10']:.4f}` | `{alpha_fit_q['p25']:.4f}` | **`{alpha_fit_q['p50']:.4f}`** | `{alpha_fit_q['p75']:.4f}` | `{alpha_fit_q['p90']:.4f}` | `{alpha_fit_q['max']:.4f}` |",
        f"| Độ chênh tuyệt đối $\\Delta\\alpha = |\\alpha_{{\\text{{poly}}}} - \\alpha_{{\\text{{center}}}}|$ | tỷ số | `{d_alpha_abs_q['mean']:.5f} ± {d_alpha_abs_q['std']:.5f}` | `{d_alpha_abs_q['min']:.5f}` | `{d_alpha_abs_q['p10']:.5f}` | `{d_alpha_abs_q['p25']:.5f}` | **`{d_alpha_abs_q['p50']:.5f}`** | `{d_alpha_abs_q['p75']:.5f}` | `{d_alpha_abs_q['p90']:.5f}` | `{d_alpha_abs_q['max']:.5f}` |",
        f"| Độ chênh tương đối $\\Delta\\alpha / \\alpha_{{\\text{{center}}}}$ | % | `{d_alpha_rel_q['mean']:.3f}% ± {d_alpha_rel_q['std']:.3f}%` | `{d_alpha_rel_q['min']:.3f}%` | `{d_alpha_rel_q['p10']:.3f}%` | `{d_alpha_rel_q['p25']:.3f}%` | **`{d_alpha_rel_q['p50']:.3f}%`** | `{d_alpha_rel_q['p75']:.3f}%` | `{d_alpha_rel_q['p90']:.3f}%` | `{d_alpha_rel_q['max']:.3f}%` |",
        f"| Độ chênh so với khớp mặt phẳng $|\\alpha_{{\\text{{poly}}}} - \\alpha_{{\\text{{fit}}}}|$ | tỷ số | `{d_alpha_fit_q['mean']:.5f} ± {d_alpha_fit_q['std']:.5f}` | `{d_alpha_fit_q['min']:.5f}` | `{d_alpha_fit_q['p10']:.5f}` | `{d_alpha_fit_q['p25']:.5f}` | **`{d_alpha_fit_q['p50']:.5f}`** | `{d_alpha_fit_q['p75']:.5f}` | `{d_alpha_fit_q['p90']:.5f}` | `{d_alpha_fit_q['max']:.5f}` |",
        "",
        "---",
        "",
        "## 2. Bảng Tổng Hợp Kết Quả 5 Cấu Hình (`N = 96` Nhóm Khối Lượng, Oracle GT Mask)",
        "",
        "| Cấu hình | Nguồn `mm_per_px` | Bản đồ độ sâu | `Intrinsics Area` | `relief_scale` | MAE (g) ↓ | MAPE (%) ↓ | Median APE (%) ↓ | RMSE (g) ↓ | MPE (%) | Bias_mean (%) | Số Fallback | Median `mm_per_px` | Median `scale_anchor` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    short_cfg_meta = {
        "Config_1_GT_Assisted_Plate_Heuristic": ("GT-Assisted Heur (77)", "`raw_depth_mm`", "`False`", "`1.0`"),
        "Config_2_ArUco_XY_Only": ("ArUco (76)+Heur (1)", "`raw_depth_mm`", "`False`", "`1.0`"),
        "Config_3a_ArUco_XY_Depth_Anchor": ("ArUco (76)+Heur (1)", "`anchored_depth_mm`", "`False`", "`1.0`"),
        "Config_3a_Intrinsics": ("ArUco (76)+Heur (1)", "`anchored_depth_mm`", "`True`", "`1.0`"),
        "Config_3b_Full_ArUco": ("ArUco (76)+Heur (1)", "`anchored_depth_mm`", "`True`", "`0.79` (76) / `1.0` (1)"),
    }

    for cid, _desc in configs_spec:
        s = summaries_by_cfg[cid]
        recs = records_by_cfg[cid]
        med_mm = st.median(r["mm_per_pixel"] for r in recs)
        med_anc = st.median(r["scale_anchor"] for r in recs)
        m_src, m_dep, m_intr, m_rel = short_cfg_meta[cid]
        bold = "**" if cid == "Config_3b_Full_ArUco" else ""
        lines.append(
            f"| {bold}`{cid}`{bold} | {m_src} | {m_dep} | {m_intr} | {m_rel} | "
            f"{bold}`{s['mae_g']:.2f}`{bold} | {bold}`{s['mape_pct']:.2f}%`{bold} | "
            f"`{s['median_ape_pct']:.2f}%` | {bold}`{s['rmse_g']:.2f}`{bold} | "
            f"`{s['mpe_pct']:+.2f}%` | `{s['bias_mean_pct']:+.2f}%` | `{s['fallback_count']}` | "
            f"`{med_mm:.4f}` | `{med_anc:.4f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Phân Rã Đóng Góp Từng Bước Chuyển Đơn Biến (Stepwise Attribution)",
        "",
        "| Bước chuyển cấu hình | Thành phần thay đổi | $\\Delta\\text{MAE}$ (g) | $\\Delta\\text{MAPE}$ (điểm %) | $\\Delta\\text{RMSE}$ (g) | $\\Delta\\text{MPE}$ (điểm %) | Số nhóm giảm lỗi / tăng lỗi / giữ nguyên |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    step_pairs = [
        ("Config_1_GT_Assisted_Plate_Heuristic", "Config_2_ArUco_XY_Only", "Chuyển cơ chế tỷ lệ ngang & $D_{\\text{rim}}$ từ GT-Assisted Heuristic sang ArUco XY"),
        ("Config_2_ArUco_XY_Only", "Config_3a_ArUco_XY_Depth_Anchor", "Bật neo độ sâu dọc `scale_anchor = tvec_z / d_pred` (`raw` $\\rightarrow$ `anchored`)"),
        ("Config_3a_ArUco_XY_Depth_Anchor", "Config_3a_Intrinsics", "Bật công thức diện tích phần nhô miệng bát `Z²/(fx·fy)`"),
        ("Config_3a_Intrinsics", "Config_3b_Full_ArUco", "Áp dụng hệ số hiệu chỉnh thực nghiệm `relief_scale = 0.79` (76 ảnh ArUco, overlap `2/96`)"),
        ("Config_1_GT_Assisted_Plate_Heuristic", "Config_3a_Intrinsics", "**Tổng tác động các thành phần ArUco (XY + neo độ sâu) & intrinsics trước `relief_scale = 0.79`**"),
        ("Config_1_GT_Assisted_Plate_Heuristic", "Config_3b_Full_ArUco", "**Tổng hợp toàn bộ pipeline hiện hành (ArUco XY + Neo độ sâu + Intrinsics + `relief_scale = 0.79`)**"),
    ]

    for c_from, c_to, label in step_pairs:
        s0 = summaries_by_cfg[c_from]
        s1 = summaries_by_cfg[c_to]
        d_mae = s1["mae_g"] - s0["mae_g"]
        d_mape = s1["mape_pct"] - s0["mape_pct"]
        d_rmse = s1["rmse_g"] - s0["rmse_g"]
        d_mpe = s1["mpe_pct"] - s0["mpe_pct"]

        r0_map = {r["mass_group_id"]: r for r in records_by_cfg[c_from]}
        r1_map = {r["mass_group_id"]: r for r in records_by_cfg[c_to]}
        n_improved = n_worse = n_same = 0
        for gid in r0_map:
            e0 = abs(r0_map[gid]["estimated_mass"] - r0_map[gid]["GT_mass"])
            e1 = abs(r1_map[gid]["estimated_mass"] - r1_map[gid]["GT_mass"])
            if e1 < e0 - 0.05:
                n_improved += 1
            elif e1 > e0 + 0.05:
                n_worse += 1
            else:
                n_same += 1

        lines.append(
            f"| `{c_from}` $\\rightarrow$ `{c_to}` | {label} | "
            f"`{d_mae:+.2f} g` | `{d_mape:+.2f}%` | `{d_rmse:+.2f} g` | `{d_mpe:+.2f}%` | "
            f"`{n_improved}` giảm / `{n_worse}` tăng / `{n_same}` bằng |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Phân Tích Chi Tiết Theo Từng Lớp Món Ăn (14 Lớp, 96 Nhóm Khối Lượng)",
        "",
        "| Lớp món ăn (`dish_class`) | `N` | GT Mean (g) | MAE C1 (g) | MAE C2 (g) | MAE C3a (g) | MAE C3a-Intr (g) | MAE C3b (g) | MAPE C1 (%) | MAPE C2 (%) | MAPE C3a (%) | MAPE C3a-Intr (%) | MAPE C3b (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    all_classes = sorted(summaries_by_cfg["Config_3b_Full_ArUco"]["per_class"].keys())
    for c_name in all_classes:
        c1 = summaries_by_cfg["Config_1_GT_Assisted_Plate_Heuristic"]["per_class"][c_name]
        c2 = summaries_by_cfg["Config_2_ArUco_XY_Only"]["per_class"][c_name]
        c3a = summaries_by_cfg["Config_3a_ArUco_XY_Depth_Anchor"]["per_class"][c_name]
        c3ai = summaries_by_cfg["Config_3a_Intrinsics"]["per_class"][c_name]
        c3b = summaries_by_cfg["Config_3b_Full_ArUco"]["per_class"][c_name]
        lines.append(
            f"| **`{c_name}`** | `{c3b['n_groups']}` | `{c3b['gt_mean_g']:.1f}` | "
            f"`{c1['mae_g']:.1f}` | `{c2['mae_g']:.1f}` | `{c3a['mae_g']:.1f}` | `{c3ai['mae_g']:.1f}` | **`{c3b['mae_g']:.1f}`** | "
            f"`{c1['mape_pct']:.1f}%` | `{c2['mape_pct']:.1f}%` | `{c3a['mape_pct']:.1f}%` | `{c3ai['mape_pct']:.1f}%` | **`{c3b['mape_pct']:.1f}%`** |"
        )

    s_c1 = summaries_by_cfg["Config_1_GT_Assisted_Plate_Heuristic"]
    s_c2 = summaries_by_cfg["Config_2_ArUco_XY_Only"]
    s_c3a = summaries_by_cfg["Config_3a_ArUco_XY_Depth_Anchor"]
    s_c3ai = summaries_by_cfg["Config_3a_Intrinsics"]
    s_c3b = summaries_by_cfg["Config_3b_Full_ArUco"]
    lines.append(
        f"| **Toàn tập (`Total`)** | **`96`** | **`346.5`** | "
        f"`{s_c1['mae_g']:.2f}` | `{s_c2['mae_g']:.2f}` | `{s_c3a['mae_g']:.2f}` | `{s_c3ai['mae_g']:.2f}` | **`{s_c3b['mae_g']:.2f}`** | "
        f"`{s_c1['mape_pct']:.2f}%` | `{s_c2['mape_pct']:.2f}%` | `{s_c3a['mape_pct']:.2f}%` | `{s_c3ai['mape_pct']:.2f}%` | **`{s_c3b['mape_pct']:.2f}%`** |"
    )

    # 4.1 Dynamic class-level improvement / regression breakdown per transition
    def _class_step_summary(c_from: str, c_to: str) -> tuple:
        p0 = summaries_by_cfg[c_from]["per_class"]
        p1 = summaries_by_cfg[c_to]["per_class"]
        improved, worsened = [], []
        for cls_k in all_classes:
            d_mae = p1[cls_k]["mae_g"] - p0[cls_k]["mae_g"]
            d_mape = p1[cls_k]["mape_pct"] - p0[cls_k]["mape_pct"]
            entry = (d_mae, d_mape, cls_k, p0[cls_k]["mae_g"], p1[cls_k]["mae_g"], p0[cls_k]["mape_pct"], p1[cls_k]["mape_pct"])
            if d_mae < -0.05 or d_mape < -0.05:
                improved.append(entry)
            elif d_mae > 0.05 or d_mape > 0.05:
                worsened.append(entry)
        improved.sort(key=lambda t: t[0])
        worsened.sort(key=lambda t: t[0], reverse=True)
        return improved, worsened

    def _fmt_cls_entries(entries: list) -> str:
        return ", ".join(
            f"`{cls_k}` (MAE `{m0:.1f} → {m1:.1f} g`, MAPE `{p0_v:.1f}% → {p1_v:.1f}%`)"
            for _dm, _dp, cls_k, m0, m1, p0_v, p1_v in entries
        )

    lines.extend([
        "",
        "### 4.1. Phân tích lớp món ăn cải thiện và bị tác động ngược theo từng bước chuyển",
    ])
    step_labels_41 = [
        ("Config_1_GT_Assisted_Plate_Heuristic", "Config_2_ArUco_XY_Only", "Bước 1 (`Config 1 → Config 2`: GT-Assisted Heuristic → ArUco XY)"),
        ("Config_2_ArUco_XY_Only", "Config_3a_ArUco_XY_Depth_Anchor", "Bước 2 (`Config 2 → Config 3a`: `raw_depth_mm` → `anchored_depth_mm`)"),
        ("Config_3a_ArUco_XY_Depth_Anchor", "Config_3a_Intrinsics", "Bước 3 (`Config 3a → Config 3a-Intrinsics`: `mm_per_px²` → `Z²/(fx·fy)`)"),
        ("Config_3a_Intrinsics", "Config_3b_Full_ArUco", "Bước 4 (`Config 3a-Intrinsics → Config 3b`: `relief_scale = 1.0 → 0.79`)"),
    ]
    for c_from, c_to, title in step_labels_41:
        imp, wrs = _class_step_summary(c_from, c_to)
        imp_txt = _fmt_cls_entries(imp) if imp else "không có"
        wrs_txt = _fmt_cls_entries(wrs) if wrs else "không có (`0` lớp tăng lỗi)"
        lines.append(f"- **{title}:**")
        lines.append(f"  - *Cải thiện (`{len(imp)}` lớp):* {imp_txt}.")
        lines.append(f"  - *Tác động ngược (`{len(wrs)}` lớp):* {wrs_txt}.")

    p_c2 = summaries_by_cfg["Config_2_ArUco_XY_Only"]["per_class"]
    p_c3a = summaries_by_cfg["Config_3a_ArUco_XY_Depth_Anchor"]["per_class"]
    p_c3ai = summaries_by_cfg["Config_3a_Intrinsics"]["per_class"]
    p_c3b = summaries_by_cfg["Config_3b_Full_ArUco"]["per_class"]

    lines.extend([
        "",
        "### 4.2. Ghi chú giới hạn tập dữ liệu kiểm định (`77` ảnh / `96` nhóm khối lượng)",
        "- Toàn bộ kết quả trong báo cáo này được đo trên **tập dữ liệu kiểm định 77 ảnh / 96 nhóm khối lượng** (cùng tập đã dùng để thiết kế chuỗi `A1–A5`, **không phải tập kiểm thử độc lập hoàn toàn / held-out test set**). Giá trị của thực nghiệm là chứng minh **cơ chế vật lý và đóng góp từng thành phần** trong điều kiện `Oracle GT Polygon Mask`.",
        f"- Riêng bước `Config 1 → Config 2` cho thấy 2 lớp đĩa phẳng (`Banh cuon` và `Com chien ga`) tăng lỗi khi chuyển sang `ArUco XY` trên nền `raw_depth_mm`, trước khi hồi phục mạnh ở `Config 3a` (`Banh cuon`: `{p_c2['Banh cuon']['mae_g']:.1f} → {p_c3a['Banh cuon']['mae_g']:.1f} g`) và `Config 3b` (`Com chien ga`: `{p_c3ai['Com chien ga']['mae_g']:.1f} → {p_c3b['Com chien ga']['mae_g']:.1f} g`): nguyên nhân là ở `Config 1`, `mm_per_pixel` heuristic lớn hơn thực tế vô tình bù trừ cho độ sâu chưa neo.",
        "",
        "---",
        "",
        "## 5. Truy Vết Nguồn Gốc `relief_scale = 0.79`, Kiểm Định Độ Nhạy & Phân Tích Lỗi Còn Lại (`Config 3b`)",
        "",
    ])

    # 5.1 & 5.2 Dynamic sensitivity checks on N=96, N=94 (excluding banh-mi1/2), N=80 (excluding Banh mi), and N=2 (banh-mi1/2)
    bm_calib_imgs = {"banh-mi1.jpg", "banh-mi2.jpg"}
    rec_c3ai = records_by_cfg["Config_3a_Intrinsics"]
    rec_c3b = records_by_cfg["Config_3b_Full_ArUco"]

    def _subset_pair(pred_fn):
        sub_a = [r for r in rec_c3ai if pred_fn(r)]
        sub_b = [r for r in rec_c3b if pred_fn(r)]
        return len(sub_a), summarize_mass_groups(sub_a), summarize_mass_groups(sub_b)

    n96, s96_a, s96_b = _subset_pair(lambda r: True)
    n94, s94_a, s94_b = _subset_pair(lambda r: r["image_id"] not in bm_calib_imgs)
    n80, s80_a, s80_b = _subset_pair(lambda r: r["dish_class"] != "Banh mi")
    n2, s2_a, s2_b = _subset_pair(lambda r: r["image_id"] in bm_calib_imgs)

    sum_ae_96_diff = sum(abs(r["estimated_mass"] - r["GT_mass"]) for r in rec_c3b) - sum(
        abs(r["estimated_mass"] - r["GT_mass"]) for r in rec_c3ai
    )
    sum_ae_80_diff = sum(
        abs(r["estimated_mass"] - r["GT_mass"]) for r in rec_c3b if r["dish_class"] != "Banh mi"
    ) - sum(
        abs(r["estimated_mass"] - r["GT_mass"]) for r in rec_c3ai if r["dish_class"] != "Banh mi"
    )
    non_bm_share_pct = (sum_ae_80_diff / sum_ae_96_diff * 100.0) if sum_ae_96_diff != 0 else 0.0
    sum_ae_2_diff = sum(
        abs(r["estimated_mass"] - r["GT_mass"]) for r in rec_c3b if r["image_id"] in bm_calib_imgs
    ) - sum(
        abs(r["estimated_mass"] - r["GT_mass"]) for r in rec_c3ai if r["image_id"] in bm_calib_imgs
    )
    dev_vs_gt_gap_pct = (242.6 - 225.0) / 225.0 * 100.0

    bm12_a_map = {r["image_id"]: r for r in rec_c3ai if r["image_id"] in bm_calib_imgs}
    bm12_b_map = {r["image_id"]: r for r in rec_c3b if r["image_id"] in bm_calib_imgs}
    rec_by_gid_c3b = {r["mass_group_id"]: r for r in rec_c3b}

    lines.extend([
        "### 5.1. Truy vết nguồn gốc hằng số `relief_scale = 0.79`",
        "- **Bản chất:** `relief_scale = 0.79` là **hệ số hiệu chỉnh thực nghiệm từ giai đoạn phát triển (`31/08/2026`, commit `3905bc1`) có chồng lấn `2/96` nhóm với tập đánh giá**, không phải hằng số được tối ưu (sweep) trên 96 nhóm và cũng **không phải** hệ số từ một tập calibration độc lập.",
        f"- **Lịch sử & chênh lệch mốc tham chiếu:** Tại commit `3905bc1`, hệ số được chỉnh từ `0.62 → 0.79` trên hai ảnh `banh-mi1.jpg` và `banh-mi2.jpg` hướng tới mốc dev `~500 cm³ / 225 g` (`ρ = 0.45 g/cm³`). Khi đóng băng benchmark ở commit `18cd148` (`27/09/2026`), hai ảnh này có GT cân thực tế là **`242.6 g`** (chênh lệch **`+{dev_vs_gt_gap_pct:.2f}%`** so với `225 g`), và `0.79` được giữ nguyên không đổi qua toàn bộ chuỗi `A1–A5` và ArUco ablation.",
        "",
        "### 5.2. Kiểm định độ nhạy loại bỏ mẫu chồng lấn (`Config 3a-Intrinsics → Config 3b`)",
        "",
        "| Tập con kiểm định | `N` | MAE C3a-Intr → C3b (g) | $\\Delta\\text{MAE}$ (g) | MAPE C3a-Intr → C3b (%) | $\\Delta\\text{MAPE}$ | MPE C3a-Intr → C3b (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| Toàn tập benchmark | `{n96}` | `{s96_a['mae_g']:.2f} → {s96_b['mae_g']:.2f}` | `{s96_b['mae_g'] - s96_a['mae_g']:+.2f}` | `{s96_a['mape_pct']:.2f}% → {s96_b['mape_pct']:.2f}%` | `{s96_b['mape_pct'] - s96_a['mape_pct']:+.2f}%` | `{s96_a['mpe_pct']:+.2f}% → {s96_b['mpe_pct']:+.2f}%` |",
        f"| **Trường hợp A:** Loại bỏ 2 ảnh calibration (`banh-mi1/2`) | `{n94}` | `{s94_a['mae_g']:.2f} → {s94_b['mae_g']:.2f}` | **`{s94_b['mae_g'] - s94_a['mae_g']:+.2f}`** | `{s94_a['mape_pct']:.2f}% → {s94_b['mape_pct']:.2f}%` | **`{s94_b['mape_pct'] - s94_a['mape_pct']:+.2f}%`** | `{s94_a['mpe_pct']:+.2f}% → {s94_b['mpe_pct']:+.2f}%` |",
        f"| **Trường hợp B:** Loại bỏ toàn bộ lớp `Banh mi` (`13` lớp khác) | `{n80}` | `{s80_a['mae_g']:.2f} → {s80_b['mae_g']:.2f}` | **`{s80_b['mae_g'] - s80_a['mae_g']:+.2f}`** | `{s80_a['mape_pct']:.2f}% → {s80_b['mape_pct']:.2f}%` | **`{s80_b['mape_pct'] - s80_a['mape_pct']:+.2f}%`** | `{s80_a['mpe_pct']:+.2f}% → {s80_b['mpe_pct']:+.2f}%` |",
        f"| Riêng 2 ảnh calibration (`banh-mi1.jpg`, `banh-mi2.jpg`) | `{n2}` | `{s2_a['mae_g']:.2f} → {s2_b['mae_g']:.2f}` | `{s2_b['mae_g'] - s2_a['mae_g']:+.2f}` | `{s2_a['mape_pct']:.2f}% → {s2_b['mape_pct']:.2f}%` | `{s2_b['mape_pct'] - s2_a['mape_pct']:+.2f}%` | `{s2_a['mpe_pct']:+.2f}% → {s2_b['mpe_pct']:+.2f}%` |",
        "",
        f"- Trên chính `banh-mi1.jpg` (`{bm12_a_map['banh-mi1.jpg']['estimated_mass']:.1f} → {bm12_b_map['banh-mi1.jpg']['estimated_mass']:.1f} g`, GT `242.6 g`) và `banh-mi2.jpg` (`{bm12_a_map['banh-mi2.jpg']['estimated_mass']:.1f} → {bm12_b_map['banh-mi2.jpg']['estimated_mass']:.1f} g`, GT `242.6 g`), `0.79` làm tổng sai số tuyệt đối tăng nhẹ **`{sum_ae_2_diff:+.2f} g`**; ngược lại, **`{non_bm_share_pct:.1f}%`** mức giảm `Σ|AE|` (`{sum_ae_80_diff:.2f} g / {sum_ae_96_diff:.2f} g`) đến từ `{n80}` nhóm ngoài `Banh mi`.",
        f"- **Lưu ý diễn giải:** MPE trên `N = {n80}` nhóm ngoài `Banh mi` dịch chuyển từ `{s80_a['mpe_pct']:+.2f}% → {s80_b['mpe_pct']:+.2f}%`, cho thấy `0.79` làm giảm sai lệch tổng thể nhưng tăng nhẹ xu hướng ước lượng thiếu ở một số món phẳng/trũng, chưa phải hằng số tối ưu đồng đều cho mọi lớp món ăn.",
        "",
        "### 5.3. Tổng hợp lỗi còn lại trên `Config_3b_Full_ArUco` (`N = 96`)",
        "",
        "| Hạng | Nhóm khối lượng (`mass_group_id`) | Lớp (`dish_class`) | GT (g) | Dự đoán C3b (g) | Sai số tuyệt đối (g) | APE (%) | Nguồn scale | Fallback |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    top_errs = sorted(
        rec_c3b,
        key=lambda r: abs(r["estimated_mass"] - r["GT_mass"]),
        reverse=True,
    )[:8]
    for rank, r in enumerate(top_errs, 1):
        ae = abs(r["estimated_mass"] - r["GT_mass"])
        ape = ae / r["GT_mass"] * 100.0
        lines.append(
            f"| `{rank}` | `{r['mass_group_id']}` | `{r['dish_class']}` | "
            f"`{r['GT_mass']:.1f}` | `{r['estimated_mass']:.1f}` | `{ae:.1f}` | "
            f"`{ape:.1f}%` | `{r['scale_source']}` | `{r['fallback_used']}` |"
        )

    sup5 = rec_by_gid_c3b["SupCua (5).jpg::ann_58"]
    pho5 = rec_by_gid_c3b["Pho1 (5).jpg::ann_53"]
    sup5_ae = abs(sup5["estimated_mass"] - sup5["GT_mass"])
    pho5_ae = abs(pho5["estimated_mass"] - pho5["GT_mass"])

    lines.extend([
        "",
        "- **Các cụm sai số chính cần phân tích sâu ở bước kế tiếp:** "
        f"(1) **`Banh mi` (`MAPE = {p_c3b['Banh mi']['mape_pct']:.1f}%`, `N = {p_c3b['Banh mi']['n_groups']}`)**: đặc biệt `BanhMi_Trung` (GT `99.3 g` là 2 nửa ổ bánh mì rỗng ruột áp chảo nhưng dùng chung mật độ `0.45 g/cm³` của bánh mì thịt và chịu ảnh hưởng góc nghiêng); "
        f"(2) **`Com chien ga` (`MAE = {p_c3b['Com chien ga']['mae_g']:.1f} g`, `MPE = {p_c3b['Com chien ga']['mpe_pct']:+.1f}%`)**: các ảnh chụp đĩa sứ trũng bị ước lượng thiếu (`ComGa2 (1)`, `ComGa1 (5)`, `ComGa2 (3)`); "
        f"(3) **`Sup cua (5)` (`AE = {sup5_ae:.1f} g`)**: góc nghiêng mạnh trên chất lỏng; "
        f"(4) **`Pho1 (5)` (`AE = {pho5_ae:.1f} g`)**: mẫu duy nhất không phát hiện được ArUco (`{pho5['scale_source']}`).",
    ])

    with open(OUT_REPORT_PATH, "w", encoding="utf-8") as f_rep:
        f_rep.write("\n".join(lines) + "\n")
    print(f"Saved Markdown report to {OUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
