# -*- coding: utf-8 -*-
"""_build_aruco_ablation_cache.py

Builds and validates an isolated, provenance-tracked cache for the
5-Configuration ArUco & Depth-Anchoring Ablation study in:
    scratch/bench_cache_full/depth_ctx_aruco_ablation/scene_<stem>.npz
    scratch/bench_cache_full/depth_ctx_aruco_ablation/cache_manifest.json

Strictly preserves all existing caches (`scratch/bench_cache_full/depth_ctx/`)
and records (`scratch/volume_ablation_records.jsonl`).

Important Design & Data-Source Note on Config 1 (`GT-Assisted Plate Heuristic`):
  All 5 configurations in this ablation study are evaluated under the
  `Oracle GT Polygon Mask` setting (matching the primary A1-A5 geometric ablation
  in `scratch/run_volume_ablation.py`), where instance masks, bounding boxes
  (`bbox_xyxy`), and food classes (`vietfood67_class_name`) come from
  `scratch/gt_mapping_verified.json`.
  Accordingly, `scaled_dets` passed to `recover_heuristic_only()` uses these same
  Oracle GT bounding boxes and class labels so that:
    (1) No YOLO detection misses or class misclassifications confound the comparison
        between Config 1 and Configs 2-3b; and
    (2) Config 1 is explicitly named and interpreted as `Config_1_GT_Assisted_Plate_Heuristic`
        (an upper-bound container/plate heuristic assisted by Oracle GT boxes/classes,
        NOT a ground-truth-independent end-to-end detector heuristic).

For each of the 77 benchmark scenes, stores:
  - schema_version                : str cache schema identifier ("aruco_ablation_cache_v4")
  - raw_depth_mm                  : float32 (H, W) raw Depth Anything V2 Metric Indoor output in mm
  - anchored_depth_mm             : float32 (H, W) depth map after multiplying by scale_anchor_poly
  - valid_depth_px_ratio          : float64 fraction of pixels in raw_depth_mm with finite depth > 0
  - K                             : float64 (3, 3) camera intrinsic matrix at volume resolution
  - dist                          : float64 (5,) camera distortion coefficients
  - aruco_mm_per_pixel            : float64 lateral scale from ScaleRecovery.recover()
  - aruco_scale_source            : str ("aruco_marker" | "checkerboard" | "plate_heuristic" | ...)
  - depth_anchored                : bool True iff ArUco/checkerboard solvePnP depth anchoring succeeded
  - heuristic_mm_per_pixel        : float64 lateral scale from stateless GT-assisted heuristic recovery
  - heuristic_scale_source        : str ("plate_heuristic" | "image_heuristic")
  - heuristic_submethod           : str ("dish_guided_ellipse" | "hough_circle" | "image_dimension_fallback")
  - scale_anchor_poly             : float64 z_marker_tvec_z / d_pred_poly_median (1.0 if unanchored)
  - scale_anchor_center           : float64 z_marker_tvec_z / d_pred_center_5x5 (NaN if <15 valid px)
  - scale_anchor_plane_fit        : float64 median(Z_plane / Z_DA2_raw) across marker polygon (diagnostic only)
  - z_marker_tvec_z               : float64 solvePnP tvec[2] along camera optical axis Z_c (mm)
  - z_marker_euclid               : float64 ||tvec||_2 Euclidean distance from camera center (mm)
  - d_pred_poly_median            : float64 median raw depth inside 4-corner marker polygon (mm)
  - d_pred_center_5x5             : float64 median raw depth in 5x5 window around projected center (mm)
  - marker_valid_px               : int total valid depth pixels inside marker polygon
  - center_5x5_valid_px           : int valid depth pixels inside 5x5 projected center window (0..25)
  - center_in_polygon             : bool True iff projected (0,0,0) lies inside detected marker quad
  - marker_tilt_deg               : float64 arccos(|R[2,2]|) in degrees (angle between marker normal and Z_c)
  - marker_depth_std_raw_mm       : float64 Std(Z_DA2_raw) across marker polygon (raw DA2 spread)
  - marker_plane_std_mm           : float64 Std(Z_plane) across marker polygon (pure geometric 3D tilt spread)
  - marker_res_std_plane_fit_mm   : float64 Std(alpha_fit * Z_DA2_raw - Z_plane) (pure non-planar noise in mm)
  - marker_res_std_plane_fit_pct  : float64 Std(alpha_fit * Z_DA2_raw - Z_plane) / tvec_z * 100 (%)
  - marker_res_std_anc_poly_mm    : float64 Std(scale_anchor_poly * Z_DA2_raw - Z_plane) (anchored residual in mm)
"""

import sys
import os
import json
import math
import hashlib
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from depth_estimation import DepthEstimator, estimate_camera_intrinsics
from scale_recovery import ScaleRecovery

COCO_JSON_PATH = Path(
    r"C:\Users\huynh\OneDrive\Pictures\test-image\FoodDetector\annotations\instances_default.json"
)
VERIFIED_MAPPING_PATH = PROJECT_ROOT / "scratch" / "gt_mapping_verified.json"
CACHE_DIR = PROJECT_ROOT / "scratch" / "bench_cache_full"
ORIG_DEPTH_CACHE_DIR = CACHE_DIR / "depth_ctx"
ARUCO_ABLATION_CACHE_DIR = CACHE_DIR / "depth_ctx_aruco_ablation"
CACHE_MANIFEST_PATH = ARUCO_ABLATION_CACHE_DIR / "cache_manifest.json"

CACHE_SCHEMA_VERSION = "aruco_ablation_cache_v4"
# Note on scale_anchor bounds in the codebase:
#   - `scale_recovery.py` does NOT perform depth anchoring; it only returns ScaleResult.
#   - `scratch/_cache_depth_scenes.py` (lines 121-124, which built the A1-A5 benchmark cache)
#     uses the acceptance gate: `if 0.25 <= scale_z <= 4.0: final_depth_mm *= scale_z`.
#   - `food_volume_pipeline.py` (lines 277-283, interactive app orchestrator)
#     uses: `scale_anchor = float(np.clip(scale_anchor, 0.3, 3.0))`.
# On the 76 ArUco-anchored scenes in the current 77-image benchmark run, the
# empirically observed range is `scale_z_observed in [0.3151, 0.7359]` (an
# observation of this dataset run, NOT a general algorithm bound). It lies strictly
# inside BOTH control intervals [0.30, 3.00] and [0.25, 4.00], which remain unchanged.
# We enforce the `_cache_depth_scenes.py` gate [0.25, 4.00] to match the A1-A5 anchored
# cache within numerical verification tolerance AND explicitly assert in Gate 1 that every
# anchored scene also satisfies [0.30, 3.00] (exact clamp equivalence on this run).
ANCHOR_CLIP_MIN = 0.25
ANCHOR_CLIP_MAX = 4.0
PIPELINE_CLIP_MIN = 0.30
PIPELINE_CLIP_MAX = 3.00
MIN_VALID_DEPTH_PX_RATIO = 0.95

REQUIRED_NPZ_KEYS = (
    "schema_version",
    "raw_depth_mm",
    "anchored_depth_mm",
    "valid_depth_px_ratio",
    "K",
    "dist",
    "aruco_mm_per_pixel",
    "aruco_scale_source",
    "depth_anchored",
    "heuristic_mm_per_pixel",
    "heuristic_scale_source",
    "heuristic_submethod",
    "scale_anchor_poly",
    "scale_anchor_center",
    "scale_anchor_plane_fit",
    "z_marker_tvec_z",
    "z_marker_euclid",
    "d_pred_poly_median",
    "d_pred_center_5x5",
    "marker_valid_px",
    "center_5x5_valid_px",
    "center_in_polygon",
    "marker_tilt_deg",
    "marker_depth_std_raw_mm",
    "marker_plane_std_mm",
    "marker_res_std_plane_fit_mm",
    "marker_res_std_plane_fit_pct",
    "marker_res_std_anc_poly_mm",
    "h_vol",
    "w_vol",
    "sx",
    "sy",
    "encoder",
    "model_id",
)


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file for provenance verification."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_expected_cache_manifest(target_files: list) -> dict:
    """Construct expected cache manifest dictionary for provenance validation."""
    calib_file_str = str(getattr(pc, "CAMERA_CALIB_FILE", ""))
    calib_path = Path(calib_file_str) if calib_file_str else None
    if calib_path is not None and not calib_path.is_absolute():
        calib_path = PROJECT_ROOT / calib_path
    calib_sha256 = (
        compute_file_sha256(calib_path)
        if (calib_path is not None and calib_path.is_file())
        else "none_default_fov"
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "num_scenes": len(target_files),
        "target_files": list(target_files),
        "verified_mapping_sha256": compute_file_sha256(VERIFIED_MAPPING_PATH),
        "depth_encoder": str(pc.DEPTH_ENCODER),
        "depth_model_id": str(pc.DEPTH_HF_MODELS.get(pc.DEPTH_ENCODER, "vits")),
        "aruco_dict_type": str(getattr(pc, "ARUCO_DICT_TYPE", "DICT_4X4_50")),
        "aruco_marker_length_mm": float(getattr(pc, "ARUCO_MARKER_LENGTH_MM", 30.0)),
        "camera_calib_file": calib_file_str,
        "camera_calib_sha256": calib_sha256,
        "anchor_clip_bounds": [ANCHOR_CLIP_MIN, ANCHOR_CLIP_MAX],
        "pipeline_clip_bounds": [PIPELINE_CLIP_MIN, PIPELINE_CLIP_MAX],
        "config_1_box_source": "oracle_gt_bbox_and_class_from_gt_mapping_verified",
    }


def recover_heuristic_only(
    image_bgr: np.ndarray,
    detected_classes: list,
    scaled_dets: list,
) -> tuple:
    """Stateless GT-assisted heuristic-only scale recovery for Config 1
    (`Config_1_GT_Assisted_Plate_Heuristic`).

    Instantiates a fresh `ScaleRecovery` object and explicitly bypasses
    `_try_aruco` and `_try_checkerboard` so no ArUco/checkerboard information
    can ever leak into Config 1. Note that `scaled_dets` and `detected_classes`
    come from the Oracle GT annotations (`gt_mapping_verified.json`) so that all
    5 configurations share identical Oracle GT boxes/classes.

    Returns:
        (ScaleResult, submethod_str)
        where submethod_str in:
          - "dish_guided_ellipse"
          - "hough_circle"
          - "image_dimension_fallback"
    """
    sr_heur = ScaleRecovery()

    if scaled_dets:
        dish_res = sr_heur._try_dish_guided_scale(
            image_bgr, yolo_detections=scaled_dets, seg_results=None
        )
        if dish_res is not None:
            return dish_res, "dish_guided_ellipse"

    plate_res = sr_heur._try_plate_heuristic(image_bgr, detected_classes)
    if plate_res is not None:
        return plate_res, "hough_circle"

    img_res = sr_heur._image_heuristic(image_bgr)
    return img_res, "image_dimension_fallback"


def compute_marker_plane_depths(
    xs: np.ndarray,
    ys: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Compute exact expected camera-axis depth Z_plane(x, y) (in mm) on the
    3D marker plane defined by solvePnP (rvec, tvec) for pixel coordinates (xs, ys).

    Derivation:
      In the marker object frame, the marker lies on z_m = 0 with origin (0,0,0)
      at the marker center. In the camera coordinate frame:
        - Plane point : t = tvec.reshape(3) (mm)
        - Plane normal: n = R[:, 2] where R = Rodrigues(rvec)
      Any 3D point P_c = Z_c * [x_norm, y_norm, 1]^T on the marker plane satisfies:
        n^T (P_c - t) = 0  ==>  Z_plane(x, y) = (n^T t) / (n_x * x_norm + n_y * y_norm + n_z)
      where (x_norm, y_norm) are undistorted normalized camera coordinates from
      cv2.undistortPoints.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    n = R[:, 2].reshape(3)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    d0 = float(np.dot(n, t))

    pts_uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)]).reshape(-1, 1, 2)
    dist_arr = np.asarray(dist, dtype=np.float64) if dist is not None else np.zeros(5, dtype=np.float64)
    norm_pts = cv2.undistortPoints(pts_uv, K.astype(np.float64), dist_arr).reshape(-1, 2)

    denom = n[0] * norm_pts[:, 0] + n[1] * norm_pts[:, 1] + n[2]
    valid_denom = np.abs(denom) > 1e-6
    z_plane = np.full(len(xs), np.nan, dtype=np.float64)
    z_plane[valid_denom] = d0 / denom[valid_denom]
    return z_plane


def compute_marker_diagnostics(
    raw_depth_mm: np.ndarray,
    sres,
    K: np.ndarray,
    dist: np.ndarray,
) -> dict:
    """Compute comprehensive geometric and residual diagnostics on the detected marker.

    Includes an independent plane-fitted scale factor:
        alpha_fit = median(Z_plane(u, v) / Z_DA2_raw(u, v))
    so that the non-planar surface residual:
        res_plane_fit = alpha_fit * Z_DA2_raw(u, v) - Z_plane(u, v)
    measures DA2's local surface noise on the marker plane after removing 3D tilt,
    independently of the single-point/median anchor factor `scale_anchor_poly`.
    Note: `alpha_fit` is used strictly for diagnostic reporting in Section 1.2 and
    never alters `scale_anchor_poly` or `anchored_depth_mm` in Configs 3a-3b.
    """
    h, w = raw_depth_mm.shape[:2]
    diag = {
        "depth_anchored": False,
        "scale_anchor_poly": 1.0,
        "scale_anchor_center": float("nan"),
        "scale_anchor_plane_fit": float("nan"),
        "z_marker_tvec_z": float("nan"),
        "z_marker_euclid": float("nan"),
        "d_pred_poly_median": float("nan"),
        "d_pred_center_5x5": float("nan"),
        "marker_valid_px": 0,
        "center_5x5_valid_px": 0,
        "center_in_polygon": False,
        "marker_tilt_deg": float("nan"),
        "marker_depth_std_raw_mm": float("nan"),
        "marker_plane_std_mm": float("nan"),
        "marker_res_std_plane_fit_mm": float("nan"),
        "marker_res_std_plane_fit_pct": float("nan"),
        "marker_res_std_anc_poly_mm": float("nan"),
    }

    if (
        sres.scale_source not in ("aruco_marker", "checkerboard")
        or sres.marker_tvec is None
        or sres.marker_corners is None
    ):
        return diag

    tvec = np.asarray(sres.marker_tvec, dtype=np.float64).reshape(3)
    z_marker_true = float(tvec[2])
    z_marker_euclid = float(np.linalg.norm(tvec))
    diag["z_marker_tvec_z"] = z_marker_true
    diag["z_marker_euclid"] = z_marker_euclid

    if sres.marker_rvec is not None:
        R, _ = cv2.Rodrigues(np.asarray(sres.marker_rvec, dtype=np.float64))
        cos_tilt = float(np.clip(abs(R[2, 2]), 0.0, 1.0))
        diag["marker_tilt_deg"] = float(np.degrees(np.arccos(cos_tilt)))

    corners = np.asarray(sres.marker_corners, dtype=np.float32).reshape(-1, 2)
    marker_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(marker_mask, [corners.astype(np.int32)], 1)

    ys, xs = np.where(marker_mask > 0)
    raw_vals = raw_depth_mm[ys, xs].astype(np.float64)
    valid_mask = (raw_vals > 0.0) & np.isfinite(raw_vals)
    ys_v, xs_v, valid_md = ys[valid_mask], xs[valid_mask], raw_vals[valid_mask]

    diag["marker_valid_px"] = int(len(valid_md))
    if len(valid_md) == 0 or z_marker_true <= 0:
        return diag

    d_pred_poly = float(np.median(valid_md))
    diag["d_pred_poly_median"] = d_pred_poly
    diag["marker_depth_std_raw_mm"] = float(np.std(valid_md, ddof=0))

    scale_z = z_marker_true / d_pred_poly
    if ANCHOR_CLIP_MIN <= scale_z <= ANCHOR_CLIP_MAX:
        diag["depth_anchored"] = True
        diag["scale_anchor_poly"] = float(scale_z)

    # Plane residual decomposition (separating 3D geometric tilt from DA2 non-planar noise)
    if sres.marker_rvec is not None and K is not None and len(valid_md) >= 4:
        z_plane = compute_marker_plane_depths(xs_v, ys_v, sres.marker_rvec, tvec, K, dist)
        ok_p = np.isfinite(z_plane) & (z_plane > 0.0) & (valid_md > 0.0)
        if np.sum(ok_p) >= 4:
            zp_v = z_plane[ok_p]
            zd_v = valid_md[ok_p]
            diag["marker_plane_std_mm"] = float(np.std(zp_v, ddof=0))

            # Independent per-pixel ratio fit on the marker plane (diagnostic only)
            alpha_fit = float(np.median(zp_v / zd_v))
            diag["scale_anchor_plane_fit"] = alpha_fit
            res_plane_fit_mm = (alpha_fit * zd_v) - zp_v
            std_fit_mm = float(np.std(res_plane_fit_mm, ddof=0))
            diag["marker_res_std_plane_fit_mm"] = std_fit_mm
            diag["marker_res_std_plane_fit_pct"] = float((std_fit_mm / z_marker_true) * 100.0)

            # Residual of the production polygon-anchored depth against Z_plane
            if scale_z > 0:
                res_anc_poly_mm = (scale_z * zd_v) - zp_v
                diag["marker_res_std_anc_poly_mm"] = float(np.std(res_anc_poly_mm, ddof=0))

    # Project marker origin (0, 0, 0) to image and sample 5x5 window
    if sres.marker_rvec is not None and K is not None:
        dist_arr = np.asarray(dist, dtype=np.float64) if dist is not None else np.zeros(5, dtype=np.float64)
        origin_3d = np.zeros((1, 1, 3), dtype=np.float64)
        proj_pts, _ = cv2.projectPoints(
            origin_3d,
            np.asarray(sres.marker_rvec, dtype=np.float64),
            tvec.reshape(3, 1),
            K.astype(np.float64),
            dist_arr,
        )
        uc, vc = float(proj_pts[0, 0, 0]), float(proj_pts[0, 0, 1])
        inside_test = cv2.pointPolygonTest(corners, (uc, vc), False)
        diag["center_in_polygon"] = bool(inside_test >= 0)

        rc, cc = int(round(vc)), int(round(uc))
        r0, r1 = max(0, rc - 2), min(h, rc + 3)
        c0, c1 = max(0, cc - 2), min(w, cc + 3)
        if (r1 - r0) == 5 and (c1 - c0) == 5 and diag["center_in_polygon"]:
            patch = raw_depth_mm[r0:r1, c0:c1].astype(np.float64)
            valid_patch = patch[(patch > 0.0) & np.isfinite(patch)]
            n_val_5x5 = int(len(valid_patch))
            diag["center_5x5_valid_px"] = n_val_5x5
            # Strict requirement: at least 15/25 valid pixels, no silent fallback
            if n_val_5x5 >= 15:
                d_center = float(np.median(valid_patch))
                if d_center > 0:
                    diag["d_pred_center_5x5"] = d_center
                    diag["scale_anchor_center"] = float(z_marker_true / d_center)

    return diag


def validate_scene_dict_and_cross_check(
    fn: str,
    scene_dict: dict,
    orig_npz_path: Path,
    expected_manifest: dict,
):
    """Run strict schema, provenance, Gate 1, Gate 2, and original-cache cross-checks
    on a single scene dictionary (executed identically whether loaded from disk or newly built).
    """
    missing_keys = set(REQUIRED_NPZ_KEYS) - set(scene_dict.keys())
    if missing_keys:
        raise RuntimeError(
            f"Scene cache for {fn} is missing required keys {sorted(missing_keys)}. "
            f"Re-run with force_rebuild=True."
        )

    if str(scene_dict["schema_version"]) != expected_manifest["schema_version"]:
        raise RuntimeError(
            f"Schema version mismatch on {fn}: {scene_dict['schema_version']} "
            f"!= {expected_manifest['schema_version']}. Re-run with force_rebuild=True."
        )
    if str(scene_dict["encoder"]) != expected_manifest["depth_encoder"]:
        raise RuntimeError(
            f"Depth encoder mismatch on {fn}: {scene_dict['encoder']} "
            f"!= {expected_manifest['depth_encoder']}. Re-run with force_rebuild=True."
        )
    if str(scene_dict["model_id"]) != expected_manifest["depth_model_id"]:
        raise RuntimeError(
            f"Depth model_id mismatch on {fn}: {scene_dict['model_id']} "
            f"!= {expected_manifest['depth_model_id']}. Re-run with force_rebuild=True."
        )

    # Gate 1: Raw & anchored depth map integrity + valid pixel ratio check
    raw_d = scene_dict["raw_depth_mm"]
    anc_d = scene_dict["anchored_depth_mm"]
    h_v, w_v = int(scene_dict["h_vol"]), int(scene_dict["w_vol"])
    assert raw_d.shape == (h_v, w_v) and anc_d.shape == (h_v, w_v), f"Shape mismatch on {fn}"
    assert raw_d.dtype == np.float32 and anc_d.dtype == np.float32, f"Dtype mismatch on {fn}"
    assert np.all(np.isfinite(raw_d)) and np.all(np.isfinite(anc_d)), f"Non-finite depth on {fn}"
    assert float(raw_d.max()) > 0.0, f"Non-positive max depth on {fn}"

    valid_px_ratio = float(np.mean((raw_d > 0.0) & np.isfinite(raw_d)))
    stored_ratio = float(scene_dict["valid_depth_px_ratio"])
    assert math.isclose(valid_px_ratio, stored_ratio, rel_tol=1e-6, abs_tol=1e-6), (
        f"valid_depth_px_ratio mismatch on {fn}: {valid_px_ratio} vs {stored_ratio}"
    )
    assert valid_px_ratio >= MIN_VALID_DEPTH_PX_RATIO, (
        f"Gate 1 failed on {fn}: valid_depth_px_ratio={valid_px_ratio:.4f} < {MIN_VALID_DEPTH_PX_RATIO}"
    )

    sc_poly = float(scene_dict["scale_anchor_poly"])
    is_anc = bool(scene_dict["depth_anchored"])
    aruco_src = str(scene_dict["aruco_scale_source"])
    if is_anc:
        assert aruco_src in ("aruco_marker", "checkerboard"), (
            f"Gate 1 failed on {fn}: depth_anchored=True but aruco_scale_source={aruco_src}"
        )
        assert ANCHOR_CLIP_MIN <= sc_poly <= ANCHOR_CLIP_MAX, (
            f"Gate 1 failed on {fn}: scale_anchor_poly={sc_poly} outside [{ANCHOR_CLIP_MIN}, {ANCHOR_CLIP_MAX}]"
        )
        assert PIPELINE_CLIP_MIN <= sc_poly <= PIPELINE_CLIP_MAX, (
            f"Gate 1 failed on {fn}: scale_anchor_poly={sc_poly} outside food_volume_pipeline bounds [{PIPELINE_CLIP_MIN}, {PIPELINE_CLIP_MAX}]"
        )
        # Verify empirically that `_cache_depth_scenes.py` acceptance gate [0.25, 4.00]
        # and `food_volume_pipeline.py` clamp np.clip(scale_anchor, 0.30, 3.00) yield
        # the exact same anchor factor and anchored depth map on this scene:
        sc_poly_pipeline_clamped = float(np.clip(sc_poly, PIPELINE_CLIP_MIN, PIPELINE_CLIP_MAX))
        assert math.isclose(sc_poly, sc_poly_pipeline_clamped, rel_tol=1e-12, abs_tol=1e-12), (
            f"Gate 1 failed on {fn}: gating scale_anchor_poly ({sc_poly}) != clamped ({sc_poly_pipeline_clamped})"
        )
        assert np.allclose(anc_d, raw_d * np.float32(sc_poly), rtol=1e-5, atol=1e-3), (
            f"Gate 1 failed on anchored scene {fn}: anchored_depth_mm != raw_depth_mm * scale_anchor_poly"
        )
    else:
        assert aruco_src in ("plate_heuristic", "image_heuristic"), (
            f"Gate 1 failed on {fn}: depth_anchored=False but aruco_scale_source={aruco_src}"
        )
        assert np.array_equal(anc_d, raw_d) and sc_poly == 1.0, (
            f"Gate 1 failed on unanchored scene {fn}: anchored_depth_mm != raw_depth_mm or scale_anchor_poly != 1.0"
        )

    # Gate 2: Stateless heuristic scale source & value check
    heur_src = str(scene_dict["heuristic_scale_source"])
    heur_sub = str(scene_dict["heuristic_submethod"])
    heur_mm = float(scene_dict["heuristic_mm_per_pixel"])
    assert heur_src in ("plate_heuristic", "image_heuristic"), (
        f"Gate 2 failed on {fn}: invalid heuristic_scale_source={heur_src}"
    )
    assert heur_sub in ("dish_guided_ellipse", "hough_circle", "image_dimension_fallback"), (
        f"Gate 2 failed on {fn}: invalid heuristic_submethod={heur_sub}"
    )
    assert math.isfinite(heur_mm) and heur_mm > 0.0, (
        f"Gate 2 failed on {fn}: invalid heuristic_mm_per_pixel={heur_mm}"
    )

    # Cross-check against original reference depth_ctx cache (executed on both build AND load!)
    if orig_npz_path.exists():
        orig_d = np.load(orig_npz_path, allow_pickle=True)
        orig_anchored = orig_d["depth_mm"].astype(np.float32)
        orig_mm_px = float(orig_d["mm_per_pixel"])
        orig_src = str(orig_d["scale_source"])
        orig_flag = bool(orig_d["depth_anchored"])
        orig_d.close()

        assert orig_src == aruco_src, (
            f"Scale source mismatch against orig cache on {fn}: {orig_src} vs {aruco_src}"
        )
        assert orig_flag == is_anc, (
            f"depth_anchored mismatch against orig cache on {fn}: {orig_flag} vs {is_anc}"
        )
        assert math.isclose(orig_mm_px, float(scene_dict["aruco_mm_per_pixel"]), rel_tol=1e-6), (
            f"mm_per_pixel mismatch against orig cache on {fn}: {orig_mm_px} vs {scene_dict['aruco_mm_per_pixel']}"
        )
        assert np.allclose(anc_d, orig_anchored, rtol=1e-4, atol=1e-2), (
            f"Anchored depth mismatch against original cache on {fn}"
        )


def verify_cache_manifest_state(
    cache_dir: Path,
    manifest_path: Path,
    target_files: list,
    expected_manifest: dict,
    force_rebuild: bool,
) -> dict:
    """Validate provenance manifest whenever any pre-existing .npz cache file exists.

    Policy (prevents silently mixing stale .npz files with newly built .npz files):
      1. If `force_rebuild=True`: return `mode="force_rebuild"` (all targets rebuilt).
      2. If `force_rebuild=False` and ANY `.npz` file exists in `cache_dir` (even if
         one or more target `.npz` files are missing):
           - Refuse immediately with RuntimeError if `manifest_path` is missing.
           - Refuse immediately with RuntimeError if `manifest_path` is unreadable/invalid
             or differs from `expected_manifest` on any key or key set.
      3. Only if `manifest_path` exists AND matches `expected_manifest` cleanly:
           - Reuse existing verified `.npz` files and return only `missing_files`
             to rebuild (`mode="full_reuse"` or `"partial_resume"`).
      4. If `force_rebuild=False` and ZERO `.npz` files exist in `cache_dir`:
           - Return `mode="fresh_build"` so all targets will be built from scratch.
    """
    target_npz_paths = {
        fn: cache_dir / f"scene_{Path(fn).stem}.npz" for fn in target_files
    }
    existing_target_files = [fn for fn, p in target_npz_paths.items() if p.exists()]
    missing_files = [fn for fn, p in target_npz_paths.items() if not p.exists()]
    all_npz_in_dir = sorted(cache_dir.glob("*.npz")) if cache_dir.exists() else []

    if force_rebuild:
        return {
            "mode": "force_rebuild",
            "existing_count": len(existing_target_files),
            "missing_files": list(target_files),
        }

    if len(all_npz_in_dir) > 0 or len(existing_target_files) > 0:
        if not manifest_path.exists():
            raise RuntimeError(
                f"Cache manifest {manifest_path} is missing while "
                f"{len(all_npz_in_dir)} .npz file(s) already exist in {cache_dir} "
                f"({len(missing_files)} target .npz missing). "
                f"Refusing to load or mix unverified cache files; re-run with force_rebuild=True."
            )
        try:
            with open(manifest_path, encoding="utf-8") as f_man:
                loaded_manifest = json.load(f_man)
        except Exception as exc:
            raise RuntimeError(
                f"Cache manifest {manifest_path} could not be parsed ({exc}) while "
                f"{len(all_npz_in_dir)} .npz file(s) exist. Re-run with force_rebuild=True."
            ) from exc

        if not isinstance(loaded_manifest, dict):
            raise RuntimeError(
                f"Cache manifest {manifest_path} must be a JSON object. "
                f"Re-run with force_rebuild=True."
            )

        for mk, mv in expected_manifest.items():
            if loaded_manifest.get(mk) != mv:
                raise RuntimeError(
                    f"Cache manifest mismatch on key '{mk}' (existing={len(all_npz_in_dir)}, "
                    f"missing={len(missing_files)}): found {loaded_manifest.get(mk)!r}, "
                    f"expected {mv!r}. Refusing to reuse or mix stale cache; "
                    f"re-run with force_rebuild=True."
                )

        extra_keys = set(loaded_manifest.keys()) - set(expected_manifest.keys())
        if extra_keys:
            raise RuntimeError(
                f"Cache manifest has unexpected key(s) {sorted(extra_keys)}. "
                f"Re-run with force_rebuild=True."
            )

        mode = "full_reuse" if len(missing_files) == 0 else "partial_resume"
        return {
            "mode": mode,
            "existing_count": len(existing_target_files),
            "missing_files": missing_files,
        }

    return {
        "mode": "fresh_build",
        "existing_count": 0,
        "missing_files": list(target_files),
    }


def build_or_verify_aruco_ablation_cache(force_rebuild: bool = False) -> dict:
    """Build or load the 77-scene ArUco ablation cache and run Gate 1 & Gate 2 checks."""
    os.makedirs(ARUCO_ABLATION_CACHE_DIR, exist_ok=True)

    mp = json.load(open(VERIFIED_MAPPING_PATH, encoding="utf-8"))
    inst_by_fn = defaultdict(list)
    for x in mp["instances"]:
        inst_by_fn[x["file_name"]].append(x)

    target_files = sorted(inst_by_fn.keys())
    assert len(target_files) == 77, f"Expected 77 target images, got {len(target_files)}"

    expected_manifest = build_expected_cache_manifest(target_files)
    manifest_state = verify_cache_manifest_state(
        cache_dir=ARUCO_ABLATION_CACHE_DIR,
        manifest_path=CACHE_MANIFEST_PATH,
        target_files=target_files,
        expected_manifest=expected_manifest,
        force_rebuild=force_rebuild,
    )
    print(
        f"[aruco-cache] Manifest check passed: mode={manifest_state['mode']}, "
        f"existing={manifest_state['existing_count']}/{len(target_files)}, "
        f"to_build={len(manifest_state['missing_files'])}.",
        flush=True,
    )

    de = None
    sr_aruco = None
    cached_scenes = {}
    rebuilt_any = False

    for idx, fn in enumerate(target_files):
        stem = Path(fn).stem
        npz_path = ARUCO_ABLATION_CACHE_DIR / f"scene_{stem}.npz"
        orig_npz_path = ORIG_DEPTH_CACHE_DIR / f"scene_{stem}.npz"
        insts = inst_by_fn[fn]

        if npz_path.exists() and not force_rebuild:
            data = np.load(npz_path, allow_pickle=True)
            scene_dict = {k: data[k] for k in data.files}
            data.close()
        else:
            rebuilt_any = True
            if de is None:
                de = DepthEstimator()
            if sr_aruco is None:
                sr_aruco = ScaleRecovery()

            img_path = insts[0]["image_path"]
            img_bgr = cv2.imread(img_path)
            assert img_bgr is not None, f"Failed to read image: {img_path}"
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_full, w_full = img_rgb.shape[:2]

            vol_pil = Image.fromarray(img_rgb)
            vol_pil.thumbnail((2000, 2000))
            vol_rgb = np.asarray(vol_pil)
            vol_bgr = cv2.cvtColor(vol_rgb, cv2.COLOR_RGB2BGR)
            h_vol, w_vol = vol_rgb.shape[:2]
            sx, sy = w_vol / float(w_full), h_vol / float(h_full)

            K, dist = estimate_camera_intrinsics(
                (h_vol, w_vol), image_path=img_path, calib_file=pc.CAMERA_CALIB_FILE
            )
            # Note: scaled_dets uses Oracle GT bbox_xyxy and vietfood67_class_name
            # from gt_mapping_verified.json so all 5 configurations in the Oracle GT Mask
            # study share identical boxes/classes (Config 1 = GT-Assisted Plate Heuristic).
            scaled_dets = [
                {
                    "bbox": (
                        int(round(x["bbox_xyxy"][0] * sx)),
                        int(round(x["bbox_xyxy"][1] * sy)),
                        int(round(x["bbox_xyxy"][2] * sx)),
                        int(round(x["bbox_xyxy"][3] * sy)),
                    ),
                    "class_name": x["vietfood67_class_name"],
                    "confidence": 1.0,
                }
                for x in insts
            ]
            detected_classes = [d["class_name"] for d in scaled_dets]

            # 1. Standard ScaleRecovery (with ArUco/checkerboard priority)
            sres_aruco = sr_aruco.recover(
                vol_bgr,
                K=K,
                dist=dist,
                detected_classes=detected_classes,
                yolo_detections=scaled_dets,
                seg_results=None,
            )

            # 2. Stateless GT-Assisted Heuristic-Only ScaleRecovery (Config 1)
            sres_heur, heur_submethod = recover_heuristic_only(
                vol_bgr, detected_classes=detected_classes, scaled_dets=scaled_dets
            )
            # Gate 2 check at creation time: Config 1 must never contain marker state
            assert (
                sres_heur.marker_corners is None
                and sres_heur.marker_tvec is None
                and sres_heur.marker_rvec is None
            ), f"Gate 2 violation on {fn}: heuristic ScaleResult leaked marker pose!"
            assert sres_heur.scale_source in ("plate_heuristic", "image_heuristic")
            assert math.isfinite(sres_heur.mm_per_pixel) and sres_heur.mm_per_pixel > 0

            # 3. Raw depth map from DepthEstimator
            dres = de.estimate(vol_rgb)
            raw_depth_mm = dres.depth_mm.astype(np.float32).copy()
            valid_depth_px_ratio = float(
                np.mean((raw_depth_mm > 0.0) & np.isfinite(raw_depth_mm))
            )

            diag = compute_marker_diagnostics(raw_depth_mm, sres_aruco, K, dist)
            scale_anchor_poly = float(diag["scale_anchor_poly"])
            depth_anchored = bool(diag["depth_anchored"])

            if depth_anchored:
                anchored_depth_mm = (raw_depth_mm * np.float32(scale_anchor_poly)).astype(np.float32)
            else:
                anchored_depth_mm = raw_depth_mm.copy()

            scene_dict = {
                "schema_version": str(CACHE_SCHEMA_VERSION),
                "raw_depth_mm": raw_depth_mm,
                "anchored_depth_mm": anchored_depth_mm,
                "valid_depth_px_ratio": np.float64(valid_depth_px_ratio),
                "K": K.astype(np.float64),
                "dist": np.asarray(dist, dtype=np.float64),
                "aruco_mm_per_pixel": np.float64(sres_aruco.mm_per_pixel),
                "aruco_scale_source": str(sres_aruco.scale_source),
                "depth_anchored": np.bool_(depth_anchored),
                "heuristic_mm_per_pixel": np.float64(sres_heur.mm_per_pixel),
                "heuristic_scale_source": str(sres_heur.scale_source),
                "heuristic_submethod": str(heur_submethod),
                "scale_anchor_poly": np.float64(scale_anchor_poly),
                "scale_anchor_center": np.float64(diag["scale_anchor_center"]),
                "scale_anchor_plane_fit": np.float64(diag["scale_anchor_plane_fit"]),
                "z_marker_tvec_z": np.float64(diag["z_marker_tvec_z"]),
                "z_marker_euclid": np.float64(diag["z_marker_euclid"]),
                "d_pred_poly_median": np.float64(diag["d_pred_poly_median"]),
                "d_pred_center_5x5": np.float64(diag["d_pred_center_5x5"]),
                "marker_valid_px": np.int32(diag["marker_valid_px"]),
                "center_5x5_valid_px": np.int32(diag["center_5x5_valid_px"]),
                "center_in_polygon": np.bool_(diag["center_in_polygon"]),
                "marker_tilt_deg": np.float64(diag["marker_tilt_deg"]),
                "marker_depth_std_raw_mm": np.float64(diag["marker_depth_std_raw_mm"]),
                "marker_plane_std_mm": np.float64(diag["marker_plane_std_mm"]),
                "marker_res_std_plane_fit_mm": np.float64(diag["marker_res_std_plane_fit_mm"]),
                "marker_res_std_plane_fit_pct": np.float64(diag["marker_res_std_plane_fit_pct"]),
                "marker_res_std_anc_poly_mm": np.float64(diag["marker_res_std_anc_poly_mm"]),
                "h_vol": np.int32(h_vol),
                "w_vol": np.int32(w_vol),
                "sx": np.float64(sx),
                "sy": np.float64(sy),
                "encoder": str(pc.DEPTH_ENCODER),
                "model_id": str(pc.DEPTH_HF_MODELS.get(pc.DEPTH_ENCODER, "vits")),
            }
            np.savez_compressed(npz_path, **scene_dict)
            print(
                f"[aruco-cache] [{idx+1}/77] Saved {fn} "
                f"(src={sres_aruco.scale_source}, scale_z={scale_anchor_poly:.4f})",
                flush=True,
            )

        # Always run full validation + orig cache cross-check on loaded or newly built scene
        validate_scene_dict_and_cross_check(fn, scene_dict, orig_npz_path, expected_manifest)
        cached_scenes[fn] = scene_dict

    if rebuilt_any or not CACHE_MANIFEST_PATH.exists():
        with open(CACHE_MANIFEST_PATH, "w", encoding="utf-8") as f_man:
            json.dump(expected_manifest, f_man, indent=2, ensure_ascii=False)

    # Gate 1 explicit per-source counting across all 77 scenes
    n_aruco_marker = sum(
        1 for s in cached_scenes.values() if str(s["aruco_scale_source"]) == "aruco_marker"
    )
    n_checkerboard = sum(
        1 for s in cached_scenes.values() if str(s["aruco_scale_source"]) == "checkerboard"
    )
    n_heuristic_fallback = sum(
        1
        for s in cached_scenes.values()
        if str(s["aruco_scale_source"]) in ("plate_heuristic", "image_heuristic")
    )
    n_depth_anchored = sum(1 for s in cached_scenes.values() if bool(s["depth_anchored"]))

    assert n_aruco_marker == 76, f"Gate 1: Expected 76 aruco_marker scenes, got {n_aruco_marker}"
    assert n_checkerboard == 0, f"Gate 1: Expected 0 checkerboard scenes, got {n_checkerboard}"
    assert n_heuristic_fallback == 1, (
        f"Gate 1: Expected 1 heuristic fallback scene, got {n_heuristic_fallback}"
    )
    assert n_depth_anchored == 76, (
        f"Gate 1: Expected 76 depth_anchored scenes, got {n_depth_anchored}"
    )
    assert "Pho1 (5).jpg" in cached_scenes, "Expected fallback scene 'Pho1 (5).jpg' in cached_scenes"
    assert str(cached_scenes["Pho1 (5).jpg"]["aruco_scale_source"]) == "plate_heuristic", (
        f"Expected 'Pho1 (5).jpg' to have aruco_scale_source=='plate_heuristic', "
        f"got {cached_scenes['Pho1 (5).jpg']['aruco_scale_source']}"
    )
    assert bool(cached_scenes["Pho1 (5).jpg"]["depth_anchored"]) is False, (
        "Expected 'Pho1 (5).jpg' to have depth_anchored==False"
    )

    print(
        f"[Gate 1 & Gate 2 PASS] All 77 scenes verified: "
        f"aruco_marker={n_aruco_marker}, checkerboard={n_checkerboard}, "
        f"heuristic_fallback={n_heuristic_fallback} ('Pho1 (5).jpg'), "
        f"depth_anchored={n_depth_anchored}."
    )
    return cached_scenes


def run_manifest_guard_smoke_tests():
    """Offline smoke tests in an isolated TemporaryDirectory verifying cache manifest
    provenance enforcement without running DepthEstimator or touching real caches.

    Covers:
      1. Missing manifest with full cache AND with 1 missing NPZ -> must raise RuntimeError.
      2. Stale/mismatched/corrupt manifest with full cache AND with 1 missing NPZ -> must raise RuntimeError.
      3. Partial cache (1 NPZ missing, remaining NPZs exist) + VALID manifest -> allows partial_resume
         of only the missing file while keeping verified NPZs reusable.
      4. Empty directory (fresh_build) and force_rebuild=True bypass.
    """
    dummy_targets = ["img_a.jpg", "img_b.jpg", "img_c.jpg"]
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "num_scenes": 3,
        "target_files": dummy_targets,
        "verified_mapping_sha256": "abc123",
        "camera_calib_sha256": "calib123",
    }

    def _write_dummy_npzs(cdir: Path, names: list):
        for fn in names:
            np.savez(cdir / f"scene_{Path(fn).stem}.npz", stub=np.array([1], dtype=np.int32))

    def _assert_raises_runtime(fn_call, label: str):
        try:
            fn_call()
        except RuntimeError:
            return
        raise AssertionError(f"Smoke test FAILED ({label}): expected RuntimeError, got none!")

    # Case 1a: Full cache, missing manifest
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        _write_dummy_npzs(cdir, dummy_targets)
        _assert_raises_runtime(
            lambda: verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False),
            "1a_full_cache_missing_manifest",
        )

    # Case 1b: Partial cache (1 NPZ missing, 2 NPZs exist), missing manifest
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        _write_dummy_npzs(cdir, dummy_targets[:-1])  # img_c missing
        _assert_raises_runtime(
            lambda: verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False),
            "1b_partial_cache_missing_manifest",
        )

    # Case 2a: Full cache, mismatched manifest (camera_calib_sha256 changed)
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        _write_dummy_npzs(cdir, dummy_targets)
        bad_manifest = dict(expected, camera_calib_sha256="OLD_STALE_CALIB_HASH")
        mpath.write_text(json.dumps(bad_manifest), encoding="utf-8")
        _assert_raises_runtime(
            lambda: verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False),
            "2a_full_cache_wrong_manifest",
        )

    # Case 2b: Partial cache (1 NPZ missing, 2 NPZs exist), mismatched manifest
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        _write_dummy_npzs(cdir, dummy_targets[1:])  # img_a missing
        bad_manifest = dict(expected, verified_mapping_sha256="OLD_STALE_GT_HASH")
        mpath.write_text(json.dumps(bad_manifest), encoding="utf-8")
        _assert_raises_runtime(
            lambda: verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False),
            "2b_partial_cache_wrong_manifest",
        )

    # Case 2c: Partial cache with corrupt JSON manifest
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        _write_dummy_npzs(cdir, dummy_targets[:1])
        mpath.write_text("{corrupt_json", encoding="utf-8")
        _assert_raises_runtime(
            lambda: verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False),
            "2c_partial_cache_corrupt_manifest",
        )

    # Case 2d: Full cache with corrupt JSON manifest
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        _write_dummy_npzs(cdir, dummy_targets)
        mpath.write_text("{corrupt_json", encoding="utf-8")
        _assert_raises_runtime(
            lambda: verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False),
            "2d_full_cache_corrupt_manifest",
        )

    # Case 3: Partial cache (1 NPZ missing, 2 NPZs exist) + VALID matching manifest
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        _write_dummy_npzs(cdir, ["img_a.jpg", "img_c.jpg"])  # img_b missing
        mpath.write_text(json.dumps(expected), encoding="utf-8")
        res_partial = verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False)
        assert res_partial["mode"] == "partial_resume", res_partial
        assert res_partial["existing_count"] == 2, res_partial
        assert res_partial["missing_files"] == ["img_b.jpg"], res_partial

        # Verify force_rebuild=True still allows full rebuild even if manifest differs
        mpath.write_text(json.dumps(dict(expected, schema_version="old")), encoding="utf-8")
        res_force = verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, True)
        assert res_force["mode"] == "force_rebuild", res_force
        assert res_force["missing_files"] == dummy_targets, res_force

    # Case 4: Fresh empty directory (0 NPZs)
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        mpath = cdir / "cache_manifest.json"
        res_fresh = verify_cache_manifest_state(cdir, mpath, dummy_targets, expected, False)
        assert res_fresh["mode"] == "fresh_build", res_fresh
        assert res_fresh["missing_files"] == dummy_targets, res_fresh

    print(
        "[Manifest Smoke Test PASS] Verified all cases: missing manifest (full/partial), "
        "wrong/corrupt manifest (full/partial), valid partial resume, force_rebuild, fresh_build."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/verify ArUco ablation scene cache.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding all 77 scene .npz files and cache_manifest.json.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run offline manifest guard smoke tests in a temporary directory.",
    )
    args = parser.parse_args()
    if args.smoke_test:
        run_manifest_guard_smoke_tests()
    else:
        build_or_verify_aruco_ablation_cache(force_rebuild=args.force_rebuild)
