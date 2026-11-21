import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from scratch.run_volume_ablation import (
    build_or_load_depth_cache,
    VolumeNutritionEstimator,
    coco_poly_to_mask,
    run_oracle_ablation_config,
    fit_plane_to_boundary,
    fit_plane_stabilized,
    relief_scale_factor,
)

cached_scenes, ann_by_id = build_or_load_depth_cache()
vol_estimator = VolumeNutritionEstimator()

print(f"{'Image':22s} | {'Class':14s} | {'GT_g':>6s} | {'Area':>6s} | {'D_mm':>5s} | {'pos%':>5s} | {'h_med':>6s} | {'h_p85':>6s} | {'h_stab_med':>10s} | {'delta_z':>7s}")
print("-" * 112)

for fn in sorted(cached_scenes.keys()):
    sc = cached_scenes[fn]
    h_vol, w_vol = sc["h_vol"], sc["w_vol"]
    sx, sy = sc["sx"], sc["sy"]
    mm_px = sc["mm_per_pixel"]
    d_mm = sc["depth_mm"]
    K = sc["K"]
    alpha = 0.79 if sc["depth_anchored"] else 1.0

    for inst in sc["insts"]:
        if inst["gt_weight_g"] is None:
            continue
        cls = inst["short_class"]
        v_cls = inst["vietfood67_class_name"]
        dt = pc.CLASS_TO_DISH_TYPE.get(v_cls, "default")
        h_full, w_full = int(inst["height"]), int(inst["width"])
        gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
        mask = cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
        bx1, by1, bx2, by2 = inst["bbox_xyxy"]
        bw, bh = (bx2 - bx1) * sx, (by2 - by1) * sy
        eff_mm = mm_px
        if sc["scale_source"] not in ("aruco_marker", "checkerboard") and dt in pc.CONTAINER_DISH_TYPES:
            eff_mm = pc.PLATE_DIAMETER_HEURISTICS.get(dt, 220.0) / max(10.0, max(bw, bh))
        d_food = max(bw, bh) * eff_mm
        area_cm2 = mask.sum() * (eff_mm ** 2) / 100.0

        p_base = fit_plane_to_boundary(d_mm, mask)
        p_stab = fit_plane_stabilized(d_mm, mask)
        md = d_mm[mask]
        raw_h = (p_base[mask] - md) * alpha
        stab_h = (p_stab[mask] - md) * alpha
        pos_pct = (raw_h > 2.0).mean() * 100.0
        dz = (np.median(md) - np.median(p_base[mask])) * alpha

        if cls in ("Thit ga", "Com chien ga", "Banh cuon", "Banh mi", "Sup cua", "Canh", "Thit kho", "Trung", "Xoi"):
            print(
                f"{fn:22s} | {cls:14s} | {inst['gt_weight_g']:6.1f} | {area_cm2:6.1f} | {d_food:5.1f} | "
                f"{pos_pct:5.1f} | {np.median(raw_h):+6.1f} | {np.percentile(raw_h, 85):+6.1f} | "
                f"{np.median(stab_h):+10.1f} | {dz:+7.1f}"
            )

