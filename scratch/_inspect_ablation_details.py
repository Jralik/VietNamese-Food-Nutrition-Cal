import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask
from scratch._cache_depth_scenes import build_or_load_depth_cache
from volume_nutrition import VolumeNutritionEstimator, fit_plane_to_boundary, relief_scale_factor

cached_scenes, ann_by_id = build_or_load_depth_cache()
vol_est = VolumeNutritionEstimator()

# Let's inspect:
# 1. All soup_bowl / rice_bowl instances: D_food_mm, delta_z, baseline vol vs GT mass
# 2. All Trung instances: area_cm2, d_maj_mm, d_min_mm, solidity, mask_fill_ratio, GT mass
# 3. All multi-instance scenes (ComGa1, ComGa2, ThitKho1, BunBoHue1, com-suon1, BanhMi_Trung):
#    overlap / adjacency between masks, median depths, MAD
# 4. All Banh cuon, Banh mi, Com tam, Com chien ga, Xoi instances: raw_heights, pos_frac, mean_h, GT mass

print("=== 1. BOWL INSTANCES (Pho, Bun bo Hue, Bun rieu, Sup cua, Canh, Thit kho, Xoi) ===")
by_cls = defaultdict(list)
for fn, sc in sorted(cached_scenes.items()):
    h_vol, w_vol = sc["h_vol"], sc["w_vol"]
    sx, sy = sc["sx"], sc["sy"]
    mm_px = sc["mm_per_pixel"]
    for inst in sc["insts"]:
        cls = inst["short_class"]
        v_cls = inst["vietfood67_class_name"]
        dt = pc.CLASS_TO_DISH_TYPE.get(v_cls, "default")
        if dt in ("soup_bowl", "large_bowl", "rice_bowl"):
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            bw = (bx2 - bx1) * sx
            bh = (by2 - by1) * sy
            d_box_mm = max(bw, bh) * mm_px
            h_full, w_full = int(inst["height"]), int(inst["width"])
            gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
            gt_vol = cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
            fill_r = gt_vol.sum() / max(1.0, bw * bh)
            by_cls[cls].append((fn, d_box_mm, fill_r, inst["gt_weight_g"], sc["scale_source"]))

for cls, items in sorted(by_cls.items()):
    d_vals = [x[1] for x in items]
    gt_vals = [x[3] for x in items if x[3] is not None]
    f_vals = [x[2] for x in items]
    print(f"  {cls:15s} (n={len(items):2d}): D_box={np.mean(d_vals):5.1f}mm (min={np.min(d_vals):5.1f}, max={np.max(d_vals):5.1f}), bbox_fill={np.mean(f_vals):.3f}, GT={np.mean(gt_vals):5.1f}g")

print("\n=== 2. TRUNG INSTANCES ===")
for fn, sc in sorted(cached_scenes.items()):
    h_vol, w_vol = sc["h_vol"], sc["w_vol"]
    sx, sy = sc["sx"], sc["sy"]
    mm_px = sc["mm_per_pixel"]
    for inst in sc["insts"]:
        if inst["short_class"] == "Trung":
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            bw = (bx2 - bx1) * sx
            bh = (by2 - by1) * sy
            h_full, w_full = int(inst["height"]), int(inst["width"])
            gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
            gt_vol = cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
            area_cm2 = gt_vol.sum() * (mm_px ** 2) / 100.0
            d_maj = max(bw, bh) * mm_px
            d_min = min(bw, bh) * mm_px
            fill_r = gt_vol.sum() / max(1.0, bw * bh)
            d_mm = sc["depth_mm"]
            plane_z = fit_plane_to_boundary(d_mm, gt_vol)
            alpha = 0.79 if sc["depth_anchored"] else 1.0
            raw_h = (plane_z[gt_vol] - d_mm[gt_vol]) * alpha
            print(f"  {fn:22s}: area={area_cm2:5.1f}cm2, d_maj={d_maj:5.1f}mm, d_min={d_min:5.1f}mm, fill={fill_r:.3f}, raw_h_med={np.median(raw_h):+5.1f}mm, raw_h_p90={np.percentile(raw_h,90):+5.1f}mm, GT={inst['gt_weight_g']}g")

print("\n=== 3. MULTI-INSTANCE SPATIAL RELATIONSHIPS (ComGa, ThitKho) ===")
for fn, sc in sorted(cached_scenes.items()):
    if not (fn.startswith("ComGa") or fn.startswith("ThitKho")):
        continue
    h_vol, w_vol = sc["h_vol"], sc["w_vol"]
    sx, sy = sc["sx"], sc["sy"]
    mm_px = sc["mm_per_pixel"]
    d_mm = sc["depth_mm"]
    masks = []
    for inst in sc["insts"]:
        h_full, w_full = int(inst["height"]), int(inst["width"])
        gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
        gt_vol = cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append((inst["short_class"], gt_vol, inst["gt_weight_g"]))
    if len(masks) >= 2:
        c0, m0, g0 = masks[0]
        c1, m1, g1 = masks[1]
        inter = (m0 & m1).sum()
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # Ensure m_small is the smaller mask
        if m0.sum() < m1.sum():
            cs, ms, gs, cl, ml, gl = c0, m0, g0, c1, m1, g1
        else:
            cs, ms, gs, cl, ml, gl = c1, m1, g1, c0, m0, g0
        ring_s = (cv2.dilate(ms.astype(np.uint8)*255, k_out) > 0) & (cv2.dilate(ms.astype(np.uint8)*255, k_in) == 0)
        adj = ring_s & ml
        adj_frac = adj.sum() / max(1, ring_s.sum())
        zs = d_mm[ms]
        zl_adj = d_mm[adj] if adj.any() else d_mm[ml]
        zl_all = d_mm[ml]
        print(f"  {fn:16s}: small={cs}({ms.sum()}px, GT={gs}g) vs large={cl}({ml.sum()}px, GT={gl}g) | inter={inter}px, adj_frac={adj_frac:.3f}, med(Z_small)={np.median(zs):.1f}mm, med(Z_adj)={np.median(zl_adj):.1f}mm, med(Z_large)={np.median(zl_all):.1f}mm, MAD(Z_adj)={np.median(np.abs(zl_adj - np.median(zl_adj)))*1.4826:.1f}mm")

