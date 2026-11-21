"""
Progressive + Independent Ablation of Physical/Geometric Improvements (C1-C6)
on the 418-image / 500-group dataset (Oracle GT Mask + Oracle MobileSAM Config C).
"""
import gc, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask
from volume_nutrition import (
    VolumeNutritionEstimator,
    compute_volume_bowl_container,
    compute_volume_column_integration,
    compute_volume_side_vegetables,
    compute_volume_small_ingredient,
    compute_volume_bbox_approximation,
    fit_plane_to_boundary,
    relief_scale_factor,
    SATURATION_CAP_RATIO,
)

COCO_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = ROOT / "scratch" / "bench_cache_v2"
DEPTH_DIR = CACHE_DIR / "depth_ctx"
Z_MARKER_PATH = CACHE_DIR / "z_marker_cache.json"

BANH_MI_DIP_PARTNERS = ("Bo kho", "Trung", "Ca ri", "Pate", "Xiu mai")
THIN_SLICED_MEATS = ("Thit nuong", "Thit heo", "Thit bo", "Ca (Fish)")
LOOSE_GRAIN_KEYS = ("Com", "Banh cuon", "Nui", "Bun", "Banh beo", "Banh trang tron", "Xoi")


def eval_instance_c(
    vol_est: VolumeNutritionEstimator,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    scene_class_names: list,
    mm_px: float,
    scale_src: str,
    bbox: tuple,
    K: np.ndarray,
    depth_anchored: bool,
    z_marker_depth_mm: float,
    c1: bool,
    c2: bool,
    c3: bool,
    c4: bool,
    c5: bool,
    c6: bool,
) -> float:
    dish_type = pc.CLASS_TO_DISH_TYPE.get(class_name, "default")
    if c2 and "Kho qua thit" in class_name:
        dish_type = "small_plate"

    relief_scale = 0.79 if depth_anchored else None

    # C6: Perspective depth-ratio scale adjustment (Z_food / Z_marker)
    mm_px_eff = mm_px
    if c6 and scale_src == "aruco_marker" and z_marker_depth_mm is not None and z_marker_depth_mm > 0 and mask.any():
        fd = depth_mm[mask.astype(bool)]
        vfd = fd[fd > 0]
        if len(vfd) > 20:
            z_food = float(np.median(vfd))
            ratio = float(np.clip(z_food / z_marker_depth_mm, 0.72, 1.35))
            mm_px_eff = mm_px * ratio

    if scale_src == "plate_heuristic" and dish_type in pc.CONTAINER_DISH_TYPES and bbox is not None:
        exp_dia = pc.PLATE_DIAMETER_HEURISTICS.get(dish_type, 220.0)
        max_dim = max(10, bbox[2] - bbox[0], bbox[3] - bbox[1])
        if max_dim > 20:
            mm_px_eff = exp_dia / float(max_dim)

    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if mask.any() and (mask.sum() / float(bbox_area) > 0.92):
        vol = float(np.clip(compute_volume_bbox_approximation(bbox, mm_px_eff), 20.0, 1500.0))
    elif dish_type in {"soup_bowl", "large_bowl", "rice_bowl"}:
        vol, _ = compute_volume_bowl_container(
            depth_map_mm=depth_mm, food_mask=mask, bbox=bbox,
            mm_per_pixel=mm_px_eff, dish_type=dish_type, class_name=class_name,
            K=K, scale_source=scale_src, relief_scale=relief_scale,
        )
    elif dish_type == "ingredient_small":
        area_cm2 = float(mask.sum() * (mm_px_eff ** 2) / 100.0) if mask.any() else 0.0
        if c3 and "Ca chua" in class_name and area_cm2 > 20.0:
            vol = compute_volume_column_integration(
                depth_map_mm=depth_mm, food_mask=mask, mm_per_pixel=mm_px_eff,
                K=K, scale_source=scale_src, max_height_mm=55.0,
                relief_scale=relief_scale, min_mean_height_mm=0.0,
                mean_height_cap=SATURATION_CAP_RATIO * 55.0,
            )
        else:
            vol, _ = compute_volume_small_ingredient(
                bbox=bbox, mm_per_pixel=mm_px_eff, class_name=class_name,
                depth_map_mm=depth_mm, food_mask=mask, K=K, relief_scale=relief_scale,
            )
    elif dish_type == "side_vegetables":
        if not c4:
            vol, _ = compute_volume_side_vegetables(
                depth_map_mm=depth_mm, food_mask=mask, bbox=bbox,
                mm_per_pixel=mm_px_eff, K=K, relief_scale=relief_scale,
            )
        else:
            mask_bool = mask.astype(bool)
            area_mm2 = float(mask_bool.sum()) * (mm_px_eff ** 2)
            mean_h_mm = 22.0
            if mask_bool.any():
                plane_z = fit_plane_to_boundary(depth_mm, mask, alpha=relief_scale or 0.79)
                plane_d = plane_z[mask_bool]
                z_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
                alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_med, mm_px_eff, K)
                raw_h = (plane_d - depth_mm[mask_bool]) * alpha
                # C4: require >=35% positive pixels before overriding default mound height (fixes recessed herb bowls!)
                if raw_h.size > 50 and float((raw_h > 0).mean()) > 0.35:
                    pos_h = np.clip(raw_h, 0.0, 45.0)
                    mean_h_mm = float(pos_h.mean())
            # Cap fluffy leafy mound effective solid height at 28 mm (air gaps between leaves)
            mean_h_mm = float(np.clip(mean_h_mm, 14.0, 28.0))
            vol = float(np.clip((area_mm2 * mean_h_mm) / 1000.0, 20.0, 450.0))
    else:
        is_loose_grain = any(k in class_name for k in LOOSE_GRAIN_KEYS)
        is_banh_trang = class_name.startswith("Banh trang (")
        is_banh_xeo = class_name.startswith("Banh xeo")
        is_thin_meat = any(class_name.startswith(k) for k in THIN_SLICED_MEATS)

        min_mean_h = 0.0
        if is_banh_trang:
            max_h = 6.0 if c5 else 8.0
            mean_cap = 4.0 if c5 else 5.0
        elif is_banh_xeo:
            max_h = 16.0
            mean_cap = 8.5
        elif "Banh mi" in class_name or "Hamburger" in class_name:
            max_h = 55.0
            mean_cap = SATURATION_CAP_RATIO * max_h
        elif "Kho qua thit" in class_name and c2:
            max_h = 35.0
            mean_cap = SATURATION_CAP_RATIO * max_h
        elif dish_type == "small_plate":
            max_h = 24.0
            mean_cap = SATURATION_CAP_RATIO * max_h
        else:
            max_h = 45.0
            min_mean_h = 15.0 if is_loose_grain else 0.0
            mean_cap = SATURATION_CAP_RATIO * max_h

        if not c5 or (not is_banh_trang and not is_thin_meat):
            vol = compute_volume_column_integration(
                depth_map_mm=depth_mm, food_mask=mask, mm_per_pixel=mm_px_eff,
                K=K, scale_source=scale_src, max_height_mm=max_h,
                relief_scale=relief_scale, min_mean_height_mm=min_mean_h,
                mean_height_cap=mean_cap,
            )
        else:
            mask_bool = mask.astype(bool)
            masked_d = depth_mm[mask_bool]
            valid_d = masked_d[masked_d > 0]
            if len(valid_d) == 0:
                vol = 0.0
            else:
                plane_z = fit_plane_to_boundary(depth_mm, mask, alpha=relief_scale or 0.79)
                plane_d = plane_z[mask_bool]
                z_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
                alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_med, mm_px_eff, K)
                raw_h = (plane_d - masked_d) * alpha
                if is_banh_trang:
                    # Rice paper sheet never sits in a deep bowl cavity; use actual thin sheet height [1.5, 6.0] mm
                    food_h = np.clip(raw_h, 2.0, max_h)
                else:
                    # Thin sliced meat: gentler recessed offset (+1 * FOOD_HEIGHT_OFFSET_MM = 5mm instead of 10mm)
                    med_h = float(np.median(raw_h))
                    p85_h = float(np.percentile(raw_h, 85))
                    neg_flat = float((raw_h <= 2.5).mean())
                    area_cm2 = float(mask_bool.sum() * (mm_px_eff ** 2) / 100.0) if mm_px_eff > 0 else 150.0
                    if area_cm2 <= 225.0 and ((med_h <= 4.0 and p85_h <= 16.0) or (med_h <= 11.5 and p85_h <= 27.0 and neg_flat >= 0.22)):
                        shift = min(max(0.0, float(np.percentile(-raw_h, 85))), 10.0) + 1.0 * pc.FOOD_HEIGHT_OFFSET_MM
                        food_h = np.clip(raw_h + shift, 2.0, max_h)
                    else:
                        food_h = np.clip(raw_h, 2.0, max_h)
                if mean_cap is not None and food_h.size > 0:
                    sat_f = float((food_h >= max_h - 1.5).mean())
                    hi_f = float((food_h >= 0.78 * max_h).mean())
                    if sat_f > 0.35 or hi_f > 0.65:
                        mh = float(food_h.mean())
                        if mh > mean_cap > 0:
                            food_h = np.clip(food_h * (mean_cap / mh), 2.0, max_h)
                med_z = float(np.median(valid_d))
                px_area = (mm_px_eff * (masked_d / med_z)) ** 2 if (mm_px_eff > 0 and med_z > 0) else mm_px_eff ** 2
                vol = float(np.sum(food_h * px_area) / 1000.0)

    vol = float(np.clip(vol, 0.5, 2500.0))
    m_g, _, density_mean = vol_est._mass_from_volume(vol, class_name, [])

    # C1: Co-occurrence physical density for plain dipping baguette (bánh mì không)
    if c1 and "Banh mi" in class_name:
        has_dip_partner = any(
            any(k in other for k in BANH_MI_DIP_PARTNERS)
            for other in scene_class_names
            if other != class_name
        )
        if has_dip_partner and density_mean > 0:
            # Plain empty Vietnamese baguette density = 0.20 g/cm3 (vs 0.45 g/cm3 stuffed sandwich)
            m_g = vol * 0.20

    return round(m_g, 1)


def main():
    mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
    coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    del coco
    gc.collect()

    z_marker_map = json.loads(Z_MARKER_PATH.read_text(encoding="utf-8"))
    inst_by_file = defaultdict(list)
    for x in mp["instances"]:
        inst_by_file[x["file_name"]].append(x)
    target_files = sorted(inst_by_file.keys())
    vol_est = VolumeNutritionEstimator()

    configs = [
        ("Current (A1-A5+B1-B2)", False, False, False, False, False, False),
        ("+ C1 (Plain Banh mi)", True, False, False, False, False, False),
        ("+ C1+C2 (KhoQuaThit)", True, True, False, False, False, False),
        ("+ C1..C3 (CaChua col)", True, True, True, False, False, False),
        ("+ C1..C4 (Rau fix)", True, True, True, True, False, False),
        ("+ C1..C5 (Thin/Sheet)", True, True, True, True, True, False),
        ("+ C1..C6 (+Z_food/Z_m)", True, True, True, True, True, True),
        ("Only C6 (Z_food/Z_m)", False, False, False, False, False, True),
    ]

    recs_gt = defaultdict(list)
    recs_ms = defaultdict(list)

    for idx, fn in enumerate(target_files):
        stem = Path(fn).stem
        insts = inst_by_file[fn]
        h_full, w_full = int(insts[0]["height"]), int(insts[0]["width"])

        sc = np.load(DEPTH_DIR / f"scene_{stem}.npz", allow_pickle=True)
        depth_mm = sc["depth_mm"].astype(np.float32)
        K = sc["K"]
        mm_px = float(sc["mm_per_pixel"])
        scale_src = str(sc["scale_source"])
        d_anc = bool(sc["depth_anchored"])
        h_vol, w_vol = int(sc["h_vol"]), int(sc["w_vol"])
        sx, sy = float(sc["sx"]), float(sc["sy"])
        sc.close()

        zm = z_marker_map.get(fn, {}).get("z_marker_depth_mm")
        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        oracle_c = ms_npz["oracle_cand_raw"].astype(bool)
        ms_npz.close()

        gt_masks_v, ms_masks_v, b_vols = [], [], []
        scene_cnames = [x["vietfood67_class_name"] for x in insts]
        for i, inst in enumerate(insts):
            gf = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
            gv = cv2.resize(gf.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if gf.shape[:2] != (h_vol, w_vol) else gf
            mv = cv2.resize(oracle_c[i].astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if oracle_c[i].shape[:2] != (h_vol, w_vol) else oracle_c[i]
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            gt_masks_v.append(gv)
            ms_masks_v.append(mv)
            b_vols.append((int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy))))

        for cname, c1, c2, c3, c4, c5, c6 in configs:
            g_pred = [
                eval_instance_c(vol_est, depth_mm, gt_masks_v[i], insts[i]["vietfood67_class_name"], scene_cnames, mm_px, scale_src, b_vols[i], K, d_anc, zm, c1, c2, c3, c4, c5, c6)
                for i in range(len(insts))
            ]
            m_pred = (
                [
                    eval_instance_c(vol_est, depth_mm, ms_masks_v[i], insts[i]["vietfood67_class_name"], scene_cnames, mm_px, scale_src, b_vols[i], K, d_anc, zm, c1, c2, c3, c4, c5, c6)
                    for i in range(len(insts))
                ]
                if cname in ("Current (A1-A5+B1-B2)", "+ C1..C5 (Thin/Sheet)", "+ C1..C6 (+Z_food/Z_m)")
                else None
            )

            grouped = defaultdict(list)
            for gi, inst in enumerate(insts):
                if inst["mapping_status"] == "exact_1to1":
                    gt_w = float(inst["gt_weight_g"])
                    recs_gt[cname].append((inst["short_class"], g_pred[gi], gt_w))
                    if m_pred is not None:
                        recs_ms[cname].append((inst["short_class"], m_pred[gi], gt_w))
                elif inst["mapping_status"] == "grouped_multi_instance":
                    grouped[inst["short_class"]].append(gi)
            for s_cls, gis in grouped.items():
                gt_w = float(insts[gis[0]]["gt_weight_g"])
                recs_gt[cname].append((s_cls, sum(g_pred[i] for i in gis), gt_w))
                if m_pred is not None:
                    recs_ms[cname].append((s_cls, sum(m_pred[i] for i in gis), gt_w))

        del depth_mm, oracle_c, gt_masks_v, ms_masks_v
        if (idx + 1) % 100 == 0:
            gc.collect()

    print("=== PROGRESSIVE ABLATION C1..C6 ON 418 IMAGES / 500 MASS GROUPS (ORACLE GT MASK) ===")
    for cname, *_ in configs:
        rs = recs_gt[cname]
        mae = st.mean(abs(p - g) for _, p, g in rs)
        mape = st.mean(abs(p - g) / g * 100.0 for _, p, g in rs)
        rmse = float(np.sqrt(st.mean((p - g) ** 2 for _, p, g in rs)))
        mpe = st.mean((p - g) / g * 100.0 for _, p, g in rs)
        bias = (sum(p for _, p, _ in rs) - sum(g for _, _, g in rs)) / sum(g for _, _, g in rs) * 100.0
        print(f"  {cname:26s} | MAE={mae:6.2f}g | MAPE={mape:5.2f}% | RMSE={rmse:6.2f}g | MPE={mpe:+6.2f}% | Bias={bias:+6.2f}%")

    print("\n=== ORACLE MOBILESAM CONFIG C (500 MASS GROUPS) ===")
    for cname in ("Current (A1-A5+B1-B2)", "+ C1..C5 (Thin/Sheet)", "+ C1..C6 (+Z_food/Z_m)"):
        rs = recs_ms[cname]
        mae = st.mean(abs(p - g) for _, p, g in rs)
        mape = st.mean(abs(p - g) / g * 100.0 for _, p, g in rs)
        mpe = st.mean((p - g) / g * 100.0 for _, p, g in rs)
        print(f"  {cname:26s} | MAE={mae:6.2f}g | MAPE={mape:5.2f}% | MPE={mpe:+6.2f}%")

    best_cfg = "+ C1..C6 (+Z_food/Z_m)" if st.mean(abs(p-g)/g for _,p,g in recs_gt["+ C1..C6 (+Z_food/Z_m)"]) < st.mean(abs(p-g)/g for _,p,g in recs_gt["+ C1..C5 (Thin/Sheet)"]) else "+ C1..C5 (Thin/Sheet)"
    print(f"\n=== PER-CLASS CHANGES: Current -> {best_cfg} ===")
    b_cls, n_cls = defaultdict(list), defaultdict(list)
    for c, p, g in recs_gt["Current (A1-A5+B1-B2)"]:
        b_cls[c].append((p, g))
    for c, p, g in recs_gt[best_cfg]:
        n_cls[c].append((p, g))
    for c in sorted(b_cls.keys()):
        bm = st.mean(abs(p - g) / g * 100.0 for p, g in b_cls[c])
        nm = st.mean(abs(p - g) / g * 100.0 for p, g in n_cls[c])
        if abs(nm - bm) >= 1.5:
            ba = st.mean(abs(p - g) for p, g in b_cls[c])
            na = st.mean(abs(p - g) for p, g in n_cls[c])
            print(f"  {c:20s} (N={len(b_cls[c]):2d}): MAE {ba:5.1f} -> {na:5.1f}g | MAPE {bm:5.1f}% -> {nm:5.1f}%")


if __name__ == "__main__":
    main()
