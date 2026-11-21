"""
Fast Ablation of Physical/Geometric Improvements (B1-B5) on the 418-image / 500-group dataset.
Uses cached Depth Anything V2 + ArUco scale + MobileSAM Config C masks (streaming, low RAM).
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
    fit_plane_to_boundary,
    relief_scale_factor,
    SATURATION_CAP_RATIO,
)

COCO_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = ROOT / "scratch" / "bench_cache_v2"
DEPTH_DIR = CACHE_DIR / "depth_ctx"

SMALL_PLATE_CLASSES_B1 = {
    "Dau hu (Tofu)",
    "Muc (Squid)",
    "Ca (Fish)",
    "Thit heo (Pork)",
    "Thit nuong (Grilled meat)",
    "Thit bo (Beef)",
    "Thit ga (Chicken)",
    "Tom (Shrimp)",
    "Cha gio (Spring rolls)",
    "Long heo (Pork offal)",
    "Banh khot (Mini savory pancakes)",
    "Bo la lot (Grilled beef wrapped in betel leaves)",
    "Goi cuon (Fresh spring rolls)",
}

LOOSE_GRAIN_FLOOR_CLASSES = (
    "Com", "Banh cuon", "Nui", "Bun", "Banh beo", "Banh trang tron", "Xoi"
)


def estimate_with_flags(
    vol_est: VolumeNutritionEstimator,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    mm_px: float,
    scale_src: str,
    bbox: tuple,
    K: np.ndarray,
    depth_anchored: bool,
    b1: bool,
    b2: bool,
    b3: bool,
    b4: bool,
    b5: bool,
) -> float:
    dish_type = pc.CLASS_TO_DISH_TYPE.get(class_name, "default")
    if b1:
        if class_name == "Bun cha (Grilled pork with vermicelli)":
            dish_type = "soup_bowl"
        elif class_name in SMALL_PLATE_CLASSES_B1:
            dish_type = "small_plate"

    relief_scale = 0.79 if depth_anchored else None

    if scale_src == "plate_heuristic" and dish_type in pc.CONTAINER_DISH_TYPES and bbox is not None:
        exp_dia = pc.PLATE_DIAMETER_HEURISTICS.get(dish_type, 220.0)
        if b1 and dish_type == "small_plate":
            exp_dia = 130.0
        max_dim = max(10, bbox[2] - bbox[0], bbox[3] - bbox[1])
        if max_dim > 20:
            mm_px = exp_dia / float(max_dim)

    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if mask.any() and (mask.sum() / float(bbox_area) > 0.92):
        from volume_nutrition import compute_volume_bbox_approximation
        vol = float(np.clip(compute_volume_bbox_approximation(bbox, mm_px), 20.0, 1500.0))
        m_g, _, _ = vol_est._mass_from_volume(vol, class_name, [])
        return round(m_g, 1)

    if dish_type in {"soup_bowl", "large_bowl", "rice_bowl"}:
        vol, _ = compute_volume_bowl_container(
            depth_map_mm=depth_mm,
            food_mask=mask,
            bbox=bbox,
            mm_per_pixel=mm_px,
            dish_type=dish_type,
            class_name=class_name,
            K=K,
            scale_source=scale_src,
            relief_scale=relief_scale,
        )
    elif dish_type == "ingredient_small":
        if b3 and "Ca chua" in class_name and mask.any():
            x1, y1, x2, y2 = bbox
            w_mm = max(2.0, (x2 - x1) * mm_px)
            h_mm = max(2.0, (y2 - y1) * mm_px)
            d_maj = max(w_mm, h_mm)
            d_min = min(w_mm, h_mm)
            area_cm2 = float(mask.sum() * (mm_px ** 2) / 100.0)
            if area_cm2 > 22.0 or d_maj > 60.0:
                # Whole tomato / multiple tomatoes / wedges bowl: spheroid without 35 cm3 slice cap
                t_mm = d_min * 0.78
                vol_mm3 = (4.0 / 3.0) * np.pi * (w_mm / 2.0) * (h_mm / 2.0) * (t_mm / 2.0) * 0.85
                vol = float(np.clip(vol_mm3 / 1000.0, 20.0, 280.0))
            else:
                vol, _ = compute_volume_small_ingredient(
                    bbox=bbox, mm_per_pixel=mm_px, class_name=class_name,
                    depth_map_mm=depth_mm, food_mask=mask, K=K, relief_scale=relief_scale,
                )
        else:
            vol, _ = compute_volume_small_ingredient(
                bbox=bbox, mm_per_pixel=mm_px, class_name=class_name,
                depth_map_mm=depth_mm, food_mask=mask, K=K, relief_scale=relief_scale,
            )
    elif dish_type == "side_vegetables":
        if not b4:
            vol, _ = compute_volume_side_vegetables(
                depth_map_mm=depth_mm, food_mask=mask, bbox=bbox,
                mm_per_pixel=mm_px, K=K, relief_scale=relief_scale,
            )
        else:
            mask_bool = mask.astype(bool)
            area_mm2 = float(mask_bool.sum()) * (mm_px ** 2)
            mean_h_mm = 20.0
            if mask_bool.any():
                plane_z = fit_plane_to_boundary(depth_mm, mask, alpha=relief_scale or 0.79)
                plane_d = plane_z[mask_bool]
                z_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
                alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_med, mm_px, K)
                raw_h = (plane_d - depth_mm[mask_bool]) * alpha
                # B4: recessed cavity shift for vegetables in bowl/plate (e.g. BongCai)
                med_h = float(np.median(raw_h))
                p85_h = float(np.percentile(raw_h, 85))
                if med_h <= 4.0 and p85_h <= 16.0:
                    shift = min(max(0.0, float(np.percentile(-raw_h, 85))), 18.0) + 2.0 * pc.FOOD_HEIGHT_OFFSET_MM
                    raw_h = raw_h + shift
                pos_h = np.clip(raw_h, 0.0, 55.0)
                if pos_h.size > 50 and (raw_h > 0).sum() > 50:
                    mean_h_mm = float(pos_h.mean())
            mean_h_mm = float(np.clip(mean_h_mm, 10.0, 55.0))
            vol = float(np.clip((area_mm2 * mean_h_mm) / 1000.0, 20.0, 900.0))
    else:
        # Column integration
        is_loose_grain = any(k in class_name for k in LOOSE_GRAIN_FLOOR_CLASSES)
        is_banh_trang = class_name.startswith("Banh trang (")
        is_banh_xeo = class_name.startswith("Banh xeo")

        if b2:
            if is_banh_trang:
                max_h = 8.0
                min_h = 0.0
                mean_cap = 5.0
            elif is_banh_xeo:
                max_h = 16.0
                min_h = 0.0
                mean_cap = 8.5
            elif "Banh mi" in class_name or "Hamburger" in class_name:
                max_h = 55.0
                min_h = 0.0
                mean_cap = SATURATION_CAP_RATIO * max_h
            elif dish_type == "small_plate":
                max_h = 24.0
                min_h = 0.0
                mean_cap = SATURATION_CAP_RATIO * max_h
            else:
                max_h = 45.0
                min_h = 15.0 if is_loose_grain else 0.0
                mean_cap = SATURATION_CAP_RATIO * max_h
        else:
            height_floors = {"flat_plate": 15.0, "small_plate": 12.0}
            if "Banh mi" in class_name or "Hamburger" in class_name:
                max_h = 55.0
            elif "Thit ga" in class_name:
                max_h = 28.0
                height_floors[dish_type] = 10.0
            else:
                max_h = 35.0 if dish_type == "small_plate" else 45.0
            min_h = height_floors.get(dish_type, 0.0)
            mean_cap = SATURATION_CAP_RATIO * max_h

        if not b5:
            vol = compute_volume_column_integration(
                depth_map_mm=depth_mm,
                food_mask=mask,
                mm_per_pixel=mm_px,
                K=K,
                scale_source=scale_src,
                max_height_mm=max_h,
                relief_scale=relief_scale,
                min_mean_height_mm=min_h,
                mean_height_cap=mean_cap,
            )
        else:
            # B5: Recessed-cavity shift ONLY for concave/loose-grain plates, NOT flat discrete items
            mask_bool = mask.astype(bool)
            masked_d = depth_mm[mask_bool]
            valid_d = masked_d[masked_d > 0]
            if len(valid_d) == 0:
                vol = 0.0
            else:
                plane_z = fit_plane_to_boundary(depth_mm, mask, alpha=relief_scale or 0.79)
                plane_d = plane_z[mask_bool]
                z_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
                alpha = relief_scale if relief_scale is not None else relief_scale_factor(z_med, mm_px, K)
                raw_h = (plane_d - masked_d) * alpha
                area_cm2 = float(mask_bool.sum() * (mm_px ** 2) / 100.0) if mm_px > 0 else 150.0
                if is_loose_grain and area_cm2 <= 225.0:
                    med_h = float(np.median(raw_h))
                    p85_h = float(np.percentile(raw_h, 85))
                    neg_flat = float((raw_h <= 2.5).mean())
                    if (med_h <= 4.0 and p85_h <= 16.0) or (med_h <= 11.5 and p85_h <= 27.0 and neg_flat >= 0.22):
                        shift = min(max(0.0, float(np.percentile(-raw_h, 85))), 18.0) + 2.0 * pc.FOOD_HEIGHT_OFFSET_MM
                        raw_h = raw_h + shift
                food_h = np.clip(raw_h, 2.0, max_h)
                if min_h > 0 and food_h.size > 0:
                    mh = float(food_h.mean())
                    if 0 < mh < min_h:
                        food_h = np.clip(food_h * (min_h / mh), 2.0, max_h)
                if mean_cap is not None and food_h.size > 0:
                    sat_f = float((food_h >= max_h - 1.5).mean())
                    hi_f = float((food_h >= 0.78 * max_h).mean())
                    if sat_f > 0.35 or hi_f > 0.65:
                        mh = float(food_h.mean())
                        if mh > mean_cap > 0:
                            food_h = np.clip(food_h * (mean_cap / mh), 2.0, max_h)
                med_z = float(np.median(valid_d))
                px_area = (mm_px * (masked_d / med_z)) ** 2 if (mm_px > 0 and med_z > 0) else mm_px ** 2
                vol = float(np.sum(food_h * px_area) / 1000.0)

    vol = float(np.clip(vol, 0.5, 2500.0))
    m_g, _, _ = vol_est._mass_from_volume(vol, class_name, [])
    return round(m_g, 1)


def main():
    mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
    coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    del coco
    gc.collect()

    inst_by_file = defaultdict(list)
    for x in mp["instances"]:
        inst_by_file[x["file_name"]].append(x)
    target_files = sorted(inst_by_file.keys())
    vol_est = VolumeNutritionEstimator()

    configs = [
        ("Current (A1-A5)", False, False, False, False, False),
        ("+ B1 (Dish routing)", True, False, False, False, False),
        ("+ B1+B2 (Height floors/thin)", True, True, False, False, False),
        ("+ B1..B3 (CaChua whole)", True, True, True, False, False),
        ("+ B1..B4 (Veg cap+recess)", True, True, True, True, False),
        ("+ B1..B5 (Recess guard)", True, True, True, True, True),
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

        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        oracle_c = ms_npz["oracle_cand_raw"].astype(bool)
        ms_npz.close()

        gt_masks_v = []
        ms_masks_v = []
        b_vols = []
        for i, inst in enumerate(insts):
            gf = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
            gv = cv2.resize(gf.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if gf.shape[:2] != (h_vol, w_vol) else gf
            mv = cv2.resize(oracle_c[i].astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if oracle_c[i].shape[:2] != (h_vol, w_vol) else oracle_c[i]
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            gt_masks_v.append(gv)
            ms_masks_v.append(mv)
            b_vols.append((int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy))))

        for cname, b1, b2, b3, b4, b5 in configs:
            g_pred = [
                estimate_with_flags(vol_est, depth_mm, gt_masks_v[i], insts[i]["vietfood67_class_name"], mm_px, scale_src, b_vols[i], K, d_anc, b1, b2, b3, b4, b5)
                for i in range(len(insts))
            ]
            # Only compute MobileSAM C on Baseline and Full B1..B5 for speed
            m_pred = (
                [
                    estimate_with_flags(vol_est, depth_mm, ms_masks_v[i], insts[i]["vietfood67_class_name"], mm_px, scale_src, b_vols[i], K, d_anc, b1, b2, b3, b4, b5)
                    for i in range(len(insts))
                ]
                if cname in ("Current (A1-A5)", "+ B1..B5 (Recess guard)")
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

    print("=== PROGRESSIVE ABLATION B1..B5 ON 418 IMAGES / 500 MASS GROUPS (ORACLE GT MASK) ===")
    for cname, *_ in configs:
        rs = recs_gt[cname]
        mae = st.mean(abs(p - g) for _, p, g in rs)
        mape = st.mean(abs(p - g) / g * 100.0 for _, p, g in rs)
        rmse = float(np.sqrt(st.mean((p - g) ** 2 for _, p, g in rs)))
        mpe = st.mean((p - g) / g * 100.0 for _, p, g in rs)
        bias = (sum(p for _, p, _ in rs) - sum(g for _, _, g in rs)) / sum(g for _, _, g in rs) * 100.0
        print(f"  {cname:30s} | MAE={mae:6.2f}g | MAPE={mape:5.2f}% | RMSE={rmse:6.2f}g | MPE={mpe:+6.2f}% | Bias={bias:+6.2f}%")

    print("\n=== ORACLE MOBILESAM CONFIG C (500 MASS GROUPS) ===")
    for cname in ("Current (A1-A5)", "+ B1..B5 (Recess guard)"):
        rs = recs_ms[cname]
        mae = st.mean(abs(p - g) for _, p, g in rs)
        mape = st.mean(abs(p - g) / g * 100.0 for _, p, g in rs)
        mpe = st.mean((p - g) / g * 100.0 for _, p, g in rs)
        print(f"  {cname:30s} | MAE={mae:6.2f}g | MAPE={mape:5.2f}% | MPE={mpe:+6.2f}%")

    # Print classes that changed most
    base_cls = defaultdict(list)
    new_cls = defaultdict(list)
    for c, p, g in recs_gt["Current (A1-A5)"]:
        base_cls[c].append((p, g))
    for c, p, g in recs_gt["+ B1..B5 (Recess guard)"]:
        new_cls[c].append((p, g))
    print("\n=== PER-CLASS CHANGES (Base -> B1..B5) ===")
    for c in sorted(base_cls.keys()):
        b_mape = st.mean(abs(p - g) / g * 100.0 for p, g in base_cls[c])
        n_mape = st.mean(abs(p - g) / g * 100.0 for p, g in new_cls[c])
        if abs(n_mape - b_mape) >= 1.0:
            b_mae = st.mean(abs(p - g) for p, g in base_cls[c])
            n_mae = st.mean(abs(p - g) for p, g in new_cls[c])
            print(f"  {c:20s} (N={len(base_cls[c]):2d}): MAE {b_mae:5.1f} -> {n_mae:5.1f}g | MAPE {b_mape:5.1f}% -> {n_mape:5.1f}%")


if __name__ == "__main__":
    main()
