"""Compare C6 perspective scaling variants: raw Z_food/Z_m vs alpha-damped Z_food vs alpha-damped Z_plane."""
import gc, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask
from volume_nutrition import VolumeNutritionEstimator, fit_plane_to_boundary
from scratch.run_ablation_v2_c1_c6 import eval_instance_c

COCO_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = ROOT / "scratch" / "bench_cache_v2"
DEPTH_DIR = CACHE_DIR / "depth_ctx"
Z_MARKER_PATH = CACHE_DIR / "z_marker_cache.json"

mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
ann_by_id = {a["id"]: a for a in coco["annotations"]}
del coco

z_marker_map = json.loads(Z_MARKER_PATH.read_text(encoding="utf-8"))
inst_by_file = defaultdict(list)
for x in mp["instances"]:
    inst_by_file[x["file_name"]].append(x)
target_files = sorted(inst_by_file.keys())
vol_est = VolumeNutritionEstimator()

variants = ["C1..C5 (no C6)", "C6_raw_Zfood", "C6_alpha_Zfood", "C6_raw_Zplane", "C6_alpha_Zplane"]
recs = defaultdict(list)

for fn in target_files:
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
    scene_cnames = [x["vietfood67_class_name"] for x in insts]
    gt_masks_v, b_vols, z_foods, z_planes = [], [], [], []
    for inst in insts:
        gf = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
        gv = cv2.resize(gf.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if gf.shape[:2] != (h_vol, w_vol) else gf
        bx1, by1, bx2, by2 = inst["bbox_xyxy"]
        gt_masks_v.append(gv)
        b_vols.append((int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy))))
        fd = depth_mm[gv]
        vfd = fd[fd > 0]
        z_foods.append(float(np.median(vfd)) if len(vfd) > 20 else None)
        if zm is not None and scale_src == "aruco_marker" and gv.any():
            pz = fit_plane_to_boundary(depth_mm, gv, alpha=0.79 if d_anc else 1.0)
            pd = pz[gv]
            vpd = pd[pd > 0]
            z_planes.append(float(np.median(vpd)) if len(vpd) > 20 else z_foods[-1])
        else:
            z_planes.append(None)

    for vname in variants:
        preds = []
        for i, inst in enumerate(insts):
            mm_eff = mm_px
            if vname != "C1..C5 (no C6)" and scale_src == "aruco_marker" and zm and zm > 0:
                zf = z_foods[i]
                zp = z_planes[i]
                alpha = 0.79 if d_anc else 0.85
                if vname == "C6_raw_Zfood" and zf:
                    ratio = float(np.clip(zf / zm, 0.72, 1.35))
                elif vname == "C6_alpha_Zfood" and zf:
                    ratio = float(np.clip(1.0 + alpha * (zf - zm) / zm, 0.72, 1.35))
                elif vname == "C6_raw_Zplane" and zp:
                    ratio = float(np.clip(zp / zm, 0.72, 1.35))
                elif vname == "C6_alpha_Zplane" and zp:
                    ratio = float(np.clip(1.0 + alpha * (zp - zm) / zm, 0.72, 1.35))
                else:
                    ratio = 1.0
                mm_eff = mm_px * ratio

            preds.append(eval_instance_c(
                vol_est, depth_mm, gt_masks_v[i], inst["vietfood67_class_name"], scene_cnames,
                mm_eff, scale_src, b_vols[i], K, d_anc, None,
                True, True, True, True, True, False
            ))

        grouped = defaultdict(list)
        for gi, inst in enumerate(insts):
            if inst["mapping_status"] == "exact_1to1":
                recs[vname].append((inst["short_class"], preds[gi], float(inst["gt_weight_g"])))
            elif inst["mapping_status"] == "grouped_multi_instance":
                grouped[inst["short_class"]].append(gi)
        for s_cls, gis in grouped.items():
            recs[vname].append((s_cls, sum(preds[i] for i in gis), float(insts[gis[0]]["gt_weight_g"])))
    del depth_mm

for vname in variants:
    rs = recs[vname]
    mae = st.mean(abs(p - g) for _, p, g in rs)
    mape = st.mean(abs(p - g) / g * 100.0 for _, p, g in rs)
    rmse = float(np.sqrt(st.mean((p - g) ** 2 for _, p, g in rs)))
    mpe = st.mean((p - g) / g * 100.0 for _, p, g in rs)
    bias = (sum(p for _, p, _ in rs) - sum(g for _, _, g in rs)) / sum(g for _, _, g in rs) * 100.0
    print(f"  {vname:18s} | MAE={mae:6.2f}g | MAPE={mape:5.2f}% | RMSE={rmse:6.2f}g | MPE={mpe:+6.2f}% | Bias={bias:+6.2f}%")
