import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask
from volume_nutrition import VolumeNutritionEstimator, fit_plane_to_boundary, relief_scale_factor

COCO_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
DEPTH_DIR = ROOT / "scratch" / "bench_cache_v2" / "depth_ctx"

mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
ann_by_id = {a["id"]: a for a in coco["annotations"]}
del coco

inst_by_file = defaultdict(list)
for x in mp["instances"]:
    inst_by_file[x["file_name"]].append(x)

vol_est = VolumeNutritionEstimator()

TARGET_CLASSES = {
    "DauHu", "BanhTrang", "KhoQuaThit", "BanhXeo", "Muc", "ChaGio",
    "Banh mi", "ThitHeo", "ThitNuong", "ComChienDuongChau", "Ca",
    "BunCha", "BanhKhot", "CaChua", "Thit bo", "MiQuang", "BoKho",
    "NuiXaoBo", "Rau", "BongCai"
}

rows_by_cls = defaultdict(list)

for fn in sorted(inst_by_file.keys()):
    insts = inst_by_file[fn]
    if not any(x["short_class"] in TARGET_CLASSES for x in insts):
        continue
    stem = Path(fn).stem
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

    for inst in insts:
        cls = inst["short_class"]
        if cls not in TARGET_CLASSES or inst["gt_weight_g"] is None:
            continue
        gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
        gt_vol = cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if gt_full.shape[:2] != (h_vol, w_vol) else gt_full
        bx1, by1, bx2, by2 = inst["bbox_xyxy"]
        b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
        est = vol_est.estimate(
            depth_map_mm=depth_mm,
            food_mask=gt_vol,
            class_name=inst["vietfood67_class_name"],
            mm_per_pixel=mm_px,
            scale_source=scale_src,
            bbox=b_vol,
            K=K,
            depth_anchored=d_anc,
        )
        area_cm2 = float(gt_vol.sum() * (mm_px ** 2) / 100.0)
        d_maj_mm = float(max(b_vol[2] - b_vol[0], b_vol[3] - b_vol[1]) * mm_px)
        implied_h_mm = (est.volume_cm3 / max(0.1, area_cm2)) * 10.0
        rows_by_cls[cls].append({
            "fn": fn,
            "status": inst["mapping_status"],
            "dtype": inst["dish_type"],
            "v_method": est.volume_method,
            "scale_src": scale_src,
            "area_cm2": round(area_cm2, 1),
            "d_maj": round(d_maj_mm, 1),
            "imp_h": round(implied_h_mm, 1),
            "rho": round(est.density_g_per_cm3, 3),
            "pred_g": est.mass_g,
            "gt_g": inst["gt_weight_g"],
        })

for cls in sorted(rows_by_cls.keys()):
    rs = rows_by_cls[cls]
    print(f"\n=== {cls} (n={len(rs)}, dtype={rs[0]['dtype']}, method={rs[0]['v_method']}, rho={rs[0]['rho']}) ===")
    print(f"  Mean area={st.mean(r['area_cm2'] for r in rs):.1f} cm2 | d_maj={st.mean(r['d_maj'] for r in rs):.1f} mm | imp_h={st.mean(r['imp_h'] for r in rs):.1f} mm | pred={st.mean(r['pred_g'] for r in rs):.1f}g vs gt={st.mean(r['gt_g'] for r in rs):.1f}g")
    for r in rs[:5]:
        print(f"    {r['fn']:32s} | {r['status'][:6]} | {r['scale_src'][:5]} | area={r['area_cm2']:5.1f} | d={r['d_maj']:5.1f} | h={r['imp_h']:4.1f} | pred={r['pred_g']:6.1f} | gt={r['gt_g']:6.1f}")
