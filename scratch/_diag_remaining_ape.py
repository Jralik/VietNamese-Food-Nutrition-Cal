import gc, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask
from volume_nutrition import VolumeNutritionEstimator

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
groups = []

for fn in sorted(inst_by_file.keys()):
    insts = inst_by_file[fn]
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

    ests = []
    areas = []
    dmajs = []
    for inst in insts:
        gf = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
        gv = cv2.resize(gf.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if gf.shape[:2] != (h_vol, w_vol) else gf
        bx1, by1, bx2, by2 = inst["bbox_xyxy"]
        b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
        e = vol_est.estimate(
            depth_map_mm=depth_mm, food_mask=gv, class_name=inst["vietfood67_class_name"],
            mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol, K=K, depth_anchored=d_anc,
        )
        ests.append(e)
        areas.append(float(gv.sum() * (mm_px ** 2) / 100.0))
        dmajs.append(float(max(b_vol[2] - b_vol[0], b_vol[3] - b_vol[1]) * mm_px))

    grouped = defaultdict(list)
    scene_classes = sorted({x["short_class"] for x in insts})
    for gi, inst in enumerate(insts):
        if inst["mapping_status"] == "exact_1to1":
            gt = float(inst["gt_weight_g"])
            pred = ests[gi].mass_g
            ape = abs(pred - gt) / gt * 100.0
            groups.append({
                "fn": fn, "cls": inst["short_class"], "v_method": ests[gi].volume_method,
                "scale_src": scale_src, "d_anc": d_anc, "mm_px": mm_px,
                "area": areas[gi], "dmaj": dmajs[gi], "rho": ests[gi].density_g_per_cm3,
                "pred": pred, "gt": gt, "ape": ape, "scene_classes": scene_classes,
            })
        elif inst["mapping_status"] == "grouped_multi_instance":
            grouped[inst["short_class"]].append(gi)
    for s_cls, gis in grouped.items():
        gt = float(insts[gis[0]]["gt_weight_g"])
        pred = sum(ests[i].mass_g for i in gis)
        ape = abs(pred - gt) / gt * 100.0
        groups.append({
            "fn": fn, "cls": s_cls, "v_method": ests[gis[0]].volume_method,
            "scale_src": scale_src, "d_anc": d_anc, "mm_px": mm_px,
            "area": sum(areas[i] for i in gis), "dmaj": st.mean(dmajs[i] for i in gis),
            "rho": ests[gis[0]].density_g_per_cm3,
            "pred": pred, "gt": gt, "ape": ape, "scene_classes": scene_classes,
        })
    del depth_mm

N = len(groups)
tot_mape = sum(g["ape"] for g in groups) / N
print(f"Total groups: {N} | Overall MAPE: {tot_mape:.2f}%")

by_cls = defaultdict(list)
for g in groups:
    by_cls[g["cls"]].append(g)

ranked = sorted(by_cls.items(), key=lambda kv: -sum(x["ape"] for x in kv[1]))
print("\n=== TOP 18 CONTRIBUTORS TO OVERALL MAPE (contribution in pct pts = sum_APE / 500) ===")
for c, items in ranked[:18]:
    contrib = sum(x["ape"] for x in items) / N
    c_mape = st.mean(x["ape"] for x in items)
    c_pred = st.mean(x["pred"] for x in items)
    c_gt = st.mean(x["gt"] for x in items)
    print(f"  {c:20s} | N={len(items):2d} | contrib={contrib:5.2f} pts | MAPE={c_mape:6.1f}% | pred={c_pred:5.1f} vs gt={c_gt:5.1f} | method={items[0]['v_method']}")

print("\n=== WORST 25 INDIVIDUAL GROUPS BY APE ===")
for g in sorted(groups, key=lambda x: -x["ape"])[:25]:
    print(f"  {g['fn']:32s} | {g['cls']:18s} | {g['scale_src'][:5]} | area={g['area']:5.1f} | d={g['dmaj']:5.1f} | pred={g['pred']:6.1f} vs gt={g['gt']:5.1f} | APE={g['ape']:6.1f}% | scene={g['scene_classes']}")
