import json
import statistics as st
from collections import defaultdict
import sys

sys.path.insert(0, r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
import pipeline_config as pc

recs = [
    json.loads(l)
    for l in open(
        r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\scratch\full_seg_benchmark.jsonl",
        encoding="utf-8",
    )
    if l.strip()
]
mapping = json.load(
    open(
        r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\scratch\gt_mapping_verified.json",
        encoding="utf-8",
    )
)
inst_meta = {x["ann_id"]: x for x in mapping["instances"]}

by_cls = defaultdict(list)
for r in recs:
    sc = r["scale_source"]
    mmp = r["mm_per_pixel"]
    for inst in r["oracle_instances"]:
        if inst["mapping_status"] not in ("exact_1to1", "grouped_multi_instance"):
            continue
        meta = inst_meta[inst["ann_id"]]
        bx = meta["bbox_xyxy"]
        w_img, h_img = meta["width"], meta["height"]
        s = 2000.0 / max(w_img, h_img) if max(w_img, h_img) > 2000 else 1.0
        bw_mm = (bx[2] - bx[0]) * s * mmp
        bh_mm = (bx[3] - bx[1]) * s * mmp
        v_cls = meta["vietfood67_class_name"]
        dt = pc.CLASS_TO_DISH_TYPE.get(v_cls, "default")
        prior_d = pc.PLATE_DIAMETER_HEURISTICS.get(dt, 200.0)
        by_cls[inst["short_class"]].append({
            "fn": r["file_name"],
            "scale": sc,
            "mmp": mmp,
            "dish_type": dt,
            "prior_d": prior_d,
            "d_maj_mm": max(bw_mm, bh_mm),
            "d_min_mm": min(bw_mm, bh_mm),
            "gt_g": inst["gt_weight_g"],
            "pred_g": inst["gt_mask_mass_g"],
            "msam_g": inst["models"]["mobilesam"]["mass_cand_raw_g"],
        })

print(f"{'Class':<15} {'n':>3} {'DishType':<17} {'PriorD':>6} {'MeasD_maj':>9} {'MeasD_min':>9} {'ScaleSources':<28} {'GT(g)':>7} {'PredGT(g)':>9} {'Bias%':>8}")
print("-" * 125)
for c, items in sorted(by_cls.items()):
    scs = sorted(set(x["scale"] for x in items))
    dt = items[0]["dish_type"]
    pd = items[0]["prior_d"]
    d_med = st.median(x["d_maj_mm"] for x in items)
    d_min_med = st.median(x["d_min_mm"] for x in items)
    gt_m = st.mean(x["gt_g"] for x in items)
    pr_m = st.mean(x["pred_g"] for x in items)
    bias = st.mean((x["pred_g"] - x["gt_g"]) / x["gt_g"] * 100 for x in items)
    print(f"{c:<15} {len(items):>3} {dt:<17} {pd:>6.0f} {d_med:>9.1f} {d_min_med:>9.1f} {str(scs):<28} {gt_m:>7.1f} {pr_m:>9.1f} {bias:>+7.1f}%")
