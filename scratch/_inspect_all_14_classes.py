import sys
import json
import statistics as st
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

from scratch.run_volume_ablation import (
    build_or_load_depth_cache,
    VolumeNutritionEstimator,
    coco_poly_to_mask,
    run_oracle_ablation_config,
)

cached_scenes, ann_by_id = build_or_load_depth_cache()
vol_estimator = VolumeNutritionEstimator()

prepared_scenes = {}
for fn, sc in cached_scenes.items():
    h_vol, w_vol = sc["h_vol"], sc["w_vol"]
    sx, sy = sc["sx"], sc["sy"]
    insts = sc["insts"]
    h_full, w_full = int(insts[0]["height"]), int(insts[0]["width"])
    items = []
    for idx, inst in enumerate(insts):
        gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
        gt_vol = (
            cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
            if gt_full.shape[:2] != (h_vol, w_vol)
            else gt_full
        )
        bx1, by1, bx2, by2 = inst["bbox_xyxy"]
        b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
        ic = dict(inst)
        ic["gt_mask"] = gt_vol
        ic["bbox_vol"] = b_vol
        items.append(ic)
    prepared_scenes[fn] = {
        "depth_mm": sc["depth_mm"],
        "K": sc["K"],
        "mm_per_pixel": sc["mm_per_pixel"],
        "scale_source": sc["scale_source"],
        "depth_anchored": sc["depth_anchored"],
        "h_vol": h_vol,
        "w_vol": w_vol,
        "sx": sx,
        "sy": sy,
        "items": items,
    }

summ, recs = run_oracle_ablation_config("Baseline", False, False, False, False, False, prepared_scenes, vol_estimator, "gt_mask")
print(f"BASELINE OVERALL: MAE={summ['mae_g']:.2f}g, MAPE={summ['mape_pct']:.2f}%, RMSE={summ['rmse_g']:.2f}g, MPE={summ['mpe_pct']:+.2f}%, Bias={summ['bias_mean_pct']:+.2f}%")
print(f"{'Class':16s} | {'N':>2s} | {'GT_g':>6s} | {'Pred_g':>6s} | {'MAE_g':>6s} | {'MAPE%':>6s} | {'MPE%':>7s} | {'Bias%':>7s} | {'D_food':>6s} | {'mean_h':>6s}")
print("-" * 98)
for c, cs in sorted(summ["per_class"].items(), key=lambda kv: -kv[1]["mape_pct"]):
    print(f"{c:16s} | {cs['n_groups']:2d} | {cs['gt_mean_g']:6.1f} | {cs['pred_mean_g']:6.1f} | {cs['mae_g']:6.1f} | {cs['mape_pct']:6.1f} | {cs['mpe_pct']:+7.1f} | {cs['bias_mean_pct']:+7.1f} | {cs['D_food_mean']:6.1f} | {cs['mean_h']:6.1f}")

print("\n--- Check bbox & mask distances in ComGa2 and ThitKho1 ---")
for fn in ["ComGa2 (1).jpg", "ComGa2 (2).jpg", "ComGa2 (3).jpg", "ThitKho1 (2).jpg", "ThitKho1 (6).jpg"]:
    sc = prepared_scenes[fn]
    it0, it1 = sc["items"][0], sc["items"][1]
    print(f"{fn}: {it0['short_class']} bbox={it0['bbox_vol']} vs {it1['short_class']} bbox={it1['bbox_vol']}")

