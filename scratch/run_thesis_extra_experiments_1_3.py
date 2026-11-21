"""
Thesis Extra Experiments 1, 2, 3 on the 418-image / 55-class dataset:
  - Exp 1: Calorie & Macronutrient Accuracy (Calories, Protein, Fat, Carbs)
           on 500 mass groups & 415 full meals (Oracle MobileSAM C vs End-to-End).
  - Exp 2: ArUco Calibration Ablation on 402 ArUco scenes
           (No ArUco Plate Heuristic vs ArUco w/o C6 vs ArUco + C6).
  - Exp 3: Multi-View Repeatability / Consistency (CV% across viewpoints of same dish).

Outputs:
  - results/thesis_extra_experiments_report.md
  - results/thesis_figures_v2/fig6_calorie_and_aruco_ablation_v2.png
  - results/thesis_figures_v2/fig7_multiview_consistency_v2.png
"""
import gc
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc
from density_db import estimate_nutrition
from eval_segmentation_benchmark import match_boxes_one_to_one
from volume_nutrition import VolumeNutritionEstimator

MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = ROOT / "scratch" / "bench_cache_v2"
DEPTH_DIR = CACHE_DIR / "depth_ctx"
Z_MARKER_PATH = CACHE_DIR / "z_marker_cache.json"
FIG_DIR = ROOT / "results" / "thesis_figures_v2"
ARTIFACT_DIR = Path(r"C:\Users\huynh\.gemini\antigravity\brain\c0224c8e-e00c-45d5-bd8d-70ec03b619ce")
OUT_JSON = CACHE_DIR / "extra_experiments_1_3.json"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def norm_cls(s: str) -> str:
    return s.split("(")[0].strip().lower().replace(" ", "")


def dish_session_key(image_path: str, short_class: str, gt_w: float) -> str:
    p = Path(image_path)
    folder = p.parent.name
    stem = p.stem
    # Strip trailing view numbers like " (1)", "(2)", " 10", etc.
    base = re.sub(r"(\s*\(\d+\)|\s+\d+|\d+)$", "", stem).strip()
    return f"{folder}::{base}::{short_class}::{round(gt_w, 1)}"


def calc_metrics(preds, gts):
    p = np.array(preds, dtype=float)
    g = np.array(gts, dtype=float)
    valid = g > 0
    p, g = p[valid], g[valid]
    mae = float(np.mean(np.abs(p - g)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mape = float(np.mean(np.abs(p - g) / g) * 100.0)
    mpe = float(np.mean((p - g) / g) * 100.0)
    bias = float((np.sum(p) - np.sum(g)) / np.sum(g) * 100.0)
    r = float(np.corrcoef(p, g)[0, 1]) if len(p) > 2 else 0.0
    return {"n": int(len(p)), "mae": mae, "rmse": rmse, "mape": mape, "mpe": mpe, "bias": bias, "r": r, "r2": r * r}


def main():
    mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
    yolo_data = json.loads((CACHE_DIR / "yolo_detections.json").read_text(encoding="utf-8"))
    z_marker_map = json.loads(Z_MARKER_PATH.read_text(encoding="utf-8"))

    inst_by_file = defaultdict(list)
    for x in mp["instances"]:
        inst_by_file[x["file_name"]].append(x)
    target_files = sorted(inst_by_file.keys())
    vol_est = VolumeNutritionEstimator()

    NUTR_KEYS = ("Calories", "Protein", "Fat", "Carbs")

    # Exp 1 containers
    ms_meal_nutr = {k: ([], []) for k in NUTR_KEYS}
    e2e_meal_nutr = {k: ([], []) for k in NUTR_KEYS}
    ms_group_nutr = {k: ([], []) for k in NUTR_KEYS}
    e2e_group_nutr = {k: ([], []) for k in NUTR_KEYS}

    # Exp 2 containers (402 ArUco scenes only)
    aruco_modes = ("no_aruco_heuristic", "aruco_no_c6", "aruco_with_c6")
    exp2_recs = {m: [] for m in aruco_modes}

    # Exp 3 containers: session_key -> mode -> list of predicted masses
    mv_sessions = defaultdict(lambda: {m: [] for m in aruco_modes})
    mv_gt_map = {}

    for idx, fn in enumerate(target_files):
        insts = inst_by_file[fn]
        stem = Path(fn).stem

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
        yolo_c = ms_npz["yolo_cand_raw"].astype(bool)
        ms_npz.close()

        zm = z_marker_map.get(fn, {}).get("z_marker_depth_mm")
        gt_scene_classes = [x["vietfood67_class_name"] for x in insts]

        # Prepare resized MobileSAM C masks & bboxes
        ms_masks_v, b_vols = [], []
        for i_inst, inst in enumerate(insts):
            m = oracle_c[i_inst]
            mv = cv2.resize(m.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if m.shape[:2] != (h_vol, w_vol) else m
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            ms_masks_v.append(mv)
            b_vols.append((int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy))))

        # Run MobileSAM C (production mode = aruco_with_c6 if aruco else plate_heuristic)
        ms_ests_c6 = []
        ms_ests_noc6 = []
        ms_ests_heur = []
        for i_inst, inst in enumerate(insts):
            cname = inst["vietfood67_class_name"]
            e_c6 = vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=ms_masks_v[i_inst], class_name=cname,
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vols[i_inst],
                K=K, depth_anchored=d_anc, z_marker_mm=zm, scene_classes=gt_scene_classes,
            )
            ms_ests_c6.append(e_c6)

            if scale_src == "aruco_marker":
                e_noc6 = vol_est.estimate(
                    depth_map_mm=depth_mm, food_mask=ms_masks_v[i_inst], class_name=cname,
                    mm_per_pixel=mm_px, scale_source="aruco_marker", bbox=b_vols[i_inst],
                    K=K, depth_anchored=d_anc, z_marker_mm=None, scene_classes=gt_scene_classes,
                )
                # Artificial no-ArUco mode: plate_heuristic, depth_anchored=False, z_marker_mm=None
                dtype = pc.CLASS_TO_DISH_TYPE.get(cname, "default")
                exp_d = pc.PLATE_DIAMETER_HEURISTICS.get(dtype, 200.0)
                bw = max(10, b_vols[i_inst][2] - b_vols[i_inst][0])
                bh = max(10, b_vols[i_inst][3] - b_vols[i_inst][1])
                heur_mm_px = exp_d / float(max(bw, bh))
                e_heur = vol_est.estimate(
                    depth_map_mm=depth_mm, food_mask=ms_masks_v[i_inst], class_name=cname,
                    mm_per_pixel=heur_mm_px, scale_source="plate_heuristic", bbox=b_vols[i_inst],
                    K=K, depth_anchored=False, z_marker_mm=None, scene_classes=gt_scene_classes,
                )
                ms_ests_noc6.append(e_noc6)
                ms_ests_heur.append(e_heur)

        # YOLO End-to-End predictions
        y_preds = yolo_data[fn]["preds"]
        y_scene_classes = [p["class_name"] for p in y_preds]
        y_ests = []
        for pi, p in enumerate(y_preds):
            ym = yolo_c[pi]
            ym_v = cv2.resize(ym.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if ym.shape[:2] != (h_vol, w_vol) else ym
            bx1, by1, bx2, by2 = p["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
            e_y = vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=ym_v, class_name=p["class_name"],
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol,
                K=K, depth_anchored=d_anc, z_marker_mm=zm, scene_classes=y_scene_classes,
            )
            y_ests.append(e_y)

        # Match YOLO boxes (normalized class names)
        y_norm = [{**p, "short_class": norm_cls(p["short_class"])} for p in y_preds]
        g_norm = [{**g, "short_class": norm_cls(g["short_class"])} for g in insts]
        awr_m, rem_p1, rem_g1 = match_boxes_one_to_one(y_norm, g_norm, iou_thr=0.5, require_same_class=True)
        sub_agn, _, _ = match_boxes_one_to_one(
            [y_norm[i] for i in rem_p1], [g_norm[j] for j in rem_g1], iou_thr=0.5, require_same_class=False
        )
        pred_for_gt = {gi: pi for pi, gi, _ in awr_m}
        for spi, sgi, _ in sub_agn:
            pred_for_gt[rem_g1[sgi]] = rem_p1[spi]

        # Aggregate Exp 1 (Nutrition), Exp 2 (ArUco), Exp 3 (Multi-View)
        grouped = defaultdict(list)
        meal_gt_nutr = {k: 0.0 for k in NUTR_KEYS}
        meal_ms_nutr = {k: sum(float(e.nutrition.get(k, 0.0)) for e in ms_ests_c6) for k in NUTR_KEYS}
        meal_e2e_nutr = {k: sum(float(e.nutrition.get(k, 0.0)) for e in y_ests) for k in NUTR_KEYS}
        meal_complete = True
        has_meal_only = False

        for gi, inst in enumerate(insts):
            stt = inst["mapping_status"]
            if stt == "exact_1to1":
                gt_w = float(inst["gt_weight_g"])
                gt_n = estimate_nutrition(gt_w, inst["vietfood67_class_name"])
                for k in NUTR_KEYS:
                    meal_gt_nutr[k] += float(gt_n.get(k, 0.0))
                    ms_group_nutr[k][0].append(float(ms_ests_c6[gi].nutrition.get(k, 0.0)))
                    ms_group_nutr[k][1].append(float(gt_n.get(k, 0.0)))
                    if gi in pred_for_gt:
                        e2e_group_nutr[k][0].append(float(y_ests[pred_for_gt[gi]].nutrition.get(k, 0.0)))
                        e2e_group_nutr[k][1].append(float(gt_n.get(k, 0.0)))

                if scale_src == "aruco_marker":
                    dt = pc.CLASS_TO_DISH_TYPE.get(inst["vietfood67_class_name"], "default")
                    exp2_recs["no_aruco_heuristic"].append((dt, ms_ests_heur[gi].mass_g, gt_w))
                    exp2_recs["aruco_no_c6"].append((dt, ms_ests_noc6[gi].mass_g, gt_w))
                    exp2_recs["aruco_with_c6"].append((dt, ms_ests_c6[gi].mass_g, gt_w))

                    skey = dish_session_key(inst["image_path"], inst["short_class"], gt_w)
                    mv_sessions[skey]["no_aruco_heuristic"].append(ms_ests_heur[gi].mass_g)
                    mv_sessions[skey]["aruco_no_c6"].append(ms_ests_noc6[gi].mass_g)
                    mv_sessions[skey]["aruco_with_c6"].append(ms_ests_c6[gi].mass_g)
                    mv_gt_map[skey] = gt_w
            elif stt == "grouped_multi_instance":
                grouped[inst["short_class"]].append(gi)
            elif stt == "meal_total_only":
                has_meal_only = True
            else:
                meal_complete = False

        for s_cls, gis in grouped.items():
            gt_w = float(insts[gis[0]]["gt_weight_g"])
            cname = insts[gis[0]]["vietfood67_class_name"]
            gt_n = estimate_nutrition(gt_w, cname)
            for k in NUTR_KEYS:
                meal_gt_nutr[k] += float(gt_n.get(k, 0.0))
                ms_group_nutr[k][0].append(sum(float(ms_ests_c6[i].nutrition.get(k, 0.0)) for i in gis))
                ms_group_nutr[k][1].append(float(gt_n.get(k, 0.0)))
                if all(i in pred_for_gt for i in gis):
                    e2e_group_nutr[k][0].append(sum(float(y_ests[pred_for_gt[i]].nutrition.get(k, 0.0)) for i in gis))
                    e2e_group_nutr[k][1].append(float(gt_n.get(k, 0.0)))

            if scale_src == "aruco_marker":
                dt = pc.CLASS_TO_DISH_TYPE.get(cname, "default")
                m_h = sum(ms_ests_heur[i].mass_g for i in gis)
                m_n = sum(ms_ests_noc6[i].mass_g for i in gis)
                m_c = sum(ms_ests_c6[i].mass_g for i in gis)
                exp2_recs["no_aruco_heuristic"].append((dt, m_h, gt_w))
                exp2_recs["aruco_no_c6"].append((dt, m_n, gt_w))
                exp2_recs["aruco_with_c6"].append((dt, m_c, gt_w))

                skey = dish_session_key(insts[gis[0]]["image_path"], s_cls, gt_w)
                mv_sessions[skey]["no_aruco_heuristic"].append(m_h)
                mv_sessions[skey]["aruco_no_c6"].append(m_n)
                mv_sessions[skey]["aruco_with_c6"].append(m_c)
                mv_gt_map[skey] = gt_w

        if meal_complete and not has_meal_only and meal_gt_nutr["Calories"] > 0:
            for k in NUTR_KEYS:
                ms_meal_nutr[k][0].append(meal_ms_nutr[k])
                ms_meal_nutr[k][1].append(meal_gt_nutr[k])
                e2e_meal_nutr[k][0].append(meal_e2e_nutr[k])
                e2e_meal_nutr[k][1].append(meal_gt_nutr[k])

        del depth_mm, oracle_c, yolo_c
        if (idx + 1) % 100 == 0:
            gc.collect()

    # Summarize Exp 1
    exp1_summary = {
        "oracle_ms_group": {k: calc_metrics(*ms_group_nutr[k]) for k in NUTR_KEYS},
        "e2e_group": {k: calc_metrics(*e2e_group_nutr[k]) for k in NUTR_KEYS},
        "oracle_ms_meal": {k: calc_metrics(*ms_meal_nutr[k]) for k in NUTR_KEYS},
        "e2e_meal": {k: calc_metrics(*e2e_meal_nutr[k]) for k in NUTR_KEYS},
    }

    # Summarize Exp 2
    exp2_summary = {}
    for m in aruco_modes:
        rs = exp2_recs[m]
        overall = calc_metrics([x[1] for x in rs], [x[2] for x in rs])
        by_dt = defaultdict(list)
        for dt, pr, gt in rs:
            by_dt[dt].append((pr, gt))
        per_dt = {dt: calc_metrics([x[0] for x in v], [x[1] for x in v]) for dt, v in sorted(by_dt.items())}
        exp2_summary[m] = {"overall": overall, "per_dish_type": per_dt}

    # Summarize Exp 3 (sessions with >= 3 views)
    valid_sessions = {k: v for k, v in mv_sessions.items() if len(v["aruco_with_c6"]) >= 3}
    exp3_summary = {"n_sessions": len(valid_sessions), "n_views_total": sum(len(v["aruco_with_c6"]) for v in valid_sessions.values())}
    cv_lists = {m: [] for m in aruco_modes}
    spread_lists = {m: [] for m in aruco_modes}
    for m in aruco_modes:
        for skey, d in valid_sessions.items():
            vals = d[m]
            mean_v = st.mean(vals)
            std_v = st.stdev(vals) if len(vals) > 1 else 0.0
            cv_pct = (std_v / mean_v * 100.0) if mean_v > 0 else 0.0
            cv_lists[m].append(cv_pct)
            spread_lists[m].append(max(vals) - min(vals))
        exp3_summary[m] = {
            "mean_cv_pct": st.mean(cv_lists[m]),
            "median_cv_pct": st.median(cv_lists[m]),
            "p75_cv_pct": float(np.percentile(cv_lists[m], 75)),
            "mean_spread_g": st.mean(spread_lists[m]),
        }

    # Save JSON
    OUT_JSON.write_text(
        json.dumps({"exp1": exp1_summary, "exp2": exp2_summary, "exp3": exp3_summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print concise summary
    print("=== EXP 1: NUTRITION ACCURACY (FULL-MEAL, N={} meals) ===".format(exp1_summary["oracle_ms_meal"]["Calories"]["n"]))
    for k in NUTR_KEYS:
        om = exp1_summary["oracle_ms_meal"][k]
        em = exp1_summary["e2e_meal"][k]
        unit = "kcal" if k == "Calories" else "g"
        print(f"  {k:10s} | Oracle MS: MAE={om['mae']:6.1f} {unit}, MAPE={om['mape']:5.2f}%, r={om['r']:.3f} | E2E: MAE={em['mae']:6.1f} {unit}, MAPE={em['mape']:5.2f}%, r={em['r']:.3f}")

    print("\n=== EXP 2: ARUCO ABLATION ON 402 ARUCO SCENES (N={} groups) ===".format(exp2_summary["aruco_with_c6"]["overall"]["n"]))
    for m in aruco_modes:
        ov = exp2_summary[m]["overall"]
        print(f"  {m:20s} | MAE={ov['mae']:6.2f}g | MAPE={ov['mape']:5.2f}% | RMSE={ov['rmse']:6.2f}g | MPE={ov['mpe']:+6.2f}% | r={ov['r']:.3f}")

    print(f"\n=== EXP 3: MULTI-VIEW REPEATABILITY ({exp3_summary['n_sessions']} sessions, {exp3_summary['n_views_total']} views) ===")
    for m in aruco_modes:
        s = exp3_summary[m]
        print(f"  {m:20s} | Mean CV={s['mean_cv_pct']:5.2f}% | Median CV={s['median_cv_pct']:5.2f}% | P75 CV={s['p75_cv_pct']:5.2f}% | Mean Max-Min={s['mean_spread_g']:5.1f}g")

    # Figure 6: Calorie Scatter (Full-Meal) + ArUco Ablation by Dish Type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=300)

    cal_p, cal_g = ms_meal_nutr["Calories"]
    cal_p = np.array(cal_p, dtype=float)
    cal_g = np.array(cal_g, dtype=float)
    ax1.scatter(cal_g, cal_p, c="#2980b9", alpha=0.62, s=26, edgecolors="white", linewidth=0.3, label="Bữa ăn (Oracle MobileSAM C)")
    mv = max(cal_g.max(), cal_p.max()) * 1.05
    grid = np.linspace(0, mv, 200)
    ax1.plot(grid, grid, "k--", linewidth=1.3, label="y = x")
    ax1.fill_between(grid, grid * 0.70, grid * 1.30, color="#95a5a6", alpha=0.14, label="Biên ±30%")
    cm = exp1_summary["oracle_ms_meal"]["Calories"]
    ax1.text(
        0.04, 0.96,
        f"N = {cm['n']} bữa ăn (Full-Meal)\nPearson r = {cm['r']:.3f} (R² = {cm['r2']:.3f})\nMAE = {cm['mae']:.1f} kcal\nMAPE = {cm['mape']:.2f}%",
        transform=ax1.transAxes, va="top", ha="left", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#2c3e50", alpha=0.92),
    )
    ax1.set_xlim(0, 1200)
    ax1.set_ylim(0, 1200)
    ax1.set_xlabel("Năng lượng thực tế — GT Calories (kcal)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Năng lượng dự đoán — Pred Calories (kcal)", fontsize=10, fontweight="bold")
    ax1.set_title("Tương quan Năng lượng Bữa ăn (409 Full-Meal)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Subplot 2: ArUco Ablation by Dish Type
    dts = ["soup_bowl", "rice_bowl", "flat_plate", "small_plate", "side_vegetables", "ingredient_small"]
    dt_labels = ["Tô nước", "Chén cơm", "Đĩa chính", "Đĩa nhỏ", "Rau lá", "Món nhỏ"]
    x = np.arange(len(dts))
    w = 0.26
    v_heur = [exp2_summary["no_aruco_heuristic"]["per_dish_type"][d]["mape"] for d in dts]
    v_noc6 = [exp2_summary["aruco_no_c6"]["per_dish_type"][d]["mape"] for d in dts]
    v_c6 = [exp2_summary["aruco_with_c6"]["per_dish_type"][d]["mape"] for d in dts]

    ax2.bar(x - w, v_heur, width=w, color="#e74c3c", alpha=0.88, label=f"Không ArUco ({exp2_summary['no_aruco_heuristic']['overall']['mape']:.1f}%)")
    ax2.bar(x, v_noc6, width=w, color="#f39c12", alpha=0.88, label=f"Có ArUco, không C6 ({exp2_summary['aruco_no_c6']['overall']['mape']:.1f}%)")
    ax2.bar(x + w, v_c6, width=w, color="#27ae60", alpha=0.88, label=f"Có ArUco + C6 ({exp2_summary['aruco_with_c6']['overall']['mape']:.1f}%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dt_labels, fontsize=9)
    ax2.set_ylabel("MAPE (%) ↓", fontsize=10, fontweight="bold")
    ax2.set_title("Đối chứng Có vs Không có ArUco (402 ảnh / 483 nhóm)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.legend(fontsize=8.3, loc="upper left")

    fig.tight_layout()
    out6 = FIG_DIR / "fig6_calorie_and_aruco_ablation_v2.png"
    fig.savefig(out6, dpi=300)
    fig.savefig(ARTIFACT_DIR / out6.name, dpi=300)
    plt.close(fig)
    print(f"Saved {out6}")

    # Figure 7: Multi-view CV% Boxplot
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    bp = ax.boxplot(
        [cv_lists["no_aruco_heuristic"], cv_lists["aruco_no_c6"], cv_lists["aruco_with_c6"]],
        tick_labels=[
            f"Không ArUco\n(Median {exp3_summary['no_aruco_heuristic']['median_cv_pct']:.1f}%)",
            f"Có ArUco, không C6\n(Median {exp3_summary['aruco_no_c6']['median_cv_pct']:.1f}%)",
            f"Có ArUco + C6\n(Median {exp3_summary['aruco_with_c6']['median_cv_pct']:.1f}%)",
        ],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="white", markersize=6),
    )
    for patch, col in zip(bp["boxes"], ["#e74c3c", "#f39c12", "#27ae60"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.set_ylabel("Hệ số biến thiên đa góc chụp — CV (%) ↓", fontsize=10.5, fontweight="bold")
    ax.set_title(
        f"Độ ổn định đa góc chụp giữa các ảnh cùng một phần ăn\n({exp3_summary['n_sessions']} phiên chụp, {exp3_summary['n_views_total']} góc chụp, ≥ 3 ảnh/phần ăn)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    out7 = FIG_DIR / "fig7_multiview_consistency_v2.png"
    fig.savefig(out7, dpi=300)
    fig.savefig(ARTIFACT_DIR / out7.name, dpi=300)
    plt.close(fig)
    print(f"Saved {out7}")


if __name__ == "__main__":
    main()
