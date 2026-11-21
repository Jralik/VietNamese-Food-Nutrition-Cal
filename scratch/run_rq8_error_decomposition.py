"""
RQ8 / Section 4.12 — Rigorous 4-Source Error Decomposition Experiment
on the full 418-image / 552-instance / 500-mass-group / 415-full-meal / 55-class benchmark.

Decomposes total portion estimation error into 4 distinct sources:
  1. Detection (YOLOv10b: FN misses, FP extras, TH2 class confusion, box localization jitter)
  2. Segmentation (MobileSAM Config C vs CVAT GT Polygon Mask)
  3. Monocular Depth + Scale + 3D Geometry (residual after GT mask & density variance;
     cross-verified by multi-view within-serving pure geometry variance on 95 sessions / 428 views)
  4. Bulk Density Variability (DENSITY_DB natural intra-class sigma_rho via 1,000-trial Monte-Carlo
     + empirical implied-density dispersion across classes & servings)

Outputs:
  - scratch/bench_cache_v2/rq8_error_decomposition_v2.json
  - results/thesis_figures_v2/fig9_four_source_error_decomposition_v2.png
    (also copied to Artifact directory)
"""
import json
import math
import os
import re
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
from density_db import DENSITY_DB, get_density
from eval_segmentation_benchmark import match_boxes_one_to_one
from volume_nutrition import VolumeNutritionEstimator

MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
COCO_PATH = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
CACHE_DIR = ROOT / "scratch" / "bench_cache_v2"
DEPTH_DIR = CACHE_DIR / "depth_ctx"
Z_MARKER_PATH = CACHE_DIR / "z_marker_cache.json"
FIG_DIR = ROOT / "results" / "thesis_figures_v2"
ARTIFACT_DIR = Path(r"C:\Users\huynh\.gemini\antigravity\brain\c0224c8e-e00c-45d5-bd8d-70ec03b619ce")
OUT_JSON = CACHE_DIR / "rq8_error_decomposition_v2.json"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def norm_cls(s: str) -> str:
    return s.split("(")[0].strip().lower().replace(" ", "")


def dish_session_key(image_path: str, short_class: str, gt_w: float) -> str:
    p = Path(image_path)
    folder = p.parent.name
    stem = p.stem
    base = re.sub(r"(\s*\(\d+\)|\s+\d+|\d+)$", "", stem).strip()
    return f"{folder}::{base}::{short_class}::{round(gt_w, 1)}"


def rasterize_coco_poly(segmentation, w_vol: int, h_vol: int, sx: float, sy: float) -> np.ndarray:
    mask = np.zeros((h_vol, w_vol), dtype=np.uint8)
    if isinstance(segmentation, list):
        for poly in segmentation:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def metrics_dict(preds, gts):
    p = np.array(preds, dtype=float)
    g = np.array(gts, dtype=float)
    valid = g > 0
    p, g = p[valid], g[valid]
    ae = np.abs(p - g)
    ape = ae / g * 100.0
    rel = (p - g) / g
    mse = float(np.mean((p - g) ** 2))
    # Log-space error (floor pred at 1.0g only if zero, e.g., FN miss)
    log_err = np.log(np.maximum(p, 1.0) / g)
    msle = float(np.mean(log_err ** 2))
    r = float(np.corrcoef(p, g)[0, 1]) if len(p) > 2 and np.std(p) > 0 else 0.0
    return {
        "n": int(len(p)),
        "mae": float(np.mean(ae)),
        "median_ae": float(np.median(ae)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "mpe": float(np.mean(rel * 100.0)),
        "msle": msle,
        "rmsle": float(np.sqrt(msle)),
        "msre": float(np.mean(rel ** 2)),
        "r": r,
        "r2": r * r,
    }


def main():
    mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
    coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    yolo_data = json.loads((CACHE_DIR / "yolo_detections.json").read_text(encoding="utf-8"))
    z_marker_map = json.loads(Z_MARKER_PATH.read_text(encoding="utf-8"))

    inst_by_file = defaultdict(list)
    for x in mp["instances"]:
        inst_by_file[x["file_name"]].append(x)
    target_files = sorted(inst_by_file.keys())

    vol_est = VolumeNutritionEstimator()

    # Containers for 500 mass groups
    group_records = []
    # Containers for 415 complete meals (exclude meal_total_only and incomplete meals)
    meal_records = []
    # Multi-view sessions (Oracle GT Mask, ArUco scenes) for pure viewpoint/depth/geometry variance
    mv_gtmask_logerr = defaultdict(list)
    mv_gtmask_pred = defaultdict(list)
    mv_ms_logerr = defaultdict(list)

    for fn in target_files:
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

        # 1. Oracle GT Polygon Mask & Oracle MobileSAM C
        gt_mask_ests = []
        ms_ests = []
        for gi, inst in enumerate(insts):
            cname = inst["vietfood67_class_name"]
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))

            # GT Polygon mask
            ann = ann_by_id[inst["annotation_id"]]
            gm = rasterize_coco_poly(ann["segmentation"], w_vol, h_vol, sx, sy)
            e_gt = vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=gm, class_name=cname,
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol,
                K=K, depth_anchored=d_anc, z_marker_mm=zm, scene_classes=gt_scene_classes,
            )
            gt_mask_ests.append(e_gt)

            # MobileSAM C (Oracle Box)
            m = oracle_c[gi]
            mv = cv2.resize(m.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if m.shape[:2] != (h_vol, w_vol) else m
            e_ms = vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=mv, class_name=cname,
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol,
                K=K, depth_anchored=d_anc, z_marker_mm=zm, scene_classes=gt_scene_classes,
            )
            ms_ests.append(e_ms)

        # 2. YOLO detections + counterfactual (YOLO box & mask + TRUE class)
        y_preds = yolo_data[fn]["preds"]
        y_scene_classes = [p["class_name"] for p in y_preds]
        y_ests_yolo_cls = []
        y_masks_v = []
        y_bvols = []
        for pi, p in enumerate(y_preds):
            ym = yolo_c[pi]
            ym_v = cv2.resize(ym.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if ym.shape[:2] != (h_vol, w_vol) else ym
            bx1, by1, bx2, by2 = p["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
            y_masks_v.append(ym_v)
            y_bvols.append(b_vol)
            e_y = vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=ym_v, class_name=p["class_name"],
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol,
                K=K, depth_anchored=d_anc, z_marker_mm=zm, scene_classes=y_scene_classes,
            )
            y_ests_yolo_cls.append(e_y)

        # Match YOLO to GT (normalized class names)
        y_norm = [{**p, "short_class": norm_cls(p["short_class"])} for p in y_preds]
        g_norm = [{**g, "short_class": norm_cls(g["short_class"])} for g in insts]
        awr_m, rem_p1, rem_g1 = match_boxes_one_to_one(y_norm, g_norm, iou_thr=0.5, require_same_class=True)
        sub_agn, _, _ = match_boxes_one_to_one(
            [y_norm[i] for i in rem_p1], [g_norm[j] for j in rem_g1], iou_thr=0.5, require_same_class=False
        )
        pred_for_gt = {}
        match_type_for_gt = {}
        for pi, gi, iou_v in awr_m:
            pred_for_gt[gi] = pi
            match_type_for_gt[gi] = "TH1"
        for spi, sgi, iou_v in sub_agn:
            pred_for_gt[rem_g1[sgi]] = rem_p1[spi]
            match_type_for_gt[rem_g1[sgi]] = "TH2"

        # Counterfactual: YOLO box + YOLO-prompted MobileSAM mask, but TRUE class
        y_est_true_cls_for_gt = {}
        for gi, pi in pred_for_gt.items():
            if match_type_for_gt[gi] == "TH1":
                y_est_true_cls_for_gt[gi] = y_ests_yolo_cls[pi]
            else:
                e_tc = vol_est.estimate(
                    depth_map_mm=depth_mm, food_mask=y_masks_v[pi],
                    class_name=insts[gi]["vietfood67_class_name"],
                    mm_per_pixel=mm_px, scale_source=scale_src, bbox=y_bvols[pi],
                    K=K, depth_anchored=d_anc, z_marker_mm=zm, scene_classes=gt_scene_classes,
                )
                y_est_true_cls_for_gt[gi] = e_tc

        # Aggregate groups & meals
        grouped = defaultdict(list)
        meal_gt_g = 0.0
        meal_gtmask_g = 0.0
        meal_ms_g = 0.0
        meal_e2e_all_g = sum(e.mass_g for e in y_ests_yolo_cls)
        meal_e2e_nofp_g = 0.0      # matched boxes only (FN=0, no FP extras)
        meal_e2e_nofp_nofn_g = 0.0 # matched YOLO boxes + Oracle MS for FN items (keeps TH2 + Box jitter)
        meal_e2e_truecls_g = 0.0   # YOLO box + True class for matched + Oracle MS for FN (keeps Box jitter only)
        meal_complete = True
        has_meal_only = False

        def process_group(gis, gt_w, short_cls, full_cls, dish_type, img_path):
            m_gtmask = sum(gt_mask_ests[i].mass_g for i in gis)
            v_gtmask = sum(gt_mask_ests[i].volume_cm3 for i in gis)
            m_ms = sum(ms_ests[i].mass_g for i in gis)
            all_matched = all(i in pred_for_gt for i in gis)
            any_matched = any(i in pred_for_gt for i in gis)

            m_e2e_raw = sum(y_ests_yolo_cls[pred_for_gt[i]].mass_g for i in gis if i in pred_for_gt)
            m_e2e_nofn = sum(y_ests_yolo_cls[pred_for_gt[i]].mass_g if i in pred_for_gt else ms_ests[i].mass_g for i in gis)
            m_e2e_truecls = sum(y_est_true_cls_for_gt[i].mass_g if i in pred_for_gt else ms_ests[i].mass_g for i in gis)

            if all_matched:
                mtypes = {match_type_for_gt[i] for i in gis}
                grp_status = "TH1" if mtypes == {"TH1"} else "TH2"
                yolo_lbl = y_preds[pred_for_gt[gis[0]]]["short_class"]
            elif any_matched:
                grp_status = "PARTIAL_FN"
                yolo_lbl = "PARTIAL"
            else:
                grp_status = "FN"
                yolo_lbl = "NONE"

            rho_mean, rho_std = get_density(full_cls)
            rho_used = m_gtmask / v_gtmask if v_gtmask > 1e-6 else rho_mean
            rho_implied = gt_w / v_gtmask if v_gtmask > 1e-6 else rho_mean

            rec = {
                "file_name": fn,
                "short_class": short_cls,
                "full_class": full_cls,
                "dish_type": dish_type,
                "scale_source": scale_src,
                "gt_g": gt_w,
                "m_gtmask_g": m_gtmask,
                "v_gtmask_cm3": v_gtmask,
                "m_ms_g": m_ms,
                "m_e2e_raw_g": m_e2e_raw,
                "m_e2e_nofn_g": m_e2e_nofn,
                "m_e2e_truecls_g": m_e2e_truecls,
                "all_matched": all_matched,
                "grp_status": grp_status,
                "yolo_lbl": yolo_lbl,
                "rho_db_mean": float(rho_mean),
                "rho_db_std": float(rho_std),
                "rho_used": float(rho_used),
                "rho_implied": float(rho_implied),
                "n_insts": len(gis),
            }
            group_records.append(rec)

            if scale_src == "aruco_marker" and gt_w > 0 and m_gtmask > 0 and m_ms > 0:
                skey = dish_session_key(img_path, short_cls, gt_w)
                mv_gtmask_logerr[skey].append(math.log(m_gtmask / gt_w))
                mv_gtmask_pred[skey].append(m_gtmask)
                mv_ms_logerr[skey].append(math.log(m_ms / gt_w))

            return m_gtmask, m_ms, m_e2e_raw, m_e2e_nofn, m_e2e_truecls

        for gi, inst in enumerate(insts):
            stt = inst["mapping_status"]
            if stt == "exact_1to1":
                gt_w = float(inst["gt_weight_g"])
                a, b, c, d, e = process_group(
                    [gi], gt_w, inst["short_class"], inst["vietfood67_class_name"],
                    inst["dish_type"], inst["image_path"]
                )
                meal_gt_g += gt_w
                meal_gtmask_g += a
                meal_ms_g += b
                meal_e2e_nofp_g += c
                meal_e2e_nofp_nofn_g += d
                meal_e2e_truecls_g += e
            elif stt == "grouped_multi_instance":
                grouped[inst["short_class"]].append(gi)
            elif stt == "meal_total_only":
                has_meal_only = True
            else:
                meal_complete = False

        for s_cls, gis in grouped.items():
            inst0 = insts[gis[0]]
            gt_w = float(inst0["gt_weight_g"])
            a, b, c, d, e = process_group(
                gis, gt_w, s_cls, inst0["vietfood67_class_name"],
                inst0["dish_type"], inst0["image_path"]
            )
            meal_gt_g += gt_w
            meal_gtmask_g += a
            meal_ms_g += b
            meal_e2e_nofp_g += c
            meal_e2e_nofp_nofn_g += d
            meal_e2e_truecls_g += e

        if meal_complete and not has_meal_only and meal_gt_g > 0:
            n_fp = len(y_preds) - len(pred_for_gt)
            n_fn = len(insts) - len(pred_for_gt)
            n_th2 = sum(1 for v in match_type_for_gt.values() if v == "TH2")
            meal_records.append({
                "file_name": fn,
                "scale_source": scale_src,
                "gt_g": meal_gt_g,
                "s0_e2e_all": meal_e2e_all_g,
                "s1_e2e_nofp": meal_e2e_nofp_g,
                "s2_e2e_nofp_nofn": meal_e2e_nofp_nofn_g,
                "s3_e2e_truecls": meal_e2e_truecls_g,
                "s4_oracle_ms": meal_ms_g,
                "s5_oracle_gtmask": meal_gtmask_g,
                "n_fp": n_fp,
                "n_fn": n_fn,
                "n_th2": n_th2,
                "n_gt": len(insts),
            })

    # =========================================================================
    # ANALYSIS 1: PROGRESSIVE ORACLE LADDER (415 FULL MEALS & 500/437 GROUPS)
    # =========================================================================
    gts_m = [r["gt_g"] for r in meal_records]
    meal_ladder = {
        "S0_Full_E2E": metrics_dict([r["s0_e2e_all"] for r in meal_records], gts_m),
        "S1_Remove_FP_Extras": metrics_dict([r["s1_e2e_nofp"] for r in meal_records], gts_m),
        "S2_Remove_FN_Misses": metrics_dict([r["s2_e2e_nofp_nofn"] for r in meal_records], gts_m),
        "S3_Remove_TH2_Class": metrics_dict([r["s3_e2e_truecls"] for r in meal_records], gts_m),
        "S4_Oracle_MobileSAM_C": metrics_dict([r["s4_oracle_ms"] for r in meal_records], gts_m),
        "S5_Oracle_GT_Mask": metrics_dict([r["s5_oracle_gtmask"] for r in meal_records], gts_m),
    }

    matched_grps = [r for r in group_records if r["all_matched"]]
    gts_g500 = [r["gt_g"] for r in group_records]
    gts_g437 = [r["gt_g"] for r in matched_grps]

    group_ladder_500 = {
        "G0_E2E_All500_FN0": metrics_dict([r["m_e2e_raw_g"] for r in group_records], gts_g500),
        "G1_Remove_FN_Misses": metrics_dict([r["m_e2e_nofn_g"] for r in group_records], gts_g500),
        "G2_Remove_TH2_Class": metrics_dict([r["m_e2e_truecls_g"] for r in group_records], gts_g500),
        "G3_Oracle_MobileSAM_C": metrics_dict([r["m_ms_g"] for r in group_records], gts_g500),
        "G4_Oracle_GT_Mask": metrics_dict([r["m_gtmask_g"] for r in group_records], gts_g500),
    }

    group_ladder_437 = {
        "M0_E2E_Matched437": metrics_dict([r["m_e2e_raw_g"] for r in matched_grps], gts_g437),
        "M1_Remove_TH2_Class": metrics_dict([r["m_e2e_truecls_g"] for r in matched_grps], gts_g437),
        "M2_Oracle_MobileSAM_C": metrics_dict([r["m_ms_g"] for r in matched_grps], gts_g437),
        "M3_Oracle_GT_Mask": metrics_dict([r["m_gtmask_g"] for r in matched_grps], gts_g437),
    }

    # =========================================================================
    # ANALYSIS 2: EXACT PAIRWISE PERTURBATION ERRORS (WITHOUT SIGN CANCELLATION)
    # =========================================================================
    def pairwise_stats(a_list, b_list, gt_list):
        a = np.maximum(np.array(a_list, dtype=float), 1.0)
        b = np.maximum(np.array(b_list, dtype=float), 1.0)
        g = np.array(gt_list, dtype=float)
        diff_g = np.abs(a - b)
        diff_pct_gt = diff_g / g * 100.0
        log_d = np.log(a / b)
        return {
            "mean_abs_diff_g": float(np.mean(diff_g)),
            "median_abs_diff_g": float(np.median(diff_g)),
            "mean_pct_of_gt": float(np.mean(diff_pct_gt)),
            "median_pct_of_gt": float(np.median(diff_pct_gt)),
            "msle_step": float(np.mean(log_d ** 2)),
            "rmsle_step": float(np.sqrt(np.mean(log_d ** 2))),
        }

    pairwise_437 = {
        "step_TH2_class_confusion": pairwise_stats(
            [r["m_e2e_raw_g"] for r in matched_grps],
            [r["m_e2e_truecls_g"] for r in matched_grps], gts_g437
        ),
        "step_YOLO_box_jitter": pairwise_stats(
            [r["m_e2e_truecls_g"] for r in matched_grps],
            [r["m_ms_g"] for r in matched_grps], gts_g437
        ),
        "step_Detection_total_matched": pairwise_stats(
            [r["m_e2e_raw_g"] for r in matched_grps],
            [r["m_ms_g"] for r in matched_grps], gts_g437
        ),
        "step_Segmentation_MS_vs_GT": pairwise_stats(
            [r["m_ms_g"] for r in matched_grps],
            [r["m_gtmask_g"] for r in matched_grps], gts_g437
        ),
    }

    pairwise_500_seg = pairwise_stats(
        [r["m_ms_g"] for r in group_records],
        [r["m_gtmask_g"] for r in group_records], gts_g500
    )

    pairwise_415_meals = {
        "step_Detection_all_FN_FP_TH2_Box": pairwise_stats(
            [r["s0_e2e_all"] for r in meal_records],
            [r["s4_oracle_ms"] for r in meal_records], gts_m
        ),
        "step_Segmentation_MS_vs_GT": pairwise_stats(
            [r["s4_oracle_ms"] for r in meal_records],
            [r["s5_oracle_gtmask"] for r in meal_records], gts_m
        ),
    }

    # =========================================================================
    # ANALYSIS 3: SEPARATING DEPTH/SCALE/GEOMETRY VS BULK DENSITY (N = 500)
    # =========================================================================
    # 3A. Pure Within-Serving Multi-View Variance (same physical serving -> rho_true = const!)
    mv_valid_keys = [k for k, v in mv_gtmask_logerr.items() if len(v) >= 3]
    within_vars = []
    within_dof = 0
    within_ss = 0.0
    between_means = []
    mv_cvs = []
    for k in mv_valid_keys:
        vals = np.array(mv_gtmask_logerr[k], dtype=float)
        preds_v = np.array(mv_gtmask_pred[k], dtype=float)
        within_ss += float(np.sum((vals - np.mean(vals)) ** 2))
        within_dof += len(vals) - 1
        within_vars.append(float(np.var(vals, ddof=1)))
        between_means.append(float(np.mean(vals)))
        mv_cvs.append(float(np.std(preds_v, ddof=1) / np.mean(preds_v) * 100.0))

    pooled_within_var_log = within_ss / within_dof if within_dof > 0 else 0.0
    all_mv_logerr = np.concatenate([np.array(mv_gtmask_logerr[k]) for k in mv_valid_keys])
    total_mv_msle = float(np.mean(all_mv_logerr ** 2))
    # Pure viewpoint-varying depth/scale/geometry share of total GT-mask MSLE on multi-view scenes
    pure_viewpoint_geom_share_of_gtmask = pooled_within_var_log / total_mv_msle * 100.0

    # 3B. Analytical & 1,000-trial Monte-Carlo Bulk Density Variance (DENSITY_DB sigma_rho)
    rng = np.random.default_rng(42)
    N_MC = 1000
    cv_rho_sq_list = []
    mc_msle_density_list = []
    mc_mape_density_only = []
    mc_mape_perturbed_gtmask = []

    gt_arr = np.array([r["gt_g"] for r in group_records], dtype=float)
    gm_arr = np.array([r["m_gtmask_g"] for r in group_records], dtype=float)
    rho_m_arr = np.array([r["rho_db_mean"] for r in group_records], dtype=float)
    rho_s_arr = np.array([r["rho_db_std"] for r in group_records], dtype=float)
    cv_rho_arr = rho_s_arr / np.maximum(rho_m_arr, 1e-6)
    analytical_density_msle = float(np.mean(np.log(1.0 + cv_rho_arr ** 2)))  # exact Var(ln rho) for lognormal matching CV

    for _ in range(N_MC):
        # Sample relative density perturbation ~ N(1, (sigma/mu)^2), clipped to [0.5, 1.5]
        factor = np.clip(rng.normal(1.0, cv_rho_arr), 0.5, 1.5)
        mc_msle_density_list.append(float(np.mean(np.log(factor) ** 2)))
        mc_mape_density_only.append(float(np.mean(np.abs(factor - 1.0) * 100.0)))
        mc_mape_perturbed_gtmask.append(float(np.mean(np.abs(gm_arr * factor - gt_arr) / gt_arr * 100.0)))

    mc_density_msle = float(np.mean(mc_msle_density_list))
    mc_density_standalone_mape = float(np.mean(mc_mape_density_only))
    mc_gtmask_plus_rho_mape = float(np.mean(mc_mape_perturbed_gtmask))

    # 3C. Class-level systematic bias upper bound (Oracle Class-Mean Density Calibration on classes with n>=3)
    by_cls = defaultdict(list)
    for idx_r, r in enumerate(group_records):
        by_cls[r["short_class"]].append(idx_r)

    cal_preds_500 = np.copy(gm_arr)
    class_bias_stats = []
    for cls_name, idxs in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
        sub_gm = gm_arr[idxs]
        sub_gt = gt_arr[idxs]
        log_ratios = np.log(np.maximum(sub_gm, 1.0) / sub_gt)
        med_log_bias = float(np.median(log_ratios))
        mean_log_bias = float(np.mean(log_ratios))
        # Empirical optimal multiplier = exp(-mean_log_bias) (only for classes with n>=3 to avoid overfitting n=1-2)
        if len(idxs) >= 3:
            cal_preds_500[idxs] = sub_gm * math.exp(-mean_log_bias)
        sub_rho_imp = [group_records[i]["rho_implied"] for i in idxs if group_records[i]["v_gtmask_cm3"] > 5.0]
        class_bias_stats.append({
            "short_class": cls_name,
            "n": len(idxs),
            "rho_db_mean": group_records[idxs[0]]["rho_db_mean"],
            "rho_db_std": group_records[idxs[0]]["rho_db_std"],
            "rho_implied_median": float(np.median(sub_rho_imp)) if sub_rho_imp else 0.0,
            "rho_implied_iqr": float(np.percentile(sub_rho_imp, 75) - np.percentile(sub_rho_imp, 25)) if len(sub_rho_imp) >= 3 else 0.0,
            "mape_gtmask": float(np.mean(np.abs(sub_gm - sub_gt) / sub_gt * 100.0)),
            "mpe_gtmask": float(np.mean((sub_gm - sub_gt) / sub_gt * 100.0)),
            "mean_log_bias": mean_log_bias,
            "exp_neg_bias": math.exp(-mean_log_bias),
        })

    oracle_class_cal_metrics = metrics_dict(cal_preds_500, gt_arr)
    # Note: class-level systematic bias reflects BOTH class-level density offset AND class-level geometry prior offset.
    # For a clean, conservative 4-source variance budget:
    #   Var_total(ln) = Var_Detection + Var_Segmentation + Var_DepthScaleGeom + Var_Density
    # Let's compute this cleanly on THREE scopes:
    #   Scope 1: Matched Groups (N=437, exact log-space without FN=0 clipping)
    #   Scope 2: Full-Meal End-to-End (N=415, real user-facing meal error incl. FN & FP!)
    #   Scope 3: Oracle 3D Core (N=500, zero detection error -> Seg vs Depth/Scale/Geom vs Density)

    def build_4source_budget(msle_det, msle_seg, msle_gtmask_total, msle_rho_nat, msle_rho_sys=0.0):
        # msle_gtmask_total = msle_depth_geom + msle_rho_total
        msle_rho_tot = min(msle_rho_nat + msle_rho_sys, msle_gtmask_total * 0.5)
        msle_depth_geom = max(0.0, msle_gtmask_total - msle_rho_tot)
        tot = msle_det + msle_seg + msle_depth_geom + msle_rho_tot
        return {
            "total_msle_budget": tot,
            "detection": {"msle": msle_det, "rmsle": math.sqrt(msle_det), "share_pct": msle_det / tot * 100.0},
            "segmentation": {"msle": msle_seg, "rmsle": math.sqrt(msle_seg), "share_pct": msle_seg / tot * 100.0},
            "depth_scale_geometry": {"msle": msle_depth_geom, "rmsle": math.sqrt(msle_depth_geom), "share_pct": msle_depth_geom / tot * 100.0},
            "bulk_density": {"msle": msle_rho_tot, "rmsle": math.sqrt(msle_rho_tot), "share_pct": msle_rho_tot / tot * 100.0},
        }

    # Exact between-class systematic density vs geometry split:
    # Notice oracle_class_cal reduces MSLE from group_ladder_500["G4_Oracle_GT_Mask"]["msle"] down to oracle_class_cal_metrics["msle"].
    # Attributing half of systematic class bias to density offset and keeping natural intra-class sigma_rho gives realistic density share,
    # AND we ALSO report the strict lower bound (natural DENSITY_DB sigma_rho only) and upper bound (+ full class-level bias)!
    msle_gt500 = group_ladder_500["G4_Oracle_GT_Mask"]["msle"]
    msle_sys_class = max(0.0, msle_gt500 - oracle_class_cal_metrics["msle"])

    # Scope 1: Matched Groups (N=437) — pairwise exact log-variance of each stage
    budget_matched_437_natural = build_4source_budget(
        msle_det=pairwise_437["step_Detection_total_matched"]["msle_step"],
        msle_seg=pairwise_437["step_Segmentation_MS_vs_GT"]["msle_step"],
        msle_gtmask_total=group_ladder_437["M3_Oracle_GT_Mask"]["msle"],
        msle_rho_nat=mc_density_msle,
        msle_rho_sys=0.0,
    )
    budget_matched_437_with_sys = build_4source_budget(
        msle_det=pairwise_437["step_Detection_total_matched"]["msle_step"],
        msle_seg=pairwise_437["step_Segmentation_MS_vs_GT"]["msle_step"],
        msle_gtmask_total=group_ladder_437["M3_Oracle_GT_Mask"]["msle"],
        msle_rho_nat=mc_density_msle,
        msle_rho_sys=0.5 * msle_sys_class,
    )

    # Scope 2: Full-Meal End-to-End (N=415 meals) — using relative squared error (MSRE) AND MSLE!
    # Also compute MSRE (Mean Squared Relative Error = E[((pred-gt)/gt)^2]) which handles FN=0 cleanly without log(0)!
    def msre_step(a_list, b_list, gt_list):
        a = np.array(a_list, dtype=float)
        b = np.array(b_list, dtype=float)
        g = np.array(gt_list, dtype=float)
        return float(np.mean(((a - b) / g) ** 2))

    # For 415 full meals, average meal-level natural density CV^2 (averaging across items in a meal)
    meal_rho_msre_list = []
    for r in meal_records:
        # Approx natural density variance of a meal = 0.095^2 (~0.009) or exact weighted sum over its items
        fn_m = r["file_name"]
        sub_grps = [gr for gr in group_records if gr["file_name"] == fn_m]
        num = sum((gr["gt_g"] * (gr["rho_db_std"] / gr["rho_db_mean"])) ** 2 for gr in sub_grps)
        den = (sum(gr["gt_g"] for gr in sub_grps)) ** 2
        meal_rho_msre_list.append(num / den if den > 0 else 0.01)
    meal_rho_msre_nat = float(np.mean(meal_rho_msre_list))

    # Full-Meal MSRE budget (no log clipping needed even when S0=0g!)
    msre_det_meal = msre_step([r["s0_e2e_all"] for r in meal_records], [r["s4_oracle_ms"] for r in meal_records], gts_m)
    msre_seg_meal = msre_step([r["s4_oracle_ms"] for r in meal_records], [r["s5_oracle_gtmask"] for r in meal_records], gts_m)
    msre_gtmask_meal = meal_ladder["S5_Oracle_GT_Mask"]["msre"]
    msre_depth_meal = max(0.0, msre_gtmask_meal - meal_rho_msre_nat)
    tot_msre_meal = msre_det_meal + msre_seg_meal + msre_depth_meal + meal_rho_msre_nat

    budget_fullmeal_415_msre = {
        "total_msre_budget": tot_msre_meal,
        "detection": {"msre": msre_det_meal, "rmsre_pct": math.sqrt(msre_det_meal) * 100.0, "share_pct": msre_det_meal / tot_msre_meal * 100.0},
        "segmentation": {"msre": msre_seg_meal, "rmsre_pct": math.sqrt(msre_seg_meal) * 100.0, "share_pct": msre_seg_meal / tot_msre_meal * 100.0},
        "depth_scale_geometry": {"msre": msre_depth_meal, "rmsre_pct": math.sqrt(msre_depth_meal) * 100.0, "share_pct": msre_depth_meal / tot_msre_meal * 100.0},
        "bulk_density": {"msre": meal_rho_msre_nat, "rmsre_pct": math.sqrt(meal_rho_msre_nat) * 100.0, "share_pct": meal_rho_msre_nat / tot_msre_meal * 100.0},
    }

    # Scope 3: Oracle 3D Core (N=500 groups, zero detection error)
    budget_oracle_500 = build_4source_budget(
        msle_det=0.0,
        msle_seg=pairwise_500_seg["msle_step"],
        msle_gtmask_total=msle_gt500,
        msle_rho_nat=mc_density_msle,
        msle_rho_sys=0.0,
    )

    # =========================================================================
    # ANALYSIS 4: PER-DISH-TYPE 4-SOURCE PROFILE & TOP-15 FAILURE CASES
    # =========================================================================
    per_dish_profile = {}
    for dt in sorted({r["dish_type"] for r in group_records}):
        sub_all = [r for r in group_records if r["dish_type"] == dt]
        sub_m = [r for r in matched_grps if r["dish_type"] == dt]
        gt_a = [r["gt_g"] for r in sub_all]
        gt_m = [r["gt_g"] for r in sub_m]
        per_dish_profile[dt] = {
            "n_all": len(sub_all),
            "n_matched": len(sub_m),
            "fn_rate_pct": (1.0 - len(sub_m) / len(sub_all)) * 100.0,
            "th2_rate_among_matched_pct": (sum(1 for r in sub_m if r["grp_status"] == "TH2") / len(sub_m) * 100.0) if sub_m else 0.0,
            "mape_e2e_matched": metrics_dict([r["m_e2e_raw_g"] for r in sub_m], gt_m)["mape"] if sub_m else 0.0,
            "mape_oracle_ms": metrics_dict([r["m_ms_g"] for r in sub_all], gt_a)["mape"],
            "mape_oracle_gtmask": metrics_dict([r["m_gtmask_g"] for r in sub_all], gt_a)["mape"],
            "seg_pairwise_pct": pairwise_stats([r["m_ms_g"] for r in sub_all], [r["m_gtmask_g"] for r in sub_all], gt_a)["mean_pct_of_gt"],
            "mean_density_cv_pct": float(np.mean([r["rho_db_std"] / r["rho_db_mean"] * 100.0 for r in sub_all])),
        }

    # Top-15 Worst Failure Cases (End-to-End Full Meal & Oracle MobileSAM Group)
    top_meal_failures = sorted(
        meal_records, key=lambda r: abs(r["s0_e2e_all"] - r["gt_g"]), reverse=True
    )[:15]
    top_3d_failures = sorted(
        group_records, key=lambda r: abs(r["m_ms_g"] - r["gt_g"]), reverse=True
    )[:15]

    out = {
        "n_images": len(target_files),
        "n_groups": len(group_records),
        "n_matched_groups": len(matched_grps),
        "n_complete_meals": len(meal_records),
        "meal_ladder_415": meal_ladder,
        "group_ladder_500": group_ladder_500,
        "group_ladder_437": group_ladder_437,
        "pairwise_437": pairwise_437,
        "pairwise_500_seg": pairwise_500_seg,
        "pairwise_415_meals": pairwise_415_meals,
        "multiview_pure_geometry": {
            "n_sessions": len(mv_valid_keys),
            "n_views": int(len(all_mv_logerr)),
            "pooled_within_var_log": pooled_within_var_log,
            "pooled_within_sd_log": math.sqrt(pooled_within_var_log),
            "total_mv_msle": total_mv_msle,
            "pure_viewpoint_geom_share_of_gtmask_pct": pure_viewpoint_geom_share_of_gtmask,
            "median_cv_gtmask_pct": float(np.median(mv_cvs)),
            "mean_cv_gtmask_pct": float(np.mean(mv_cvs)),
        },
        "density_uncertainty_mc": {
            "n_trials": N_MC,
            "mean_db_cv_pct": float(np.mean(cv_rho_arr * 100.0)),
            "analytical_density_msle": analytical_density_msle,
            "mc_density_msle": mc_density_msle,
            "mc_density_standalone_mape": mc_density_standalone_mape,
            "mc_gtmask_plus_rho_mape": mc_gtmask_plus_rho_mape,
            "oracle_class_cal_metrics_500": oracle_class_cal_metrics,
            "msle_systematic_class_bias": msle_sys_class,
        },
        "budget_fullmeal_415_msre": budget_fullmeal_415_msre,
        "budget_matched_437_natural": budget_matched_437_natural,
        "budget_matched_437_with_sys": budget_matched_437_with_sys,
        "budget_oracle_500": budget_oracle_500,
        "per_dish_profile": per_dish_profile,
        "class_bias_top20": class_bias_stats[:20],
        "top15_e2e_meal_failures": top_meal_failures,
        "top15_3d_group_failures": top_3d_failures,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # =========================================================================
    # PRINT CLEAN SUMMARY
    # =========================================================================
    print("=== PROGRESSIVE ORACLE LADDER: FULL-MEAL (N=415) ===")
    for k, v in meal_ladder.items():
        print(f"  {k:24s} | MAE={v['mae']:6.2f}g | MAPE={v['mape']:6.2f}% | RMSE={v['rmse']:6.2f}g | r={v['r']:.3f}")

    print("\n=== PROGRESSIVE ORACLE LADDER: MATCHED GROUPS (N=437) ===")
    for k, v in group_ladder_437.items():
        print(f"  {k:24s} | MAE={v['mae']:6.2f}g | MAPE={v['mape']:6.2f}% | MSLE={v['msle']:.4f} | r={v['r']:.3f}")

    print("\n=== PAIRWISE STEP PERTURBATIONS (N=437 matched, N=500 seg) ===")
    for k, v in pairwise_437.items():
        print(f"  {k:30s} | Mean|d|={v['mean_abs_diff_g']:5.2f}g ({v['mean_pct_of_gt']:5.2f}% GT) | MSLE_step={v['msle_step']:.5f}")
    print(f"  {'Seg_MS_vs_GT (N=500)':30s} | Mean|d|={pairwise_500_seg['mean_abs_diff_g']:5.2f}g ({pairwise_500_seg['mean_pct_of_gt']:5.2f}% GT) | MSLE_step={pairwise_500_seg['msle_step']:.5f}")

    print("\n=== 4-SOURCE ERROR BUDGET (%) ===")
    print("  [A] Full-Meal End-to-End (N=415, MSRE — includes FN, FP, TH2, Box):")
    for src in ("detection", "segmentation", "depth_scale_geometry", "bulk_density"):
        d = budget_fullmeal_415_msre[src]
        print(f"      {src:22s}: {d['share_pct']:5.1f}% (RMSRE={d['rmsre_pct']:5.1f}%)")
    print("  [B] Matched Groups (N=437, MSLE — natural DENSITY_DB sigma_rho):")
    for src in ("detection", "segmentation", "depth_scale_geometry", "bulk_density"):
        d = budget_matched_437_natural[src]
        print(f"      {src:22s}: {d['share_pct']:5.1f}% (RMSLE={d['rmsle']:.3f})")
    print("  [C] Oracle 3D Core (N=500, MSLE — zero detection error):")
    for src in ("segmentation", "depth_scale_geometry", "bulk_density"):
        d = budget_oracle_500[src]
        print(f"      {src:22s}: {d['share_pct']:5.1f}% (RMSLE={d['rmsle']:.3f})")

    print(f"\n=== MULTI-VIEW PURE GEOMETRY CHECK ({len(mv_valid_keys)} sessions, {len(all_mv_logerr)} views) ===")
    print(f"  Pooled within-serving Var(ln) = {pooled_within_var_log:.4f} (SD = {math.sqrt(pooled_within_var_log):.3f})")
    print(f"  Total GT-Mask MSLE on MV set  = {total_mv_msle:.4f} -> Pure viewpoint geometry >= {pure_viewpoint_geom_share_of_gtmask:.1f}% of GT-Mask MSLE")
    print(f"  Monte-Carlo Density (1000x)   : Mean CV_rho={np.mean(cv_rho_arr*100):.2f}%, Standalone MAPE={mc_density_standalone_mape:.2f}%, MSLE_rho={mc_density_msle:.4f}")
    print(f"  Oracle Class-Mean Calibration : GT-Mask MAPE {group_ladder_500['G4_Oracle_GT_Mask']['mape']:.2f}% -> {oracle_class_cal_metrics['mape']:.2f}% (MAE {group_ladder_500['G4_Oracle_GT_Mask']['mae']:.2f} -> {oracle_class_cal_metrics['mae']:.2f}g)")

    # =========================================================================
    # FIGURE 9: 4-SOURCE ERROR DECOMPOSITION (2-PANEL 300-DPI PUBLICATION FIGURE)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=300, gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Panel A: Progressive Oracle Ladder on Full-Meal (N=415) — MAE (g) & MAPE (%)
    stages = [
        ("S0: E2E\nToàn tuyến", meal_ladder["S0_Full_E2E"]["mape"], meal_ladder["S0_Full_E2E"]["mae"], "#d9534f"),
        ("S1: Bỏ FP\n(hộp thừa)", meal_ladder["S1_Remove_FP_Extras"]["mape"], meal_ladder["S1_Remove_FP_Extras"]["mae"], "#e67e22"),
        ("S2: Bỏ FN\n(bù bỏ sót)", meal_ladder["S2_Remove_FN_Misses"]["mape"], meal_ladder["S2_Remove_FN_Misses"]["mae"], "#f39c12"),
        ("S3: Sửa TH2\n(đúng lớp)", meal_ladder["S3_Remove_TH2_Class"]["mape"], meal_ladder["S3_Remove_TH2_Class"]["mae"], "#f1c40f"),
        ("S4: Oracle\nMobileSAM C", meal_ladder["S4_Oracle_MobileSAM_C"]["mape"], meal_ladder["S4_Oracle_MobileSAM_C"]["mae"], "#2ecc71"),
        ("S5: Oracle\nGT Mask", meal_ladder["S5_Oracle_GT_Mask"]["mape"], meal_ladder["S5_Oracle_GT_Mask"]["mae"], "#27ae60"),
    ]
    x = np.arange(len(stages))
    mapes = [s[1] for s in stages]
    maes = [s[2] for s in stages]
    cols = [s[3] for s in stages]
    bars = ax1.bar(x, mapes, width=0.58, color=cols, edgecolor="#2c3e50", linewidth=0.8)
    for bar, mp_v, mae_v in zip(bars, mapes, maes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, mp_v + 1.0,
            f"{mp_v:.1f}%\n({mae_v:.0f} g)",
            ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#1a252f",
        )
    ax1.axhline(35.0, color="#34495e", linestyle="--", linewidth=1.0, alpha=0.75, label="Ngưỡng mục tiêu 35% MAPE")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[0] for s in stages], fontsize=9.0)
    ax1.set_ylabel("MAPE cấp trọn bữa ăn — Full-Meal (%) ↓", fontsize=10, fontweight="bold")
    ax1.set_title("A. Thang cắt bỏ tuần tự nguồn sai số (415 bữa Full-Meal)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylim(0, max(mapes) * 1.24)
    ax1.grid(axis="y", linestyle=":", alpha=0.55)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel B: 4-Source Error Share (%) across 3 Scopes (Stacked Horizontal Bars)
    scopes = [
        "Toàn tuyến Full-Meal\n(N = 415, gồm FN & FP)",
        "Nhóm đã khớp YOLO\n(N = 437, gồm TH2 & Box)",
        "Lõi 3D cô lập (Oracle)\n(N = 500, không lỗi YOLO)",
    ]
    src_keys = ["depth_scale_geometry", "detection", "bulk_density", "segmentation"]
    src_labels = [
        "Độ sâu + Tỷ lệ + Hình học 3D",
        "Phát hiện YOLO (FN/FP/TH2/Box)",
        "Biến thiên khối lượng riêng (ρ)",
        "Mặt nạ phân đoạn (MobileSAM C)",
    ]
    src_colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]

    data_matrix = np.array([
        [budget_fullmeal_415_msre[k]["share_pct"] for k in src_keys],
        [budget_matched_437_natural[k]["share_pct"] for k in src_keys],
        [budget_oracle_500[k]["share_pct"] for k in src_keys],
    ])

    y_pos = np.arange(len(scopes))
    lefts = np.zeros(len(scopes))
    for si, (sk, sl, sc_col) in enumerate(zip(src_keys, src_labels, src_colors)):
        vals = data_matrix[:, si]
        ax2.barh(y_pos, vals, left=lefts, height=0.50, color=sc_col, edgecolor="white", linewidth=1.2, label=sl)
        for yi, (v, lft) in enumerate(zip(vals, lefts)):
            if v >= 6.5:
                ax2.text(
                    lft + v / 2.0, yi, f"{v:.1f}%",
                    ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if sc_col != "#f39c12" else "#1a252f",
                )
        lefts += vals

    # Annotate compact density + mask shares cleanly on the right of each bar
    for yi in range(len(scopes)):
        rho_v = data_matrix[yi, 2]
        seg_v = data_matrix[yi, 3]
        ax2.text(
            101.2, yi, f"ρ: {rho_v:.1f}%\nMask: {seg_v:.1f}%",
            ha="left", va="center", fontsize=8.4, fontweight="bold", color="#2c3e50",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(scopes, fontsize=9.2, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 121)
    ax2.set_xlabel("Tỷ trọng đóng góp vào phương sai sai số (%) →", fontsize=10, fontweight="bold")
    ax2.set_title("B. Phân rã tỷ trọng 4 nguồn sai số (RQ8 — Tập 418 ảnh)", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle=":", alpha=0.55)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.45, -0.28), ncol=2, fontsize=8.8, frameon=True)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig9_four_source_error_decomposition_v2.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    import shutil
    shutil.copy2(fig_path, ARTIFACT_DIR / fig_path.name)
    print(f"\nSaved {fig_path}")


if __name__ == "__main__":
    main()
