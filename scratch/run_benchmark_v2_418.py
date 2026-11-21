"""
Full-Scale Benchmark (418 images / 552 instances / 506 mass groups / 55+ classes)
using MobileSAM Config C + Volume A1-A5.

Stages:
  --stage yolo       : Run YOLOv10b detection on 418 images -> scratch/bench_cache_v2/yolo_detections.json
  --stage mobilesam  : Run MobileSAM (Raw + Config C) -> scratch/bench_cache_v2/mobilesam_{stem}.npz + meta
  --stage eval       : Streaming evaluation of Seg + Oracle GT / Oracle MobileSAM C / E2E Volume & Mass
"""

import argparse
import gc
import json
import os
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from eval_segmentation_benchmark import (
    YOLO_WEIGHTS,
    coco_poly_to_mask,
    compute_mask_metrics,
    get_cpu_ram_mib,
    match_boxes_one_to_one,
)

COCO_JSON_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
VERIFIED_MAPPING_PATH = PROJECT_ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = PROJECT_ROOT / "scratch" / "bench_cache_v2"
DEPTH_CACHE_DIR = CACHE_DIR / "depth_ctx"
REPORT_OUT_PATH = PROJECT_ROOT / "results" / "benchmark_v2_418_report.md"
REPORT_SCRATCH_PATH = PROJECT_ROOT / "scratch" / "benchmark_v2_418_report.md"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROJECT_ROOT / "results", exist_ok=True)


def load_v2_targets():
    mp = json.loads(VERIFIED_MAPPING_PATH.read_text(encoding="utf-8"))
    inst_by_file = defaultdict(list)
    for inst in mp["instances"]:
        inst_by_file[inst["file_name"]].append(inst)
    return sorted(inst_by_file.keys()), inst_by_file


def stage_yolo():
    sys.modules["pyarrow"] = None  # Avoid Windows pyarrow DLL access violation in pandas
    import av  # noqa: F401
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    from class_names import class_names
    from utils import predict_with_dual_preresize

    out_json = CACHE_DIR / "yolo_detections.json"
    if out_json.exists():
        print(f"[stage:yolo] Using cached {out_json.name}")
        return

    target_files, inst_by_file = load_v2_targets()
    full_to_short = {}
    for item in class_names:
        fn_cls = item["name"] if isinstance(item, dict) else str(item)
        full_to_short[fn_cls] = fn_cls.split(" (")[0]

    model = YOLO(str(YOLO_WEIGHTS))
    model.overrides["device"] = "cpu"

    yolo_data = {}
    for idx, fn in enumerate(target_files):
        img_path = inst_by_file[fn][0]["image_path"]
        pil_upright = ImageOps.exif_transpose(Image.open(img_path))
        t0 = time.perf_counter()
        res = predict_with_dual_preresize(model, pil_upright, conf=0.30)[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        preds = []
        for b in res.boxes:
            cid = int(b.cls[0].item())
            if cid < 0 or cid >= len(class_names):
                continue
            full_name = class_names[cid]["name"]
            if full_name == "Con nguoi (Human)":
                continue
            conf_val = float(b.conf[0].item())
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            preds.append({
                "bbox_xyxy": [round(float(v), 2) for v in xyxy[:4]],
                "class_id": cid,
                "class_name": full_name,
                "short_class": full_to_short.get(full_name, full_name),
                "confidence": round(conf_val, 4),
            })
        yolo_data[fn] = {"detect_ms": round(dt_ms, 2), "preds": preds}
        if (idx + 1) % 50 == 0 or (idx + 1) == len(target_files):
            print(f"[stage:yolo] [{idx+1}/{len(target_files)}] {fn}: {len(preds)} dets ({dt_ms:.1f} ms)", flush=True)

    out_json.write_text(json.dumps(yolo_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage:yolo] Saved {len(yolo_data)} scenes to {out_json}")


def stage_mobilesam():
    import torch
    import av  # noqa: F401
    from ultralytics.utils import ops
    from food_segmentation import MobileSAMSegmenter

    meta_path = CACHE_DIR / "mobilesam_meta.json"
    if meta_path.exists():
        print("[stage:mobilesam] Using cached mobilesam_meta.json")
        return

    target_files, inst_by_file = load_v2_targets()
    yolo_data = json.loads((CACHE_DIR / "yolo_detections.json").read_text(encoding="utf-8"))

    msam_seg = MobileSAMSegmenter()
    msam_seg._load_model()
    predictor = msam_seg._predictor
    device = str(predictor.device)

    def run_boxes(image_rgb, boxes_xyxy):
        h_orig, w_orig = image_rgb.shape[:2]
        if not boxes_xyxy:
            return {
                "raw_masks": np.zeros((0, h_orig, w_orig), dtype=bool),
                "cand_raw_masks": np.zeros((0, h_orig, w_orig), dtype=bool),
                "timing_ms": {"preprocess": 0.0, "encode": 0.0, "decode": 0.0, "postprocess": 0.0, "total": 0.0},
                "peak_vram_mib": 0.0,
                "cpu_ram_mib": get_cpu_ram_mib(),
            }
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        predictor.reset_image()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        predictor.batch = [["in_memory.jpg"], [image_bgr], None]
        with torch.inference_mode():
            im_tensor = predictor.preprocess([image_bgr])
            predictor.im = im_tensor
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pre = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            predictor.features = predictor.get_im_features(im_tensor)
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_enc = (time.perf_counter() - t0) * 1000.0

        decoded_list = []
        t_dec = 0.0
        for bbox in boxes_xyxy:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
            t0 = time.perf_counter()
            with torch.inference_mode():
                pred_logits, pred_scores = predictor.prompt_inference(
                    predictor.im,
                    bboxes=[[x1, y1, x2, y2]],
                    multimask_output=True,
                )
            if "cuda" in device and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_dec += (time.perf_counter() - t0) * 1000.0
            decoded_list.append((pred_logits, pred_scores, bbox_area))

        t0 = time.perf_counter()
        raw_masks, cand_raw_masks = [], []
        with torch.inference_mode():
            for pred_logits, pred_scores, bbox_area in decoded_list:
                scaled_logits = ops.scale_masks(
                    pred_logits[None].float(), (h_orig, w_orig), padding=False
                )[0]
                cand_masks = (
                    (scaled_logits > predictor.model.mask_threshold)
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
                scores_np = pred_scores.cpu().numpy()
                best_idx = int(np.argmax(scores_np))
                raw_masks.append(cand_masks[best_idx].copy())

                cand_list = [cand_masks[i] for i in range(cand_masks.shape[0])]
                valid_cands = [m for m in cand_list if (m.sum() / bbox_area) >= msam_seg.min_mask_threshold]
                cand_full = (
                    min(valid_cands, key=lambda m: m.sum()).copy()
                    if valid_cands
                    else max(cand_list, key=lambda m: m.sum()).copy()
                )
                cand_raw_masks.append(cand_full)

        predictor.reset_image()
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_post = (time.perf_counter() - t0) * 1000.0
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
        return {
            "raw_masks": np.stack(raw_masks, axis=0),
            "cand_raw_masks": np.stack(cand_raw_masks, axis=0),
            "timing_ms": {
                "preprocess": round(t_pre, 2),
                "encode": round(t_enc, 2),
                "decode": round(t_dec, 2),
                "postprocess": round(t_post, 2),
                "total": round(t_pre + t_enc + t_dec + t_post, 2),
            },
            "peak_vram_mib": peak_vram,
            "cpu_ram_mib": get_cpu_ram_mib(),
        }

    # Warm-up on first 2 images
    for w_fn in target_files[:min(2, len(target_files))]:
        w_insts = inst_by_file[w_fn]
        w_img = cv2.cvtColor(cv2.imread(w_insts[0]["image_path"]), cv2.COLOR_BGR2RGB)
        run_boxes(w_img, [x["bbox_xyxy"] for x in w_insts])

    meta = {}
    for idx, fn in enumerate(target_files):
        stem = Path(fn).stem
        npz_path = CACHE_DIR / f"mobilesam_{stem}.npz"
        insts = inst_by_file[fn]
        y_preds = yolo_data[fn]["preds"]

        img_rgb = cv2.cvtColor(cv2.imread(insts[0]["image_path"]), cv2.COLOR_BGR2RGB)
        oracle_out = run_boxes(img_rgb, [x["bbox_xyxy"] for x in insts])
        yolo_out = run_boxes(img_rgb, [p["bbox_xyxy"] for p in y_preds])

        np.savez_compressed(
            npz_path,
            oracle_raw=oracle_out["raw_masks"],
            oracle_cand_raw=oracle_out["cand_raw_masks"],
            yolo_cand_raw=yolo_out["cand_raw_masks"],
        )
        meta[fn] = {
            "oracle_timing": oracle_out["timing_ms"],
            "oracle_peak_vram_mib": oracle_out["peak_vram_mib"],
            "oracle_cpu_ram_mib": oracle_out["cpu_ram_mib"],
            "yolo_timing": yolo_out["timing_ms"],
            "yolo_peak_vram_mib": yolo_out["peak_vram_mib"],
        }
        del img_rgb, oracle_out, yolo_out
        if (idx + 1) % 50 == 0 or (idx + 1) == len(target_files):
            print(
                f"[stage:mobilesam] [{idx+1}/{len(target_files)}] {fn}: "
                f"{meta[fn]['oracle_timing']['total']:.1f} ms | VRAM={meta[fn]['oracle_peak_vram_mib']} MiB",
                flush=True,
            )

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[stage:mobilesam] Saved meta for {len(meta)} scenes")


def summarize_mass_groups(records):
    abs_errs = [abs(r["estimated_mass"] - r["GT_mass"]) for r in records]
    apes = [abs(r["estimated_mass"] - r["GT_mass"]) / r["GT_mass"] * 100.0 for r in records]
    pes = [(r["estimated_mass"] - r["GT_mass"]) / r["GT_mass"] * 100.0 for r in records]
    sq_errs = [(r["estimated_mass"] - r["GT_mass"]) ** 2 for r in records]
    sum_pred = sum(r["estimated_mass"] for r in records)
    sum_gt = sum(r["GT_mass"] for r in records)

    by_cls = defaultdict(list)
    for r in records:
        by_cls[r["dish_class"]].append(r)

    per_class = {}
    for c, items in sorted(by_cls.items()):
        c_abs = [abs(x["estimated_mass"] - x["GT_mass"]) for x in items]
        c_ape = [abs(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in items]
        c_pe = [(x["estimated_mass"] - x["GT_mass"]) / x["GT_mass"] * 100.0 for x in items]
        per_class[c] = {
            "n_groups": len(items),
            "gt_mean_g": st.mean(x["GT_mass"] for x in items),
            "pred_mean_g": st.mean(x["estimated_mass"] for x in items),
            "mae_g": st.mean(c_abs),
            "mape_pct": st.mean(c_ape),
            "mpe_pct": st.mean(c_pe),
        }

    return {
        "n_groups": len(records),
        "mae_g": st.mean(abs_errs),
        "mape_pct": st.mean(apes),
        "rmse_g": float(np.sqrt(st.mean(sq_errs))),
        "mpe_pct": st.mean(pes),
        "bias_mean_pct": (sum_pred - sum_gt) / sum_gt * 100.0 if sum_gt > 0 else 0.0,
        "per_class": per_class,
    }


def stage_eval():
    from volume_nutrition import VolumeNutritionEstimator

    target_files, inst_by_file = load_v2_targets()
    coco = json.loads(COCO_JSON_PATH.read_text(encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    del coco
    gc.collect()

    yolo_data = json.loads((CACHE_DIR / "yolo_detections.json").read_text(encoding="utf-8"))
    ms_meta = json.loads((CACHE_DIR / "mobilesam_meta.json").read_text(encoding="utf-8"))
    z_marker_map = json.loads((CACHE_DIR / "z_marker_cache.json").read_text(encoding="utf-8"))
    vol_est = VolumeNutritionEstimator()

    # Detection counters
    th1_total = th2_total = fn_total = fp_total = total_preds = total_gts = 0
    # Segmentation metrics
    raw_metrics_list = []
    c_metrics_list = []
    # Volume/Mass records
    gt_group_recs = []
    ms_group_recs = []
    e2e_group_recs = []

    gt_meal_recs = []
    ms_meal_recs = []
    e2e_meal_recs = []

    for idx, fn in enumerate(target_files):
        stem = Path(fn).stem
        insts = inst_by_file[fn]
        h_full, w_full = int(insts[0]["height"]), int(insts[0]["width"])

        sc_npz = np.load(DEPTH_CACHE_DIR / f"scene_{stem}.npz", allow_pickle=True)
        depth_mm = sc_npz["depth_mm"].astype(np.float32)
        K = sc_npz["K"]
        mm_per_pixel = float(sc_npz["mm_per_pixel"])
        scale_source = str(sc_npz["scale_source"])
        depth_anchored = bool(sc_npz["depth_anchored"])
        h_vol = int(sc_npz["h_vol"])
        w_vol = int(sc_npz["w_vol"])
        sx = float(sc_npz["sx"])
        sy = float(sc_npz["sy"])
        sc_npz.close()

        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        oracle_raw = ms_npz["oracle_raw"].astype(bool)
        oracle_c = ms_npz["oracle_cand_raw"].astype(bool)
        yolo_c = ms_npz["yolo_cand_raw"].astype(bool)
        ms_npz.close()

        # 1. Seg metrics + Oracle GT / MobileSAM C volume
        gt_ests = []
        ms_ests = []
        zm = z_marker_map.get(fn, {}).get("z_marker_depth_mm")
        gt_scene_classes = [x["vietfood67_class_name"] for x in insts]
        for i_inst, inst in enumerate(insts):
            gt_full = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
            raw_m = compute_mask_metrics(oracle_raw[i_inst], gt_full, inst["bbox_xyxy"])
            c_m = compute_mask_metrics(oracle_c[i_inst], gt_full, inst["bbox_xyxy"])
            raw_metrics_list.append(raw_m)
            c_metrics_list.append(c_m)

            gt_vol = (
                cv2.resize(gt_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                if gt_full.shape[:2] != (h_vol, w_vol)
                else gt_full
            )
            ms_vol = (
                cv2.resize(oracle_c[i_inst].astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                if oracle_c[i_inst].shape[:2] != (h_vol, w_vol)
                else oracle_c[i_inst]
            )
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))

            e_gt = vol_est.estimate(
                depth_map_mm=depth_mm,
                food_mask=gt_vol,
                class_name=inst["vietfood67_class_name"],
                mm_per_pixel=mm_per_pixel,
                scale_source=scale_source,
                bbox=b_vol,
                K=K,
                depth_anchored=depth_anchored,
                z_marker_mm=zm,
                scene_classes=gt_scene_classes,
            )
            e_ms = vol_est.estimate(
                depth_map_mm=depth_mm,
                food_mask=ms_vol,
                class_name=inst["vietfood67_class_name"],
                mm_per_pixel=mm_per_pixel,
                scale_source=scale_source,
                bbox=b_vol,
                K=K,
                depth_anchored=depth_anchored,
                z_marker_mm=zm,
                scene_classes=gt_scene_classes,
            )
            gt_ests.append(e_gt.mass_g)
            ms_ests.append(e_ms.mass_g)

        # 2. YOLO Detection + E2E Volume (normalize CamelCase vs space class names for TH1)
        def _norm_cls(s: str) -> str:
            return s.split("(")[0].strip().lower().replace(" ", "")

        y_preds = yolo_data[fn]["preds"]
        y_scene_classes = [p["class_name"] for p in y_preds]
        total_preds += len(y_preds)
        total_gts += len(insts)
        y_preds_norm = [{**p, "short_class": _norm_cls(p["short_class"])} for p in y_preds]
        insts_norm = [{**g, "short_class": _norm_cls(g["short_class"])} for g in insts]
        awr_m, rem_p1, rem_g1 = match_boxes_one_to_one(y_preds_norm, insts_norm, iou_thr=0.5, require_same_class=True)
        rem_preds = [y_preds_norm[i] for i in rem_p1]
        rem_gts = [insts_norm[j] for j in rem_g1]
        sub_agn, rem_p2, rem_g2 = match_boxes_one_to_one(rem_preds, rem_gts, iou_thr=0.5, require_same_class=False)
        th2_m = [(rem_p1[spi], rem_g1[sgi], biou) for spi, sgi, biou in sub_agn]

        th1_total += len(awr_m)
        th2_total += len(th2_m)
        fn_total += len(rem_g2)
        fp_total += len(rem_p2)
        pred_for_gt = {gi: pi for pi, gi, _ in (awr_m + th2_m)}

        y_ests = []
        for pi, p in enumerate(y_preds):
            ym_full = yolo_c[pi]
            ym_vol = (
                cv2.resize(ym_full.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool)
                if ym_full.shape[:2] != (h_vol, w_vol)
                else ym_full
            )
            bx1, by1, bx2, by2 = p["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
            e_y = vol_est.estimate(
                depth_map_mm=depth_mm,
                food_mask=ym_vol,
                class_name=p["class_name"],
                mm_per_pixel=mm_per_pixel,
                scale_source=scale_source,
                bbox=b_vol,
                K=K,
                depth_anchored=depth_anchored,
                z_marker_mm=zm,
                scene_classes=y_scene_classes,
            )
            y_ests.append(e_y.mass_g)

        # 3. Aggregate into Mass Groups & Full-Meal
        grouped_buckets = defaultdict(list)
        meal_gt_g = 0.0
        meal_complete = True
        meal_only_gt = None

        for gi, inst in enumerate(insts):
            stt = inst["mapping_status"]
            if stt == "exact_1to1":
                gt_w = float(inst["gt_weight_g"])
                meal_gt_g += gt_w
                gt_group_recs.append({"dish_class": inst["short_class"], "estimated_mass": gt_ests[gi], "GT_mass": gt_w})
                ms_group_recs.append({"dish_class": inst["short_class"], "estimated_mass": ms_ests[gi], "GT_mass": gt_w})
                if gi in pred_for_gt:
                    e2e_group_recs.append({
                        "dish_class": inst["short_class"],
                        "estimated_mass": y_ests[pred_for_gt[gi]],
                        "GT_mass": gt_w,
                    })
            elif stt == "grouped_multi_instance":
                grouped_buckets[inst["short_class"]].append(gi)
            elif stt == "meal_total_only":
                meal_only_gt = float(inst["meal_total_g"])
            else:
                meal_complete = False

        for s_cls, g_indices in grouped_buckets.items():
            gt_w = float(insts[g_indices[0]]["gt_weight_g"])
            meal_gt_g += gt_w
            gt_group_recs.append({"dish_class": s_cls, "estimated_mass": sum(gt_ests[i] for i in g_indices), "GT_mass": gt_w})
            ms_group_recs.append({"dish_class": s_cls, "estimated_mass": sum(ms_ests[i] for i in g_indices), "GT_mass": gt_w})
            if all(i in pred_for_gt for i in g_indices):
                e2e_group_recs.append({
                    "dish_class": s_cls,
                    "estimated_mass": sum(y_ests[pred_for_gt[i]] for i in g_indices),
                    "GT_mass": gt_w,
                })

        if meal_only_gt is not None:
            gt_meal_recs.append((sum(gt_ests), meal_only_gt))
            ms_meal_recs.append((sum(ms_ests), meal_only_gt))
            e2e_meal_recs.append((sum(y_ests), meal_only_gt))
        elif meal_complete and meal_gt_g > 0:
            gt_meal_recs.append((sum(gt_ests), meal_gt_g))
            ms_meal_recs.append((sum(ms_ests), meal_gt_g))
            e2e_meal_recs.append((sum(y_ests), meal_gt_g))

        del depth_mm, oracle_raw, oracle_c, yolo_c
        if (idx + 1) % 100 == 0:
            gc.collect()

    # Compute Seg Summary
    def seg_summary(mlist):
        inter_sum = sum(round(m["iou"] * (m["pred_area"] + m["gt_area"]) / (1.0 + m["iou"])) if m["iou"] > 0 else 0 for m in mlist)
        # Exact micro mIoU via pred_area/gt_area/iou: inter / (pred + gt - inter) = iou => inter = iou*(pred+gt)/(1+iou)
        # Better: compute exact inter & union
        tot_inter = 0.0
        tot_union = 0.0
        for m in mlist:
            if m["iou"] > 0:
                inter = (m["iou"] * (m["pred_area"] + m["gt_area"])) / (1.0 + m["iou"])
                union = m["pred_area"] + m["gt_area"] - inter
            else:
                inter = 0.0
                union = m["pred_area"] + m["gt_area"]
            tot_inter += inter
            tot_union += union
        return {
            "micro_miou": tot_inter / tot_union if tot_union > 0 else 0.0,
            "macro_miou": st.mean(m["iou"] for m in mlist),
            "dice": st.mean(m["dice"] for m in mlist),
            "biou": st.mean(m["biou"] for m in mlist),
            "seg_success_pct": sum(1 for m in mlist if m["seg_success_050"]) / len(mlist) * 100.0,
        }

    raw_seg = seg_summary(raw_metrics_list)
    c_seg = seg_summary(c_metrics_list)
    latencies = [ms_meta[fn]["oracle_timing"]["total"] for fn in target_files]
    vrams = [ms_meta[fn]["oracle_peak_vram_mib"] for fn in target_files]

    gt_vol_summ = summarize_mass_groups(gt_group_recs)
    ms_vol_summ = summarize_mass_groups(ms_group_recs)
    e2e_vol_summ = summarize_mass_groups(e2e_group_recs)

    def meal_summary(pairs):
        errs = [abs(p - g) for p, g in pairs]
        apes = [abs(p - g) / g * 100.0 for p, g in pairs]
        return len(pairs), st.mean(errs), st.mean(apes)

    gt_m_n, gt_m_mae, gt_m_mape = meal_summary(gt_meal_recs)
    ms_m_n, ms_m_mae, ms_m_mape = meal_summary(ms_meal_recs)
    e2e_m_n, e2e_m_mae, e2e_m_mape = meal_summary(e2e_meal_recs)

    print("=== BENCHMARK V2 (418 IMAGES / 552 INSTANCES) ===")
    print(f"YOLO Det: GT={total_gts}, Pred={total_preds}, TH1={th1_total}, TH2={th2_total}, FN={fn_total}, FP={fp_total}")
    print(f"MobileSAM Raw: Micro mIoU={raw_seg['micro_miou']:.4f}, BIoU={raw_seg['biou']:.4f}, Succ={raw_seg['seg_success_pct']:.1f}%")
    print(f"MobileSAM Config C: Micro mIoU={c_seg['micro_miou']:.4f}, Macro={c_seg['macro_miou']:.4f}, BIoU={c_seg['biou']:.4f}, Succ={c_seg['seg_success_pct']:.1f}%")
    print(f"Latency: Mean={st.mean(latencies):.1f} ms, P50={st.median(latencies):.1f} ms | Peak VRAM={max(vrams):.1f} MiB")
    print(f"Oracle GT Mask ({gt_vol_summ['n_groups']} groups): MAE={gt_vol_summ['mae_g']:.2f}g | MAPE={gt_vol_summ['mape_pct']:.2f}% | RMSE={gt_vol_summ['rmse_g']:.2f}g | MPE={gt_vol_summ['mpe_pct']:+.2f}% | Bias={gt_vol_summ['bias_mean_pct']:+.2f}%")
    print(f"Oracle MobileSAM C ({ms_vol_summ['n_groups']} groups): MAE={ms_vol_summ['mae_g']:.2f}g | MAPE={ms_vol_summ['mape_pct']:.2f}% | RMSE={ms_vol_summ['rmse_g']:.2f}g | MPE={ms_vol_summ['mpe_pct']:+.2f}% | Bias={ms_vol_summ['bias_mean_pct']:+.2f}%")
    print(f"E2E YOLO+MobileSAM C ({e2e_vol_summ['n_groups']} matched groups): MAE={e2e_vol_summ['mae_g']:.2f}g | MAPE={e2e_vol_summ['mape_pct']:.2f}% | MPE={e2e_vol_summ['mpe_pct']:+.2f}%")
    print(f"Full-Meal ({gt_m_n} meals): GT MAE={gt_m_mae:.2f}g ({gt_m_mape:.2f}%) | MS MAE={ms_m_mae:.2f}g ({ms_m_mape:.2f}%) | E2E MAE={e2e_m_mae:.2f}g ({e2e_m_mape:.2f}%)")

    # Write Markdown Report
    md = [
        "# Báo Cáo Benchmark Mở Rộng Toàn Hệ Thống (418 Ảnh / 552 Instances / 55 Lớp)",
        "",
        "> [!IMPORTANT]",
        "> **Tóm tắt điều hành (Executive Summary)**",
        f"> Đánh giá toàn diện hệ thống **YOLOv10b + MobileSAM (Config C) + Depth Anything V2 + Volume A1–A5** trên tập dữ liệu mở rộng gồm **418 ảnh** (**552 GT instances**, **55 lớp món ăn** có nhãn phân đoạn, **500 nhóm khối lượng đơn/gộp** + **6 bữa `BunDau` tổng bữa**, tổng **418 bữa ăn** có khối lượng thực tế).",
        "> - **Nhận diện ArUco 4x4 (30 mm):** Thành công **402 / 418 ảnh (`96.2%`)**, neo độ sâu thành công **359 / 418 ảnh (`85.9%`)**.",
        f"> - **Phân đoạn MobileSAM (Config C):** Micro mIoU = **`{c_seg['micro_miou']:.4f}`**, Macro mIoU = **`{c_seg['macro_miou']:.4f}`**, Boundary IoU = **`{c_seg['biou']:.4f}`**, Seg Success (`IoU >= 0.50`) = **`{c_seg['seg_success_pct']:.1f}%`**, Median P50 Latency = **`{st.median(latencies):.1f} ms`** (Mean **`{st.mean(latencies):.1f} ms`**), Peak VRAM = **`{max(vrams):.1f} MiB`**.",
        f"> - **Ước lượng Khối lượng (`Volume A1–A5`) trên {gt_vol_summ['n_groups']} nhóm khối lượng:**",
        f">   - **Oracle GT Mask:** MAE = **`{gt_vol_summ['mae_g']:.2f} g`**, MAPE = **`{gt_vol_summ['mape_pct']:.2f}%`**, MPE = **`{gt_vol_summ['mpe_pct']:+.2f}%`**, Bias = **`{gt_vol_summ['bias_mean_pct']:+.2f}%`**.",
        f">   - **Oracle MobileSAM Config C:** MAE = **`{ms_vol_summ['mae_g']:.2f} g`**, MAPE = **`{ms_vol_summ['mape_pct']:.2f}%`**, MPE = **`{ms_vol_summ['mpe_pct']:+.2f}%`**.",
        f">   - **End-to-End (`YOLO + MobileSAM C`) Matched-Only ({e2e_vol_summ['n_groups']} nhóm):** MAE = **`{e2e_vol_summ['mae_g']:.2f} g`**, MAPE = **`{e2e_vol_summ['mape_pct']:.2f}%`**.",
        f">   - **End-to-End Full-Meal ({e2e_m_n} bữa ăn):** MAE = **`{e2e_m_mae:.2f} g`**, MAPE = **`{e2e_m_mape:.2f}%`**.",
        "",
        "---",
        "",
        "## 1. Kết Quả Nhận Diện (YOLOv10b) & Phân Đoạn (MobileSAM Config C)",
        "",
        "### 1.1. Nhận diện đối tượng (YOLOv10b trên 418 ảnh / 552 GT boxes)",
        "| Chỉ số Detection | Giá trị |",
        "| :--- | :---: |",
        f"| Tổng GT boxes | `{total_gts}` |",
        f"| Tổng YOLO predictions | `{total_preds}` |",
        f"| **TH1 (Class-Aware TP, IoU ≥ 0.50)** | **`{th1_total}`** (`{th1_total/total_gts*100:.1f}%`) |",
        f"| **TH2 (Class-Agnostic TP, IoU ≥ 0.50)** | **`{th2_total}`** (`{th2_total/total_gts*100:.1f}%`) |",
        f"| Tổng Recall (`TH1 + TH2`) | **`{(th1_total+th2_total)/total_gts*100:.1f}%`** (`{th1_total+th2_total}/{total_gts}`) |",
        f"| Precision (`(TH1 + TH2) / Pred`) | **`{(th1_total+th2_total)/max(1,total_preds)*100:.1f}%`** |",
        f"| Bỏ sót (`FN`) / Dương tính giả (`FP`) | `{fn_total}` / `{fp_total}` |",
        "",
        "### 1.2. Chất lượng phân đoạn (`MobileSAM` Raw Top-1 vs `Config C` trên 552 instances)",
        "| Cấu hình MobileSAM | Micro mIoU | Macro mIoU | Dice | Boundary IoU | Seg Success (`≥ 0.50`) | Median P50 (ms) | Mean Latency (ms) | Peak VRAM (MiB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Raw Top-1** | `{raw_seg['micro_miou']:.4f}` | `{raw_seg['macro_miou']:.4f}` | `{raw_seg['dice']:.4f}` | `{raw_seg['biou']:.4f}` | `{raw_seg['seg_success_pct']:.1f}%` | `{st.median(latencies):.1f}` | `{st.mean(latencies):.1f}` | `{max(vrams):.1f}` |",
        f"| **Config C (`cand_filter`, no erosion)** | **`{c_seg['micro_miou']:.4f}`** | **`{c_seg['macro_miou']:.4f}`** | **`{c_seg['dice']:.4f}`** | **`{c_seg['biou']:.4f}`** | **`{c_seg['seg_success_pct']:.1f}%`** | **`{st.median(latencies):.1f}`** | **`{st.mean(latencies):.1f}`** | **`{max(vrams):.1f}`** |",
        "",
        "---",
        "",
        "## 2. Kết Quả Ước Lượng Thể Tích & Khối Lượng (`Volume A1–A5`)",
        "",
        "| Chế độ đánh giá Pipeline | Số đơn vị (`N`) | MAE (g) ↓ | MAPE (%) ↓ | RMSE (g) ↓ | MPE (%) | Bias_mean (%) | Full-Meal MAE (g) | Full-Meal MAPE (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **1. Oracle GT Polygon Mask** | `{gt_vol_summ['n_groups']}` nhóm | **`{gt_vol_summ['mae_g']:.2f}`** | **`{gt_vol_summ['mape_pct']:.2f}%`** | `{gt_vol_summ['rmse_g']:.2f}` | `{gt_vol_summ['mpe_pct']:+.2f}%` | `{gt_vol_summ['bias_mean_pct']:+.2f}%` | `{gt_m_mae:.2f}` (`N={gt_m_n}`) | `{gt_m_mape:.2f}%` |",
        f"| **2. Oracle Box + MobileSAM Config C** | `{ms_vol_summ['n_groups']}` nhóm | **`{ms_vol_summ['mae_g']:.2f}`** | **`{ms_vol_summ['mape_pct']:.2f}%`** | `{ms_vol_summ['rmse_g']:.2f}` | `{ms_vol_summ['mpe_pct']:+.2f}%` | `{ms_vol_summ['bias_mean_pct']:+.2f}%` | `{ms_m_mae:.2f}` (`N={ms_m_n}`) | `{ms_m_mape:.2f}%` |",
        f"| **3. End-to-End (`YOLO + MobileSAM C`)** | `{e2e_vol_summ['n_groups']}` nhóm khớp | **`{e2e_vol_summ['mae_g']:.2f}`** | **`{e2e_vol_summ['mape_pct']:.2f}%`** | `{e2e_vol_summ['rmse_g']:.2f}` | `{e2e_vol_summ['mpe_pct']:+.2f}%` | `{e2e_vol_summ['bias_mean_pct']:+.2f}%` | `{e2e_m_mae:.2f}` (`N={e2e_m_n}`) | `{e2e_m_mape:.2f}%` |",
        "",
        "---",
        "",
        "## 3. Chi Tiết Theo Từng Lớp Món Ăn (`Oracle GT Mask` vs `Oracle MobileSAM Config C`)",
        "",
        "| Lớp món ăn (`dish_class`) | `N` nhóm | GT Mean (g) | GT-Mask Pred (g) | GT-Mask MAE (g) | GT-Mask MAPE (%) | MobileSAM C MAE (g) | MobileSAM C MAPE (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for c in sorted(gt_vol_summ["per_class"].keys()):
        g = gt_vol_summ["per_class"][c]
        m = ms_vol_summ["per_class"][c]
        md.append(
            f"| **`{c}`** | `{g['n_groups']}` | `{g['gt_mean_g']:.1f}` | `{g['pred_mean_g']:.1f}` | `{g['mae_g']:.1f}` | `{g['mape_pct']:.1f}%` | `{m['mae_g']:.1f}` | `{m['mape_pct']:.1f}%` |"
        )

    report_str = "\n".join(md) + "\n"
    REPORT_OUT_PATH.write_text(report_str, encoding="utf-8")
    REPORT_SCRATCH_PATH.write_text(report_str, encoding="utf-8")
    print(f"Saved report to {REPORT_OUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["yolo", "mobilesam", "eval"], required=True)
    args = parser.parse_args()
    if args.stage == "yolo":
        stage_yolo()
    elif args.stage == "mobilesam":
        stage_mobilesam()
    elif args.stage == "eval":
        stage_eval()


if __name__ == "__main__":
    main()
