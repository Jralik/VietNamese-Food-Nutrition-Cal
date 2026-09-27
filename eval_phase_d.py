"""Phase D — 54-image validation, single protocol (frozen metric definitions).

Configs (all through the same evaluator, same GT, same matching rules):
  v26m-dual      : v26m ONNX, upright stretch+letterbox union
  v10b-dual      : v10b .pt,  upright stretch+letterbox union
  v10b-dual+raw  : v10b-dual + raw-orientation stretch run (orient != 1),
                   boxes mapped to upright coords via the EXIF transform

Inference runs ONCE per image/variant at conf=0.05; thresholds
{0.50, 0.40, 0.30} are applied locally on the shared prediction pools.

Metrics (frozen definitions):
  Recall (class-level GT): greedy one-to-one matching per class by confidence;
      matched = min(n_gt_c, n_det_c).
  FP/Extras: detections after merge not matched to any GT target
      (= total dets - total matched).
  0-box: images with zero detections at T.
  Same-class duplicate: kept-box pairs, same class, IoU > 0.7 (invariant: 0).
  Cross-class overlap: kept-box pairs, different class, IoU > 0.7 (diagnostic
      only — legitimate multi-class overlap exists; audit before any rule).
  Raw-only recovery: GT target class missed by the upright pool at T, with a
      raw-pool box of that class at >= T whose IoU <= 0.7 against EVERY
      upright-pool box (excludes conf-flips on already-found objects).
  Latency: mean wall time of the config's inference runs per image (CPU).

Writes scratch/phase_d_results.md in the frozen table format.
"""

import os
import time

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from eval_val_detection import (VAL_FILE, parse_val_file, pre_stretch,
                                pre_letterbox, to_orig_xyxy, iou_matrix)
from utils import _nms_class_aware, _map_boxes_exif_to_upright

ONNX26 = "./model/yolov26/best.onnx"
PT10B = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
OUT_PATH = os.path.join("scratch", "phase_d_results.md")
THRESHOLDS = (0.50, 0.40, 0.30)
RUN_CONF = 0.05
NMS_IOU = 0.55
DUP_IOU = 0.7
CONFIGS = ("v26m-dual", "v10b-dual", "v10b-dual+raw")


def predict(model, src):
    t0 = time.time()
    r = model.predict(src, conf=RUN_CONF, imgsz=640, half=False, verbose=False)[0]
    ms = (time.time() - t0) * 1000
    data = r.boxes.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    return data, ms


def merge_pools(pools):
    """pools: list of (data, xyxy_upright). Returns (N,6) [xyxy conf cls] after
    class-aware NMS."""
    datas = [d for d, _ in pools if len(d)]
    maps = [m for d, m in pools if len(d)]
    if not datas:
        return np.zeros((0, 6))
    xyxy = np.concatenate(maps, axis=0)
    extra = np.concatenate([d[:, 4:] for d in datas], axis=0)
    keep = _nms_class_aware(xyxy, extra[:, 0], extra[:, 1])
    return np.concatenate([xyxy[keep], extra[keep]], axis=1)


def evaluate_dets(final, expected, up_dets, raw_pool, thr):
    """Metrics for one image at threshold `thr` (frozen definitions).

    `expected` = {class_id: gt_box_count}; greedy one-to-one per class:
    matched_c = min(n_gt_c, n_det_c).
    """
    out = {}
    n_det = len(final)
    det_cls = final[:, 5].astype(int) if n_det else np.array([], int)
    matched = 0
    gt_total = sum(expected.values())
    for c, n_gt in expected.items():
        n_dc = int((det_cls == c).sum())
        matched += min(n_gt, n_dc)
    out["recall_num"] = matched
    out["recall_den"] = gt_total
    out["extras"] = n_det - matched
    out["zero_box"] = 1 if n_det == 0 else 0
    # duplicates / overlaps among kept boxes
    same_dup = cross_ov = 0
    if n_det >= 2:
        ious = iou_matrix(final[:, :4], final[:, :4])
        for i in range(n_det):
            for j in range(i + 1, n_det):
                if ious[i, j] > DUP_IOU:
                    if det_cls[i] == det_cls[j]:
                        same_dup += 1
                    else:
                        cross_ov += 1
    out["same_dup"] = same_dup
    out["cross_ov"] = cross_ov
    # raw-only recovery: upright-missed target class, raw box correct class at
    # >= thr, IoU <= 0.7 vs EVERY upright-pool box (excludes conf-flips)
    up_cls = up_dets[:, 5].astype(int) if len(up_dets) else np.array([], int)
    recovery_num = 0
    recovery_den = 0
    if raw_pool is not None:
        raw_cls = raw_pool[:, 5].astype(int) if len(raw_pool) else np.array([], int)
        for c, n_gt in expected.items():
            if c in up_cls:
                continue  # upright found it — not a recovery candidate
            raw_c = raw_pool[raw_cls == c] if len(raw_pool) else np.zeros((0, 6))
            raw_c = raw_c[raw_c[:, 4] >= thr]
            if not len(raw_c):
                continue
            recovery_den += n_gt
            if len(up_dets):
                ious = iou_matrix(raw_c[:, :4], up_dets[:, :4])
                novel = (ious <= DUP_IOU).all(axis=1)
            else:
                novel = np.ones(len(raw_c), dtype=bool)
            recovery_num += min(n_gt, int(novel.sum())) if novel.any() else 0
    out["rec_num"] = recovery_num
    out["rec_den"] = recovery_den
    return out


def main():
    entries = parse_val_file(VAL_FILE)
    m26 = YOLO(ONNX26, task="detect")
    m10 = YOLO(PT10B)
    for m in (m26, m10):
        try:
            m.overrides["device"] = "cpu"
        except Exception:
            pass

    # per config per T: accumulators
    acc = {c: {t: {"recall_num": 0, "recall_den": 0, "extras": 0, "zero_box": 0,
                   "same_dup": 0, "cross_ov": 0, "rec_num": 0, "rec_den": 0}
               for t in THRESHOLDS}
           for c in CONFIGS}
    lat = {c: [] for c in CONFIGS}
    per_class = {c: {} for c in CONFIGS}  # config -> {(T, cls): [hit, tot]}
    n_images = 0

    for idx, (img_path, expected) in enumerate(entries):
        if not os.path.exists(img_path):
            continue
        n_images += 1
        raw = Image.open(img_path)
        try:
            orient = int(raw.getexif().get(274, 1) or 1)
        except Exception:
            orient = 1
        upright = ImageOps.exif_transpose(raw)
        w_up, h_up = upright.size
        w_raw, h_raw = raw.size

        # inference: v26m upright runs; v10b upright runs; v10b raw run (orient != 1)
        dS26, t1 = predict(m26, pre_stretch(upright))
        dL26, t2 = predict(m26, pre_letterbox(upright))
        dS10, t3 = predict(m10, pre_stretch(upright))
        dL10, t4 = predict(m10, pre_letterbox(upright))
        dR10, t5 = (np.zeros((0, 6)), 0.0)
        if orient != 1:
            dR10, t5 = predict(m10, pre_stretch(raw))

        # upright pools (already in upright coords)
        up26 = merge_pools([(dS26, to_orig_xyxy(dS26, "stretch", w_up, h_up)),
                            (dL26, to_orig_xyxy(dL26, "letterbox", w_up, h_up))])
        up10 = merge_pools([(dS10, to_orig_xyxy(dS10, "stretch", w_up, h_up)),
                            (dL10, to_orig_xyxy(dL10, "letterbox", w_up, h_up))])
        raw10 = np.zeros((0, 6))
        if orient != 1 and len(dR10):
            xyxy_r = to_orig_xyxy(dR10, "stretch", w_raw, h_raw)
            xyxy_r, _ = _map_boxes_exif_to_upright(xyxy_r, orient, w_raw, h_raw)
            raw10 = np.concatenate([np.asarray(xyxy_r, dtype=np.float64),
                                    dR10[:, 4:]], axis=1)

        finals = {
            "v26m-dual": up26,
            "v10b-dual": up10,
            "v10b-dual+raw": merge_pools([(up10, up10[:, :4]),
                                          (raw10, raw10[:, :4])]),
        }
        lats = {
            "v26m-dual": t1 + t2,
            "v10b-dual": t3 + t4,
            "v10b-dual+raw": t3 + t4 + t5,
        }

        for cfg in CONFIGS:
            final = finals[cfg]
            lat[cfg].append(lats[cfg])
            up_dets = up26 if cfg == "v26m-dual" else up10
            raw_pool = raw10 if cfg == "v10b-dual+raw" else None
            for T in THRESHOLDS:
                sel = final[final[:, 4] >= T] if len(final) else np.zeros((0, 6))
                up_sel = up_dets[up_dets[:, 4] >= T] if len(up_dets) else np.zeros((0, 6))
                m = evaluate_dets(sel, expected, up_sel, raw_pool, T)
                for k, v in m.items():
                    acc[cfg][T][k] += v
                # per-class recall (GT now carries per-class box counts)
                det_cls = sel[:, 5].astype(int) if len(sel) else []
                for c, cnt in expected.items():
                    key = (T, c)
                    pc = per_class[cfg].setdefault(key, [0, 0])
                    pc[1] += cnt
                    n_dc = int((np.asarray(det_cls) == c).sum())
                    pc[0] += min(cnt, n_dc)
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(entries)}")

    # ── report ──
    lines = ["", "# Phase D — 54-image validation (single protocol, frozen metrics)", ""]
    lines.append("Matching: class-level GT **with per-class box counts**, greedy one-to-one "
                 "per class by confidence (matched_c = min(n_gt_c, n_det_c)). "
                 "Raw-only recovery: upright-missed target class with a raw box of the correct "
                 "class at >= T and IoU <= 0.7 vs every upright box. Inference once at "
                 f"conf={RUN_CONF}; thresholds applied locally on shared pools. CPU runtime. "
                 "v10b-dual+raw is kept as a documented ablation — the app path is dual-only "
                 "(raw branch rejected: all audited raw-only boxes were position FPs).")
    lines.append("")
    lines.append("| Threshold | Config | Recall | FP/Extras | 0-box | Same-class dup | Cross-class overlap | Raw-only recovery | Latency ms/img |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        for T in THRESHOLDS:
            a = acc[cfg][T]
            rec = a["recall_num"] / a["recall_den"] if a["recall_den"] else 0
            rec_s = f"{a['rec_num']}/{a['rec_den']}" if cfg == "v10b-dual+raw" else "n/a"
            lines.append(
                f"| {T:.2f} | {cfg} | {rec:.3f} ({a['recall_num']}/{a['recall_den']}) "
                f"| {a['extras']} | {a['zero_box']}/{n_images} | {a['same_dup']} "
                f"| {a['cross_ov']} | {rec_s} | {sum(lat[cfg]) / len(lat[cfg]):.0f} |")
    lines.append("")

    lines.append("## Per-class recall @0.50 (hits/targets)")
    lines.append("")
    lines.append("| Class | " + " | ".join(CONFIGS) + " |")
    lines.append("|---|" + "---|" * len(CONFIGS))
    all_keys = sorted({k for c in CONFIGS for k in per_class[c] if k[0] == 0.50})
    for (T, c) in all_keys:
        if T != 0.50:
            continue
        cells = []
        for cfg in CONFIGS:
            hit, tot = per_class[cfg].get((T, c), [0, 0])
            cells.append(f"{hit}/{tot}" if tot else "-")
        lines.append(f"| {class_names[c]['name']} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Per-class recall @0.30 (v10b-dual+raw)")
    lines.append("")
    lines.append("| Class | hits/targets |")
    lines.append("|---|---|")
    for (T, c) in sorted(k for k in per_class["v10b-dual+raw"] if k[0] == 0.30):
        hit, tot = per_class["v10b-dual+raw"][(T, c)]
        lines.append(f"| {class_names[c]['name']} | {hit}/{tot} |")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
