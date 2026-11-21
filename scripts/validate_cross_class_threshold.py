"""Cross-class IoU threshold validation (post-freeze correctness fix).

Bug being fixed: a single dish region can carry TWO surviving detections of
DIFFERENT classes (reproduced on bun-rieu (1).jpg: Bun bo Hue 0.955 + Bun rieu
0.354, IoU 0.988) because both the ultralytics internal NMS and the app-level
`_nms_class_aware` are class-aware. The duplicate matches the same volume
estimation and its nutrition is counted twice.

This script measures the cross-class IoU landscape on the validation set to
justify the suppression threshold. The detection pool is built exactly like
the production path BEFORE cross-class suppression:

    EXIF-transposed RGB
      -> two PIL BILINEAR pre-resizes (stretch 640x640 + letterbox 640x640)
      -> YOLOv10b PT predict @ conf 0.30, imgsz 640
      -> boxes mapped back to original-image coordinates
      -> class-aware NMS iou 0.55 (the frozen `_nms_class_aware`)

Every surviving pair of detections with DIFFERENT classes and IoU > 0 is
recorded and classified against the image GT class-set:

    true_pair  : both classes present in GT  -> risk of false suppression.
                 GT is class+count only (NO bounding boxes), so this is NOT
                 spatial false-suppression evidence; it bounds the IoU of
                 pairs that plausibly correspond to two real dishes.
    candidate  : exactly one class in GT     -> likely cross-class duplicate
                 of a true dish (the bun-rieu failure mode).
    noise      : neither class in GT.

Outputs
    scratch/cross_class_val_records.jsonl                   (one row per pair)
    evaluation_final/post_freeze/cross_class_threshold_validation.md

Run:  .venv/Scripts/python.exe scripts/validate_cross_class_threshold.py
"""
import argparse
import json
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from eval_val_detection import VAL_FILE, parse_val_file
from utils import load_model, _nms_class_aware

IMGSZ = 640
NMS_CLASS_AWARE_IOU = 0.55
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
OUT_JSONL = os.path.join(ROOT_DIR, "scratch", "cross_class_val_records.jsonl")
OUT_MD_DIR = os.path.join(ROOT_DIR, "evaluation_final", "post_freeze")


def detection_pool(model, pil_image, conf):
    """Production pool before cross-class suppression (see module docstring)."""
    from PIL import Image

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    w, h = pil_image.size

    stretch_img = pil_image.resize((IMGSZ, IMGSZ), Image.BILINEAR)
    r = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    pad_x, pad_y = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    letter_img = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    letter_img.paste(pil_image.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))

    res_s = model.predict(stretch_img, conf=conf, imgsz=IMGSZ, half=False, verbose=False)[0]
    res_l = model.predict(letter_img, conf=conf, imgsz=IMGSZ, half=False, verbose=False)[0]

    def data(res):
        d = res.boxes.data
        return d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)

    ds, dl = data(res_s), data(res_l)
    xyxy_s = ds[:, :4].copy()
    xyxy_s[:, [0, 2]] *= w / float(IMGSZ)
    xyxy_s[:, [1, 3]] *= h / float(IMGSZ)
    xyxy_l = dl[:, :4].copy()
    xyxy_l[:, [0, 2]] = (xyxy_l[:, [0, 2]] - pad_x) / r
    xyxy_l[:, [1, 3]] = (xyxy_l[:, [1, 3]] - pad_y) / r

    if len(ds) + len(dl) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = np.concatenate([xyxy_s, xyxy_l], axis=0)
    extra = np.concatenate([ds[:, 4:], dl[:, 4:]], axis=0)
    keep = _nms_class_aware(xyxy, extra[:, 0], extra[:, 1])
    return np.concatenate([xyxy[keep], extra[keep]], axis=1)


def pair_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.30, help="app default slider value")
    args = ap.parse_args()

    entries = parse_val_file(os.path.join(ROOT_DIR, VAL_FILE))
    print(f"val entries: {len(entries)}")

    model = load_model()
    records = []
    n_images_used, n_images_missing, n_dets = 0, 0, 0
    t0 = time.time()

    for img_path, gt_counts in entries:
        if not os.path.exists(img_path):
            n_images_missing += 1
            print(f"  [missing] {img_path}")
            continue
        from PIL import Image, ImageOps

        pil = ImageOps.exif_transpose(Image.open(img_path))
        pool = detection_pool(model, pil, args.conf)
        n_images_used += 1
        n_dets += len(pool)
        gt_set = set(int(c) for c in gt_counts)

        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if pool[i, 5] == pool[j, 5]:
                    continue
                iou = pair_iou(pool[i, :4], pool[j, :4])
                if iou <= 0:
                    continue
                in_gt = (int(pool[i, 5]) in gt_set) + (int(pool[j, 5]) in gt_set)
                category = {2: "true_pair", 1: "candidate", 0: "noise"}[in_gt]
                records.append({
                    "image": os.path.basename(img_path),
                    "folder": os.path.basename(os.path.dirname(img_path)),
                    "iou": round(float(iou), 4),
                    "category": category,
                    "class_i": int(pool[i, 5]), "conf_i": round(float(pool[i, 4]), 4),
                    "class_j": int(pool[j, 5]), "conf_j": round(float(pool[j, 4]), 4),
                    "gt_class_ids": sorted(gt_set),
                })
        done = n_images_used + n_images_missing
        if done % 10 == 0:
            print(f"  [{done}/{len(entries)}] {time.time() - t0:.0f}s elapsed")

    os.makedirs(OUT_MD_DIR, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    true_pairs = [r for r in records if r["category"] == "true_pair"]
    candidates = [r for r in records if r["category"] == "candidate"]
    noise = [r for r in records if r["category"] == "noise"]
    true_ious = sorted((r["iou"] for r in true_pairs), reverse=True)

    lines = [
        "# Cross-class IoU threshold validation",
        "",
        "**Post-freeze correctness evaluation** — không thay thế hay sửa số liệu Chapter 4 frozen.",
        "",
        f"- Protocol: dual-preresize (stretch + letterbox, PIL BILINEAR) + conf {args.conf}"
        f" + class-aware NMS IoU {NMS_CLASS_AWARE_IOU} — pool TRƯỚC cross-class suppression.",
        f"- Images: {n_images_used} dùng, {n_images_missing} thiếu file.",
        f"- Detections sau class-aware NMS: {n_dets}.",
        f"- Cross-class pairs (IoU > 0): {len(records)} — "
        f"true_pair {len(true_pairs)}, candidate {len(candidates)}, noise {len(noise)}.",
        "",
        "## Phân bố IoU của true_pair (cả hai lớp cùng có trong GT class-set)",
        "",
        "GT chỉ có lớp + số lượng (KHÔNG có bounding box), nên phân bố này **không phải**",
        "bằng chứng về spatial false-suppression; nó giới hạn trên IoU của các cặp",
        "có khả năng tương ứng hai món thật — ngưỡng phải nằm trên mức này.",
        "",
    ]
    if true_ious:
        p99 = float(np.percentile(true_ious, 99))
        lines += [
            f"- n = {len(true_ious)}, max = {true_ious[0]:.3f}, P99 = {p99:.3f}",
            "- Toàn bộ true_pair IoU (giảm dần): "
            + ", ".join(f"{v:.3f}" for v in true_ious),
        ]
    else:
        lines.append("- Không có cặp true_pair nào (các món thật trong val set không chồng nhau).")
    lines += ["", "## Số cặp bị suppress theo ngưỡng (pairwise approximation)", "",
              "| Ngưỡng | true_pair bị suppress (rủi ro) | candidate bị loại (lợi ích) | noise bị loại |", "|---|---|---|---|"]
    for t in THRESHOLDS:
        lines.append(
            f"| {t:.2f} "
            f"| {sum(1 for r in true_pairs if r['iou'] >= t)} "
            f"| {sum(1 for r in candidates if r['iou'] >= t)} "
            f"| {sum(1 for r in noise if r['iou'] >= t)} |")
    lines += ["", "## Chi tiết candidate (duplicate) tại ngưỡng 0.70 trở lên", ""]
    hi = [r for r in candidates if r["iou"] >= 0.70]
    if hi:
        lines += ["| Ảnh | IoU | lớp giữ | conf giữ | lớp bị loại | conf bị loại |", "|---|---|---|---|---|---|"]
        from class_names import class_names

        def cname(cid):
            c = class_names[int(cid)]
            return c["name"] if isinstance(c, dict) else str(c)
        for r in sorted(hi, key=lambda x: -x["iou"]):
            lines.append(
                f"| {r['folder']}/{r['image']} | {r['iou']:.3f} "
                f"| {cname(r['class_i'] if r['conf_i'] >= r['conf_j'] else r['class_j'])} "
                f"| {max(r['conf_i'], r['conf_j']):.2f} "
                f"| {cname(r['class_i'] if r['conf_i'] < r['conf_j'] else r['class_j'])} "
                f"| {min(r['conf_i'], r['conf_j']):.2f} |")
    else:
        lines.append("(không có)")
    lines += ["", f"Records: `scratch/cross_class_val_records.jsonl` ({len(records)} pairs).", ""]

    out_md = os.path.join(OUT_MD_DIR, "cross_class_threshold_validation.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_JSONL} ({len(records)} pairs)")
    print(f"wrote {out_md}")
    print(f"true_pair IoUs: {[f'{v:.3f}' for v in true_ious] or 'none'}")


if __name__ == "__main__":
    main()
