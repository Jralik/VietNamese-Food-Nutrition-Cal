"""Val-set detection evaluation: choose the preprocessing config with data.

Uses portion-estimation_val.txt (60 images, 10 dish groups, expected dish
classes per image) to score candidate preprocessing configs of the v26m ONNX
through the app-identical predict() path:

  D  app-current  : no pre-resize (cv2 letterbox inside ultralytics)
  A  stretch-BIL  : PIL BILINEAR stretch to 640x640 (approved plan)
  B  letterbox-BIL: PIL BILINEAR letterbox to 640x640 + 114 pad
  AB dual-union   : A + B boxes inverse-mapped to original coords, merged
                    with cross-run IoU NMS (0.55)
  REF v10b-stretch: the reference model + its reference preprocessing (ceiling)

Metrics per config and per dish group: class-level recall @0.50 and @0.30,
plus the number of images with ZERO boxes at 0.50 (the "Khong phat hien mon
an nao" case).

The inverse mappings implemented here (stretch: x*w/640; letterbox:
(x-pad)/scale) are exactly what the app fix will use.
"""

import os
import re
import time

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names

ONNX26 = "./model/yolov26/best.onnx"
PT10B = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
VAL_FILE = "portion-estimation_val.txt"
OUT_PATH = os.path.join("scratch", "ab_preprocess_results.md")

# val-file dish token -> class id (exact short-name match)
SHORT_TO_ID = {}
for _i, _item in enumerate(class_names):
    _short = (_item.get("name", "") if isinstance(_item, dict) else str(_item)).split(" (")[0]
    SHORT_TO_ID[_short] = _i


# val-file dish token aliases (GT text -> class short name)
TOKEN_ALIASES = {"Soup": "Canh"}


def parse_val_file(path):
    """Returns list of (img_path, {class_id: gt_box_count}).

    Counts all GT targets per class; supports "(tong)" totals, "?" count-only
    targets, and the legacy formats.
    """
    from eval_full_pipeline import parse_gt as _parse_full
    entries = []
    for img_path, gt in _parse_full(path):
        counts = {}
        for cid, v in gt.items():
            counts[cid] = counts.get(cid, 0) + v["count"]
        entries.append((img_path, counts))
    return entries


def pre_stretch(pil):
    return pil.resize((640, 640), Image.BILINEAR)


def pre_letterbox(pil):
    w, h = pil.size
    r = min(640 / w, 640 / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(pil.resize((nw, nh), Image.BILINEAR), ((640 - nw) // 2, (640 - nh) // 2))
    return canvas


def letterbox_params(w, h):
    r = min(640 / w, 640 / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (640 - nw) // 2, (640 - nh) // 2
    return r, px, py


def to_orig_xyxy(data, mode, w, h):
    """Map (N,6) boxes from 640-space to original-image coords (xyxy float)."""
    if len(data) == 0:
        return data[:, :4].reshape(0, 4)
    b = data[:, :4].copy()
    if mode == "stretch":
        b[:, [0, 2]] *= w / 640.0
        b[:, [1, 3]] *= h / 640.0
    else:  # letterbox
        r, px, py = letterbox_params(w, h)
        b[:, [0, 2]] = (b[:, [0, 2]] - px) / r
        b[:, [1, 3]] = (b[:, [1, 3]] - py) / r
    return b


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms_merge(boxes, scores, classes, iou_thr=0.55):
    """Greedy NMS over one merged pool; returns kept indices."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=int)


def predict(model, src):
    r = model.predict(src, conf=0.05, imgsz=640, half=False, verbose=False)[0]
    data = r.boxes.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    return data


def main():
    entries = parse_val_file(VAL_FILE)
    n_empty_e = sum(1 for _, e in entries if not e)
    print(f"parsed {len(entries)} val images ({n_empty_e} with no mappable dish token)")
    m26 = YOLO(ONNX26, task="detect")
    m10 = YOLO(PT10B)

    # per config: list of (expected_set, detected_classes_050, detected_030, n_boxes_050, empty)
    results = {k: [] for k in ("D app-current", "A stretch-BIL", "B letterbox-BIL",
                               "AB dual-union", "REF v10b-stretch")}
    t_start = time.time()
    for idx, (img_path, expected) in enumerate(entries):
        if not os.path.exists(img_path):
            print("MISSING:", img_path)
            continue
        pil = ImageOps.exif_transpose(Image.open(img_path))
        w, h = pil.size

        dA = predict(m26, pre_stretch(pil))
        dB = predict(m26, pre_letterbox(pil))
        dD = predict(m26, pil)

        xyxyA = to_orig_xyxy(dA, "stretch", w, h)
        xyxyB = to_orig_xyxy(dB, "letterbox", w, h)
        merged = np.concatenate([dA, dB], axis=0) if len(dA) + len(dB) else np.zeros((0, 6))
        if len(merged):
            mxy = np.concatenate([xyxyA, xyxyB], axis=0)
            keep = nms_merge(mxy, merged[:, 4], merged[:, 5])
            dU = merged[keep]
        else:
            dU = merged

        dR = predict(m10, pre_stretch(pil))  # reference model, reference preprocessing

        for key, data in (("D app-current", dD), ("A stretch-BIL", dA),
                          ("B letterbox-BIL", dB), ("AB dual-union", dU),
                          ("REF v10b-stretch", dR)):
            cls050 = {int(b[5]) for b in data if b[4] >= 0.50}
            cls030 = {int(b[5]) for b in data if b[4] >= 0.30}
            n50 = int(sum(1 for b in data if b[4] >= 0.50))
            results[key].append((img_path, expected, cls050, cls030, n50, len(expected)))
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(entries)} images done ({time.time() - t_start:.0f}s)")

    def summarize(key):
        rows = results[key]
        scored = [r for r in rows if r[1]]
        n = len(scored)
        rec050 = sum(len(e & d) / len(e) for _, e, d, _, _, _ in scored) / max(n, 1)
        rec030 = sum(len(e & c) / len(e) for _, e, c, _, _, _ in scored) / max(n, 1)
        empties = sum(1 for _, _, _, _, nb, _ in rows if nb == 0)
        return rec050, rec030, empties, len(rows)

    lines = ["", f"# Val-set evaluation (54 anh, portion-estimation_val.txt)", ""]
    lines.append("| Config | Recall@0.50 | Recall@0.30 | Anh 0 box @0.50 | So anh |")
    lines.append("|---|---|---|---|---|")
    for key in results:
        r5, r3, emp, n = summarize(key)
        lines.append(f"| {key} | {r5:.3f} | {r3:.3f} | {emp}/{n} | {n} |")
    lines.append("")

    lines.append("## Chi tiet theo nhom mon (Recall@0.50 / so anh)")

    def group_key(img_path):
        m = re.search(r"test-image[\\/]([^\\/]+)[\\/]", img_path)
        return m.group(1) if m else os.path.basename(img_path)

    groups = {}
    for key, rows in results.items():
        for img_path, expected, cls050, _, _, _ in rows:
            groups.setdefault(group_key(img_path), {}).setdefault(key, []).append(
                (img_path, expected, cls050))
    lines.append("")
    lines.append("| Nhom (folder) | " + " | ".join(results) + " |")
    lines.append("|---|" + "---|" * len(results))
    for gname in sorted(groups):
        cells = []
        for key in results:
            rows = groups[gname].get(key, [])
            scored = [r for r in rows if r[1]]
            if not scored:
                cells.append("-")
                continue
            rec = sum(len(e & d) / len(e) for _, e, d in scored) / len(scored)
            cells.append(f"{rec:.2f} ({len(scored)} anh)")
        lines.append(f"| {gname} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Cac anh con thieu class @0.50 (config AB dual-union)")
    for img_path, expected, cls050, _, _, _ in results["AB dual-union"]:
        miss = expected - cls050
        if miss:
            miss_names = ", ".join(class_names[c]["name"].split(" (")[0] for c in sorted(miss))
            lines.append(f"- {os.path.basename(img_path)}: thieu {miss_names}")

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
