"""A/B benchmark: letterbox (current app) vs stretch 640x640 (reference-style).

Runs the 7 diagnostic images through the SAME ultralytics predict call the
Streamlit app uses, in two modes:
  - letterbox: predict on the full-res original (current app behaviour,
    ultralytics LetterBox keeps aspect ratio -> tall images become a sliver)
  - stretch:   predict on a PIL stretch-resize to 640x640 (proposed fix,
    same preprocessing as the h-nam01/FoodDetector reference)

One predict run per mode at conf=0.05; all thresholds (0.50/0.40/0.30/0.20/0.10)
are applied locally to the returned box confidences. For end2end ONNX output
(1,300,6) the ultralytics postprocess is a pure confidence filter
(ultralytics/utils/ops.py:219-223), so a local re-threshold is equivalent to
calling predict with that conf.

Writes scratch/ab_preprocess_results.md.
"""

import os
import time

# Must precede ultralytics: in this venv, ultralytics ONNX predict segfaults
# unless av's FFmpeg DLLs are loaded first (same import order as utils.py).
import av  # noqa: F401

from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names

MODEL_PATH = "./model/yolov26/best.onnx"
CONF_LEVELS = [0.50, 0.40, 0.30, 0.20, 0.10]
RUN_CONF = 0.05  # single run per mode; thresholds applied locally afterwards
OUT_PATH = os.path.join("scratch", "ab_preprocess_results.md")

IMAGES = [
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (2).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (3).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (2).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (3).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (4).jpg",
]

CLS_BANH_MI = 4    # "Banh mi"
CLS_TRUNG = 56     # "Trung"
CLS_SUP_CUA = 67   # "Sup cua"


def cname(cls_id: int) -> str:
    if 0 <= cls_id < len(class_names):
        item = class_names[cls_id]
        return item.get("name", str(cls_id)) if isinstance(item, dict) else str(item)
    return f"ID_{cls_id}"


def run_mode(model, pil_img, mode):
    """One predict run; returns (data (N,6) numpy [xyxy conf cls], elapsed_ms)."""
    if mode == "letterbox":
        source = pil_img
    else:  # stretch
        source = pil_img.resize((640, 640), Image.BILINEAR)
    t0 = time.time()
    res = model.predict(source, conf=RUN_CONF, imgsz=640, half=False, verbose=False)[0]
    elapsed_ms = (time.time() - t0) * 1000
    data = res.boxes.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    return data, elapsed_ms


def counts_at(data, levels):
    confs = data[:, 4] if len(data) else []
    return {c: int(sum(1 for x in confs if x >= c)) for c in levels}


def fmt_dets(data, conf, limit=8):
    rows = [b for b in data if b[4] >= conf]
    rows.sort(key=lambda b: b[4], reverse=True)
    parts = []
    for b in rows[:limit]:
        parts.append(f"{cname(int(b[5]))}: {b[4]:.3f}")
    if len(rows) > limit:
        parts.append(f"... (+{len(rows) - limit})")
    return ", ".join(parts) if parts else "(none)"


def main():
    model = YOLO(MODEL_PATH, task="detect")
    lines = []
    lines.append("# A/B preprocess benchmark — letterbox (app hiện tại) vs stretch 640x640 (reference)")
    lines.append("")
    lines.append(f"Model: `{MODEL_PATH}` (end2end ONNX, output (1,300,6)). "
                 f"Mỗi mode chạy 1 lần predict ở conf={RUN_CONF}, các ngưỡng "
                 f"{CONF_LEVELS} áp dụng local (postprocess end2end chỉ là lọc conf).")
    lines.append("")

    for path in IMAGES:
        if not os.path.exists(path):
            lines.append(f"**MISSING**: {path}")
            lines.append("")
            continue
        pil = ImageOps.exif_transpose(Image.open(path))
        w, h = pil.size
        name = os.path.basename(path)
        lines.append(f"## {name} — {w}x{h}")
        lines.append("")
        lines.append("| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        per_mode_classmax = {}
        for mode in ("letterbox", "stretch"):
            data, ms = run_mode(model, pil, mode)
            counts = counts_at(data, CONF_LEVELS)
            if len(data):
                best = data[int(data[:, 4].argmax())]
                max_conf_s = f"{best[4]:.3f} ({cname(int(best[5]))})"
            else:
                max_conf_s = "n/a"
            lines.append(
                f"| {mode} | {counts[0.50]} | {fmt_dets(data, 0.50)} | {max_conf_s} "
                f"| {counts[0.40]} | {counts[0.30]} | {counts[0.20]} | {counts[0.10]} | {ms:.0f} |")
            cls_max = {}
            for b in data:
                cid = int(b[5])
                if cid not in cls_max or b[4] > cls_max[cid]:
                    cls_max[cid] = float(b[4])
            per_mode_classmax[mode] = cls_max
        lines.append("")
        # Per-class max conf for the classes relevant to this image
        interesting = [CLS_SUP_CUA] if "SupCua" in name else [CLS_BANH_MI, CLS_TRUNG]
        lines.append("Max conf theo class then chốt (mọi slot, conf>" + f"{RUN_CONF}):")
        lines.append("")
        lines.append("| Mode | " + " | ".join(cname(c) for c in interesting) + " |")
        lines.append("|---|" + "---|" * len(interesting))
        for mode in ("letterbox", "stretch"):
            cells = [f"{per_mode_classmax[mode].get(c, 0.0):.4f}" for c in interesting]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines.append("")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved -> {OUT_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
