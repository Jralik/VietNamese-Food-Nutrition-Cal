"""Diagnosis matrix: model x orientation x preprocessing on the 7 test images.

Extends benchmark_ab_preprocess.py after it revealed that EXIF orientation
matters: the raw files are landscape (4000x1800) with an EXIF rotate tag, the
app exif_transpose's to portrait before predict, and the reference project
feeds the RAW orientation. For SupCua (1) v26m fires at 0.917 on
raw+letterbox but ~0.16 on every transposed variant.

Matrix per image:
  - v26m  ONNX (app model, 68 cls):      raw+letterbox, raw+stretch
  - v10b  PT  (reference model, 68 cls): raw+stretch (exact reference
    preprocessing), raw+letterbox (informational)

Appends a section to scratch/ab_preprocess_results.md.
"""

import os
import time

# Must precede ultralytics: in this venv, ultralytics ONNX/torch predict
# segfaults unless av's FFmpeg DLLs are loaded first (same order as utils.py).
import av  # noqa: F401

from PIL import Image
from ultralytics import YOLO

from class_names import class_names

ONNX26 = "./model/yolov26/best.onnx"
PT10B = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
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

CLS_BANH_MI, CLS_TRUNG, CLS_SUP_CUA = 4, 56, 67


def cname(cls_id: int) -> str:
    if 0 <= cls_id < len(class_names):
        item = class_names[cls_id]
        return item.get("name", str(cls_id)) if isinstance(item, dict) else str(item)
    return f"ID_{cls_id}"


def predict(model, pil_img, mode):
    src = pil_img if mode == "letterbox" else pil_img.resize((640, 640), Image.BILINEAR)
    t0 = time.time()
    r = model.predict(src, conf=0.05, imgsz=640, half=False, verbose=False)[0]
    ms = (time.time() - t0) * 1000
    data = r.boxes.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    return data, ms


def row(data, ms):
    n50 = int(sum(1 for b in data if b[4] >= 0.50))
    top = sorted(data, key=lambda b: b[4], reverse=True)[:3]
    tops = ", ".join(f"{cname(int(b[5])).split(' (')[0]}:{b[4]:.2f}" for b in top) or "-"
    cls_max = {}
    for b in data:
        c = int(b[5])
        cls_max[c] = max(cls_max.get(c, 0.0), float(b[4]))
    return (n50, tops, cls_max.get(CLS_SUP_CUA, 0.0),
            cls_max.get(CLS_TRUNG, 0.0), cls_max.get(CLS_BANH_MI, 0.0), ms)


def main():
    m26 = YOLO(ONNX26, task="detect")
    m10 = YOLO(PT10B)

    lines = ["", "# Diagnosis 2 — model x orientation x preprocessing (raw = không exif_transpose)", ""]
    lines.append("| Ảnh | Model | Preprocess | #box @0.50 | Top-3 | maxSupCua | maxTrung | maxBanhMi | ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for path in IMAGES:
        if not os.path.exists(path):
            continue
        pil = Image.open(path)  # RAW orientation — no exif_transpose
        try:
            orient = pil.getexif().get(274, None)
        except Exception:
            orient = None
        name = os.path.basename(path)
        size = pil.size
        variants = [
            ("v26m(onnx)", m26, "letterbox"),
            ("v26m(onnx)", m26, "stretch"),
            ("v10b(ref)", m10, "stretch"),
            ("v10b(ref)", m10, "letterbox"),
        ]
        lines.append(f"| **{name}** raw={size[0]}x{size[1]} EXIF-orient={orient} | | | | | | | | |")
        for label, model, mode in variants:
            data, ms = predict(model, pil, mode)
            n50, tops, msc, mtr, mbm, ms = row(data, ms)
            lines.append(f"| {name.split(' ')[0]} | {label} | raw+{mode} | {n50} | {tops} "
                         f"| {msc:.3f} | {mtr:.3f} | {mbm:.3f} | {ms:.0f} |")
        lines.append("")

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
