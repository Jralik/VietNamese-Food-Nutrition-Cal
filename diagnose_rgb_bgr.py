"""Test the RGB/BGR channel-order hypothesis for the v26m ONNX model.

Evidence so far:
  - ultralytics predict(pil) feeds BGR to the graph (LoadPilAndNumpy flips
    RGB->BGR, no exif handling in 8.2.74).
  - The user's manual ORT scripts feed RGB (no flip) and fired
    "Sup cua" @0.917 on SupCua (1) raw+letterbox, while predict() gives 0.000
    on the same geometry.
  - Color-dependent classes (Sup cua, Trung) are dead through predict();
    shape-dominant Banh mi survives.

If the model is RGB-native (training loader fed RGB), passing an RGB numpy
array directly (the loader does NOT flip ndarray inputs, _single_check only
flips PIL) should revive the dead classes.

RGB path: predict(np.asarray(pil_rgb), ...)  -> ndarray passes through unflipped.
BGR path: predict(pil, ...)                  -> PIL is converted RGB->BGR.
"""

import os
import time

import av  # noqa: F401  (must precede ultralytics in this venv)

import numpy as np
from PIL import Image
from ultralytics import YOLO

from class_names import class_names

ONNX26 = "./model/yolov26/best.onnx"

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


def predict(model, pil, mode, channel):
    src_pil = pil if mode == "letterbox" else pil.resize((640, 640), Image.BILINEAR)
    if channel == "RGB":
        source = np.asarray(src_pil.convert("RGB"))  # ndarray pass-through: model sees RGB
    else:
        source = src_pil  # PIL: loader flips RGB->BGR
    t0 = time.time()
    r = model.predict(source, conf=0.05, imgsz=640, half=False, verbose=False)[0]
    ms = (time.time() - t0) * 1000
    data = r.boxes.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    return data, ms


def summarize(data):
    n50 = int(sum(1 for b in data if b[4] >= 0.50))
    top = sorted(data, key=lambda b: b[4], reverse=True)[:4]
    tops = ", ".join(f"{cname(int(b[5])).split(' (')[0]}:{b[4]:.2f}" for b in top) or "-"
    cls_max = {}
    for b in data:
        c = int(b[5])
        cls_max[c] = max(cls_max.get(c, 0.0), float(b[4]))
    return (n50, tops, cls_max.get(CLS_SUP_CUA, 0.0),
            cls_max.get(CLS_TRUNG, 0.0), cls_max.get(CLS_BANH_MI, 0.0))


def main():
    model = YOLO(ONNX26, task="detect")
    lines = ["", "# Diagnosis 3 — RGB vs BGR channel order (v26m ONNX)", ""]
    lines.append("RGB = predict(numpy RGB array, loader pass-through). "
                 "BGR = predict(PIL, loader flips to BGR).")
    lines.append("")
    lines.append("| Ảnh | Preprocess | Channel | #box @0.50 | Top-4 | maxSupCua | maxTrung | maxBanhMi |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for path in IMAGES:
        pil = Image.open(path)
        name = os.path.basename(path)
        for mode in ("letterbox", "stretch"):
            for channel in ("RGB", "BGR"):
                data, _ = predict(model, pil, mode, channel)
                n50, tops, msc, mtr, mbm = summarize(data)
                lines.append(f"| {name.split(' ')[0]} | {mode} | {channel} | {n50} | {tops} "
                             f"| {msc:.3f} | {mtr:.3f} | {mbm:.3f} |")
        lines.append("")
    out_path = os.path.join("scratch", "ab_preprocess_results.md")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
