"""T4: PIL pre-resize (anti-aliased) through the app-identical predict path.

Story so far: predict(PIL) feeds the model RGB (loader flip + predictor
un-flip cancel out). The remaining difference vs the manual ORT scripts is
the resize: ultralytics LetterBox uses cv2.INTER_LINEAR (aliased at 4000px
downscales), while the manual scripts use PIL BILINEAR (anti-aliased).
SupCua (1) fired at 0.917 only in the PIL-letterbox config.

This script measures, for each of the 7 images in APP-display orientation
(exif_transposed), all candidate fixes through the standard predict() call:
  A. pre-PIL-stretch 640x640 (BILINEAR)   [the approved fix]
  B. pre-PIL-letterbox 640x640 (BILINEAR + 114 pad)
  C. pre-PIL-letterbox 640x640 (LANCZOS + 114 pad)
  D. no pre-resize (current app: cv2 letterbox inside predict)

Post-fix the app would call predict() on the pre-resized PIL image, with
LetterBox a no-op (input already 640x640). So these numbers ARE the app path
numbers for each candidate.
"""

import os
import time

import av  # noqa: F401  (must precede ultralytics in this venv)

from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names

ONNX26 = "./model/yolov26/best.onnx"
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


def pre_stretch(pil, resample=Image.BILINEAR):
    return pil.resize((640, 640), resample)


def pre_letterbox(pil, resample=Image.BILINEAR):
    w, h = pil.size
    scale = min(640 / w, 640 / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(pil.resize((nw, nh), resample), ((640 - nw) // 2, (640 - nh) // 2))
    return canvas


VARIANTS = {
    "A stretch-BILINEAR": lambda p: pre_stretch(p, Image.BILINEAR),
    "B letterbox-BILINEAR": lambda p: pre_letterbox(p, Image.BILINEAR),
    "C letterbox-LANCZOS": lambda p: pre_letterbox(p, Image.LANCZOS),
    "D no-preresize(cv2)": lambda p: p,
}


def main():
    model = YOLO(ONNX26, task="detect")
    lines = ["", "# T4 — PIL pre-resize qua đúng đường predict của app (model nhận RGB)", ""]
    lines.append("Orientation = exif_transpose (như app). Variant D = app hiện tại.")
    lines.append("")
    lines.append("| Ảnh | Variant | #box @0.50 | Top-4 | maxSupCua | maxTrung | maxBanhMi | ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for path in IMAGES:
        pil = ImageOps.exif_transpose(Image.open(path))
        name = os.path.basename(path)
        for label, prep in VARIANTS.items():
            src = prep(pil)
            t0 = time.time()
            r = model.predict(src, conf=0.05, imgsz=640, half=False, verbose=False)[0]
            ms = (time.time() - t0) * 1000
            data = r.boxes.data
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            n50 = int(sum(1 for b in data if b[4] >= 0.50))
            top = sorted(data, key=lambda b: b[4], reverse=True)[:4]
            tops = ", ".join(f"{cname(int(b[5])).split(' (')[0]}:{b[4]:.2f}" for b in top) or "-"
            cls_max = {}
            for b in data:
                c = int(b[5])
                cls_max[c] = max(cls_max.get(c, 0.0), float(b[4]))
            lines.append(
                f"| {name.split(' ')[0]} | {label} | {n50} | {tops} "
                f"| {cls_max.get(CLS_SUP_CUA, 0.0):.3f} | {cls_max.get(CLS_TRUNG, 0.0):.3f} "
                f"| {cls_max.get(CLS_BANH_MI, 0.0):.3f} | {ms:.0f} |")
        lines.append("")
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
