"""Phase A — isolate the orientation effect on the egg (and all classes).

The early diagnosis ran v10b on RAW-orientation images (no exif_transpose)
and saw Trung at 0.68/0.43/0.50, but the app path (upright after
exif_transpose) only reaches ~0.2-0.41. This measures both models x
{upright, raw} x {stretch, letterbox} on the 4 BanhMi_Trung images.

ALSO doubles as the Phase B checkpoint: the raw+stretch rows must reproduce
Egg 0.68 / 0.43 / 0.50 (same preprocessing as the early diagnosis: raw PIL
-> PIL BILINEAR stretch 640 -> predict, no exif handling anywhere).
"""

import os

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)

from PIL import Image, ImageOps
from ultralytics import YOLO

ONNX26 = "./model/yolov26/best.onnx"
PT10B = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"

IMAGES = [
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (2).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (3).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (4).jpg",
]


def pre(pil, mode):
    if mode == "stretch":
        return pil.resize((640, 640), Image.BILINEAR)
    w, h = pil.size
    r = min(640 / w, 640 / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(pil.resize((nw, nh), Image.BILINEAR), ((640 - nw) // 2, (640 - nh) // 2))
    return canvas


def main():
    m26 = YOLO(ONNX26, task="detect")
    m10 = YOLO(PT10B)
    for m in (m26, m10):
        try:
            m.overrides["device"] = "cpu"
        except Exception:
            pass

    for p in IMAGES:
        raw = Image.open(p)
        orient = raw.getexif().get(274, None)
        upright = ImageOps.exif_transpose(raw)
        name = os.path.basename(p)
        print(f"\n=== {name} | raw={raw.size} orient={orient} upright={upright.size} ===")
        for model, mname in ((m26, "v26m"), (m10, "v10b")):
            for orient_name, base in (("upright", upright), ("raw", raw)):
                for mode in ("stretch", "letterbox"):
                    r = model.predict(pre(base, mode), conf=0.05, imgsz=640,
                                      device="cpu", verbose=False)[0]
                    data = r.boxes.data
                    if hasattr(data, "cpu"):
                        data = data.cpu().numpy()
                    top = sorted(data, key=lambda b: b[4], reverse=True)[:5]
                    tops = ", ".join(
                        f"{m10.names[int(b[5])].split(' (')[0]}:{b[4]:.2f}" for b in top) or "-"
                    trung = max((float(b[4]) for b in data if int(b[5]) == 56), default=0.0)
                    print(f"  [{mname}] {orient_name:>7}+{mode:<9} Trung={trung:.3f} | {tops}")


if __name__ == "__main__":
    main()
