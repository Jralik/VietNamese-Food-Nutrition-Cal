"""Phase E sanity check: run the 7 diagnostic images through the app's
detection call (load_model-equivalent v10b + predict_with_dual_preresize,
conf = app default) and print per-image detections.
"""

import os
import time

# Must precede any ultralytics import: in this venv, predict segfaults unless
# av's FFmpeg DLLs are loaded first (utils.py keeps the same order).
import av  # noqa: F401

from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from utils import predict_with_dual_preresize

MODEL_PATH = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
CONF = 0.30

IMAGES = [
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (2).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (3).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (2).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (3).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (4).jpg",
]


def main():
    model = YOLO(MODEL_PATH)
    model.overrides["device"] = "cpu"
    for p in IMAGES:
        name = os.path.basename(p)
        if not os.path.exists(p):
            print(f"MISSING {p}")
            continue
        im = ImageOps.exif_transpose(Image.open(p))
        t0 = time.time()
        res = predict_with_dual_preresize(model, im, CONF)[0]
        ms = (time.time() - t0) * 1000
        dets = []
        for b in res.boxes:
            cid = int(b.cls[0].item())
            cname = class_names[cid]["name"].split(" (")[0]
            dets.append(f"{cname}: {float(b.conf[0].item()):.3f}")
        print(f"{name}: {len(res.boxes)} box | {'; '.join(dets) or '(none)'} | {ms:.0f}ms")


if __name__ == "__main__":
    main()
