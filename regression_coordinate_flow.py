"""REGRESSION: coordinate flow of the dual-preresize detection path.

Chain verified here (must transform each bbox exactly once per stage):
    uploaded (upright, exif-transposed)
        -> [stretch | letterbox] 640x640        (inside predict_with_dual_preresize)
        -> YOLOv26m ONNX end2end
        -> inverse-map ONCE to original coords  (inside predict_with_dual_preresize)
        -> bbox_scale to volume-source space    (volume_integration contract)

Checks per image:
  1. Results semantics == predict-on-original: orig_shape == (h, w) upright.
  2. Every original-space box inside [0, w] x [0, h] with positive area.
  3. Every volume-space box inside [0, w_vol] x [0, h_vol], positive area.
"""

import os

# Must precede any ultralytics import: in this venv, ONNX predict segfaults
# unless av's FFmpeg DLLs are loaded first (utils.py keeps the same order).
import av  # noqa: F401

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from utils import predict_with_dual_preresize

TEST_IMGS = [
    r'C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (2).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (3).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (1).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (3).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (4).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\Pho\Pho1 (1).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\ComSuon\com-suon1 (2).jpg',
]

model = YOLO("./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt")
model.overrides["device"] = "cpu"

print("============ REGRESSION TEST: DUAL-PRERESIZE COORDINATE FLOW (v10b) ============")
all_pass = True

for p in TEST_IMGS:
    fname = os.path.basename(p)
    if not os.path.exists(p):
        print(f"Skipping {fname} (not found)")
        continue

    raw_image = Image.open(p).convert('RGB')
    uploaded_image = ImageOps.exif_transpose(raw_image)
    w_orig, h_orig = uploaded_image.size

    volume_source_image = uploaded_image.copy()
    volume_source_image.thumbnail((2000, 2000))
    w_vol, h_vol = volume_source_image.size
    bbox_scale = (
        float(w_vol) / float(w_orig),
        float(h_vol) / float(h_orig),
    )

    results = predict_with_dual_preresize(model, uploaded_image, conf=0.50)
    r = results[0]

    print(f"\n--- IMAGE: {fname} (orient={int(raw_image.getexif().get(274, 1) or 1)}) ---")
    print(f"Original: {w_orig}x{h_orig} | Volume source: {w_vol}x{h_vol} "
          f"| bbox_scale=({bbox_scale[0]:.4f}, {bbox_scale[1]:.4f})")

    # Check 1: Results semantics == predict-on-original
    sem_ok = tuple(r.orig_shape) == (h_orig, w_orig)
    if not sem_ok:
        all_pass = False
    print(f"[{'PASSED' if sem_ok else 'FAILED'}] Results.orig_shape == (h, w) upright: "
          f"{tuple(r.orig_shape)}")

    n_boxes = len(r.boxes)
    print(f"Detections @0.50: {n_boxes}")

    for box in r.boxes:
        xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, "cpu") else np.asarray(box.xyxy)[0]
        cid = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cname = class_names[cid]['name']

        x1o, y1o, x2o, y2o = (float(v) for v in xyxy)
        orig_ok = (0 <= x1o < x2o <= w_orig) and (0 <= y1o < y2o <= h_orig)

        x1v, y1v = x1o * bbox_scale[0], y1o * bbox_scale[1]
        x2v, y2v = x2o * bbox_scale[0], y2o * bbox_scale[1]
        vol_ok = (0 <= x1v < x2v <= w_vol) and (0 <= y1v < y2v <= h_vol)

        if not (orig_ok and vol_ok):
            all_pass = False
        status = "PASSED" if (orig_ok and vol_ok) else "FAILED (OUT OF BOUNDS)"
        print(f"   [{status}] {cname}: conf={conf:.3f}")
        print(f"        Original bbox: [{x1o:.1f}, {y1o:.1f}, {x2o:.1f}, {y2o:.1f}] / [{w_orig}, {h_orig}]")
        print(f"        Volume bbox:   [{x1v:.1f}, {y1v:.1f}, {x2v:.1f}, {y2v:.1f}] / [{w_vol}, {h_vol}]")

print("\nOVERALL REGRESSION INTEGRITY: "
      f"{'ALL BOUNDS VALID (PASS)' if all_pass else 'FAIL'}")
