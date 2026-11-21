"""Phase 7 acceptance — E2E duplicate regression (slow, needs model weights).

Runs the production detector path (predict_with_dual_preresize, conf 0.30 =
app default slider, cross-class suppression enabled via pipeline_config) on
the BunRieu-CanhBun regression images and asserts NUMERIC invariants on the
surviving detections — never on display text.

Invariants
    1. Every image: no two surviving food boxes of DIFFERENT classes overlap
       with IoU >= CROSS_CLASS_SUPPRESS_IOU (the double-counting failure
       mode is structurally impossible after suppression).
    2. bun-rieu (1).jpg (the reproduced bug): exactly 1 food box, class
       Bun bo Hue, confidence > 0.5 (sanity bound, not the invariant).

Skip rules (infrastructure, never assertions):
    SKIPPED: regression image directory not found: ...
    SKIPPED: YOLO model weights not available: ...
A wrong result on an existing image FAILS.

Run:  .venv/Scripts/python.exe tests/test_phase7_e2e_duplicate_regression.py
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageOps

from utils import load_model, predict_with_dual_preresize
from class_names import class_names

IMAGES_DIR = r"C:\Users\huynh\OneDrive\Pictures\test-image\BunRieu-CanhBun"
REGRESSION_IMAGE = "bun-rieu (1).jpg"
MODEL_WEIGHTS = os.path.join(
    ROOT_DIR, "model", "yolov10", "YOLOv10b_VietFood67_SGD_new_bigger.pt")
CONF = 0.30  # app default slider (main.py)
NON_FOOD_CLASS = "Con nguoi (Human)"

# NOTE: do NOT `import pipeline_config` at module level here. It touches
# torch.cuda, and a CUDA-API call before the first YOLO predict segfaults
# bare (non-Streamlit) processes (measured: repro R2b/R2c). The first
# predict_with_dual_preresize call imports it internally AFTER its predicts
# (safe per repro R4), so the threshold is read below after that first call.


def class_name(cid):
    c = class_names[int(cid)]
    return c["name"] if isinstance(c, dict) else str(c)


def pair_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def run_all_tests():
    print(f"=== Phase 7: E2E duplicate regression (conf={CONF}) ===")

    if not os.path.isdir(IMAGES_DIR):
        print(f"SKIPPED: regression image directory not found: {IMAGES_DIR}")
        return True
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"SKIPPED: YOLO model weights not available: {MODEL_WEIGHTS}")
        return True

    model = load_model()
    images = sorted(f for f in os.listdir(IMAGES_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not images:
        print(f"SKIPPED: no images in {IMAGES_DIR}")
        return True

    failed = 0
    thresh = None
    for name in images:
        path = os.path.join(IMAGES_DIR, name)
        pil = ImageOps.exif_transpose(Image.open(path))
        result = predict_with_dual_preresize(model, pil, CONF)[0]
        if thresh is None:
            # Safe here: the call above already imported pipeline_config
            # internally (post-predict — see the module-level NOTE).
            import pipeline_config
            thresh = float(pipeline_config.CROSS_CLASS_SUPPRESS_IOU)
            print(f"  threshold from pipeline_config: {thresh}")
        assert hasattr(result, "suppressed_cross_class"), (
            "Results must carry the suppressed_cross_class audit attribute")

        food = [(b.conf[0].item(), b.xyxy[0].cpu().numpy(), int(b.cls[0].item()))
                for b in result.boxes
                if class_name(int(b.cls[0].item())) != NON_FOOD_CLASS]

        # Invariant 1 — no cross-class pair at/above the suppression threshold.
        for i in range(len(food)):
            for j in range(i + 1, len(food)):
                if food[i][2] == food[j][2]:
                    continue
                iou = pair_iou(food[i][1], food[j][1])
                assert iou < thresh, (
                    f"{name}: cross-class duplicate survived — "
                    f"{class_name(food[i][2])} ({food[i][0]:.2f}) vs "
                    f"{class_name(food[j][2])} ({food[j][0]:.2f}), IoU={iou:.3f}")

        status = f"{len(food)} food box(es)"
        if name == REGRESSION_IMAGE:
            # Invariant 2 — the reproduced bug case collapses to one box.
            assert len(food) == 1, (
                f"{name}: expected exactly 1 food box, got {len(food)}: "
                + ", ".join(f"{class_name(c)} {cf:.2f}" for cf, _, c in food))
            conf0, _, cid0 = food[0]
            assert class_name(cid0).startswith("Bun bo Hue"), (
                f"{name}: expected Bun bo Hue, got {class_name(cid0)}")
            assert conf0 > 0.5, f"{name}: surviving box conf {conf0:.2f} <= 0.5"
            status += " — Bun bo Hue, single box (regression case OK)"

        n_sup = len(result.suppressed_cross_class)
        print(f"  [PASS] {name}: {status}"
              + (f", {n_sup} cross-class duplicate(s) suppressed (audit)"
                 if n_sup else ""))

    print(f"=== Phase 7: {len(images)} images checked ===")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
