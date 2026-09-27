"""Reproduce the cross-class duplicate on bun-rieu (1).jpg.

Runs the same two pre-resize passes as predict_with_dual_preresize, prints
raw boxes per pass, then the merged pool before/after _nms_class_aware.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageOps

from utils import load_model, _nms_class_aware
from class_names import class_names

IMG = r"C:\Users\huynh\OneDrive\Pictures\test-image\BunRieu-CanhBun\bun-rieu (1).jpg"
IMGSZ = 640


def name(cid):
    c = class_names[int(cid)]
    return c["name"] if isinstance(c, dict) else c


def data(res):
    d = res.boxes.data
    return d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)


def show(tag, arr, w, h):
    print(f"\n--- {tag} ({len(arr)} boxes) ---")
    for b in sorted(arr, key=lambda x: -x[4]):
        x1, y1, x2, y2 = b[:4]
        print(f"  {name(b[5])[:38]:40s} conf={b[4]:.3f} "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


img = ImageOps.exif_transpose(Image.open(IMG)).convert("RGB")
w, h = img.size
print(f"image: {w}x{h}")

stretch_img = img.resize((IMGSZ, IMGSZ), Image.BILINEAR)
r = min(IMGSZ / w, IMGSZ / h)
nw, nh = int(round(w * r)), int(round(h * r))
pad_x, pad_y = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
letter_img = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
letter_img.paste(img.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))

model = load_model()
CONF = 0.30  # app default slider value
res_s = model.predict(stretch_img, conf=CONF, imgsz=IMGSZ, half=False, verbose=False)[0]
res_l = model.predict(letter_img, conf=CONF, imgsz=IMGSZ, half=False, verbose=False)[0]

ds, dl = data(res_s), data(res_l)
show("STRETCH pass (raw, in image coords)", ds, w, h)
show("LETTERBOX pass (raw, in image coords)", dl, w, h)

xyxy_s = ds[:, :4].copy()
xyxy_s[:, [0, 2]] *= w / float(IMGSZ)
xyxy_s[:, [1, 3]] *= h / float(IMGSZ)
xyxy_l = dl[:, :4].copy()
xyxy_l[:, [0, 2]] = (xyxy_l[:, [0, 2]] - pad_x) / r
xyxy_l[:, [1, 3]] = (xyxy_l[:, [1, 3]] - pad_y) / r

show("STRETCH pass (mapped back to original)", np.column_stack([xyxy_s, ds[:, 4:]]), w, h)
show("LETTERBOX pass (mapped back to original)", np.column_stack([xyxy_l, dl[:, 4:]]), w, h)

xyxy = np.concatenate([xyxy_s, xyxy_l], axis=0)
extra = np.concatenate([ds[:, 4:], dl[:, 4:]], axis=0)
merged = np.column_stack([xyxy, extra])
show("MERGED pool (before NMS)", merged, w, h)

keep = _nms_class_aware(xyxy, extra[:, 0], extra[:, 1])
show("AFTER _nms_class_aware(iou=0.55) — what the app shows", merged[keep], w, h)

# Cross-class overlap between the survivors
print("\n--- pairwise IoU between survivors (different classes) ---")
kb = merged[keep]
for i in range(len(kb)):
    for j in range(i + 1, len(kb)):
        if kb[i, 5] == kb[j, 5]:
            continue
        ax1, ay1, ax2, ay2 = kb[i, :4]
        bx1, by1, bx2, by2 = kb[j, :4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        print(f"  {name(kb[i,5])[:30]:32s} vs {name(kb[j,5])[:30]:32s} "
              f"IoU={inter / u if u > 0 else 0:.3f}")
