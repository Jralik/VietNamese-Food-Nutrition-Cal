"""Diagnose edge/corner false positives: attribute each FP box to its pool.

For the 5 reported images, runs v10b in the 3 ensemble variants
(upright-stretch, upright-letterbox, raw-stretch) at conf=0.05 and prints
every box >= 0.30 with:
  - canvas coordinates (640-space) and upright-space coordinates
  - overflow past the image bounds after inverse mapping (px)
  - whether the box CENTER sits in the letterbox padding region
  - which pools it appears in (pre-NMS), to attribute each FP
"""

import os

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

PT10B = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
NMS_IOU = 0.55

IMAGES = [
    r"C:\Users\huynh\OneDrive\Pictures\test-image\ComSuon\com-suon1 (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\Pho\Pho1 (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (4).jpg",
]


def stretch(pil):
    return pil.resize((640, 640), Image.BILINEAR)


def letterbox(pil):
    w, h = pil.size
    r = min(640 / w, 640 / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(pil.resize((nw, nh), Image.BILINEAR), ((640 - nw) // 2, (640 - nh) // 2))
    return canvas, r, (640 - nw) // 2, (640 - nh) // 2


def get(model, src):
    r = model.predict(src, conf=0.05, imgsz=640, device="cpu", verbose=False)[0]
    d = r.boxes.data
    return d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)


def show(tag, data, mapper, w, h, names):
    rows = []
    for b in data:
        if b[4] < 0.30:
            continue
        xy = mapper(b[:4])
        if hasattr(xy, "tolist"):
            xy = xy.tolist()
        ov = [max(0, -xy[0]), max(0, -xy[1]), max(0, xy[2] - w), max(0, xy[3] - h)]
        ovs = sum(ov)
        rows.append((b[4], int(b[5]), b[:4].tolist(), xy, ovs))
    rows.sort(key=lambda x: -x[0])
    for conf, cid, canvas, xy, ovs in rows:
        flag = f" OVERFLOW {ovs:.0f}px" if ovs > 1 else ""
        print(f"    [{tag}] {names[cid].split(' (')[0]}:{conf:.3f} "
              f"canvas={[round(v) for v in canvas]} up={[round(v, 1) for v in xy]}{flag}")


def main():
    model = YOLO(PT10B)
    model.overrides["device"] = "cpu"
    names = model.names

    for p in IMAGES:
        raw = Image.open(p)
        orient = int(raw.getexif().get(274, 1) or 1)
        upright = ImageOps.exif_transpose(raw)
        w_up, h_up = upright.size
        w_raw, h_raw = raw.size
        lb, r, pad_x, pad_y = letterbox(upright)
        print(f"\n=== {os.path.basename(p)} orient={orient} up={w_up}x{h_up} "
              f"letterbox content=({pad_x},{pad_y})-({640 - pad_x},{640 - pad_y}) ===")

        dS = get(model, stretch(upright))
        dL = get(model, lb)
        show("up-stretch  ", dS, lambda b: [b[0] * w_up / 640, b[1] * h_up / 640,
                                            b[2] * w_up / 640, b[3] * h_up / 640], w_up, h_up, names)
        show("up-letterbox", dL, lambda b: [(b[0] - pad_x) / r, (b[1] - pad_y) / r,
                                            (b[2] - pad_x) / r, (b[3] - pad_y) / r], w_up, h_up, names)
        if orient != 1:
            dR = get(model, stretch(raw))
            from utils import _map_boxes_exif_to_upright
            def raw_map(b):
                m, _ = _map_boxes_exif_to_upright(
                    np.array([b]), orient, w_raw, h_raw)
                return m[0].tolist()
            show("raw-stretch ", dR, raw_map, w_up, h_up, names)
        else:
            print("    (orient=1: raw branch inactive)")


if __name__ == "__main__":
    main()
