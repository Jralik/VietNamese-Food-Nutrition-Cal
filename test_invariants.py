"""Invariant tests locking the Phase B/C building blocks.

1. EXIF box mapping (8/8): _map_boxes_exif_to_upright must mirror
   ImageOps.exif_transpose exactly — verified with corner-marker images,
   including the width/height swap of orientations 5-8.
2. NMS invariants (5 cases): the merge must keep the highest-confidence box
   for same-class duplicates, keep different-class overlaps, and never drop
   a box just because it came from only one orientation pool.
"""

import os

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)

import numpy as np
from PIL import Image, ImageOps

from utils import _map_boxes_exif_to_upright, _nms_class_aware

W, H = 200, 100
COLORS = {"tl": (255, 0, 0), "tr": (0, 255, 0), "bl": (0, 0, 255), "br": (255, 255, 0)}
BLOCK = 40  # corner block size; sample points inside blocks


def make_tagged(orient):
    img = Image.new("RGB", (W, H), (128, 128, 128))
    px = img.load()
    for name, (cx, cy) in {"tl": (0, 0), "tr": (W - BLOCK, 0),
                           "bl": (0, H - BLOCK), "br": (W - BLOCK, H - BLOCK)}.items():
        c = COLORS[name]
        for x in range(cx, cx + BLOCK):
            for y in range(cy, cy + BLOCK):
                px[x, y] = c
    e = img.getexif()
    e[274] = orient
    img.info["exif"] = e.tobytes()
    return img


def map_pixel(px, py, orient):
    """Map a source pixel through the same formula the box mapper uses
    (pixel-center continuous coordinates, then floor)."""
    mapped, _ = _map_boxes_exif_to_upright(
        np.array([[px + 0.5, py + 0.5, px + 0.5, py + 0.5]]), orient, W, H)
    return int(np.floor(mapped[0, 0])), int(np.floor(mapped[0, 1]))


def test_exif_mapping():
    src_points = {"tl": (5, 5), "tr": (W - BLOCK + 5, 5),
                  "bl": (5, H - BLOCK + 5), "br": (W - BLOCK + 5, H - BLOCK + 5)}
    all_pass = True
    for orient in range(1, 9):
        up = ImageOps.exif_transpose(make_tagged(orient))
        w_up, h_up = up.size
        _, (ew, eh) = _map_boxes_exif_to_upright(
            np.zeros((0, 4)), orient, W, H)
        dim_ok = (w_up, h_up) == (ew, eh)
        if not dim_ok:
            all_pass = False
        corner_ok = True
        for name, (sx, sy) in src_points.items():
            dx, dy = map_pixel(sx, sy, orient)
            if up.getpixel((min(dx, w_up - 1), min(dy, h_up - 1))) != COLORS[name]:
                corner_ok = False
        bbox = np.array([[20, 15, 180, 85]], dtype=np.float64)
        mapped, _ = _map_boxes_exif_to_upright(bbox, orient, W, H)
        x1, y1, x2, y2 = mapped[0]
        bbox_ok = (0 <= x1 <= x2 <= w_up) and (0 <= y1 <= y2 <= h_up)
        status = "PASS" if (dim_ok and corner_ok and bbox_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  orient {orient}: dims {w_up}x{h_up} expected {ew}x{eh} | "
              f"corners {'ok' if corner_ok else 'MISMATCH'} | bbox {'ok' if bbox_ok else 'BAD'} -> {status}")
    try:
        _map_boxes_exif_to_upright(np.zeros((1, 4)), 9, W, H)
        print("  orient 9: expected ValueError — FAIL")
        all_pass = False
    except ValueError:
        print("  orient 9: ValueError raised as expected — PASS")
    return all_pass


def run_nms(boxes, scores, classes):
    keep = _nms_class_aware(np.asarray(boxes, dtype=np.float64),
                            np.asarray(scores), np.asarray(classes))
    keep = sorted(keep)
    return [(classes[i], scores[i]) for i in keep]


def test_nms_invariants():
    ok = True

    # 1. same-class duplicate 0.25 + 0.68 -> only 0.68 survives
    r = run_nms([[0, 0, 100, 100], [5, 5, 105, 105]], [0.25, 0.68], [56, 56])
    case = len(r) == 1 and r[0] == (56, 0.68)
    ok &= case
    print(f"  same-class dup: {r} -> {'PASS' if case else 'FAIL'}")

    # 2. different-class overlap -> both survive
    r = run_nms([[0, 0, 100, 100], [2, 2, 102, 102]], [0.70, 0.65], [4, 56])
    case = sorted(r) == [(4, 0.70), (56, 0.65)]
    ok &= case
    print(f"  diff-class overlap: {r} -> {'PASS' if case else 'FAIL'}")

    # 3. raw-only box survives
    r = run_nms([[200, 200, 300, 300]], [0.68], [56])
    case = r == [(56, 0.68)]
    ok &= case
    print(f"  raw-only: {r} -> {'PASS' if case else 'FAIL'}")

    # 4. upright-only box survives
    r = run_nms([[400, 400, 500, 500]], [0.35], [4])
    case = r == [(4, 0.35)]
    ok &= case
    print(f"  upright-only: {r} -> {'PASS' if case else 'FAIL'}")

    # 5. same object from two pools -> one box with conf = max
    r = run_nms([[10, 10, 110, 110], [12, 12, 112, 112]], [0.25, 0.68], [56, 56])
    case = len(r) == 1 and r[0] == (56, 0.68)
    ok &= case
    print(f"  same object 2 pools: {r} -> {'PASS' if case else 'FAIL'}")

    # guard: same class, NON-overlapping -> both survive
    r = run_nms([[0, 0, 100, 100], [500, 500, 600, 600]], [0.25, 0.68], [56, 56])
    case = len(r) == 2
    ok &= case
    print(f"  same class far apart: {len(r)} boxes -> {'PASS' if case else 'FAIL'}")

    return ok


if __name__ == "__main__":
    print("=== EXIF mapping (8 orientations) ===")
    exif_ok = test_exif_mapping()
    print("=== NMS invariants ===")
    nms_ok = test_nms_invariants()
    print(f"\nOVERALL: {'ALL PASS' if (exif_ok and nms_ok) else 'FAIL'}")
