"""
Real-image calibration check for nutrition estimation.

Runs the same path as the app (YOLO on 640x640 display copy + volume pipeline
on the aspect-preserving source) on the user's test photos with known ground
truth and reports PASS/FAIL checks:
  - Pho.jpg          : no false ArUco; total ~350-700 kcal (a normal bowl)
  - banh-mi1/2.jpg   : same baguette, ~225 g -> both within 225g +-35%,
                       mutually within 25%
  - com-tam3.jpg     : >= 300 kcal (rice + grilled pork plate)

Usage:
    python test_real_images.py [image_dir]
"""

import os
import sys

DEFAULT_DIR = r"C:\Users\huynh\OneDrive\Pictures\test-image"
BANH_MI_GT_G = 225.0

FAILURES = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def main():
    img_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not os.path.isdir(img_dir):
        print(f"Image dir not found: {img_dir} — skipping real-image checks.")
        return 0

    import pandas  # noqa: F401  (must precede ultralytics — DLL ordering)
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import volume_integration
    from class_names import class_names

    targets = {
        "pho": os.path.join(img_dir, "Pho.jpg"),
        "banh-mi1": os.path.join(img_dir, "BanhMi", "banh-mi1.jpg"),
        "banh-mi2": os.path.join(img_dir, "BanhMi", "banh-mi2.jpg"),
        "com-tam3": os.path.join(img_dir, "com-tam3.jpg"),
        "com-tam4": os.path.join(img_dir, "com-tam4.jpg"),
    }
    targets = {k: p for k, p in targets.items() if os.path.exists(p)}
    if not targets:
        print(f"No known test images found in {img_dir} — nothing to check.")
        return 0

    print("=" * 74)
    print("  REAL-IMAGE CALIBRATION CHECK")
    print("=" * 74)

    model = YOLO(r"./model/yolov26/best.onnx", task="detect")
    results = {}
    bbox_scales = {}
    raw_boxes = {}
    for key, path in targets.items():
        image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        source = image.copy()
        source.thumbnail((2000, 2000))
        display = image.resize((640, 640))  # app's display/YOLO path
        bbox_scale = (source.width / 640.0, source.height / 640.0)

        res = model.predict(display, conf=0.25, imgsz=640)
        # keep raw 640-space boxes for the UI-side matching check
        raw_boxes[key] = []
        for box in res[0].boxes:
            cid = int(box.cls[0].item())
            if 0 <= cid < len(class_names) and class_names[cid]["name"] != "Con nguoi (Human)":
                xy = box.xyxy.cpu().numpy()[0][:4]
                raw_boxes[key].append(xy)
        dets = volume_integration.extract_yolo_detections(
            res, class_names, bbox_scale=bbox_scale)
        r = volume_integration.estimate_nutrition_volume(source, dets, path)
        bbox_scales[key] = bbox_scale
        results[key] = r
        if r is None:
            print(f"\n[{key}] volume estimation FAILED (worker error)")
            continue
        print(f"\n[{key}] {os.path.basename(path)} {source.size}")
        print(f"  scale: {r.scale_result.scale_source} "
              f"({r.scale_result.mm_per_pixel:.4f} mm/px)")
        for e in r.estimations:
            print(f"  - {e.class_name}: {e.volume_cm3:.0f} cm3 | "
                  f"{e.mass_g:.0f}±{e.mass_std_g:.0f} g | "
                  f"{e.nutrition.get('Calories', 0):.0f}±{e.nutrition_std.get('Calories', 0):.0f} kcal "
                  f"| {e.estimation_method}/{e.confidence_level}")
        print(f"  TOTAL: {r.total_nutrition.get('Calories', 0):.0f} "
              f"± {r.total_nutrition_std.get('Calories', 0):.0f} kcal")

    print("\n" + "-" * 74)
    # UI-side matching check: every food detection must match a volume
    # estimation (this is where the reference-serving fallback bug hid).
    from utils import _find_volume_estimation
    for key, r in results.items():
        if r is None or key not in raw_boxes:
            continue
        sx, sy = bbox_scales[key]
        for i, box in enumerate(raw_boxes[key]):
            match_box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
            est = _find_volume_estimation(list(r.estimations), match_box)
            check(f"{key}: detection {i} matched to volume estimate",
                  est is not None,
                  f"matched={est.class_name if est else 'NONE'}")

    pho = results.get("pho")
    if pho is not None:
        check("pho: no false ArUco", pho.scale_result.scale_source != "aruco_marker",
              f"source={pho.scale_result.scale_source}")
        check("pho: plausible total", 300 <= pho.total_nutrition.get("Calories", 0) <= 700,
              f"{pho.total_nutrition.get('Calories', 0):.0f} kcal (normal bowl ~350-500)")
    b1, b2 = results.get("banh-mi1"), results.get("banh-mi2")
    if b1 is not None and b2 is not None and b1.estimations and b2.estimations:
        m1 = sum(e.mass_g for e in b1.estimations)
        m2 = sum(e.mass_g for e in b2.estimations)
        v1 = sum(e.volume_cm3 for e in b1.estimations)
        v2 = sum(e.volume_cm3 for e in b2.estimations)
        check("banh-mi1 volume ~500cm3", 0.7 * 500 <= v1 <= 1.3 * 500,
              f"{v1:.0f} cm3 (measured GT ~500)")
        check("banh-mi2 volume ~500cm3", 0.7 * 500 <= v2 <= 1.3 * 500,
              f"{v2:.0f} cm3 (measured GT ~500)")
        check("banh-mi1 mass ~225g", 0.65 * BANH_MI_GT_G <= m1 <= 1.35 * BANH_MI_GT_G,
              f"{m1:.0f} g (GT 225)")
        check("banh-mi2 mass ~225g", 0.65 * BANH_MI_GT_G <= m2 <= 1.35 * BANH_MI_GT_G,
              f"{m2:.0f} g (GT 225)")
        mean_m = (m1 + m2) / 2
        # Same loaf (open vs closed shot) with the robust plane fit: the two
        # shots agree within a few percent; 25% catches gross inconsistency.
        check("banh-mi shot consistency <25%", abs(m1 - m2) / mean_m < 0.25,
              f"|{m1:.0f} - {m2:.0f}| / {mean_m:.0f} = {100*abs(m1-m2)/mean_m:.0f}%")
    c3 = results.get("com-tam3")
    if c3 is not None:
        check("com-tam3: >= 300 kcal", c3.total_nutrition.get("Calories", 0) >= 300,
              f"{c3.total_nutrition.get('Calories', 0):.0f} kcal (com tam suon ~600-800)")

    print("=" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("  ALL REAL-IMAGE CHECKS PASS")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
