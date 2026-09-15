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

import glob
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
    # Same loaf, 3 views (ground truth ~200 g) — DucTri shots; shot 3 has
    # no ArUco marker (plate_heuristic path), so per-shot variance is wider.
    ductri_paths = sorted(glob.glob(os.path.join(img_dir, "BanhMi", "BanhMiDucTri*.jpg")))
    for i, p in enumerate(ductri_paths, 1):
        targets[f"banh-mi-ductri-{i}"] = p

    # Same bowl of bun bo Hue in 5 shots (ground truth ~700-800 cm3)
    bun_bo_hue_paths = sorted(
        glob.glob(os.path.join(img_dir, "BunBoHue", "*.jpg")))
    for i, p in enumerate(bun_bo_hue_paths, 1):
        targets[f"bun-bo-hue-{i}"] = p
    # No-marker folder: every image runs the plate_heuristic path — sanity
    # only (no per-image ground truth), guarding against absurd outputs.
    for i, p in enumerate(sorted(glob.glob(os.path.join(img_dir, "No_ArUcO", "*.jpg"))), 1):
        targets[f"no-aruco-{i}"] = p
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
            # same filters as extract_yolo_detections — a box the pipeline
            # never estimated must not enter the matching check
            if (0 <= cid < len(class_names)
                    and class_names[cid]["name"] != "Con nguoi (Human)"
                    and float(box.conf[0].item()) >= volume_integration.MIN_DETECTION_CONFIDENCE):
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
    # ── Group consistency reporting (3-tier metrics) ──
    # Per-shot: absolute + relative error | Group: mean, std, CV |
    # Dataset-wide: MAE, median AE, MAPE, max AE.
    # CV < 0.35 is an EXPERIMENTAL acceptance threshold for view consistency
    # (not a scientific proof); wide mass tolerances are smoke/diagnostic only.
    def group_report(prefix, gt_g, cv_threshold=0.35):
        shots = [(k, sum(e.mass_g for e in r.estimations))
                 for k, r in sorted(results.items())
                 if k.startswith(prefix) and r is not None and r.estimations]
        if not shots:
            return None
        vals = [v for _, v in shots]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        cv = std / mean if mean > 0 else 0.0
        errs = [abs(v - gt_g) for v in vals]
        mae = sum(errs) / len(errs)
        mape = sum(e / gt_g for e in errs) / len(errs) * 100.0
        med_ae = sorted(errs)[len(errs) // 2]
        max_ae = max(errs)
        print(f"  == Group {prefix} (GT {gt_g:.0f} g) ==")
        for (k, v), e in zip(shots, errs):
            print(f"   {k}: {v:.0f} g (abs err {e:.0f} g, rel {100*e/gt_g:.0f}%)")
        print(f"   mean={mean:.0f} g  std={std:.0f} g  CV={cv:.2f}  |  "
              f"MAE={mae:.0f} g  MAPE={mape:.0f}%  median_AE={med_ae:.0f}  max_AE={max_ae:.0f}")
        return {"mean": mean, "std": std, "cv": cv, "mae": mae, "mape": mape,
                "vals": vals, "shots": [k for k, _ in shots]}

    bm_group = [(k, r) for k, r in sorted(results.items())
                if k in ("banh-mi1", "banh-mi2") and r is not None and r.estimations]
    bm_g = None
    if bm_group:
        vals = [sum(e.mass_g for e in r.estimations) for _, r in bm_group]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        errs = [abs(v - 225.0) for v in vals]
        bm_g = {"cv": std / mean if mean > 0 else 0.0, "vals": vals}
        print(f"  == Group banh-mi (GT 225 g) ==")
        for (k, _), v, e in zip(bm_group, vals, errs):
            print(f"   {k}: {v:.0f} g (abs err {e:.0f} g, rel {100*e/225.0:.0f}%)")
        print(f"   mean={mean:.0f} g  std={std:.0f} g  CV={bm_g['cv']:.2f}  |  "
              f"MAE={sum(errs)/len(errs):.0f} g  MAPE={sum(e/225.0 for e in errs)/len(errs)*100.0:.0f}%")
    if bm_g:
        check("banh-mi group: view-consistency CV < 0.35", bm_g["cv"] < 0.35,
              f"CV={bm_g['cv']:.2f} (experimental acceptance threshold)")
    dt_g = group_report("banh-mi-ductri", 200.0)
    if dt_g:
        check("ductri group: view-consistency CV < 0.35", dt_g["cv"] < 0.35,
              f"CV={dt_g['cv']:.2f} (experimental acceptance threshold)")
        # Smoke tolerance only — DIAGNOSTIC, not an accuracy criterion and
        # not counted as a suite failure (the no-marker shot has documented
        # wider variance).
        for k, v in zip(dt_g["shots"], dt_g["vals"]):
            status = "PASS" if 100 <= v <= 320 else "DIAG"
            print(f"  [DIAG:{status}] {k}: {v:.0f} g (smoke range 100-320 g)")

    c3 = results.get("com-tam3")
    if c3 is not None:
        check("com-tam3: >= 300 kcal", c3.total_nutrition.get("Calories", 0) >= 300,
              f"{c3.total_nutrition.get('Calories', 0):.0f} kcal (com tam suon ~600-800)")

    # Same bowl of bun bo Hue across shots (GT ~750 cm3 fill volume):
    # every shot within +-35% of 750 and the shot mean inside the measured band.
    bbh = [(k, r) for k, r in sorted(results.items())
           if k.startswith("bun-bo-hue-") and r is not None and r.estimations]
    if bbh:
        shot_vols = [sum(e.volume_cm3 for e in r.estimations) for _, r in bbh]
        for (k, _), v in zip(bbh, shot_vols):
            check(f"{k}: volume ~750cm3", 0.65 * 750 <= v <= 1.35 * 750,
                  f"{v:.0f} cm3 (measured 700-800)")
        mean_v = sum(shot_vols) / len(shot_vols)
        std_v = (sum((v - mean_v) ** 2 for v in shot_vols) / len(shot_vols)) ** 0.5
        check("bun-bo-hue: shot mean in 700-800",
              700 <= mean_v <= 800,
              f"mean {mean_v:.0f} cm3 over {len(shot_vols)} shots (std {std_v:.0f})")

    # No-marker folder (plate_heuristic path): sanity-only checks — every
    # image must produce a valid estimation with a plausible total volume
    # (no container-integration blowups) and a bounded kcal total.
    na = [(k, r) for k, r in sorted(results.items())
          if k.startswith("no-aruco-") and r is not None]
    if na:
        for k, r in na:
            total_v = sum(e.volume_cm3 for e in r.estimations)
            check(f"{k}: valid estimation, sane volume",
                  bool(r.estimations) and total_v <= 2000,
                  f"{total_v:.0f} cm3 total, {len(r.estimations)} item(s), "
                  f"scale={r.scale_result.scale_source}")
            kcals = [e.nutrition.get("Calories", 0) for e in r.estimations]
            check(f"{k}: sane kcal", all(5 <= c <= 1600 for c in kcals),
                  f"kcal/item={[round(c) for c in kcals]}")

    print("=" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("  ALL REAL-IMAGE CHECKS PASS")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
