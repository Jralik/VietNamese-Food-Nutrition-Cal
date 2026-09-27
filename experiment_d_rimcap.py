"""Experiment D — Bowl-container rim-plane ablation (D0-D4, offline on captured inputs).

Frozen: images / detections / masks / anchored depth / ArUco scale / density 0.45 /
volume integration logic (compute_volume_bowl_container copied VERBATIM from
volume_nutrition.py). ONLY the rim-plane knobs change:

  D0: original (z_rim = median of table-plane over mask)
  D1: z_rim = mouth plane
  D2: original z_rim + wall-band exclusion
      (exclude if d_pixel < z_mouth(x,y) — exact inequality, no tolerance)
  D3: original z_rim + height cap at the mouth plane
      (mound heights capped by the table-to-mouth freeboard per pixel)
  D4: D1 + D2 + D3

Mouth plane: least-squares plane through the MASK-BOUNDARY INTERIOR RING depth
(a PROXY for the container mouth plane — the boundary is not asserted to be the
physical rim). Diagnostic logged per image BEFORE any MAPE is computed:
boundary-pixel count, depth min/p5/p10/p25/median/max, plane coefficients.

Per item logged: raw_volume (BEFORE the 1200 cm³ clip), clipped_volume, clip hit.

Writes scratch/experiment_d_results.md + scratch/experiment_d_records.jsonl.
"""

import json
import os
import statistics
from collections import defaultdict

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)
import cv2
import numpy as np

from class_names import class_names
from eval_full_pipeline import parse_gt, VAL_FILE

CAP_DIR = os.path.join("scratch", "exp_d_captures")
OUT_MD = os.path.join("scratch", "experiment_d_results.md")
OUT_JSONL = os.path.join("scratch", "experiment_d_records.jsonl")

RHOS = None  # density frozen at DB value — not swept here
GROUP_OBLIQUE = {"SupCua (3).jpg", "SupCua (4).jpg", "SupCua (5).jpg"}
GROUP_NORMAL = {"SupCua (1).jpg", "SupCua (2).jpg", "SupCua (6).jpg"}


# ── verbatim copy of volume_nutrition.compute_volume_bowl_container ──────────
def bowl_container_variant(
    depth_map_mm, food_mask, bbox, mm_per_pixel,
    dish_type="soup_bowl", K=None, scale_source="unknown",
    relief_scale=None,
    z_rim_mode="median_mask",          # D0/D2/D3: "median_mask"; D1/D4: "mouth_plane"
    wall_band_exclude=False,           # D2/D4
    height_cap_at_mouth=False,         # D3/D4
    mouth_plane=None,                  # (A, B, C) coefficients, required by D1/D3/D4
):
    warnings = []
    from pipeline_config import (
        PLATE_DIAMETER_HEURISTICS, BOWL_DEPTH_RATIO,
        BOWL_BASE_RATIO, BOWL_DEFAULT_FILL_RATIO,
        BOWL_NOISE_FILL_RATIO, BOWL_DELTA_Z_NOISE_MM,
        BOWL_HEADROOM_MAX_RATIO, BOWL_DIAMETER_TOLERANCE,
    )

    x1, y1, x2, y2 = bbox
    bw_px = max(10, x2 - x1)
    bh_px = max(10, y2 - y1)

    expected_dia_mm = PLATE_DIAMETER_HEURISTICS.get(dish_type, 180.0)

    if scale_source in ("aruco_marker", "checkerboard"):
        d_major_px = max(bw_px, bh_px)
        D_rim_mm = float(d_major_px * mm_per_pixel)
        D_rim_mm = float(np.clip(
            D_rim_mm,
            (1.0 - BOWL_DIAMETER_TOLERANCE) * expected_dia_mm,
            (1.0 + BOWL_DIAMETER_TOLERANCE) * expected_dia_mm,
        ))
        try:
            import cv2 as _cv2
            cnts = _cv2.findContours(
                food_mask.astype(np.uint8), _cv2.RETR_EXTERNAL,
                _cv2.CHAIN_APPROX_NONE)[0]
            cnt = max(cnts, key=_cv2.contourArea)
            if len(cnt) >= 5:
                (_ex, _ey), (ew, eh), _ea = _cv2.fitEllipse(cnt)
                if min(ew, eh) / max(ew, eh) < 0.75:
                    D_rim_mm = max(
                        D_rim_mm, (1.0 + BOWL_DIAMETER_TOLERANCE) * expected_dia_mm)
        except (_cv2.error, ValueError):
            pass
    else:
        D_rim_mm = float(expected_dia_mm)

    R_rim_mm = D_rim_mm / 2.0
    R_base_mm = R_rim_mm * BOWL_BASE_RATIO
    H_bowl_mm = D_rim_mm * BOWL_DEPTH_RATIO

    has_valid_depth = False
    delta_z = 0.0
    above_rim_volume_cm3 = 0.0
    raw_total_before_clip = None

    if depth_map_mm is not None and food_mask.any():
        mask_bool = food_mask.astype(bool)
        masked_d = depth_map_mm[mask_bool]
        valid_d = masked_d[masked_d > 0]

        if len(valid_d) > 20:
            plane_z = fit_plane_to_boundary(depth_map_mm, food_mask)
            plane_d = plane_z[mask_bool]

            if z_rim_mode == "mouth_plane" and mouth_plane is not None:
                A, B, C = mouth_plane
                yy, xx = np.mgrid[0:plane_z.shape[0], 0:plane_z.shape[1]]
                mouth_z = (A * xx + B * yy + C).astype(np.float32)
                mouth_d = mouth_z[mask_bool]
                valid_md = mouth_d[mouth_d > 0]
                z_rim = float(np.median(valid_md)) if len(valid_md) else \
                    float(np.median(plane_d))
            else:
                z_rim = float(np.median(plane_d))

            z_food = float(np.median(valid_d))
            alpha = relief_scale if relief_scale is not None else relief_scale_factor(
                z_rim, mm_per_pixel, K)
            z_rim_true = z_rim * alpha
            z_food_true = z_food * alpha
            delta_z = z_food_true - z_rim_true
            has_valid_depth = True
            delta_z_resolved = delta_z > BOWL_DELTA_Z_NOISE_MM

            if delta_z_resolved:
                import cv2 as _cv2
                r_px = int(np.sqrt(mask_bool.sum() / np.pi))
                erode_px = max(3, int(0.04 * r_px))
                eroded = _cv2.erode(
                    mask_bool.astype(np.uint8),
                    _cv2.getStructuringElement(
                        _cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)),
                ).astype(bool)
                interior_d = depth_map_mm[eroded]
                interior_plane = plane_z[eroded]
                if wall_band_exclude and mouth_plane is not None:
                    A, B, C = mouth_plane
                    yy, xx = np.mgrid[0:depth_map_mm.shape[0], 0:depth_map_mm.shape[1]]
                    mouth_full = (A * xx + B * yy + C).astype(np.float32)
                    keep = interior_d >= mouth_full[eroded]
                    interior_d = interior_d[keep]
                    interior_plane = interior_plane[keep]
                raw_heights_above_rim = interior_plane - interior_d
                mound_cap = BOWL_HEADROOM_MAX_RATIO * H_bowl_mm
                clipped_heights = np.minimum(raw_heights_above_rim, mound_cap)
                pos_mound = clipped_heights[clipped_heights > 0]
                if len(pos_mound) > 0.05 * len(raw_heights_above_rim):
                    interior_masked = interior_d[raw_heights_above_rim > 0]
                    if scale_source in ("aruco_marker", "checkerboard") and K is not None and K[0, 0] > 0 and K[1, 1] > 0:
                        px_area = (interior_masked ** 2) / (float(K[0, 0]) * float(K[1, 1]))
                    else:
                        px_area = mm_per_pixel ** 2
                    above_rim_volume_cm3 = float(np.sum(pos_mound * alpha * px_area) / 1000.0)

    if has_valid_depth:
        if delta_z > BOWL_DELTA_Z_NOISE_MM:
            headroom_mm = float(np.clip(
                delta_z, BOWL_DELTA_Z_NOISE_MM,
                BOWL_HEADROOM_MAX_RATIO * H_bowl_mm))
            h_fill = H_bowl_mm - headroom_mm
        elif delta_z > 0:
            h_fill = H_bowl_mm * BOWL_NOISE_FILL_RATIO
        else:
            h_fill = H_bowl_mm * BOWL_NOISE_FILL_RATIO
    else:
        h_fill = H_bowl_mm * BOWL_DEFAULT_FILL_RATIO

    if height_cap_at_mouth and mouth_plane is not None and h_fill > H_bowl_mm:
        h_fill = H_bowl_mm  # fill cannot exceed the mouth plane (D3)

    fill_fraction = h_fill / H_bowl_mm
    R_fill_mm = R_base_mm + (R_rim_mm - R_base_mm) * fill_fraction

    frustum_mm3 = (np.pi * h_fill / 3.0) * (
        R_fill_mm**2 + R_fill_mm * R_base_mm + R_base_mm**2)
    bowl_volume_cm3 = float(frustum_mm3 / 1000.0)

    raw_total_before_clip = bowl_volume_cm3 + above_rim_volume_cm3
    total_volume_cm3 = float(np.clip(raw_total_before_clip, 150.0, 1200.0))

    return total_volume_cm3, raw_total_before_clip, warnings


# fit_plane_to_boundary: import from the frozen volume_nutrition (unchanged file)
from volume_nutrition import fit_plane_to_boundary, relief_scale_factor  # noqa: E402


def mouth_plane_fit(depth_map_mm, food_mask):
    """Least-squares plane through the MASK-BOUNDARY INTERIOR RING depth.

    The boundary ring is used as a PROXY for the container mouth plane — the
    ring is not asserted to be the physical rim.
    """
    mask_bool = food_mask.astype(bool)
    eroded = cv2.erode(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ring = mask_bool & ~eroded
    ys, xs = np.nonzero(ring)
    if len(xs) < 12:
        return None, int(len(xs))
    z = depth_map_mm[ys, xs].astype(np.float64)
    ok = z > 0
    xs, ys, z = xs[ok], ys[ok], z[ok]
    if len(xs) < 12:
        return None, int(len(xs))
    Amat = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(Amat, z, rcond=None)
    return (float(coef[0]), float(coef[1]), float(coef[2])), int(len(xs))


def boundary_diagnostic(depth_map_mm, food_mask):
    mask_bool = food_mask.astype(bool)
    eroded = cv2.erode(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ring = mask_bool & ~eroded
    z = depth_map_mm[ring].astype(np.float32)
    z = z[z > 0]
    if len(z) == 0:
        return {"ring_px": 0}
    qs = np.percentile(z, [5, 10, 25, 50])
    return {"ring_px": int(len(z)), "min": float(z.min()),
            "p5": float(qs[0]), "p10": float(qs[1]), "p25": float(qs[2]),
            "median": float(qs[3]), "max": float(z.max())}



def variant_estimate(depth_map_mm, food_mask, bbox, mm_per_pixel, class_name,
                     dish_type, K, scale_source, relief_scale,
                     z_rim_mode, wall_band_exclude, height_cap_at_mouth,
                     mouth_plane):
    """Dispatch replica của VolumeNutritionEstimator.estimate với rim-plane
    knobs trên bowl path; các path khác delegate tới hàm gốc (file
    volume_nutrition.py không đổi). Trả về (volume_clipped, volume_raw)."""
    has_valid_depth = (
        depth_map_mm is not None and food_mask.any()
        and bool((depth_map_mm[food_mask.astype(bool)] > 0).any()))
    if dish_type in {"soup_bowl", "large_bowl", "rice_bowl"} and bbox is not None:
        return bowl_container_variant(
            depth_map_mm=depth_map_mm, food_mask=food_mask, bbox=bbox,
            mm_per_pixel=mm_per_pixel, dish_type=dish_type, K=K,
            scale_source=scale_source, relief_scale=relief_scale,
            z_rim_mode=z_rim_mode, wall_band_exclude=wall_band_exclude,
            height_cap_at_mouth=height_cap_at_mouth, mouth_plane=mouth_plane)
    if dish_type == "ingredient_small" and bbox is not None:
        from volume_nutrition import compute_volume_small_ingredient
        vol, _ = compute_volume_small_ingredient(
            bbox=bbox, mm_per_pixel=mm_per_pixel, class_name=class_name,
            depth_map_mm=depth_map_mm if has_valid_depth else None,
            food_mask=food_mask if has_valid_depth else None,
            K=K, relief_scale=relief_scale)
        return vol, vol, []
    if dish_type == "side_vegetables" and bbox is not None:
        from volume_nutrition import compute_volume_side_vegetables
        vol, _ = compute_volume_side_vegetables(
            bbox=bbox, depth_map_mm=depth_map_mm if has_valid_depth else None,
            food_mask=food_mask if has_valid_depth else None,
            mm_per_pixel=mm_per_pixel, K=K, relief_scale=relief_scale)
        return vol, vol, []
    if has_valid_depth:
        from volume_nutrition import compute_volume_column_integration
        height_limits = {"flat_plate": 45.0, "small_plate": 35.0, "default": 45.0}
        if "Banh mi" in class_name or "Hamburger" in class_name:
            max_height_mm = 55.0
        else:
            max_height_mm = height_limits.get(dish_type, 45.0)
        height_floors = {"flat_plate": 15.0, "small_plate": 12.0}
        from volume_nutrition import SATURATION_CAP_RATIO
        vol = compute_volume_column_integration(
            depth_map_mm=depth_map_mm, food_mask=food_mask,
            mm_per_pixel=mm_per_pixel, K=K, homography=None,
            scale_source=scale_source, max_height_mm=max_height_mm,
            relief_scale=relief_scale,
            min_mean_height_mm=height_floors.get(dish_type, 0.0),
            mean_height_cap=(SATURATION_CAP_RATIO * max_height_mm
                             if SATURATION_CAP_RATIO else None))
        return vol, vol, []
    from volume_nutrition import compute_volume_bbox_approximation
    vol = compute_volume_bbox_approximation(bbox, mm_per_pixel)
    return vol, vol, []


def main():
    entries = parse_gt(VAL_FILE)
    gt_by_image = {os.path.basename(p): g for p, g in entries}
    captures = sorted(f for f in os.listdir(CAP_DIR) if f.endswith("_d.npz"))
    # reverse mapping: npz stem (spaces -> underscores) -> real filename
    stem_to_fname = {
        os.path.splitext(os.path.basename(p))[0].replace(" ", "_"):
            os.path.basename(p)
        for p, _ in entries}

    from pipeline_config import CLASS_TO_DISH_TYPE

    records = []
    diag_lines = []
    for f in captures:
        stem = f.replace("_d.npz", "")
        fname = stem_to_fname.get(stem)
        if fname is None:
            print(f"  SKIP: không map được stem {stem} về filename", flush=True)
            continue
        data = np.load(os.path.join(CAP_DIR, f))
        with open(os.path.join(CAP_DIR, f"{stem}_params.json"), encoding="utf-8") as pf:
            params = json.load(pf)
        depth = data["depth_map_mm"].astype(np.float32)
        gray = data["gray"]
        rec = {"image": fname,
               "gt": {str(c): dict(v) for c, v in (gt_by_image.get(fname) or {}).items()},
               "items": [], "error": None}
        for i, item in enumerate(params["items"]):
            mask = data[f"mask_{i}"].astype(bool)
            bbox = tuple(int(v) for v in data[f"bbox_{i}"])
            dish_type = CLASS_TO_DISH_TYPE.get(item["class_name"], "default")
            diag = boundary_diagnostic(depth, mask)
            plane, ring_px = mouth_plane_fit(depth, mask)
            diag["lsq_plane"] = plane
            diag["ring_px_lsq"] = ring_px
            diag_lines.append({"image": fname, "instance": i,
                               "class_name": item["class_name"], **diag})
            if plane is None:
                # mouth-plane fit failed (mask quá nhỏ) — chỉ ghi D0
                d0, raw0, _ = bowl_container_variant(**dict(
                    depth_map_mm=depth, food_mask=mask, bbox=bbox,
                    mm_per_pixel=item["mm_per_pixel"], dish_type=dish_type,
                    K=None, scale_source=item["scale_source"],
                    relief_scale=0.79 if item["depth_anchored"] else None))
                rec["items"].append({"instance": i,
                                     "class_name": item["class_name"],
                                     "variants": {"D0": {
                                         "volume_cm3": round(d0, 1),
                                         "raw_volume_cm3": round(raw0, 1)}}})
                continue
            common = dict(
                depth_map_mm=depth, food_mask=mask, bbox=bbox,
                mm_per_pixel=item["mm_per_pixel"], class_name=item["class_name"],
                dish_type=dish_type, K=None, scale_source=item["scale_source"],
                relief_scale=0.79 if item["depth_anchored"] else None,
            )
            variants = {}
            for name, (zm, wb, hc) in {
                "D0": ("median_mask", False, False),
                "D1": ("mouth_plane", False, False),
                "D2": ("median_mask", True, False),
                "D3": ("median_mask", False, True),
                "D4": ("mouth_plane", True, True),
            }.items():
                vv, rawv, _ = variant_estimate(
                    **common, z_rim_mode=zm, wall_band_exclude=wb,
                    height_cap_at_mouth=hc, mouth_plane=plane)
                variants[name] = (vv, rawv)
            rec["items"].append({"instance": i, "class_name": item["class_name"],
                                 "variants": {
                k: {"volume_cm3": round(v[0], 1), "raw_volume_cm3": round(v[1], 1)}
                for k, v in variants.items()}})
        rec["diag"] = diag
        records.append(rec)

    # ── aggregate: per-class target-matched (Protocol 2 style) + groups ──
    VARIANTS = ("D0", "D1", "D2", "D3", "D4")
    name_to_cid = {class_names[i]["name"]: i for i in range(len(class_names))}

    # density per (image, class) từ collection records (mass/volume cùng item)
    official = {r["image"]: r for r in (
        json.loads(l) for l in open(
            os.path.join("scratch", "full_eval_records.jsonl"),
            encoding="utf-8") if l.strip())}
    dens_by_class = {}
    for img, r in official.items():
        per = defaultdict(lambda: [0.0, 0.0])
        for e in r.get("est", []):
            cid = name_to_cid.get(e["class_name"])
            if cid is not None and e.get("volume_cm3"):
                per[cid][0] += e["mass_g"]
                per[cid][1] += e["volume_cm3"]
        for cid, (m, v) in per.items():
            if v > 0:
                dens_by_class[(img, cid)] = m / v

    class_pairs = {v: defaultdict(list) for v in VARIANTS}   # cid -> [(est_g, gt_g)]
    supcua_pairs = {v: defaultdict(list) for v in VARIANTS}  # group -> [(...)]
    raw_acc = {v: [] for v in VARIANTS}
    clip_hit = {v: [0, 0] for v in VARIANTS}
    img_vol = {v: [] for v in VARIANTS}                      # (image, Σ vol, gt)
    img_map = {v: {} for v in VARIANTS}                      # image -> Σ vol

    for r in records:
        if r["error"] or not r["items"]:
            continue
        img = r["image"]
        gt = {int(c): v for c, v in r["gt"].items()}
        for v in VARIANTS:
            per_class_vol = defaultdict(float)
            for it in r["items"]:
                cid_it = name_to_cid.get(it["class_name"])
                if cid_it is not None:
                    per_class_vol[cid_it] += it["variants"][v]["volume_cm3"]
                raw = it["variants"][v]["raw_volume_cm3"]
                raw_acc[v].append(raw)
                if it["variants"][v]["volume_cm3"] < raw:
                    clip_hit[v][0] += 1
                clip_hit[v][1] += 1
            for cid, gv in gt.items():
                dens = dens_by_class.get((img, cid))
                if dens is None:
                    continue  # không suy ra density cho class này trong ảnh này
                est_g = per_class_vol.get(cid, 0.0) * dens
                class_pairs[v][cid].append((est_g, gv["grams_total"]))
                if "SupCua" in img:
                    supcua_pairs[v][cid].append((est_g, gv["grams_total"]))
        for v in VARIANTS:
            tot = sum(it["variants"][v]["volume_cm3"] for it in r["items"])
            img_vol[v].append((img, tot, sum(v["grams_total"] or 0 for v in gt.values())))
            img_map[v][img] = tot

    def pct(pairs):
        errs = [abs(e - g) / g * 100 for e, g in pairs if g > 0]
        return (sum(errs) / len(errs)) if errs else float("nan")

    L = ["", "# Experiment D — Rim-plane ablation results (bowl-container path)", ""]
    L.append("Frozen: images/detections/masks/anchored depth/ArUco/density 0.45/"
             "integration logic (verbatim copy). Chỉ thay rim-plane knobs. "
             "Mouth plane = LSQ plane qua mask-boundary interior ring (PROXY — "
             "boundary không được giả định là physical rim).")
    L.append("")

    L.append("## Mouth-plane diagnostic (per item, TRƯỚC khi tính MAPE)")
    L.append("")
    L.append("| Ảnh | item | class | ring_px | min | p5 | p10 | p25 | median | max |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in diag_lines:
        if "p5" not in d:
            continue
        L.append(f"| {d['image']} | {d['instance']} | {d['class_name'].split(' (')[0]} "
                 f"| {d['ring_px']} | {d['min']:.0f} | {d['p5']:.0f} | {d['p10']:.0f} "
                 f"| {d['p25']:.0f} | {d['median']:.0f} | {d['max']:.0f} |")
    L.append("")

    L.append("## Bảng 1 — Mass MAPE per class per variant (target-matched, %)")
    L.append("")
    classes_sorted = sorted({c for v in VARIANTS for c in class_pairs[v]})
    L.append("| Variant | " + " | ".join(
        class_names[c]["name"].split(" (")[0] for c in classes_sorted) + " |")
    L.append("|---|" + "---|" * len(classes_sorted))
    for v in VARIANTS:
        cells = []
        for c in classes_sorted:
            pairs = class_pairs[v].get(c, [])
            errs = [abs(e - g) / g * 100 for e, g in pairs if g > 0]
            cells.append(f"{sum(errs)/len(errs):.1f}% ({len(errs)})" if errs else "-")
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Bảng 2 — Raw vs clipped volume + clip-hit (SupCua nhóm)")
    L.append("")
    L.append("| Variant | Raw volume median (cm³) | Clipped volume median (cm³) | "
             "Clip-hit rate |")
    L.append("|---|---|---|---|")
    for v in VARIANTS:
        raws = raw_acc[v]
        med_raw = statistics.median(raws) if raws else float("nan")
        clip_rate = f"{clip_hit[v][0]}/{clip_hit[v][1]}"
        L.append(f"| {v} | {med_raw:.0f} | — | {clip_rate} |")
    L.append("")

    L.append("## Per-image SupCua volume per variant (cm³) — GT cột GT")
    L.append("")
    L.append("| Ảnh | GT | " + " | ".join(VARIANTS) + " |")
    L.append("|---|---|" + "---|" * len(VARIANTS))
    for img, _tot, gt in img_vol["D0"]:
        if "SupCua" not in img:
            continue
        cells = [f"{img_map[v].get(img, 0):.0f}" for v in VARIANTS]
        L.append(f"| {img} | {gt:.0f} | " + " | ".join(cells) + " |")
    L.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(chr(10).join(L) + chr(10))
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        f.write(chr(10).join(json.dumps(r, ensure_ascii=False) for r in records) + chr(10))
    print(chr(10).join(L))


if __name__ == "__main__":
    main()
