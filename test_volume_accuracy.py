"""
Synthetic ground-truth accuracy tests for the volume estimation math.

Renders synthetic top-down scenes (known geometry -> known volume) into
depth maps, feeds them through the volume computation functions, and reports
the % error per case. Used to quantify and verify accuracy improvements.

A metric-depth bias factor is applied to the rendered depth to simulate the
known bias of monocular metric depth models (Depth Anything V2 metric).

Usage:
    python test_volume_accuracy.py
"""

import numpy as np

from volume_nutrition import (
    compute_volume_column_integration,
    compute_volume_bowl_container,
    compute_volume_side_vegetables,
    compute_volume_small_ingredient,
)

IMG = 640
FX = FY = 800.0
CX = CY = IMG / 2.0
Z_PLANE_MM = 600.0
MM_PER_PIXEL = Z_PLANE_MM / FX  # consistent pinhole scale at the plane

K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)

results = []


def render_depth(height_profile_mm, mask, bias=1.0):
    """Render Z = Z_plane - h(x,y), with optional multiplicative metric bias."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    height = height_profile_mm(xx, yy)
    z_true = Z_PLANE_MM - height
    z_meas = z_true * bias
    depth = np.where(mask, z_meas, Z_PLANE_MM * bias)
    return depth.astype(np.float32)


def ellipse_mask(a_mm, b_mm):
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    cx = cy = IMG / 2.0
    a_px, b_px = a_mm / MM_PER_PIXEL, b_mm / MM_PER_PIXEL
    return ((xx - cx) / a_px) ** 2 + ((yy - cy) / b_px) ** 2 <= 1.0


def disk_mask(r_mm):
    return ellipse_mask(r_mm, r_mm)


def bbox_of(mask):
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def rel_err(est, gt):
    return 100.0 * (est - gt) / gt


def record(case, est, gt, note=""):
    err = rel_err(est, gt)
    results.append((case, est, gt, err, note))
    return err


# ─── Ground-truth cases (routed through the real dispatcher) ───────────────

def make_estimator():
    from volume_nutrition import VolumeNutritionEstimator
    return VolumeNutritionEstimator()


def estimate_case(class_name, profile, mask, bias=1.0):
    depth = render_depth(profile, mask, bias)
    return make_estimator().estimate(
        depth_map_mm=depth, food_mask=mask, class_name=class_name,
        mm_per_pixel=MM_PER_PIXEL, scale_source="plate_heuristic",
        bbox=bbox_of(mask), K=K,
    )


def case_dome(bias=1.0):
    """Flat-plate dome (Com tam): h = H(1 - (x/a)^2 - (y/b)^2), peak 30mm.

    Footprint is 220mm wide so the plate-heuristic scale recompute is
    consistent with the rendered scene (plate assumption = 220mm).
    """
    a, b, H = 110.0, 77.0, 30.0
    mask = ellipse_mask(a, b)

    def profile(xx, yy):
        cx = cy = IMG / 2.0
        rho2 = ((xx - cx) / (a / MM_PER_PIXEL)) ** 2 + ((yy - cy) / (b / MM_PER_PIXEL)) ** 2
        return H * np.clip(1.0 - rho2, 0.0, 1.0)

    est = estimate_case("Com tam (Broken rice)", profile, mask, bias)
    gt = numeric_gt(profile, mask)
    record(f"column_integration dome (bias x{bias})", est.volume_cm3, gt)
    return est.volume_cm3, gt


def case_tall_block(bias=1.0):
    """Banh mi-like block 220x90mm footprint, 45mm tall (cap is 55mm now)."""
    w_mm, l_mm, h_mm = 220.0, 90.0, 45.0
    mask = ellipse_mask(w_mm / 2, l_mm / 2)  # rounded footprint

    def profile(xx, yy):
        return np.full_like(xx, h_mm)

    est = estimate_case("Banh mi (Vietnamese baguette sandwich)", profile, mask, bias)
    gt = numeric_gt(profile, mask)
    record(f"tall block 45mm (bias x{bias})", est.volume_cm3, gt)
    return est.volume_cm3, gt

def numeric_gt(height_profile_mm, mask):
    """Ground-truth volume = integral of the true height profile over the mask."""
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    h = height_profile_mm(xx, yy)
    return float(h[mask].sum() * MM_PER_PIXEL**2 / 1000.0)


def case_veg_mound(bias=1.0):
    """Spherical-cap vegetable mound (Rau), base r=66mm, height 35mm."""
    R_s, h_cap = 80.0, 35.0
    r_base = np.sqrt(R_s**2 - (R_s - h_cap) ** 2)
    mask = disk_mask(r_base)

    def profile(xx, yy):
        cx = cy = IMG / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * MM_PER_PIXEL
        return np.clip(np.sqrt(np.clip(R_s**2 - r**2, 0, None)) - (R_s - h_cap), 0, None)

    est = estimate_case("Rau (Vegetables)", profile, mask, bias)
    gt = numeric_gt(profile, mask)
    record(f"vegetable mound cap (bias x{bias})", est.volume_cm3, gt)
    return est.volume_cm3, gt


def case_bowl(bias=1.0):
    """Soup bowl (Pho): rim D=180mm, headroom 15mm. GT = frustum(R_fill, R_base, h_fill)."""
    from pipeline_config import BOWL_DEPTH_RATIO, BOWL_BASE_RATIO
    D_rim = 180.0
    R_rim, R_base = D_rim / 2, D_rim / 2 * BOWL_BASE_RATIO
    H_bowl = D_rim * BOWL_DEPTH_RATIO
    headroom = 15.0
    h_fill = H_bowl - headroom
    fill_frac = h_fill / H_bowl
    R_fill = R_base + (R_rim - R_base) * fill_frac
    gt = (np.pi * h_fill / 3.0) * (R_fill**2 + R_fill * R_base + R_base**2) / 1000.0

    mask = disk_mask(R_rim)

    def profile(xx, yy):
        # soup surface sits headroom BELOW the rim (= table plane reference)
        return np.full_like(xx, -headroom)

    est = estimate_case("Pho (Vietnamese noodle soup)", profile, mask, bias)
    record(f"bowl fill model (bias x{bias})", est.volume_cm3, gt)
    return est.volume_cm3, gt


def case_egg(bias=1.0):
    """Chicken egg 57x45mm footprint, ~55 cm3 real volume (priors, no depth)."""
    w_px = int(45 / MM_PER_PIXEL)
    h_px = int(57 / MM_PER_PIXEL)
    mask = np.zeros((IMG, IMG), dtype=bool)
    mask[IMG // 2 - h_px // 2:IMG // 2 + h_px // 2,
         IMG // 2 - w_px // 2:IMG // 2 + w_px // 2] = True
    bbox = bbox_of(mask)
    est, _ = compute_volume_small_ingredient(
        bbox, MM_PER_PIXEL, "Trung (Egg)")
    gt = 55.0
    record("egg spheroid (priors)", est, gt)
    return est, gt


def case_egg_depth(bias=1.0):
    """Egg measured with depth relief: ellipsoid dome profile, GT = 4/3*pi*a*b*c."""
    a_mm, b_mm, c_mm = 22.5, 28.5, 22.5  # semi-axes; height 2c = 45mm
    mask = ellipse_mask(a_mm, b_mm)

    def profile(xx, yy):
        cx = cy = IMG / 2.0
        rho2 = ((xx - cx) / (a_mm / MM_PER_PIXEL)) ** 2 + \
               ((yy - cy) / (b_mm / MM_PER_PIXEL)) ** 2
        return 2 * c_mm * np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))

    est = estimate_case("Trung (Egg)", profile, mask, bias)
    gt = numeric_gt(profile, mask)
    record(f"egg from depth relief (bias x{bias})", est.volume_cm3, gt)
    return est.volume_cm3, gt


def case_uncertainty():
    """nutrition_std must reflect scale-source error, not only density."""
    from volume_nutrition import VolumeNutritionEstimator

    mask = ellipse_mask(50, 50)
    depth = render_depth(lambda xx, yy: np.full_like(xx, 15.0), mask, 1.0)
    est = VolumeNutritionEstimator().estimate(
        depth_map_mm=depth, food_mask=mask, class_name="Banh mi (Vietnamese baguette sandwich)",
        mm_per_pixel=MM_PER_PIXEL, scale_source="plate_heuristic",
        bbox=bbox_of(mask), K=K,
    )
    rel = est.nutrition_std.get("Calories", 0) / max(est.nutrition.get("Calories", 1), 1e-6)
    expected = (0.18**2 + (0.05 / 0.45) ** 2) ** 0.5  # scale 18% + density 0.05/0.45
    ok = abs(rel - expected) < 0.02
    record("uncertainty ratio (banh mi, plate)", rel * 100, expected * 100,
           "OK" if ok else f"MISMATCH expected {expected:.3f}")
    return est.nutrition_std, expected


def run_all(label):
    results.clear()
    case_dome(1.0)
    case_dome(1.25)
    case_tall_block(1.0)
    case_tall_block(1.25)
    case_veg_mound(1.0)
    case_veg_mound(1.25)
    case_bowl(1.0)
    case_bowl(1.25)
    case_egg()
    case_egg_depth(1.0)
    case_egg_depth(1.25)
    case_uncertainty()
    print(f"\n{'=' * 74}\n  {label}\n{'=' * 74}")
    print(f"  {'Case':<38}{'Est':>9}{'GT':>9}{'Err %':>8}  Note")
    print("  " + "-" * 70)
    for case, est, gt, err, note in results:
        print(f"  {case:<38}{est:>9.1f}{gt:>9.1f}{err:>+8.1f}  {note}")
    mean_abs = np.mean([abs(e) for _, _, _, e, _ in results])
    print("  " + "-" * 70)
    print(f"  Mean absolute error: {mean_abs:.1f}%")
    return mean_abs


if __name__ == "__main__":
    run_all("VOLUME ESTIMATION ACCURACY (synthetic ground truth)")
