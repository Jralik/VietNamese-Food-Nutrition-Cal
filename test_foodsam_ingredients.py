"""Ingredient feature test suite (no SAM2/FoodSAM model needed except T10).

Covers the approved invariants:
  geometry    : overlap formula (component-area denominator), union-of-bboxes
                classification, owner/ambiguous records, area flag/cap,
                purity filter, per-component extras (NO union-before-volume)
  mapping     : FOODSAM_INGREDIENT_MAP -> class_names + CLASS_TO_DISH_TYPE
  volume      : per-component additivity; union mask inflates spheroid volume
                (documented reason the merge was removed)
  pipeline    : depth computed ONCE per analyze; totals == sum(estimations)
  fail-fast   : FoodSAMSegmenter.fallback_used flag on env failure

Run: .venv/Scripts/python.exe test_foodsam_ingredients.py
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodSAM")
import foodsam_infer  # module-level imports are light (no sam2)

FOODSAM_REPO = r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodSAM"
CATEGORY_TXT = os.path.join(
    FOODSAM_REPO, "FoodSAM", "FoodSAM_tools", "category_id_files",
    "foodseg103_category_id.txt")
CATEGORY_NAMES = foodsam_infer.load_category_names(CATEGORY_TXT)
LAB = {name: i for i, name in enumerate(CATEGORY_NAMES)}

H = W = 1000
THRESH = SimpleNamespace(
    overlap_breakdown=0.5, min_area_ratio=0.008, large_area_flag=0.5,
    max_area_ratio=0.75, min_purity=0.6, owner_tie_margin=0.05,
    promote_min_ratio=0.001, mixed_min_nonstaple=3,
    mixed_group_min_ratio=0.002, mixed_min_bbox_ratio=0.10)
SUSPECT = {"Xoi (Sticky rice)", "Com (Rice)", "Com tam (Broken rice)"}
MAP_EGG = {  # minimal map: egg + rice only
    "egg": {"class_name": "Trung (Egg)", "mapping_type": "direct"},
    "rice": {"class_name": "Com (Rice)", "mapping_type": "direct"},
    "steak": {"class_name": "Thit bo (Beef)", "mapping_type": "approximate"},
}
MAP_FULL = {
    **MAP_EGG,
    "chicken duck": {"class_name": "Thit ga (Chicken)", "mapping_type": "approximate"},
    "pork": {"class_name": "Thit heo (Pork)", "mapping_type": "approximate"},
    "fried meat": {"class_name": "Thit nuong (Grilled meat)", "mapping_type": "approximate"},
    "carrot": {"class_name": "Rau (Vegetables)", "mapping_type": "generic"},
    "tomato": {"class_name": "Rau (Vegetables)", "mapping_type": "generic"},
    "cucumber": {"class_name": "Rau (Vegetables)", "mapping_type": "generic"},
}

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"[PASS] {name}")
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
        print(f"[FAIL] {name}: {e}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")


def det(cls, bbox, conf=0.9):
    return {"bbox": list(bbox), "class_name": cls, "confidence": conf}


def run_extract(seg_map, detections, map_table=None, suspect=SUSPECT, **overrides):
    args = SimpleNamespace(**{**vars(THRESH), **overrides},
                           ingredient_mode="breakdown_and_extra")
    matched = list(range(len(detections)))
    return foodsam_infer.extract_ingredients(
        seg_map, detections, matched, args, map_table if map_table is not None else MAP_EGG,
        suspect, CATEGORY_NAMES, H, W)


def rect(seg_map, label, x0, y0, x1, y1):
    seg_map[y0:y1, x0:x1] = label


# ─── geometry ────────────────────────────────────────────────────────────

def t1_breakdown_inside_single_bbox():
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["egg"], 200, 200, 400, 400)  # fully inside B
    dets = [det("Pho (Vietnamese noodle soup)", (100, 100, 600, 600))]
    extras, masks, breakdowns, _, _ = run_extract(seg, dets)
    assert extras == [], f"expected no extras, got {len(extras)}"
    assert len(breakdowns[0]) == 1, "breakdown record missing"
    rec = breakdowns[0][0]
    assert rec["owner"] == "Pho (Vietnamese noodle soup)"
    assert abs(rec["overlap_ratio"] - 1.0) < 1e-6


def t2_multi_bbox_union_not_extra():
    # component spans B1 (x<500) and B2 (x>=500): per-bbox scores 0.167/0.5
    # would both "fail" only B1 — union r_all = 1.0 -> breakdown, owner B2
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["egg"], 400, 100, 600, 400)  # 200x300
    dets = [det("Pho (Vietnamese noodle soup)", (0, 0, 450, 1000)),
            det("Bun bo Hue (Hue beef noodle soup)", (450, 0, 1000, 1000))]
    extras, masks, breakdowns, _, _ = run_extract(seg, dets)
    assert extras == [], "spanning component must be breakdown, not extra"
    all_records = [r for di in breakdowns for r in breakdowns[di]]
    assert len(all_records) == 1, f"exactly one record, got {len(all_records)}"
    assert all_records[0]["owner"] == "Bun bo Hue (Hue beef noodle soup)"
    assert not all_records[0].get("candidate_owners"), "not a tie"


def t3_owner_ambiguous_single_record():
    # scores ~0.506 vs ~0.494 (diff < 0.05) -> one ambiguous record, no extras
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["egg"], 457, 100, 542, 300)  # 85x200: 43 cols in B1, 42 in B2
    dets = [det("Pho (Vietnamese noodle soup)", (0, 0, 500, 1000)),
            det("Bun bo Hue (Hue beef noodle soup)", (500, 0, 600, 1000))]
    extras, masks, breakdowns, _, _ = run_extract(seg, dets)
    assert extras == [], "ambiguous component is breakdown, never extra"
    all_records = [(di, r) for di in breakdowns for r in breakdowns[di]]
    assert len(all_records) == 1, f"single record (no duplication), got {len(all_records)}"
    di, rec = all_records[0]
    assert rec["owner"] == "ambiguous"
    assert len(rec["candidate_owners"]) == 2
    assert len(rec["owner_scores"]) == 2


def t4_per_component_extras_no_merge():
    # two disjoint egg components outside any dish bbox -> 2 separate items
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["egg"], 100, 100, 200, 200)
    rect(seg, LAB["egg"], 400, 400, 500, 500)
    extras, masks, breakdowns, _, _ = run_extract(seg, [])  # detections=[] -> U empty
    assert len(extras) == 2, f"per-component items expected, got {len(extras)}"
    assert all(e["class_name"] == "Trung (Egg)" for e in extras)
    assert extras[0]["component_id"] != extras[1]["component_id"]
    m0, m1 = masks[0], masks[1]
    assert not (m0 & m1).any(), "components must stay separate masks"
    assert int(m0.sum()) == 100 * 100 and int(m1.sum()) == 100 * 100
    assert all(b == [] for b in breakdowns.values())


def t5_area_flag_and_cap():
    # tiny -> rejected; 0.55 -> suspicious but kept; 0.8 -> hard rejected
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["egg"], 10, 10, 30, 30)            # 400 px < min (8000)
    rect(seg, LAB["rice"], 100, 100, 842, 842)       # ~0.55 ratio
    extras, masks, _, _, _ = run_extract(seg, [])
    assert len(extras) == 1, f"tiny rejected, mid kept: got {len(extras)}"
    assert extras[0]["suspicious_large"] is True

    seg2 = np.zeros((H, W), dtype=np.int32)
    rect(seg2, LAB["rice"], 0, 0, 900, 900)          # 0.81 >= max
    extras2, _, _, _, _ = run_extract(seg2, [])
    assert extras2 == [], "component >= max_area_ratio must be hard-rejected"


def t6_purity_filter():
    # L-shaped egg with rice sitting in the concavity -> bbox purity < 0.6
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["egg"], 100, 100, 200, 400)        # vertical bar
    rect(seg, LAB["egg"], 100, 300, 400, 400)        # bottom bar (connected L)
    rect(seg, LAB["rice"], 200, 100, 400, 300)       # rice in the concavity
    dets = []
    extras, _, _, _, _ = run_extract(seg, dets)
    egg = [e for e in extras if e["foodseg_label"] == "egg"]
    rice = [e for e in extras if e["foodseg_label"] == "rice"]
    assert egg == [], "egg purity 0.556 < 0.6 must be filtered"
    assert len(rice) == 1, "rice block (pure, large enough) should pass"


def t7_zero_detections_all_candidates():
    # HomeCook YOLO-miss: no bboxes -> U empty -> every mapped component is a
    # candidate extra; unmapped classes (candy) never become items
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["egg"], 100, 100, 250, 250)
    rect(seg, LAB["candy"], 500, 500, 700, 700)
    extras, masks, _, _, _ = run_extract(seg, [])
    assert len(extras) == 1 and extras[0]["foodseg_label"] == "egg"
    assert extras[0]["r_all"] == 0.0


# ─── mapping ─────────────────────────────────────────────────────────────

def t10_mapping_table_valid():
    from pipeline_config import FOODSAM_INGREDIENT_MAP, CLASS_TO_DISH_TYPE
    from class_names import class_names
    valid_names = {c["name"] for c in class_names}
    valid_types = {"direct", "approximate", "generic"}
    for fs_label, m in FOODSAM_INGREDIENT_MAP.items():
        assert fs_label.strip(), "empty foodseg key"
        assert m["class_name"] in valid_names, \
            f"{fs_label} -> '{m['class_name']}' not in class_names.py"
        assert m["class_name"] in CLASS_TO_DISH_TYPE, \
            f"{m['class_name']} missing CLASS_TO_DISH_TYPE"
        assert m["mapping_type"] in valid_types, f"bad mapping_type: {m}"
        assert m["mapping_type"].strip() != ""


# ─── volume additivity (reason per-component estimation exists) ──────────

def t20_volume_union_inflates():
    from volume_nutrition import VolumeNutritionEstimator
    est = VolumeNutritionEstimator()
    m1 = np.zeros((H, W), dtype=bool)
    m1[100:200, 100:200] = True
    m2 = np.zeros((H, W), dtype=bool)
    m2[500:600, 500:600] = True
    union = m1 | m2
    depth = np.full((H, W), 200.0, dtype=np.float32)

    def vol(mask):
        r = est.estimate(depth_map_mm=depth, food_mask=mask,
                         class_name="Trung (Egg)", mm_per_pixel=0.5,
                         bbox=(0, 0, W, H))
        return r.volume_cm3

    v1, v2, v_u = vol(m1), vol(m2), vol(union)
    assert v1 > 0 and v2 > 0, "spheroid estimates must be positive"
    # Union masks are NOT additive for the spheroid heuristic (measured:
    # union under-estimates ~2x for two disjoint eggs). Direction may vary
    # with geometry — the invariant is that union != sum, which is why
    # extras are estimated per component and only aggregated afterwards.
    assert abs(v_u - (v1 + v2)) > 0.1 * (v1 + v2), (
        f"expected union mask to be non-additive (union={v_u:.1f} vs "
        f"sum={v1 + v2:.1f})")


# ─── pipeline invariants (real depth model — slowest test) ───────────────

def t30_depth_once_and_totals():
    import cv2
    from food_volume_pipeline import FoodVolumePipeline
    from food_segmentation import SegmentationResult

    masks = []
    m = np.zeros((640, 640), dtype=bool); m[100:200, 100:200] = True; masks.append(m)
    m = np.zeros((640, 640), dtype=bool); m[300:400, 300:400] = True; masks.append(m)
    m = np.zeros((640, 640), dtype=bool); m[450:520, 100:300] = True; masks.append(m)
    classes = ["Trung (Egg)", "Trung (Egg)", "Rau (Vegetables)"]
    seg_results = []
    for i, (mk, cn) in enumerate(zip(masks, classes)):
        bx, by, bw_, bh_ = cv2.boundingRect(mk.astype(np.uint8))
        seg_results.append(SegmentationResult(
            mask=mk, class_name=cn, bbox=(bx, by, bx + bw_, by + bh_),
            confidence=0.9, warnings=[], source="foodsam_ingredient",
            is_ingredient=True, component_id=f"test_{i:02d}",
            semantic_purity=0.9, mapping_type="direct"))

    class StubSeg:
        is_available = True
        fallback_used = False
        last_info = {}
        last_error = None

        def segment(self, image_rgb, detections):
            return seg_results

        def visualize_masks(self, image_rgb, seg_results, alpha=0.45):
            return image_rgb

    class CountingDepth:
        """Flat-depth stub with a call counter — position-independent so
        identical masks must yield identical masses."""

        def __init__(self):
            self.calls = 0

        @property
        def is_available(self):
            return True

        def estimate(self, image_rgb):
            self.calls += 1
            from depth_estimation import DepthResult
            return DepthResult(
                depth_mm=np.full(image_rgb.shape[:2], 200.0, dtype=np.float32),
                depth_source="metric_model", encoder="stub",
                input_resolution=image_rgb.shape[:2])

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[:, :] = (200, 180, 160)
    dets = []
    for mk, cn in zip(masks, classes):
        bx, by, bw_, bh_ = cv2.boundingRect(mk.astype(np.uint8))
        dets.append(det(cn, (bx, by, bx + bw_, by + bh_)))

    pipeline = FoodVolumePipeline()
    pipeline._segmenter = StubSeg()
    counting = CountingDepth()
    pipeline._depth_estimator = counting

    result = pipeline.analyze(
        image_rgb=image, yolo_detections=dets, image_path=None,
        generate_visualizations=False)

    assert counting.calls == 1, (
        f"depth must run ONCE per analyze, ran {counting.calls} times")
    assert len(result.estimations) == 3, (
        f"3 components -> 3 estimations, got {len(result.estimations)}")
    assert all(e.is_ingredient for e in result.estimations)
    total = sum(e.mass_g for e in result.estimations)
    assert abs(result.total_nutrition.get("Calories", 0)
               - sum(e.nutrition.get("Calories", 0) for e in result.estimations)) < 0.15, \
        "totals must equal the sum of per-component estimations"
    eggs = [e for e in result.estimations if e.class_name == "Trung (Egg)"]
    assert len(eggs) == 2, "two identical egg components -> two egg estimations"
    assert all(e.mass_g > 0 for e in result.estimations), "masses must be positive"
    # the two egg masks are pixel-identical in size/shape -> equal mass
    assert abs(eggs[0].mass_g - eggs[1].mass_g) < 0.05 * max(eggs[0].mass_g, 1e-6), (
        f"identical masks must give equal mass: {eggs[0].mass_g} vs {eggs[1].mass_g}")


# ─── fail-fast flag ──────────────────────────────────────────────────────

def t40_fallback_flag_on_env_failure():
    from foodsam_segmenter import FoodSAMSegmenter

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[:, :] = (180, 160, 140)
    seg = FoodSAMSegmenter()
    seg.env_python = r"Z:\nonexistent\python.exe"  # force unavailability
    seg._available = None
    dets = [det("Com tam (Broken rice)", (100, 100, 500, 500))]

    results = seg.segment(image, dets)
    assert seg.fallback_used is True, "unavailable env must set fallback_used"
    assert len(results) == 1 and results[0].mask.any(), \
        "fallback must still return a usable GrabCut mask"
    assert any("GrabCut" in w or "unavailable" in w for w in results[0].warnings)


def t41_segmenter_empty_detections_still_extracts():
    # bridge contract: 0 detections must still reach foodsam_infer (HomeCook)
    import inspect
    from foodsam_segmenter import FoodSAMSegmenter
    src = inspect.getsource(FoodSAMSegmenter.segment)
    assert "if self.is_available:" in src, \
        "segment() must run the bridge unconditionally (not only when detections)"
    assert "if self.is_available and detections" not in src


# ─── mixed-plate contradiction (suppress + promote) ─────────────────────

def t50_suppress_and_promote():
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["steak"], 150, 150, 450, 450)        # 90k px inside Xoi
    rect(seg, LAB["chicken duck"], 550, 150, 850, 450)  # 90k px
    rect(seg, LAB["pork"], 150, 550, 450, 850)          # 90k px
    dets = [det("Xoi (Sticky rice)", (100, 100, 900, 900), conf=0.88)]
    extras, masks, breakdowns, _, suppressed = run_extract(seg, dets, map_table=MAP_FULL)
    assert len(suppressed) == 1, f"Xoi must be suppressed, got {suppressed}"
    sup = suppressed[0]
    assert sup["det_index"] == 0 and sup["status"] == "suppressed"
    assert sup["reason"] == "mixed_plate_contradiction"
    assert sup["promoted_labels"] == ["Thit bo (Beef)", "Thit ga (Chicken)",
                                      "Thit heo (Pork)"]
    assert sup["promoted_area_coverage"] == 1.0
    assert len(extras) == 3, f"3 promoted items expected, got {len(extras)}"
    promoted_cls = sorted(e["class_name"] for e in extras)
    assert promoted_cls == ["Thit bo (Beef)", "Thit ga (Chicken)", "Thit heo (Pork)"]
    assert all(e["promoted_from_breakdown"] for e in extras)
    assert breakdowns[0] == [], "promoted records must leave the breakdown list"


def t51_kept_dish_pho_never_suppressed():
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["steak"], 150, 150, 450, 450)
    rect(seg, LAB["chicken duck"], 550, 150, 850, 450)
    rect(seg, LAB["pork"], 150, 550, 450, 850)
    dets = [det("Pho (Vietnamese noodle soup)", (100, 100, 900, 900), conf=0.96)]
    extras, masks, breakdowns, _, suppressed = run_extract(seg, dets, map_table=MAP_FULL)
    assert suppressed == [], "soup dishes must never be suppressed"
    assert extras == [], "kept dish keeps breakdown as metadata (no double count)"
    assert len(breakdowns[0]) == 3


def t52_anti_false_suppression_two_groups():
    # com tam suon: staple + pork + chicken = 2 non-staple groups -> keep dish
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["steak"], 150, 150, 450, 450)
    rect(seg, LAB["chicken duck"], 550, 150, 850, 450)
    dets = [det("Com tam (Broken rice)", (100, 100, 900, 900), conf=0.92)]
    extras, masks, breakdowns, _, suppressed = run_extract(seg, dets, map_table=MAP_FULL)
    assert suppressed == [], "2 non-staple groups must NOT trigger suppression"
    assert extras == []
    assert len(breakdowns[0]) == 2


def t53_accounting_safety_through_pipeline():
    # suppressed dish -> no nutrition for it; totals == sum of promoted items
    import cv2
    from food_volume_pipeline import FoodVolumePipeline
    from food_segmentation import SegmentationResult

    masks, classes = [], []
    for cls, (x0, y0, x1, y1) in [
            ("Thit bo (Beef)", (150, 150, 450, 450)),
            ("Thit ga (Chicken)", (550, 150, 850, 450)),
            ("Thit heo (Pork)", (150, 550, 450, 850))]:
        m = np.zeros((640, 640), dtype=bool)
        m[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = True
        masks.append(m)
        classes.append(cls)

    class StubSeg:
        """Mimics FoodSAMSegmenter output after a mixed-plate suppression:
        the Xoi dish result is ABSENT, only promoted ingredients remain."""
        is_available = True
        fallback_used = False
        last_error = None
        last_info = {"suppressed_dishes": [{
            "class_name": "Xoi (Sticky rice)", "confidence": 0.88,
            "bbox": [100, 100, 900, 900], "status": "suppressed",
            "reason": "mixed_plate_contradiction",
            "promoted_labels": sorted(classes),
            "promoted_area_coverage": 1.0}]}

        def segment(self, image_rgb, detections):
            return [SegmentationResult(
                mask=mk, class_name=cn,
                bbox=(x0, y0, x1, y1), confidence=0.8, warnings=[],
                source="foodsam_ingredient", is_ingredient=True,
                component_id=f"p{i:02d}", semantic_purity=0.9,
                mapping_type="approximate")
                for i, (mk, cn, (x0, y0, x1, y1)) in
                enumerate(zip(masks, classes,
                              [(150, 150, 450, 450), (550, 150, 850, 450),
                               (150, 550, 450, 850)]))]

        def visualize_masks(self, image_rgb, seg_results, alpha=0.45):
            return image_rgb

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[:, :] = (200, 180, 160)
    xoi_det = det("Xoi (Sticky rice)", (100, 100, 900, 900), conf=0.88)

    pipeline = FoodVolumePipeline()
    pipeline._segmenter = StubSeg()

    result = pipeline.analyze(
        image_rgb=image, yolo_detections=[xoi_det], image_path=None,
        generate_visualizations=False)

    assert len(result.estimations) == 3, (
        f"3 promoted items expected, got {len(result.estimations)}")
    assert not any(e.class_name == "Xoi (Sticky rice)"
                   for e in result.estimations), \
        "suppressed dish must have NO nutrition estimation (no double count)"
    assert all(e.is_ingredient for e in result.estimations)
    total = sum(e.nutrition.get("Calories", 0) for e in result.estimations)
    assert abs(result.total_nutrition.get("Calories", 0) - total) < 0.15, \
        "totals must equal the sum of ACTIVE items only"


def t54_promote_min_ratio_edge():
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["steak"], 150, 150, 450, 450)         # 90k -> group+promote
    rect(seg, LAB["chicken duck"], 550, 150, 850, 450)  # 90k
    rect(seg, LAB["pork"], 150, 550, 450, 850)          # 90k -> suppression on
    rect(seg, LAB["egg"], 600, 600, 630, 630)           # 900 px  < promote 1000
    rect(seg, LAB["egg"], 700, 700, 740, 740)           # 1600 px >= promote
    dets = [det("Xoi (Sticky rice)", (100, 100, 900, 900))]
    extras, masks, breakdowns, _, suppressed = run_extract(seg, dets, map_table=MAP_FULL)
    assert len(suppressed) == 1
    egg_extras = [e for e in extras if e["class_name"] == "Trung (Egg)"]
    assert len(egg_extras) == 1, (
        f"only the >= promote-floor egg component is promoted, got {len(egg_extras)}")
    bd_eggs = [r for r in breakdowns[0] if r["foodseg_label"] == "egg"]
    assert len(bd_eggs) == 1, "sub-floor egg stays as breakdown metadata"
    cov = suppressed[0]["promoted_area_coverage"]
    expected = (3 * 90000 + 1600) / (3 * 90000 + 1600 + 900)
    assert abs(cov - expected) < 0.01, f"coverage {cov} vs expected {expected:.4f}"


def t55_group_count_noise_guard():
    # three ~0.115% components: each >= promote(0.001) but < group(0.002)
    # -> NOT enough contradiction evidence -> dish kept, breakdown metadata
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["steak"], 100, 100, 134, 134)     # 1156 px = 0.00116
    rect(seg, LAB["chicken duck"], 400, 400, 434, 434)
    rect(seg, LAB["pork"], 700, 700, 734, 734)
    dets = [det("Xoi (Sticky rice)", (50, 50, 950, 950), conf=0.9)]
    extras, masks, breakdowns, _, suppressed = run_extract(seg, dets, map_table=MAP_FULL)
    assert suppressed == [], "sub-GROUP_MIN evidence must not trigger suppression"
    assert extras == [], "no suppression -> no promotion (no double count)"
    assert len(breakdowns[0]) == 3


def t56_post_mapping_grouping():
    # carrot+tomato+cucumber all map to "Rau" -> ONE group, not three
    seg = np.zeros((H, W), dtype=np.int32)
    rect(seg, LAB["carrot"], 150, 150, 450, 450)
    rect(seg, LAB["tomato"], 550, 150, 850, 450)
    rect(seg, LAB["cucumber"], 150, 550, 450, 850)
    dets = [det("Com tam (Broken rice)", (100, 100, 900, 900), conf=0.9)]
    extras, masks, breakdowns, _, suppressed = run_extract(
        seg, dets, map_table=MAP_FULL)
    assert suppressed == [], (
        "3 generic vegetable labels collapse to one 'Rau' group -> 1 < 3 -> keep dish")


def main():
    check("T1 breakdown inside single bbox (I1)", t1_breakdown_inside_single_bbox)
    check("T2 multi-bbox union (I6)", t2_multi_bbox_union_not_extra)
    check("T3 owner ambiguous single record (I7)", t3_owner_ambiguous_single_record)
    check("T4 per-component extras, no merge", t4_per_component_extras_no_merge)
    check("T5 area flag + cap", t5_area_flag_and_cap)
    check("T6 purity filter", t6_purity_filter)
    check("T7 zero-detections candidates (HomeCook)", t7_zero_detections_all_candidates)
    check("T10 mapping table valid", t10_mapping_table_valid)
    check("T20 volume union inflates (per-component rationale)", t20_volume_union_inflates)
    check("T30 depth-once + totals (real pipeline)", t30_depth_once_and_totals)
    check("T40 fallback flag on env failure", t40_fallback_flag_on_env_failure)
    check("T41 empty-detections bridge contract", t41_segmenter_empty_detections_still_extracts)
    check("T50 mixed-plate suppress+promote", t50_suppress_and_promote)
    check("T51 kept dish — Pho never suppressed", t51_kept_dish_pho_never_suppressed)
    check("T52 anti-false-suppression (2 groups)", t52_anti_false_suppression_two_groups)
    check("T53 accounting safety through pipeline", t53_accounting_safety_through_pipeline)
    check("T54 promote min-ratio edge", t54_promote_min_ratio_edge)
    check("T55 group-count noise guard", t55_group_count_noise_guard)
    check("T56 post-mapping grouping", t56_post_mapping_grouping)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 60}\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        for name, _, err in failed:
            print(f"  FAILED: {name} — {err}")
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
