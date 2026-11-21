"""Phase 6 acceptance — cross-class duplicate suppression (deterministic).

The suppression stage is pure numpy logic on box pools: no model weights,
no images, no GPU. Run:

    .venv/Scripts/python.exe tests/test_phase6_cross_class_suppression.py

Fixture note: the bug reproduced on bun-rieu (1).jpg is encoded in
test_t8_reproduced_bug_pool — Bun bo Hue 0.955 (stretch pass) + Bun bo Hue
0.750 / Bun rieu 0.354 (letterbox pass, identical boxes), IoU 0.988.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import pipeline_config
from utils import _nms_class_aware, _suppress_cross_class_duplicates, _class_display_name

THRESH = float(pipeline_config.CROSS_CLASS_SUPPRESS_IOU)

# Synthetic class ids — resolved through class_names for record checks only.
CLS_A, CLS_B, CLS_C = 0, 1, 2


def pair_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_t1_exact_duplicate_cross_class():
    """Different classes, identical boxes -> lower-conf box suppressed."""
    boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], dtype=np.float64)
    scores = np.array([0.955, 0.354])
    classes = np.array([CLS_A, CLS_B])
    keep, records = _suppress_cross_class_duplicates(boxes, scores, classes, THRESH)
    assert list(keep) == [0], f"expected only the high-conf box, got {keep}"
    assert len(records) == 1, f"expected 1 suppression record, got {len(records)}"
    print("  [PASS] t1 exact duplicate cross-class suppressed")


def test_t2_cross_class_iou_below_threshold_kept():
    """Different classes overlapping ~0.65 (< threshold) -> both kept."""
    box_a, box_b = [0, 0, 100, 100], [21, 0, 121, 100]
    iou = pair_iou(box_a, box_b)
    assert iou < THRESH, (
        f"fixture IoU {iou:.3f} must stay below threshold {THRESH} — "
        "update this fixture if CROSS_CLASS_SUPPRESS_IOU changes")
    boxes = np.array([box_a, box_b], dtype=np.float64)
    scores = np.array([0.90, 0.35])
    classes = np.array([CLS_A, CLS_B])
    keep, records = _suppress_cross_class_duplicates(boxes, scores, classes, THRESH)
    assert sorted(keep.tolist()) == [0, 1], f"both boxes must survive, got {keep}"
    assert records == [], f"no suppression expected, got {records}"
    print(f"  [PASS] t2 cross-class IoU {iou:.3f} < {THRESH} kept")


def test_t3_same_class_pairs_untouched_by_suppression():
    """Same-class duplicates belong to _nms_class_aware, not this stage."""
    boxes = np.array([[0, 0, 100, 100], [0, 0, 95, 100]], dtype=np.float64)
    scores = np.array([0.90, 0.80])
    classes = np.array([CLS_A, CLS_A])
    keep, records = _suppress_cross_class_duplicates(boxes, scores, classes, THRESH)
    assert sorted(keep.tolist()) == [0, 1], "same-class pair must not be touched here"
    assert records == []
    nms_keep = _nms_class_aware(boxes, scores, classes, iou_thr=0.55)
    assert list(nms_keep) == [0], "class-aware NMS must still remove same-class duplicates"
    print("  [PASS] t3 same-class duplicates still owned by _nms_class_aware")


def test_t4_three_box_chain_order_independent():
    """3 identical boxes, 3 classes -> only the highest-conf box survives,
    regardless of input order (greedy is confidence-descending)."""
    pool = [
        (np.array([[0, 0, 100, 100]] * 3, dtype=np.float64),
         np.array([0.9, 0.5, 0.3]), np.array([CLS_A, CLS_B, CLS_C])),
        (np.array([[0, 0, 100, 100]] * 3, dtype=np.float64),
         np.array([0.3, 0.9, 0.5]), np.array([CLS_C, CLS_A, CLS_B])),
    ]
    for boxes, scores, classes in pool:
        keep, records = _suppress_cross_class_duplicates(boxes, scores, classes, THRESH)
        top = int(np.argmax(scores))
        assert list(keep) == [top], (
            f"only the 0.9-conf box (index {top}) must survive, got {keep}")
        assert len(records) == 2, f"expected 2 suppression records, got {len(records)}"
        for rec in records:
            assert rec["kept_class"] == _class_display_name(CLS_A)
            assert abs(rec["kept_confidence"] - 0.9) < 1e-9
    print("  [PASS] t4 3-box chain deterministic across input orders")


def test_t5_disjoint_boxes_all_kept():
    boxes = np.array([[0, 0, 100, 100], [200, 0, 300, 100], [0, 200, 100, 300]],
                     dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([CLS_A, CLS_B, CLS_C])
    keep, records = _suppress_cross_class_duplicates(boxes, scores, classes, THRESH)
    assert sorted(keep.tolist()) == [0, 1, 2]
    assert records == []
    print("  [PASS] t5 disjoint different-class boxes all kept")


def test_t6_record_values_not_just_keys():
    """Audit records must carry the actual values (class, conf, bbox, iou)."""
    box_hi = [12, 988, 1215, 2080]
    box_lo = [21, 984, 1214, 2078]
    boxes = np.array([box_hi, box_lo], dtype=np.float64)
    scores = np.array([0.955, 0.354])
    classes = np.array([CLS_A, CLS_B])
    _, records = _suppress_cross_class_duplicates(boxes, scores, classes, THRESH)
    assert len(records) == 1, f"expected 1 record, got {records}"
    rec = records[0]
    assert set(rec) == {"class_name", "confidence", "bbox", "kept_class",
                        "kept_confidence", "iou", "reason"}
    assert rec["class_name"] == _class_display_name(CLS_B)
    assert abs(rec["confidence"] - 0.354) < 1e-9
    assert rec["bbox"] == [float(v) for v in box_lo]
    assert rec["kept_class"] == _class_display_name(CLS_A)
    assert abs(rec["kept_confidence"] - 0.955) < 1e-9
    assert rec["iou"] >= 0.98
    assert rec["reason"] == "cross_class_duplicate"
    print("  [PASS] t6 record carries correct class/conf/bbox/iou values")


def test_t7_config_defaults_valid():
    assert pipeline_config.CROSS_CLASS_SUPPRESS_ENABLED is True
    assert 0.0 < THRESH < 1.0, f"threshold {THRESH} must be a valid IoU in (0, 1)"
    print(f"  [PASS] t7 config: enabled=True, IOU={THRESH}")


def test_t8_reproduced_bug_pool():
    """The exact bun-rieu (1).jpg pool: class-aware NMS leaves the cross-class
    duplicate; the new stage removes it and keeps one Bun bo Hue box."""
    xyxy = np.array([
        [12, 988, 1215, 2080],   # stretch pass, Bun bo Hue 0.955
        [21, 984, 1214, 2078],   # letterbox pass, Bun bo Hue 0.750
        [21, 984, 1214, 2078],   # letterbox pass, Bun rieu 0.354 (identical)
    ], dtype=np.float64)
    extra = np.array([
        [0.955, 56],  # class ids from the reproduction run
        [0.750, 56],
        [0.354, 35],
    ])
    keep = _nms_class_aware(xyxy, extra[:, 0], extra[:, 1], iou_thr=0.55)
    assert sorted(keep.tolist()) == [0, 2], (
        f"class-aware NMS must keep the cross-class duplicate, got {keep}")
    keep_sub, records = _suppress_cross_class_duplicates(
        xyxy[keep], extra[keep, 0], extra[keep, 1], THRESH)
    final = keep[keep_sub]
    assert list(final) == [0], f"only the 0.955 box must survive, got {final}"
    assert len(records) == 1
    assert records[0]["iou"] >= 0.98
    assert records[0]["reason"] == "cross_class_duplicate"
    print("  [PASS] t8 reproduced bug pool collapses to 1 box (Bun bo Hue 0.955)")


def run_all_tests():
    tests = [
        test_t1_exact_duplicate_cross_class,
        test_t2_cross_class_iou_below_threshold_kept,
        test_t3_same_class_pairs_untouched_by_suppression,
        test_t4_three_box_chain_order_independent,
        test_t5_disjoint_boxes_all_kept,
        test_t6_record_values_not_just_keys,
        test_t7_config_defaults_valid,
        test_t8_reproduced_bug_pool,
    ]
    print(f"=== Phase 6: cross-class duplicate suppression (threshold={THRESH}) ===")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"=== {len(tests) - failed}/{len(tests)} passed ===")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
