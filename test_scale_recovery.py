"""
Synthetic tests for scale recovery: ArUco (30mm black square) and
checkerboard (30mm squares, 8x6 inner corners).

Renders known-size targets into images and verifies the recovered mm/px.

Usage:
    python test_scale_recovery.py
"""

import numpy as np
import cv2

from scale_recovery import ScaleRecovery

FAILURES = []


def check(name, got, expected, tol_pct=2.0):
    err = 100.0 * (got - expected) / expected
    ok = abs(err) <= tol_pct
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {name}: mm/px={got:.4f} expected={expected:.4f} ({err:+.1f}%)")
    if not ok:
        FAILURES.append(name)
    return ok


def make_checkerboard_image(square_px=40.0):
    """9x7 squares (8x6 inner corners), square_px per 30mm square."""
    rows_sq, cols_sq = 7, 9
    board = np.full((int(rows_sq * square_px), int(cols_sq * square_px)), 255, np.uint8)
    for r in range(rows_sq):
        for c in range(cols_sq):
            if (r + c) % 2 == 0:
                y0, x0 = int(r * square_px), int(c * square_px)
                y1, x1 = int((r + 1) * square_px), int((c + 1) * square_px)
                board[y0:y1, x0:x1] = 0
    # margins so the board doesn't touch image edges
    canvas = np.full((board.shape[0] + 160, board.shape[1] + 160), 255, np.uint8)
    canvas[80:80 + board.shape[0], 80:80 + board.shape[1]] = board
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def make_aruco_image(side_px=120.0, dict_name="DICT_4X4_50", marker_id=7):
    """Black square of side_px on white quiet zone (OpenCV convention)."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, int(side_px))
    margin = int(side_px * 0.5)
    canvas = np.full((int(side_px) + 2 * margin, int(side_px) + 2 * margin), 255, np.uint8)
    canvas[margin:margin + int(side_px), margin:margin + int(side_px)] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def main():
    rec = ScaleRecovery()

    print("=" * 70)
    print("  SCALE RECOVERY TESTS (30mm targets)")
    print("=" * 70)

    # ── Checkerboard: 9x7 squares of 40px → each 30mm square = 40px ──
    img = make_checkerboard_image(square_px=40.0)
    result = rec._try_checkerboard(img, K=None, dist=None)
    if result is None:
        print("  [FAIL] checkerboard not detected")
        FAILURES.append("checkerboard detect")
    else:
        print(f"  checkerboard detected: source={result.scale_source}, "
              f"confidence={result.confidence}, notes={result.notes[:1]}")
        check("checkerboard mm/px (40px squares)", result.mm_per_pixel, 30.0 / 40.0)
        assert result.scale_source == "checkerboard"
        assert result.marker_corners is not None and result.marker_corners.shape == (4, 2)

    # ── Checkerboard through the public recover() API (with K) ──
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    result = rec.recover(img, K=K, dist=np.zeros(5))
    if result.scale_source != "checkerboard":
        print(f"  [FAIL] recover() returned {result.scale_source}, expected checkerboard")
        FAILURES.append("recover checkerboard")
    else:
        check("recover() checkerboard mm/px", result.mm_per_pixel, 0.75)
        if result.marker_tvec is None:
            print("  [FAIL] checkerboard tvec missing (depth anchoring needs it)")
            FAILURES.append("checkerboard tvec")
        else:
            z = float(np.ravel(result.marker_tvec)[2])
            print(f"  checkerboard plane distance (tvec z): {z:.1f} mm")
            # board ~350x270 px wide on 800px-focal camera → z ≈ 800*0.27/0.75
            if not (100.0 < z < 2000.0):
                print("  [FAIL] tvec z implausible")
                FAILURES.append("checkerboard tvec z")

    # ── ArUco: black square 120px → 30mm black square ──
    img = make_aruco_image(side_px=120)
    result = rec._try_aruco(img, K=None, dist=None)
    if result is None:
        print("  [FAIL] aruco not detected")
        FAILURES.append("aruco detect")
    else:
        print(f"  aruco detected: source={result.scale_source}, "
              f"confidence={result.confidence}")
        check("aruco mm/px (120px black square)", result.mm_per_pixel, 30.0 / 120.0)

    # ── Overlay renders for both sources ──
    for src_name, image in [("checkerboard", make_checkerboard_image()),
                            ("aruco", make_aruco_image())]:
        res = rec.recover(image, K=K, dist=np.zeros(5))
        if res.scale_source == src_name:
            overlay = rec.draw_aruco_overlay(image, res)
            assert overlay.shape[:2] == image.shape[:2]
            print(f"  overlay for {src_name}: OK")

    # ── Image without any target → fallback, no crash ──
    plain = np.full((640, 640, 3), 128, np.uint8)
    result = rec.recover(plain, K=K, dist=np.zeros(5))
    print(f"  plain image fallback: source={result.scale_source} "
          f"(expected image_heuristic/plate)")

    print("=" * 70)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("  ALL SCALE RECOVERY TESTS PASS")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
