"""
Camera Calibration Script for Food Nutrition Pipeline.

Uses OpenCV Sector-Based Chessboard Corner Detection (findChessboardCornersSB)
with automatic corner alignment and iterative outlier rejection for high precision.
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2

def calibrate_camera(
    image_dir: str,
    pattern_size: tuple = (8, 6),
    square_size_mm: float = 30.0,
    output_dir: str = None,
    outlier_threshold_px: float = 3.5,
    save_viz: bool = True
):
    """
    Perform Camera Calibration using checkerboard images.
    
    Args:
        image_dir: Path to directory containing checkerboard calibration images.
        pattern_size: Tuple (cols, rows) of inner corners on the checkerboard (default: 8x6).
        square_size_mm: Real-world side length of each square in millimeters (default: 30.0 mm).
        output_dir: Directory to save calibration results (npz, json, visualization).
        outlier_threshold_px: Maximum allowed initial reprojection error to filter blurred/flipped images.
        save_viz: Whether to save corner detection visualizations and undistorted comparisons.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_dir))
    
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "calib_visualizations")
    if save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print("=" * 70)
    print(f"CAMERA CALIBRATION: Processing {len(image_paths)} images")
    print(f"Pattern Grid (Inner corners): {cols} cols x {rows} rows")
    print(f"Square size: {square_size_mm} mm")
    print("=" * 70)

    img_shape = None
    all_objpoints = []
    all_imgpoints = []
    detected_files = []

    for idx, fname in enumerate(image_paths, start=1):
        img_name = os.path.basename(fname)
        img = cv2.imread(fname)
        if img is None:
            print(f"[{idx:02d}/{len(image_paths):02d}] [ERROR] Could not read {img_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = (gray.shape[1], gray.shape[0])  # (width, height)

        # High accuracy Sector-Based corner detection
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_ACCURACY)
        if not found:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if found and corners is not None:
            # Check corner grid orientation consistency
            c = corners.reshape(-1, 2)
            v1 = c[cols - 1] - c[0]
            v2 = c[(rows - 1) * cols] - c[0]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross < 0:
                c_grid = c.reshape(rows, cols, 2)
                c_grid = np.flip(c_grid, axis=0)
                corners = c_grid.reshape(-1, 1, 2).astype(np.float32)

            all_objpoints.append(objp)
            all_imgpoints.append(corners)
            detected_files.append(fname)
            print(f"[{idx:02d}/{len(image_paths):02d}] [OK] Detected corners: {img_name}")

            if save_viz:
                viz_img = img.copy()
                cv2.drawChessboardCorners(viz_img, pattern_size, corners, found)
                h, w = viz_img.shape[:2]
                if w > 1920:
                    scale = 1920.0 / w
                    viz_img = cv2.resize(viz_img, (0, 0), fx=scale, fy=scale)
                cv2.imwrite(os.path.join(viz_dir, f"detected_{img_name}"), viz_img)
        else:
            print(f"[{idx:02d}/{len(image_paths):02d}] [FAIL] Pattern NOT found: {img_name}")

    if len(detected_files) < 5:
        raise ValueError(f"Too few valid images ({len(detected_files)}). Calibration aborted.")

    # Pass 1: Initial Calibration
    print("\n--- Pass 1: Initial Calibration across all detected images ---")
    ret1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
        all_objpoints, all_imgpoints, img_shape, None, None
    )
    print(f"Pass 1 Reprojection Error: {ret1:.4f} px")

    # Pass 2: Outlier rejection for extreme perspective/motion blur
    inlier_objpoints = []
    inlier_imgpoints = []
    inlier_files = []
    initial_errors = []

    for i in range(len(all_objpoints)):
        imgpoints2, _ = cv2.projectPoints(all_objpoints[i], rvecs1[i], tvecs1[i], K1, dist1)
        err = cv2.norm(all_imgpoints[i], imgpoints2, cv2.NORM_L2) / np.sqrt(len(imgpoints2))
        fname = detected_files[i]
        initial_errors.append((err, fname))
        if err <= outlier_threshold_px:
            inlier_objpoints.append(all_objpoints[i])
            inlier_imgpoints.append(all_imgpoints[i])
            inlier_files.append(fname)
            print(f"  [KEEP] {os.path.basename(fname):30s} error: {err:.4f} px")
        else:
            print(f"  [REJECT OUTLIER] {os.path.basename(fname):22s} error: {err:.4f} px (> {outlier_threshold_px} px)")

    # Pass 3: Final High-Accuracy Calibration on Inliers
    print(f"\n--- Pass 2: Final Calibration with {len(inlier_files)} / {len(image_paths)} inlier images ---")
    ret_final, K_final, dist_final, rvecs_final, tvecs_final = cv2.calibrateCamera(
        inlier_objpoints, inlier_imgpoints, img_shape, None, None
    )

    per_image_results = []
    tot_sq_error = 0
    for i in range(len(inlier_objpoints)):
        imgpoints2, _ = cv2.projectPoints(inlier_objpoints[i], rvecs_final[i], tvecs_final[i], K_final, dist_final)
        err = cv2.norm(inlier_imgpoints[i], imgpoints2, cv2.NORM_L2) / np.sqrt(len(imgpoints2))
        tot_sq_error += err ** 2
        per_image_results.append({
            "image": os.path.basename(inlier_files[i]),
            "reprojection_error_px": float(err)
        })

    rmse = np.sqrt(tot_sq_error / len(inlier_objpoints))
    fx, fy = float(K_final[0, 0]), float(K_final[1, 1])
    cx, cy = float(K_final[0, 2]), float(K_final[1, 2])
    dist_coeffs = [float(v) for v in dist_final.ravel()]

    print("\n" + "=" * 70)
    print("                    CAMERA CALIBRATION RESULTS")
    print("=" * 70)
    print(f"Image Resolution (W x H)    : {img_shape[0]} x {img_shape[1]} px")
    print(f"Inlier Images Used          : {len(inlier_files)} / {len(image_paths)}")
    print(f"Overall Reprojection Error  : {ret_final:.4f} px (RMSE: {rmse:.4f} px)")
    print(f"\n[Camera Intrinsic Matrix K]")
    print(f"  fx (Focal Length X)       : {fx:10.4f} px")
    print(f"  fy (Focal Length Y)       : {fy:10.4f} px")
    print(f"  cx (Principal Point X)    : {cx:10.4f} px (Center: {img_shape[0]/2:.1f})")
    print(f"  cy (Principal Point Y)    : {cy:10.4f} px (Center: {img_shape[1]/2:.1f})")
    print(f"\n[Distortion Coefficients (RadTan)]")
    print(f"  k1 (Radial 1)             : {dist_coeffs[0]:12.6f}")
    print(f"  k2 (Radial 2)             : {dist_coeffs[1]:12.6f}")
    print(f"  p1 (Tangential 1)         : {dist_coeffs[2]:12.6f}")
    print(f"  p2 (Tangential 2)         : {dist_coeffs[3]:12.6f}")
    print(f"  k3 (Radial 3)             : {dist_coeffs[4]:12.6f}")
    print("=" * 70)

    # Save to camera_calib.npz
    npz_path = os.path.join(output_dir, "camera_calib.npz")
    np.savez(
        npz_path,
        K=K_final,
        dist=dist_final,
        image_shape=img_shape,
        reprojection_error=ret_final,
        rmse=rmse,
        num_inliers=len(inlier_files),
        num_total=len(image_paths)
    )

    # Save to camera_calib.json
    json_path = os.path.join(output_dir, "camera_calib.json")
    calib_json = {
        "image_width": int(img_shape[0]),
        "image_height": int(img_shape[1]),
        "reprojection_error": float(ret_final),
        "rmse_error": float(rmse),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_matrix_K": K_final.tolist(),
        "distortion_coefficients_dist": dist_coeffs,
        "pattern_size": list(pattern_size),
        "square_size_mm": float(square_size_mm),
        "inlier_images_count": len(inlier_files),
        "total_images_count": len(image_paths),
        "per_image_errors": per_image_results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(calib_json, f, indent=4, ensure_ascii=False)

    # Generate Undistorted Sample Comparison
    if save_viz and inlier_files:
        sample_path = inlier_files[0]
        sample_img = cv2.imread(sample_path)
        h, w = sample_img.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(K_final, dist_final, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(sample_img, K_final, dist_final, None, newcameramtx)
        
        comp_w = 960
        scale = comp_w / w
        orig_small = cv2.resize(sample_img, (0, 0), fx=scale, fy=scale)
        undist_small = cv2.resize(undistorted_img, (0, 0), fx=scale, fy=scale)
        
        cv2.putText(orig_small, "ORIGINAL (Distorted)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(undist_small, "CALIBRATED (Undistorted)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        comparison = np.hstack([orig_small, undist_small])
        comp_path = os.path.join(output_dir, "undistort_comparison.jpg")
        cv2.imwrite(comp_path, comparison)

    print(f"\n[OUTPUT FILES]")
    print(f"  1. Calibration NPZ : {npz_path}")
    print(f"  2. Calibration JSON: {json_path}")
    if save_viz:
        print(f"  3. Visualizations  : {viz_dir}")
        print(f"  4. Undistort Preview: {os.path.join(output_dir, 'undistort_comparison.jpg')}")
    print("\nCalibration completed successfully!")

    return K_final, dist_final, ret_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration using Checkerboard")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default=r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2\Camera calibration\Image",
        help="Directory containing checkerboard images"
    )
    parser.add_argument(
        "--cols", 
        type=int, 
        default=8, 
        help="Number of inner corners horizontally (cols)"
    )
    parser.add_argument(
        "--rows", 
        type=int, 
        default=6, 
        help="Number of inner corners vertically (rows)"
    )
    parser.add_argument(
        "--square_size_mm", 
        type=float, 
        default=25.0, 
        help="Side length of one square in millimeters (default: 25.0 mm)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=3.5, 
        help="Outlier error threshold in pixels"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2\Camera calibration",
        help="Directory to save calibration results"
    )
    args = parser.parse_args()

    calibrate_camera(
        image_dir=args.image_dir,
        pattern_size=(args.cols, args.rows),
        square_size_mm=args.square_size_mm,
        output_dir=args.output_dir,
        outlier_threshold_px=args.threshold,
        save_viz=True
    )
