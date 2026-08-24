"""
Scale Recovery Module — ArUco marker detection + plate heuristic fallback.

Recovers the real-world scale (mm/pixel) needed to convert pixel-level depth
and area into physical units for volume estimation.

Methods (in priority order):
  1. ArUco marker detected → pose estimation → calibrated scale
  2. Plate/bowl circle detection → heuristic diameter → approximate scale
  3. Image metadata (EXIF focal length) → rough estimate

Each method tags its output with a confidence level and source label.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScaleResult:
    """Output of scale recovery."""
    mm_per_pixel: float             # real-world scale at the food plane
    scale_source: str               # "aruco_marker" | "plate_heuristic" | "image_heuristic"
    confidence: str                 # "high" | "medium" | "low"
    marker_corners: Optional[np.ndarray] = None   # ArUco corner pixels if detected
    marker_rvec: Optional[np.ndarray] = None       # rotation vector
    marker_tvec: Optional[np.ndarray] = None       # translation vector (mm)
    homography: Optional[np.ndarray] = None        # image -> physical plane homography (mm)
    plate_center: Optional[Tuple[int, int]] = None  # detected plate center pixel
    plate_radius_px: Optional[int] = None           # major axis radius in pixels
    plate_axes_px: Optional[Tuple[int, int]] = None # (major_axis_r, minor_axis_r)
    plate_angle_deg: float = 0.0                    # ellipse orientation angle
    dish_name: Optional[str] = None                 # dish name used for scale
    dish_expected_diameter_mm: Optional[float] = None
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class ScaleRecovery:
    """Recover real-world scale from an image.
    
    Priority:
      1. ArUco marker detection (calibrated metric scale)
      2. Dish rim fitting from YOLO bbox & SAM2 mask (perspective-aware ellipse)
      3. Global plate Hough circle detection (fallback)
      4. Image geometry heuristic (last resort)
    """

    def __init__(
        self,
        marker_length_mm: float = None,
        aruco_dict_type: str = None,
    ):
        from pipeline_config import (
            ARUCO_MARKER_LENGTH_MM, ARUCO_DICT_TYPE,
            PLATE_DIAMETER_HEURISTICS, CLASS_TO_DISH_TYPE,
            CONTAINER_DISH_TYPES,
        )

        self.marker_length_mm = marker_length_mm or ARUCO_MARKER_LENGTH_MM
        self.aruco_dict_name = aruco_dict_type or ARUCO_DICT_TYPE
        self.plate_heuristics = PLATE_DIAMETER_HEURISTICS
        self.class_to_dish_type = CLASS_TO_DISH_TYPE
        self.container_dish_types = CONTAINER_DISH_TYPES

        # Prepare list of candidate ArUco dictionaries to try
        all_candidate_dict_names = [
            "DICT_ARUCO_ORIGINAL",
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_4X4_250",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_5X5_250",
            "DICT_6X6_50",
            "DICT_6X6_100",
            "DICT_6X6_250",
            "DICT_APRILTAG_36h11",
        ]
        # Put preferred dictionary first
        dict_names = [self.aruco_dict_name] + [
            d for d in all_candidate_dict_names if d != self.aruco_dict_name
        ]
        self._dictionaries = []
        for d_name in dict_names:
            if hasattr(cv2.aruco, d_name):
                dict_obj = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, d_name))
                self._dictionaries.append((d_name, dict_obj))

    def recover(
        self,
        image: np.ndarray,
        K: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
        detected_classes: Optional[List[str]] = None,
        yolo_detections: Optional[List[dict]] = None,
        seg_results: Optional[List] = None,
    ) -> ScaleResult:
        """Attempt scale recovery using available methods in strict priority order.
        
        Priority 1: ArUco marker detection (calibrated metric scale)
        Priority 2: Dish geometry from YOLO bbox & SAM2 mask (perspective ellipse fitting)
        Priority 3: Plate/bowl circle detection heuristic
        Priority 4: Image size heuristic (last resort)
        """
        # Method 1: ArUco marker
        aruco_result = self._try_aruco(image, K, dist)
        if aruco_result is not None:
            return aruco_result

        # Method 2: Dish geometry from YOLO detections & SAM2 masks (most accurate fallback)
        if yolo_detections or seg_results:
            dish_result = self._try_dish_guided_scale(
                image, yolo_detections=yolo_detections, seg_results=seg_results
            )
            if dish_result is not None:
                return dish_result

        # Method 3: Plate/bowl circle detection on full image
        plate_result = self._try_plate_heuristic(image, detected_classes)
        if plate_result is not None:
            return plate_result

        # Method 4: Image size heuristic (last resort)
        return self._image_heuristic(image)

    def get_dish_scale(
        self,
        bbox: Tuple[int, int, int, int],
        class_name: str,
        global_scale: ScaleResult,
    ) -> float:
        """Return per-dish mm_per_pixel scale.
        If ArUco is detected, scale is globally calibrated.
        If plate_heuristic is used, items in the same scene inherit the global scale.
        Only separate container dishes (soup_bowl, flat_plate, rice_bowl) can refine their scale.
        Ingredients/sides (lime, chili, side veggies) ALWAYS inherit the global scale.
        """
        if global_scale.scale_source == "aruco_marker":
            return global_scale.mm_per_pixel

        dish_type = self.class_to_dish_type.get(class_name, "default")
        
        # Non-containers (ingredients, garnishes, side vegetables) must never use independent plate scale
        if dish_type not in self.container_dish_types:
            return global_scale.mm_per_pixel

        expected_dia_mm = self.plate_heuristics.get(dish_type, None)
        if expected_dia_mm is None:
            return global_scale.mm_per_pixel

        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        dish_dim_px = max(bw, bh)
        if dish_dim_px > 80:
            return expected_dia_mm / float(dish_dim_px)
        return global_scale.mm_per_pixel

    def _try_aruco(
        self,
        image: np.ndarray,
        K: Optional[np.ndarray],
        dist: Optional[np.ndarray],
    ) -> Optional[ScaleResult]:
        """Detect ArUco marker across candidate dictionaries with adaptive parameter tuning."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h_orig, w_orig = gray.shape[:2]
        
        # Scale for detection if image is very large to prevent memory spikes
        max_dim = 1600
        if max(h_orig, w_orig) > max_dim:
            scale_aruco = max_dim / float(max(h_orig, w_orig))
            w_det = int(round(w_orig * scale_aruco))
            h_det = int(round(h_orig * scale_aruco))
            gray_detect = cv2.resize(gray, (w_det, h_det))
        else:
            scale_aruco = 1.0
            gray_detect = gray

        # Adaptive thresholding window tuned for image resolution
        max_win = min(63, max(23, int(max(gray_detect.shape[:2]) / 40) | 1))

        best_corners = None
        best_id = None
        best_dict_name = None
        best_edge_px = 0.0

        # Create detector parameters for this image resolution
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = max_win
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0

        for d_name, dict_obj in self._dictionaries:
            detector = cv2.aruco.ArucoDetector(dict_obj, params)
            corners, ids, _ = detector.detectMarkers(gray_detect)
            if ids is not None and len(ids) > 0:
                for idx, c in zip(ids.flatten(), corners):
                    pts = c[0] / scale_aruco  # Rescale corners back to original image space
                    edge_len = float(np.mean([
                        np.linalg.norm(pts[1] - pts[0]),
                        np.linalg.norm(pts[2] - pts[1]),
                        np.linalg.norm(pts[3] - pts[2]),
                        np.linalg.norm(pts[0] - pts[3]),
                    ]))
                    # Keep the most prominent marker
                    if edge_len > best_edge_px:
                        best_edge_px = edge_len
                        best_corners = pts
                        best_id = int(idx)
                        best_dict_name = d_name
                # Found marker in this dictionary -> stop scanning further dictionaries
                if best_corners is not None:
                    break

        if best_corners is None or best_id is None:
            logger.info("No ArUco marker detected in any candidate dictionary")
            return None

        logger.info(
            f"ArUco marker detected: ID={best_id} ({best_dict_name}), "
            f"edge={best_edge_px:.1f}px"
        )
        marker_corners = best_corners  # shape (4, 2)
        mm_per_pixel = self.marker_length_mm / best_edge_px

        # Compute physical homography from marker corners to physical plane (mm)
        half_l = self.marker_length_mm / 2.0
        obj_corners_2d = np.array([
            [-half_l,  half_l],
            [ half_l,  half_l],
            [ half_l, -half_l],
            [-half_l, -half_l],
        ], dtype=np.float32)
        H_mat, _ = cv2.findHomography(marker_corners.astype(np.float32), obj_corners_2d)

        # Pose estimation with camera intrinsics
        if K is not None:
            if dist is None:
                dist = np.zeros(5)
            obj_points_3d = np.array([
                [-half_l,  half_l, 0.0],
                [ half_l,  half_l, 0.0],
                [ half_l, -half_l, 0.0],
                [-half_l, -half_l, 0.0],
            ], dtype=np.float32)
            
            try:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points_3d,
                    marker_corners.astype(np.float32),
                    K.astype(np.float64),
                    dist.astype(np.float64),
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
            except Exception:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points_3d,
                    marker_corners.astype(np.float32),
                    K.astype(np.float64),
                    dist.astype(np.float64),
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            return ScaleResult(
                mm_per_pixel=mm_per_pixel,
                scale_source="aruco_marker",
                confidence="high",
                marker_corners=marker_corners,
                marker_rvec=rvec if success else None,
                marker_tvec=tvec if success else None,
                homography=H_mat,
                notes=[
                    f"ArUco marker ID={best_id} ({best_dict_name}) detected",
                    f"Marker edge = {best_edge_px:.1f}px = {self.marker_length_mm}mm",
                    f"Calibrated Scale = {mm_per_pixel:.4f} mm/px",
                ],
            )
        else:
            return ScaleResult(
                mm_per_pixel=mm_per_pixel,
                scale_source="aruco_marker",
                confidence="medium",
                marker_corners=marker_corners,
                homography=H_mat,
                notes=[
                    f"ArUco marker ID={best_id} ({best_dict_name}) detected (no camera calib)",
                    f"Scale = {mm_per_pixel:.4f} mm/px",
                ],
            )

    def _try_dish_guided_scale(
        self,
        image: np.ndarray,
        yolo_detections: Optional[List[dict]] = None,
        seg_results: Optional[List] = None,
    ) -> Optional[ScaleResult]:
        """Compute scale by fitting an ellipse directly to the detected dish.
        
        Uses YOLO bounding box & SAM2 mask contour so the circle/ellipse
        perfectly bounds the bowl/plate rim rather than floating in the air.
        """
        if not yolo_detections and not seg_results:
            return None

        # Find the main dish container (prioritize true container dishes)
        target_bbox = None
        target_class = None
        target_mask = None

        if seg_results and len(seg_results) > 0:
            candidates_seg = []
            for s in seg_results:
                d_type = self.class_to_dish_type.get(s.class_name, "default")
                is_container = d_type in self.container_dish_types
                area = s.mask.sum() if hasattr(s, 'mask') else 0
                candidates_seg.append((is_container, area, s))
            
            # Sort: containers first, then largest mask area
            candidates_seg.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if candidates_seg and candidates_seg[0][1] > 50:
                best_seg = candidates_seg[0][2]
                target_mask = best_seg.mask
                target_class = best_seg.class_name
                target_bbox = best_seg.bbox

        if target_bbox is None and yolo_detections and len(yolo_detections) > 0:
            candidates_det = []
            for d in yolo_detections:
                d_type = self.class_to_dish_type.get(d["class_name"], "default")
                is_container = d_type in self.container_dish_types
                area = (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                candidates_det.append((is_container, area, d))
            
            candidates_det.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if candidates_det:
                best_det = candidates_det[0][2]
                target_bbox = best_det["bbox"]
                target_class = best_det["class_name"]

        if target_bbox is None:
            return None

        x1, y1, x2, y2 = target_bbox
        bw = max(10, x2 - x1)
        bh = max(10, y2 - y1)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Determine expected diameter in mm
        dish_type = self.class_to_dish_type.get(target_class, "default")
        expected_diameter_mm = self.plate_heuristics.get(
            dish_type, self.plate_heuristics["default"]
        )

        # Default axes from bbox
        major_r = int(max(bw, bh) / 2)
        minor_r = int(min(bw, bh) / 2)
        angle_deg = 0.0

        # Refine ellipse using SAM2 mask contour if available
        if target_mask is not None:
            mask_u8 = target_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                if len(largest_cnt) >= 5:
                    try:
                        (fit_cx, fit_cy), (fit_w, fit_h), fit_angle = cv2.fitEllipse(largest_cnt)
                        # Sanity check fitted ellipse
                        if fit_w > 20 and fit_h > 20:
                            cx = int(fit_cx)
                            cy = int(fit_cy)
                            major_r = int(max(fit_w, fit_h) / 2)
                            minor_r = int(min(fit_w, fit_h) / 2)
                            angle_deg = fit_angle if fit_w < fit_h else (fit_angle + 90) % 180
                    except Exception:
                        pass

        # Major diameter in pixels is the true uncompressed physical diameter
        major_diameter_px = 2 * major_r
        mm_per_pixel = expected_diameter_mm / max(1.0, float(major_diameter_px))

        return ScaleResult(
            mm_per_pixel=mm_per_pixel,
            scale_source="plate_heuristic",
            confidence="medium",
            plate_center=(cx, cy),
            plate_radius_px=major_r,
            plate_axes_px=(major_r, minor_r),
            plate_angle_deg=angle_deg,
            dish_name=target_class,
            dish_expected_diameter_mm=expected_diameter_mm,
            notes=[
                f"Khớp vành bát/đĩa từ đối tượng '{target_class}' (Tâm: ({cx}, {cy}))",
                f"Kích thước vành: {major_r*2}px × {minor_r*2}px (Đường kính thực tế: Ø{expected_diameter_mm}mm)",
                f"Tỉ lệ quy đổi: {mm_per_pixel:.4f} mm/px",
                "⚠ Tỉ lệ ước lượng từ kích thước chuẩn của loại bát/đĩa này",
            ],
        )

    def _try_plate_heuristic(
        self,
        image: np.ndarray,
        detected_classes: Optional[List[str]] = None,
    ) -> Optional[ScaleResult]:
        """Fallback circle detection when no YOLO/SAM results are available."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape[:2]

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min(h, w) * 0.3),
            param1=100,
            param2=50,
            minRadius=int(min(h, w) * 0.15),
            maxRadius=int(min(h, w) * 0.48),
        )

        if circles is None:
            return None

        circles = np.round(circles[0]).astype(int)
        circles = circles[circles[:, 2].argsort()[::-1]]
        cx, cy, radius_px = circles[0]

        dish_type = "default"
        if detected_classes:
            for cls_name in detected_classes:
                if cls_name in self.class_to_dish_type:
                    dish_type = self.class_to_dish_type[cls_name]
                    break

        expected_diameter_mm = self.plate_heuristics.get(
            dish_type, self.plate_heuristics["default"]
        )
        mm_per_pixel = expected_diameter_mm / (2 * radius_px)

        return ScaleResult(
            mm_per_pixel=mm_per_pixel,
            scale_source="plate_heuristic",
            confidence="medium",
            plate_center=(int(cx), int(cy)),
            plate_radius_px=int(radius_px),
            plate_axes_px=(int(radius_px), int(radius_px)),
            dish_expected_diameter_mm=expected_diameter_mm,
            notes=[
                f"Phát hiện hình tròn đĩa/bát: ({cx}, {cy}), r={radius_px}px",
                f"Ước lượng Ø{expected_diameter_mm}mm → {mm_per_pixel:.4f} mm/px",
            ],
        )

    def _image_heuristic(self, image: np.ndarray) -> ScaleResult:
        """Last resort: estimate scale from image dimensions."""
        h, w = image.shape[:2]
        reference_mm = 300
        avg_dim = (h + w) / 2
        mm_per_pixel = reference_mm / avg_dim

        return ScaleResult(
            mm_per_pixel=mm_per_pixel,
            scale_source="image_heuristic",
            confidence="low",
            notes=[
                f"Không tìm thấy marker hoặc đĩa — dùng kích thước ảnh {w}×{h}",
                f"Tỉ lệ ước lượng: {mm_per_pixel:.4f} mm/px",
            ],
        )

    def draw_aruco_overlay(
        self,
        image: np.ndarray,
        result: ScaleResult,
    ) -> np.ndarray:
        """Draw ArUco marker or fitted dish rim overlay with high visibility."""
        overlay = image.copy()

        if result.marker_corners is not None:
            corners_int = result.marker_corners.astype(int)
            for i in range(4):
                p1 = tuple(corners_int[i])
                p2 = tuple(corners_int[(i + 1) % 4])
                cv2.line(overlay, p1, p2, (0, 255, 0), 3)
                cv2.circle(overlay, p1, 6, (0, 0, 255), -1)

            center = corners_int.mean(axis=0).astype(int)
            label = f"ArUco Marker: {result.mm_per_pixel:.4f} mm/px"
            cv2.putText(overlay, label, (center[0] - 80, max(25, center[1] - 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        if result.plate_center is not None:
            cx, cy = result.plate_center
            # Draw center crosshair
            cv2.drawMarker(overlay, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

            # Draw fitted ellipse or circle
            if result.plate_axes_px is not None:
                major_r, minor_r = result.plate_axes_px
                angle = getattr(result, "plate_angle_deg", 0.0)
                # Draw main rim ellipse in bright green
                cv2.ellipse(overlay, (cx, cy), (major_r, minor_r), angle, 0, 360, (0, 255, 0), 3)
                # Draw subtle outer outline
                cv2.ellipse(overlay, (cx, cy), (major_r + 2, minor_r + 2), angle, 0, 360, (0, 0, 0), 1)
            elif result.plate_radius_px is not None:
                r = result.plate_radius_px
                cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 3)

            # Add clear label
            dish_info = f"{result.dish_name or 'Dish'} (Ø{result.dish_expected_diameter_mm or 200:.0f}mm)"
            label = f"{dish_info} -> {result.mm_per_pixel:.4f} mm/px"
            
            # Text background badge for readability
            text_pos = (max(10, cx - 140), max(30, cy - (result.plate_axes_px[1] if result.plate_axes_px else 50) - 15))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(overlay, (text_pos[0] - 4, text_pos[1] - th - 6),
                          (text_pos[0] + tw + 4, text_pos[1] + 4), (0, 0, 0), -1)
            cv2.putText(overlay, label, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        return overlay
