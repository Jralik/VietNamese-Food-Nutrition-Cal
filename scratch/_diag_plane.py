import sys
import json
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
from depth_estimation import DepthEstimator, estimate_camera_intrinsics
from scale_recovery import ScaleRecovery
from volume_nutrition import fit_plane_to_boundary, relief_scale_factor
import pipeline_config as pc

mp = json.load(open(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\scratch\gt_mapping_verified.json", encoding="utf-8"))
coco = json.load(open(r"C:\Users\huynh\OneDrive\Pictures\test-image\FoodDetector\annotations\instances_default.json", encoding="utf-8"))
ann_by_id = {a["id"]: a for a in coco["annotations"]}

de = DepthEstimator()
sr = ScaleRecovery()

target_fns = ["banh-mi1.jpg", "banh-mi3.jpg", "banh-mi4.jpg", "banh-mi5.jpg", "BanhMi_Trung (1).jpg", "BanhMi_Trung (3).jpg", "BanhMi_Trung (4).jpg", "ComGa2 (1).jpg"]
inst_by_fn = {}
for x in mp["instances"]:
    if x["file_name"] in target_fns:
        inst_by_fn.setdefault(x["file_name"], []).append(x)

for fn in target_fns:
    insts = inst_by_fn[fn]
    img_bgr = cv2.imread(insts[0]["image_path"])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_full, w_full = img_rgb.shape[:2]
    vol_pil = Image.fromarray(img_rgb)
    vol_pil.thumbnail((2000, 2000))
    vol_rgb = np.asarray(vol_pil)
    vol_bgr = cv2.cvtColor(vol_rgb, cv2.COLOR_RGB2BGR)
    h_vol, w_vol = vol_rgb.shape[:2]
    sx, sy = w_vol / float(w_full), h_vol / float(h_full)

    K, dist = estimate_camera_intrinsics((h_vol, w_vol), image_path=insts[0]["image_path"], calib_file=pc.CAMERA_CALIB_FILE)
    dres = de.estimate(vol_rgb)
    sres = sr.recover(vol_bgr, K=K, dist=dist, detected_classes=[x["vietfood67_class_name"] for x in insts], yolo_detections=[])
    depth_mm = dres.depth_mm.copy()
    anchored = False
    if sres.scale_source in ("aruco_marker", "checkerboard") and sres.marker_tvec is not None and sres.marker_corners is not None:
        z_true = float(np.ravel(sres.marker_tvec)[2])
        mmask = np.zeros(depth_mm.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mmask, [sres.marker_corners.astype(np.int32)], 1)
        md = depth_mm[mmask > 0]
        d_pred = float(np.median(md[md > 0])) if (md > 0).any() else 0.0
        if d_pred > 0 and z_true > 0:
            scale_z = z_true / d_pred
            if 0.25 <= scale_z <= 4.0:
                depth_mm *= scale_z
                anchored = True

    for x in insts:
        ann = ann_by_id[x["annotation_id"]]
        mask = np.zeros((h_vol, w_vol), dtype=np.uint8)
        for seg in ann["segmentation"]:
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        mbool = mask.astype(bool)
        plane_z = fit_plane_to_boundary(depth_mm, mbool)
        alpha = 0.79 if anchored else relief_scale_factor(float(np.median(plane_z[mbool])), sres.mm_per_pixel, K)
        raw_h = (plane_z[mbool] - depth_mm[mbool]) * alpha
        clipped_h = np.clip(raw_h, 2.0, 55.0)
        sat_frac = float((clipped_h >= 53.5).mean())
        print(f"{fn:<22} {x['short_class']:<13} mmp={sres.mm_per_pixel:.4f} z_food={np.median(depth_mm[mbool]):6.1f} z_plane={np.median(plane_z[mbool]):6.1f} raw_h_med={np.median(raw_h):5.1f} clip_h_mean={clipped_h.mean():5.1f} sat={sat_frac:.2f} GT={x['gt_weight_g']}")

