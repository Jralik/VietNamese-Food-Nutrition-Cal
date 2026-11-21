import sys
import os
import json
import math
import statistics as st
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from depth_estimation import DepthEstimator, estimate_camera_intrinsics
from scale_recovery import ScaleRecovery
from volume_nutrition import (
    VolumeNutritionEstimator,
    fit_plane_to_boundary,
    relief_scale_factor,
    SATURATION_CAP_RATIO,
)

COCO_JSON_PATH = Path(r"C:\Users\huynh\OneDrive\Pictures\test-image\FoodDetector\annotations\instances_default.json")
VERIFIED_MAPPING_PATH = PROJECT_ROOT / "scratch" / "gt_mapping_verified.json"
BENCH_JSONL_PATH = PROJECT_ROOT / "scratch" / "full_seg_benchmark.jsonl"
CACHE_DIR = PROJECT_ROOT / "scratch" / "bench_cache_full"
DEPTH_CACHE_DIR = CACHE_DIR / "depth_ctx"
os.makedirs(DEPTH_CACHE_DIR, exist_ok=True)


def build_or_load_depth_cache():
    mp = json.load(open(VERIFIED_MAPPING_PATH, encoding="utf-8"))
    coco = json.load(open(COCO_JSON_PATH, encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}

    inst_by_fn = defaultdict(list)
    for x in mp["instances"]:
        inst_by_fn[x["file_name"]].append(x)

    target_files = sorted(inst_by_fn.keys())
    de = None
    sr = None

    cached_scenes = {}
    for idx, fn in enumerate(target_files):
        stem = Path(fn).stem
        npz_path = DEPTH_CACHE_DIR / f"scene_{stem}.npz"
        insts = inst_by_fn[fn]
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            cached_scenes[fn] = {
                "depth_mm": data["depth_mm"],
                "K": data["K"],
                "mm_per_pixel": float(data["mm_per_pixel"]),
                "scale_source": str(data["scale_source"]),
                "depth_anchored": bool(data["depth_anchored"]),
                "h_vol": int(data["h_vol"]),
                "w_vol": int(data["w_vol"]),
                "sx": float(data["sx"]),
                "sy": float(data["sy"]),
                "insts": insts,
            }
            data.close()
            continue

        if de is None:
            de = DepthEstimator()
            sr = ScaleRecovery()

        img_path = insts[0]["image_path"]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_full, w_full = img_rgb.shape[:2]
        vol_pil = Image.fromarray(img_rgb)
        vol_pil.thumbnail((2000, 2000))
        vol_rgb = np.asarray(vol_pil)
        vol_bgr = cv2.cvtColor(vol_rgb, cv2.COLOR_RGB2BGR)
        h_vol, w_vol = vol_rgb.shape[:2]
        sx, sy = w_vol / float(w_full), h_vol / float(h_full)

        K, dist = estimate_camera_intrinsics((h_vol, w_vol), image_path=img_path, calib_file=pc.CAMERA_CALIB_FILE)
        dres = de.estimate(vol_rgb)
        scaled_dets = [
            {
                "bbox": (
                    int(round(x["bbox_xyxy"][0] * sx)),
                    int(round(x["bbox_xyxy"][1] * sy)),
                    int(round(x["bbox_xyxy"][2] * sx)),
                    int(round(x["bbox_xyxy"][3] * sy)),
                ),
                "class_name": x["vietfood67_class_name"],
                "confidence": 1.0,
            }
            for x in insts
        ]
        sres = sr.recover(
            vol_bgr,
            K=K,
            dist=dist,
            detected_classes=[d["class_name"] for d in scaled_dets],
            yolo_detections=scaled_dets,
            seg_results=None,
        )
        final_depth_mm = dres.depth_mm.copy()
        depth_anchored = False
        if (
            sres.scale_source in ("aruco_marker", "checkerboard")
            and sres.marker_tvec is not None
            and sres.marker_corners is not None
        ):
            z_marker_true = float(np.ravel(sres.marker_tvec)[2])
            corners = sres.marker_corners
            marker_mask = np.zeros(final_depth_mm.shape[:2], dtype=np.uint8)
            cv2.fillPoly(marker_mask, [corners.astype(np.int32)], 1)
            marker_depths = final_depth_mm[marker_mask > 0]
            valid_md = marker_depths[marker_depths > 0]
            d_pred = float(np.median(valid_md)) if len(valid_md) > 0 else 0.0
            if d_pred > 0 and z_marker_true > 0:
                scale_z = z_marker_true / d_pred
                if 0.25 <= scale_z <= 4.0:
                    final_depth_mm *= scale_z
                    depth_anchored = True

        np.savez_compressed(
            npz_path,
            depth_mm=final_depth_mm.astype(np.float32),
            K=K,
            mm_per_pixel=float(sres.mm_per_pixel),
            scale_source=str(sres.scale_source),
            depth_anchored=bool(depth_anchored),
            h_vol=int(h_vol),
            w_vol=int(w_vol),
            sx=float(sx),
            sy=float(sy),
        )
        cached_scenes[fn] = {
            "depth_mm": final_depth_mm.astype(np.float32),
            "K": K,
            "mm_per_pixel": float(sres.mm_per_pixel),
            "scale_source": str(sres.scale_source),
            "depth_anchored": bool(depth_anchored),
            "h_vol": int(h_vol),
            "w_vol": int(w_vol),
            "sx": float(sx),
            "sy": float(sy),
            "insts": insts,
        }
        print(f"[cache] [{idx+1}/{len(target_files)}] Cached depth & scale for {fn}", flush=True)

    return cached_scenes, ann_by_id


if __name__ == "__main__":
    scenes, _ = build_or_load_depth_cache()
    print(f"All {len(scenes)} scenes cached and ready!")

