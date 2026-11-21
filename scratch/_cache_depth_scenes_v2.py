import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from depth_estimation import DepthEstimator, estimate_camera_intrinsics
from scale_recovery import ScaleRecovery

VERIFIED_MAPPING_PATH = PROJECT_ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = PROJECT_ROOT / "scratch" / "bench_cache_v2"
DEPTH_CACHE_DIR = CACHE_DIR / "depth_ctx"
SUMMARY_PATH = CACHE_DIR / "aruco_depth_audit_v2.json"
os.makedirs(DEPTH_CACHE_DIR, exist_ok=True)


def build_depth_cache_v2():
    mp = json.loads(VERIFIED_MAPPING_PATH.read_text(encoding="utf-8"))
    inst_by_fn = defaultdict(list)
    for x in mp["instances"]:
        inst_by_fn[x["file_name"]].append(x)

    target_files = sorted(inst_by_fn.keys())
    de = None
    sr = None

    scale_counts = Counter()
    anchored_count = 0
    dim_mismatches = []
    no_aruco_files = []

    for idx, fn in enumerate(target_files):
        stem = Path(fn).stem
        npz_path = DEPTH_CACHE_DIR / f"scene_{stem}.npz"
        insts = inst_by_fn[fn]

        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            s_src = str(data["scale_source"])
            d_anc = bool(data["depth_anchored"])
            mm_px = float(data["mm_per_pixel"])
            data.close()
        else:
            if de is None:
                de = DepthEstimator()
                sr = ScaleRecovery()

            img_path = insts[0]["image_path"]
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_full, w_full = img_rgb.shape[:2]
            if (h_full, w_full) != (int(insts[0]["height"]), int(insts[0]["width"])):
                dim_mismatches.append((fn, (h_full, w_full), (insts[0]["height"], insts[0]["width"])))

            vol_pil = Image.fromarray(img_rgb)
            vol_pil.thumbnail((2000, 2000))
            vol_rgb = np.asarray(vol_pil)
            vol_bgr = cv2.cvtColor(vol_rgb, cv2.COLOR_RGB2BGR)
            h_vol, w_vol = vol_rgb.shape[:2]
            sx, sy = w_vol / float(w_full), h_vol / float(h_full)

            K, dist = estimate_camera_intrinsics(
                (h_vol, w_vol), image_path=img_path, calib_file=pc.CAMERA_CALIB_FILE
            )
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
            d_anc = False
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
                        d_anc = True

            s_src = str(sres.scale_source)
            mm_px = float(sres.mm_per_pixel)
            np.savez_compressed(
                npz_path,
                depth_mm=final_depth_mm.astype(np.float32),
                K=K,
                mm_per_pixel=mm_px,
                scale_source=s_src,
                depth_anchored=bool(d_anc),
                h_vol=int(h_vol),
                w_vol=int(w_vol),
                sx=float(sx),
                sy=float(sy),
            )
            del final_depth_mm, img_bgr, img_rgb, vol_rgb, vol_bgr, dres

        scale_counts[s_src] += 1
        if d_anc:
            anchored_count += 1
        if s_src not in ("aruco_marker", "checkerboard"):
            no_aruco_files.append({"file_name": fn, "scale_source": s_src, "mm_per_pixel": round(mm_px, 4)})

        if (idx + 1) % 25 == 0 or (idx + 1) == len(target_files):
            print(
                f"[cache_v2] [{idx+1}/{len(target_files)}] scale_counts={dict(scale_counts)} | anchored={anchored_count}",
                flush=True,
            )

    summary = {
        "total_scenes": len(target_files),
        "scale_source_counts": dict(scale_counts),
        "depth_anchored_count": anchored_count,
        "dim_mismatches": dim_mismatches,
        "non_marker_files": no_aruco_files,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done! Saved audit summary to {SUMMARY_PATH}")
    print(f"Scale sources: {dict(scale_counts)} | Depth anchored: {anchored_count}/{len(target_files)}")
    print(f"Dimension mismatches: {len(dim_mismatches)} | Non-marker scenes: {len(no_aruco_files)}")


if __name__ == "__main__":
    build_depth_cache_v2()
