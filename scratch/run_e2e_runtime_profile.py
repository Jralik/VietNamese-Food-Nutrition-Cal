"""
Thesis Extra Experiment 4: End-to-End Runtime & VRAM Profile on GTX 1650 4 GB
Measures exact per-stage wall-clock latency & Peak GPU VRAM for the full production stack:
  1. YOLOv10b Detection (CPU dual-preresize)
  2. MobileSAM Config C Segmentation (CUDA TinyViT)
  3. Depth Anything V2 Small (CUDA metric indoor)
  4. ArUco Scale Recovery + Depth Anchoring (CPU OpenCV)
  5. Volume (A1-A5 + B1-B2 + C1-C6) & Nutrition Estimation (CPU NumPy)

Evaluated on 20 stratified real images after 2 warm-up images.
"""
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.modules["pyarrow"] = None  # Avoid Windows pyarrow DLL access violation in pandas

import av  # noqa: F401
import cv2
import numpy as np
import psutil
import torch
from PIL import Image, ImageOps

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))

import pipeline_config as pc
from class_names import class_names
from depth_estimation import DepthEstimator, estimate_camera_intrinsics
from food_segmentation import MobileSAMSegmenter
from scale_recovery import ScaleRecovery
from ultralytics import YOLO
from utils import predict_with_dual_preresize
from volume_nutrition import VolumeNutritionEstimator

MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
OUT_JSON = ROOT / "scratch" / "bench_cache_v2" / "e2e_runtime_profile_v2.json"
YOLO_WEIGHTS = ROOT / "model" / "yolov10" / "YOLOv10b_VietFood67_SGD_new_bigger.pt"


def main():
    mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
    inst_by_file = defaultdict(list)
    for x in mp["instances"]:
        inst_by_file[x["file_name"]].append(x)
    all_files = sorted(inst_by_file.keys())
    # Pick 22 evenly spaced images across all 418 scenes (2 warmup + 20 timed)
    indices = np.linspace(0, len(all_files) - 1, 22, dtype=int)
    selected = [all_files[i] for i in indices]

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {gpu_name} | Profiling {len(selected)-2} images after 2 warm-up runs...", flush=True)

    yolo_model = YOLO(str(YOLO_WEIGHTS))
    yolo_model.overrides["device"] = "cpu"
    msam = MobileSAMSegmenter()
    msam._load_model()
    depth_est = DepthEstimator()
    scale_rec = ScaleRecovery()
    vol_est = VolumeNutritionEstimator()

    timings = defaultdict(list)
    peak_vrams = []
    cpu_rams = []

    for i_img, fn in enumerate(selected):
        img_path = inst_by_file[fn][0]["image_path"]
        pil_upright = ImageOps.exif_transpose(Image.open(img_path))
        img_rgb = np.asarray(pil_upright.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t_all0 = time.perf_counter()

        # 1. YOLO Detection
        t0 = time.perf_counter()
        res = predict_with_dual_preresize(yolo_model, pil_upright, conf=0.30)[0]
        dets = []
        for b in res.boxes:
            cid = int(b.cls[0].item())
            if 0 <= cid < len(class_names):
                cname = class_names[cid]["name"]
                if cname != "Con nguoi (Human)":
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    dets.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "class_name": cname, "confidence": float(b.conf[0].item())})
        t_yolo = (time.perf_counter() - t0) * 1000.0

        # Resize for volume/depth stage (matches production 2000px thumbnail)
        h_full, w_full = img_rgb.shape[:2]
        vol_pil = Image.fromarray(img_rgb)
        vol_pil.thumbnail((2000, 2000))
        vol_rgb = np.asarray(vol_pil)
        vol_bgr = cv2.cvtColor(vol_rgb, cv2.COLOR_RGB2BGR)
        h_vol, w_vol = vol_rgb.shape[:2]
        sx, sy = w_vol / float(w_full), h_vol / float(h_full)
        scaled_dets = [
            {
                "bbox": (int(round(d["bbox"][0] * sx)), int(round(d["bbox"][1] * sy)), int(round(d["bbox"][2] * sx)), int(round(d["bbox"][3] * sy))),
                "class_name": d["class_name"],
                "confidence": d["confidence"],
            }
            for d in dets
        ]

        # 2. MobileSAM Config C Segmentation
        t0 = time.perf_counter()
        seg_res = msam.segment(vol_rgb, scaled_dets) if scaled_dets else []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_sam = (time.perf_counter() - t0) * 1000.0

        # 3. Depth Anything V2 Small
        t0 = time.perf_counter()
        dres = depth_est.estimate(vol_rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_depth = (time.perf_counter() - t0) * 1000.0

        # 4. ArUco Scale Recovery + Depth Anchoring
        t0 = time.perf_counter()
        K, dist = estimate_camera_intrinsics((h_vol, w_vol), image_path=img_path, calib_file=pc.CAMERA_CALIB_FILE)
        sres = scale_rec.recover(vol_bgr, K=K, dist=dist, detected_classes=[d["class_name"] for d in scaled_dets], yolo_detections=scaled_dets, seg_results=seg_res)
        final_depth = dres.depth_mm.copy()
        z_marker_mm = None
        if sres.scale_source in ("aruco_marker", "checkerboard") and sres.marker_tvec is not None and sres.marker_corners is not None:
            z_true = float(np.ravel(sres.marker_tvec)[2])
            m_mask = np.zeros(final_depth.shape[:2], dtype=np.uint8)
            cv2.fillPoly(m_mask, [sres.marker_corners.astype(np.int32)], 1)
            md = final_depth[m_mask > 0]
            vmd = md[md > 0]
            if len(vmd) > 0 and z_true > 0:
                scale_z = float(np.clip(z_true / float(np.median(vmd)), 0.3, 3.0))
                final_depth = np.clip(final_depth * scale_z, 100.0, 5000.0)
                sres.depth_anchored = True
                z_marker_mm = float(np.median(final_depth[m_mask > 0]))
        t_scale = (time.perf_counter() - t0) * 1000.0

        # 5. Volume & Nutrition
        t0 = time.perf_counter()
        scene_classes = [s.class_name for s in seg_res]
        for s in seg_res:
            dscale = scale_rec.get_dish_scale(bbox=s.bbox, class_name=s.class_name, global_scale=sres)
            vol_est.estimate(
                depth_map_mm=final_depth, food_mask=s.mask, class_name=s.class_name,
                mm_per_pixel=dscale, scale_source=sres.scale_source, bbox=s.bbox,
                K=K, homography=sres.homography, depth_anchored=sres.depth_anchored,
                z_marker_mm=z_marker_mm, scene_classes=scene_classes,
            )
        t_vol = (time.perf_counter() - t0) * 1000.0
        t_tot = (time.perf_counter() - t_all0) * 1000.0

        vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
        ram = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)

        if i_img >= 2:  # skip 2 warmups
            timings["yolo_ms"].append(t_yolo)
            timings["mobilesam_ms"].append(t_sam)
            timings["depth_ms"].append(t_depth)
            timings["scale_aruco_ms"].append(t_scale)
            timings["volume_nutrition_ms"].append(t_vol)
            timings["total_e2e_ms"].append(t_tot)
            peak_vrams.append(vram)
            cpu_rams.append(ram)
            print(
                f"  [{i_img-1:2d}/20] {fn:28s} | YOLO={t_yolo:6.1f} | SAM={t_sam:5.1f} | Depth={t_depth:5.1f} | Scale={t_scale:4.1f} | Vol={t_vol:4.1f} | Tot={t_tot:6.1f} ms | VRAM={vram} MiB",
                flush=True,
            )

    summary = {
        "gpu_name": gpu_name,
        "n_timed_images": len(timings["total_e2e_ms"]),
        "stages": {
            k: {
                "mean_ms": round(st.mean(v), 1),
                "p50_ms": round(st.median(v), 1),
                "p95_ms": round(float(np.percentile(v, 95)), 1),
            }
            for k, v in timings.items()
        },
        "peak_vram_mib": max(peak_vrams),
        "mean_cpu_ram_mib": round(st.mean(cpu_rams), 1),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== END-TO-END RUNTIME SUMMARY ===")
    for k, s in summary["stages"].items():
        print(f"  {k:22s}: Mean={s['mean_ms']:6.1f} ms | P50={s['p50_ms']:6.1f} ms | P95={s['p95_ms']:6.1f} ms")
    print(f"  Peak GPU VRAM: {summary['peak_vram_mib']} MiB | Mean CPU RAM: {summary['mean_cpu_ram_mib']} MiB")


if __name__ == "__main__":
    main()
