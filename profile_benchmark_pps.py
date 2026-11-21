"""
Empirical Profiling & Benchmark Script for Food Volume Pipeline.
Measures exact stage timings and quality metrics on real images to validate:
1. SAM2 box-prompted: Cold-start vs Inference breakdown.
2. FoodSAM: pps=32 vs pps=16 trade-off (Runtime, Masks, Ingredients, Volume, Mass, VRAM).
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

TEST_IMAGE_PATH = r"C:\Users\huynh\OneDrive\Pictures\test-image\No_ArUcO\Pho.jpg"
if not os.path.exists(TEST_IMAGE_PATH):
    # fallback to HomeCook
    alt = r"C:\Users\huynh\OneDrive\Pictures\test-image\HomeCook\HomeCook1 (1).jpg"
    if os.path.exists(alt):
        TEST_IMAGE_PATH = alt

print(f"=== BENCHMARK TARGET: {TEST_IMAGE_PATH} ===")


def profile_sam2_box():
    """Measure exact breakdown of SAM2 box-prompted via volume_worker subprocess."""
    print("\n" + "="*50)
    print("PROFILING: SAM2 Box-Prompted (Single Call via volume_worker)")
    print("="*50)

    # Prepare job
    job = {
        "image_path": TEST_IMAGE_PATH,
        "detections": [
            {"bbox": [67, 159, 833, 673], "class_name": "Pho (Vietnamese noodle soup)", "confidence": 0.96}
        ],
        "backend": "sam2"
    }
    tmp_job = PROJECT_ROOT / "temp_profile_job.json"
    tmp_out = PROJECT_ROOT / "temp_profile_out.json"
    with open(tmp_job, "w", encoding="utf-8") as f:
        json.dump(job, f)

    t_start = time.time()
    proc = subprocess.run(
        [PYTHON_EXE, str(PROJECT_ROOT / "volume_worker.py"), str(tmp_job), str(tmp_out)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    t_total = time.time() - t_start

    print(f"Total Subprocess Wall-Clock Time: {t_total:.2f}s")
    if proc.returncode != 0:
        print("Worker failed:", proc.stderr[-300:])
        return None

    with open(tmp_out, encoding="utf-8") as f:
        res = json.load(f)

    if tmp_job.exists(): tmp_job.unlink()
    if tmp_out.exists(): tmp_out.unlink()

    est = res["estimations"][0] if res["estimations"] else {}
    print(f"Result: class={est.get('class_name')} | vol={est.get('volume_cm3', 0):.1f} cm3 | mass={est.get('mass_g', 0):.1f}g")
    return {"total_s": t_total, "volume": est.get("volume_cm3", 0), "mass": est.get("mass_g", 0)}


def profile_foodsam_infer(pps: int):
    """Profile foodsam_infer.py directly inside FoodSAM env to isolate internal stages."""
    print("\n" + "-"*50)
    print(f"PROFILING FoodSAM Internal Stages (pps={pps})")
    print("-"*50)

    from pipeline_config import FOODSAM_REPO, FOODSAM_INFER_SCRIPT, FOODSAM_INGREDIENT_MAP, FOODSAM_MIXED_PLATE_SUSPECT_CLASSES
    env_python = os.path.join(FOODSAM_REPO, "env", "Scripts", "python.exe")
    out_dir = PROJECT_ROOT / f"temp_profile_out_pps{pps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dets_path = out_dir / "dets.json"
    map_path = out_dir / "map.json"
    suspect_path = out_dir / "suspect.json"

    with open(dets_path, "w", encoding="utf-8") as f:
        json.dump([{"bbox": [67, 159, 833, 673], "class_name": "Pho (Vietnamese noodle soup)", "confidence": 0.96}], f)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(FOODSAM_INGREDIENT_MAP, f)
    with open(suspect_path, "w", encoding="utf-8") as f:
        json.dump(sorted(FOODSAM_MIXED_PLATE_SUSPECT_CLASSES), f)

    cmd = [
        env_python, FOODSAM_INFER_SCRIPT,
        TEST_IMAGE_PATH, str(dets_path), str(out_dir),
        "--pps", str(pps),
        "--ingredient-mode", "breakdown_and_extra",
        "--ingredient-map", str(map_path),
        "--suspect-classes", str(suspect_path),
    ]

    t_wall_start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(FOODSAM_REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    t_wall_total = time.time() - t_wall_start

    if proc.returncode != 0:
        print(f"foodsam_infer pps={pps} FAILED:")
        print(proc.stderr[-500:])
        return None

    result_file = out_dir / "result.json"
    with open(result_file, encoding="utf-8") as f:
        data = json.load(f)

    timings = data.get("timing_s", {})
    vram = data.get("vram_peak_mib", {})
    n_sam_masks = data.get("n_sam_masks", 0)
    extras = data.get("extras", [])
    t_internal_sum = sum(timings.values())
    t_overhead = t_wall_total - t_internal_sum

    print(f"Wall-Clock Total:           {t_wall_total:.2f}s")
    print(f"  ├── Process & Import Overhead: {t_overhead:.2f}s")
    print(f"  ├── SAM2 AMG Generation:       {timings.get('sam2', 0):.2f}s")
    print(f"  ├── SETR Semantic Inference:   {timings.get('setr', 0):.2f}s")
    print(f"  ├── Dish Mask Synthesis:       {timings.get('dish', 0):.2f}s")
    print(f"  └── Ingredient Extraction:     {timings.get('extract', 0):.2f}s")
    print(f"Quality & Memory Metrics:")
    print(f"  ├── SAM Masks Discovered:      {n_sam_masks}")
    print(f"  ├── Accepted Ingredient Extras: {len(extras)} {[e.get('class_name') for e in extras]}")
    print(f"  ├── Peak VRAM SAM2:            {vram.get('sam2', 0):.1f} MiB")
    print(f"  └── Peak VRAM SETR:            {vram.get('setr', 0):.1f} MiB")

    return {
        "pps": pps,
        "wall_s": t_wall_total,
        "overhead_s": t_overhead,
        "sam2_s": timings.get("sam2", 0),
        "setr_s": timings.get("setr", 0),
        "masks": n_sam_masks,
        "extras_count": len(extras),
        "extras": [e.get("class_name") for e in extras]
    }


if __name__ == "__main__":
    # 1. Profile SAM2 Box
    res_box = profile_sam2_box()

    # 2. Profile FoodSAM pps=32
    res_32 = profile_foodsam_infer(pps=32)

    # 3. Profile FoodSAM pps=16
    res_16 = profile_foodsam_infer(pps=16)

    print("\n" + "="*60)
    print("EMPIRICAL COMPARISON SUMMARY")
    print("="*60)
    if res_32 and res_16:
        speedup_sam2 = res_32["sam2_s"] / res_16["sam2_s"] if res_16["sam2_s"] > 0 else 0
        speedup_wall = res_32["wall_s"] / res_16["wall_s"] if res_16["wall_s"] > 0 else 0
        print(f"SAM2 AMG Time:      {res_32['sam2_s']:.2f}s (pps=32) vs {res_16['sam2_s']:.2f}s (pps=16)  -> Speedup: {speedup_sam2:.2f}x")
        print(f"Total Wall-Clock:   {res_32['wall_s']:.2f}s (pps=32) vs {res_16['wall_s']:.2f}s (pps=16)  -> Speedup: {speedup_wall:.2f}x")
        print(f"Discovered Masks:   {res_32['masks']} (pps=32) vs {res_16['masks']} (pps=16)")
        print(f"Extracted Extras:   {res_32['extras_count']} vs {res_16['extras_count']}")
