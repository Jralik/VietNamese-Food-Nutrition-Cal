"""
Benchmark FoodSAM pps=32 vs pps=16 on BunDauMamTom (mixed plate with 1200x1200 resolution).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
FOODSAM_REPO = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodSAM")
ENV_PYTHON = str(FOODSAM_REPO / "env" / "Scripts" / "python.exe")
INFER_SCRIPT = str(FOODSAM_REPO / "foodsam_infer.py")
IMG_PATH = r"C:\Users\huynh\OneDrive\Pictures\test-image\No_ArUcO\BunDauMamTom.jpg"

from pipeline_config import FOODSAM_INGREDIENT_MAP, FOODSAM_MIXED_PLATE_SUSPECT_CLASSES

DETS = [{"bbox": [65, 45, 1162, 1146], "class_name": "Bun dau (Vermicelli with tofu)", "confidence": 0.97}]


def run_test(pps: int):
    out_dir = PROJECT_ROOT / f"benchmark_bundau_pps{pps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dets_path = out_dir / "dets.json"
    map_path = out_dir / "map.json"
    suspect_path = out_dir / "suspect.json"

    with open(dets_path, "w", encoding="utf-8") as f:
        json.dump(DETS, f)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(FOODSAM_INGREDIENT_MAP, f)
    with open(suspect_path, "w", encoding="utf-8") as f:
        json.dump(sorted(FOODSAM_MIXED_PLATE_SUSPECT_CLASSES), f)

    cmd = [
        ENV_PYTHON, INFER_SCRIPT,
        IMG_PATH, str(dets_path), str(out_dir),
        "--pps", str(pps),
        "--ingredient-mode", "breakdown_and_extra",
        "--ingredient-map", str(map_path),
        "--suspect-classes", str(suspect_path),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(FOODSAM_REPO), capture_output=True, text=True, encoding="utf-8", errors="replace")
    t_wall = time.time() - t0

    if proc.returncode != 0:
        print(f"FAILED pps={pps}:", proc.stderr[-400:])
        return None

    with open(out_dir / "result.json", encoding="utf-8") as f:
        data = json.load(f)

    masks = np.load(out_dir / "masks.npz")["masks"]
    return data, masks, t_wall


if __name__ == "__main__":
    print("Running BunDauMamTom pps=16...")
    d16, m16, w16 = run_test(16)
    t_sam16 = d16["timing_s"]["sam2"]
    masks16 = d16["n_sam_masks"]
    print(f"Done pps=16: sam2={t_sam16:.1f}s, wall={w16:.1f}s, masks={masks16}")

    print("\nRunning BunDauMamTom pps=32...")
    d32, m32, w32 = run_test(32)
    t_sam32 = d32["timing_s"]["sam2"]
    masks32 = d32["n_sam_masks"]
    print(f"Done pps=32: sam2={t_sam32:.1f}s, wall={w32:.1f}s, masks={masks32}")

    iou = (m16 & m32).sum() / (m16 | m32).sum() if (m16 | m32).sum() > 0 else 1.0
    cand16 = d16["ingredient_extraction"]["candidate_count"]
    cand32 = d32["ingredient_extraction"]["candidate_count"]

    print("\n" + "="*60)
    print("BUN DAU MAM TOM BENCHMARK COMPARISON (1200x1200)")
    print("="*60)
    print(f"SAM2 AMG Time:    {t_sam32:.1f}s (pps=32) vs {t_sam16:.1f}s (pps=16) -> Speedup: {t_sam32/t_sam16:.2f}x")
    print(f"Discovered Masks: {masks32} vs {masks16}")
    print(f"Candidates:       {cand32} vs {cand16}")
    print(f"Dish Mask IoU:    {iou:.4f}")
