"""
Benchmark FoodSAM pps=32 vs pps=16 on Com Tam (Broken Rice) mixed plate.
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
IMG_PATH = r"C:\Users\huynh\OneDrive\Pictures\test-image\No_ArUcO\com-tam3.jpg"

from pipeline_config import FOODSAM_INGREDIENT_MAP, FOODSAM_MIXED_PLATE_SUSPECT_CLASSES

DETS = [{"bbox": [123, 120, 574, 518], "class_name": "Com tam (Broken rice)", "confidence": 0.96}]


def run_test(pps: int):
    out_dir = PROJECT_ROOT / f"benchmark_comtam_pps{pps}"
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
    extras = np.load(out_dir / "extras.npz")["extras"]

    return {
        "pps": pps,
        "wall_s": t_wall,
        "sam2_s": data["timing_s"]["sam2"],
        "setr_s": data["timing_s"]["setr"],
        "n_sam_masks": data["n_sam_masks"],
        "candidate_count": data["ingredient_extraction"]["candidate_count"],
        "accepted_extras": data["ingredient_extraction"]["accepted_extra_count"],
        "suppressed": len(data.get("suppressed", [])),
        "promoted_labels": data["detections"][0]["promoted_labels"] if data["detections"] else [],
        "breakdown": [b["class_name"] for b in (data["detections"][0].get("ingredient_breakdown") or [])],
        "mask_area": int(masks.sum()),
        "masks": masks,
        "extras_arr": extras
    }


if __name__ == "__main__":
    print("Running pps=16 on com-tam3.jpg...")
    r16 = run_test(16)
    print(f"pps=16 complete: sam2={r16['sam2_s']:.1f}s, wall={r16['wall_s']:.1f}s, masks={r16['n_sam_masks']}")

    print("\nRunning pps=32 on com-tam3.jpg...")
    r32 = run_test(32)
    print(f"pps=32 complete: sam2={r32['sam2_s']:.1f}s, wall={r32['wall_s']:.1f}s, masks={r32['n_sam_masks']}")

    print("\n" + "="*60)
    print("COM TAM BENCHMARK COMPARISON")
    print("="*60)
    print(f"Wall-Clock:        {r32['wall_s']:.1f}s (pps=32) vs {r16['wall_s']:.1f}s (pps=16) -> Speedup: {r32['wall_s']/r16['wall_s']:.2f}x")
    print(f"SAM2 AMG Time:     {r32['sam2_s']:.1f}s (pps=32) vs {r16['sam2_s']:.1f}s (pps=16) -> Speedup: {r32['sam2_s']/r16['sam2_s']:.2f}x")
    print(f"Discovered Masks:  {r32['n_sam_masks']} vs {r16['n_sam_masks']}")
    print(f"Candidates:        {r32['candidate_count']} vs {r16['candidate_count']}")
    print(f"Breakdown Items:   {len(r32['breakdown'])} vs {len(r16['breakdown'])}")
    print(f"Breakdown Classes: {set(r32['breakdown'])} vs {set(r16['breakdown'])}")
    
    iou = (r16['masks'] & r32['masks']).sum() / (r16['masks'] | r32['masks']).sum() if (r16['masks'] | r32['masks']).sum() > 0 else 1.0
    print(f"Dish Mask IoU (pps=16 vs pps=32): {iou:.4f}")
