import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_config as pc
from scratch.run_volume_ablation import (
    build_or_load_depth_cache,
    VolumeNutritionEstimator,
    coco_poly_to_mask,
    fit_plane_to_boundary,
    relief_scale_factor,
)

cached_scenes, ann_by_id = build_or_load_depth_cache()

# Check all multi-instance scenes for any spatial relationship
print("=== Multi-instance scenes in dataset ===")
for fn in sorted(cached_scenes.keys()):
    sc = cached_scenes[fn]
    if len(sc["insts"]) > 1:
        classes = [(x["short_class"], x["gt_weight_g"], x["mapping_status"]) for x in sc["insts"]]
        print(f"  {fn:22s}: {classes}")

