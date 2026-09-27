"""Depth audit for the worst portion-estimation cases.

Runs the app pipeline on selected images and dumps the actual segmentation
mask overlay, depth map and per-item crops produced by the volume worker,
plus all estimation internals (volume_cm3, mm_per_pixel, method, warnings)
— so the mass error can be attributed: mask coverage vs depth vs scale.
"""

import base64
import os

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)
from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from utils import predict_with_dual_preresize
import volume_integration

OUT = os.path.join("scratch", "depth_audit")

IMAGES = [
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg",   # -2% (good)
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (3).jpg",   # +158%
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (5).jpg",   # +79%
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (1).jpg",  # egg 73.6 vs 221.2
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\banh-mi1.jpg",     # bread portion audit
]


def save_b64(b64, path):
    if not b64:
        return False
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    model = YOLO("./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt")
    model.overrides["device"] = "cpu"

    for p in IMAGES:
        name = os.path.basename(p).replace(".jpg", "").replace(" ", "_")
        upright = ImageOps.exif_transpose(Image.open(p))
        src = upright.copy()
        src.thumbnail((2000, 2000))
        bbox_scale = (src.width / upright.size[0], src.height / upright.size[1])
        res = predict_with_dual_preresize(model, upright, 0.30)[0]
        dets = volume_integration.extract_yolo_detections(
            [res], class_names, bbox_scale=bbox_scale)
        vol = volume_integration.estimate_nutrition_volume(src, dets, backend="sam2")
        print(f"\n=== {name} ===")
        if vol is None:
            print("  volume None")
            continue
        sr = vol.scale_result
        print(f"  scale: source={sr.scale_source} mm_per_pixel={sr.mm_per_pixel:.3f} "
              f"conf={sr.confidence}")
        print(f"  seg_backend_used={vol.seg_backend_used} fallback={vol.seg_fallback_used} "
              f"err={vol.seg_error}")
        for _i, e in enumerate(vol.estimations):
            print(f"  item {_i} {e.class_name.split(' (')[0]}: vol={e.volume_cm3:.0f}cm3 "
                  f"mass={e.mass_g:.1f}g std={e.mass_std_g:.1f} method={e.estimation_method} "
                  f"level={e.confidence_level} warn={e.warnings}")
            save_b64(e.crop_b64,
                     os.path.join(OUT, f"{name}__{e.class_name.split(' (')[0]}_{_i}_crop.jpg"))
        m1 = save_b64(vol.mask_overlay_b64, os.path.join(OUT, f"{name}__mask.jpg"))
        m2 = save_b64(vol.depth_colored_b64, os.path.join(OUT, f"{name}__depth.jpg"))
        m3 = save_b64(vol.scale_overlay_b64, os.path.join(OUT, f"{name}__scale.jpg"))
        print(f"  saved: mask={m1} depth={m2} scale={m3}")


if __name__ == "__main__":
    main()
