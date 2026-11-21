"""C5 integration test: FoodVolumePipeline with the FoodSAM backend.

Sets SEGMENTATION_BACKEND='foodsam' and runs the full volume pipeline on one
image, exercising the subprocess bridge to the FoodSAM env (SAM2 AMG + SETR
+ composite matching) end-to-end, then prints estimations + bridge info.

Usage: .venv/Scripts/python.exe test_foodsam_integration.py [image_path]
"""
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pipeline_config as pc

pc.SEGMENTATION_BACKEND = "foodsam"

import cv2
import numpy as np
from PIL import Image

from food_volume_pipeline import FoodVolumePipeline

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else Path(
    r"C:\Users\huynh\OneDrive\Pictures\test-image\No_ArUcO\Pho.jpg")
# Detection from the project's YOLOv26 stage (conf 0.96 on this image)
DETECTIONS = [
    {"bbox": (67, 159, 833, 673),
     "class_name": "Pho (Vietnamese noodle soup)",
     "confidence": 0.96},
]


def main():
    image = np.asarray(Image.open(IMG_PATH).convert("RGB"))
    h, w = image.shape[:2]
    print(f"image: {IMG_PATH} {w}x{h}")

    pipeline = FoodVolumePipeline()
    seg = pipeline.segmenter
    print(f"backend: {type(seg).__name__} | available: {seg.is_available}")

    # volume_worker convention: segment on a 640x640 copy, dets scaled to it
    seg_img = cv2.resize(image, (640, 640))
    seg_dets = [
        {"bbox": (int(d["bbox"][0] * 640.0 / w), int(d["bbox"][1] * 640.0 / h),
                  int(d["bbox"][2] * 640.0 / w), int(d["bbox"][3] * 640.0 / h)),
         "class_name": d["class_name"], "confidence": d["confidence"]}
        for d in DETECTIONS
    ]

    result = pipeline.analyze(
        image_rgb=image,
        yolo_detections=DETECTIONS,
        image_path=IMG_PATH,
        generate_visualizations=True,
        seg_image_rgb=seg_img,
        seg_yolo_detections=seg_dets,
    )

    print(f"\nbridge info: {json.dumps(getattr(seg, 'last_info', {}), ensure_ascii=False)}")
    print(f"last_error: {getattr(seg, 'last_error', None)}")
    print(f"\nitems: {len(result.estimations)}")
    for it in result.estimations:
        print(f"  class={it.class_name} conf={it.confidence}")
        print(f"  volume_cm3={it.volume_cm3:.1f}  mass_g={it.mass_g:.1f}±{it.mass_std_g:.1f}  "
              f"confidence={it.confidence_label if hasattr(it, 'confidence_label') else it.confidence}")
        print(f"  nutrition={it.nutrition}")
        for warn in it.warnings or []:
            print(f"  warning: {warn}")

    if getattr(result, "mask_overlay", None) is not None:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "test_output", "foodsam_integration_overlay.jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, cv2.cvtColor(result.mask_overlay, cv2.COLOR_RGB2BGR))
        print(f"\noverlay saved: {out}")
    print("\nINTEGRATION TEST: PASS" if result.estimations else "\nINTEGRATION TEST: NO ITEMS")


if __name__ == "__main__":
    main()
