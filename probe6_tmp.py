# Direct column-integration probe on com-tam3's SAM2 output is heavy;
# instead compute raw mean height from the estimate: volume = area * mean_h.
# Run the worker and inspect via compute with min_mean_height_mm=0 by
# temporarily calling compute_volume_column_integration on the same data the
# worker sees — simplest: run pipeline.analyze with monkeypatched floor.
import numpy as np, json, tempfile, os, sys
sys.path.insert(0, ".")
import volume_nutrition

orig = volume_nutrition.compute_volume_column_integration
raw_holder = {}
def wrapped(*args, **kwargs):
    kwargs["min_mean_height_mm"] = 0.0
    v = orig(*args, **kwargs)
    raw_holder["v"] = v
    return v
volume_nutrition.compute_volume_column_integration = wrapped

import pandas
from PIL import Image, ImageOps
from ultralytics import YOLO
import volume_integration
from class_names import class_names
from food_volume_pipeline import FoodVolumePipeline

model = YOLO(r"./model/yolov26/best.onnx", task="detect")
img = ImageOps.exif_transpose(Image.open(r"C:\Users\huynh\OneDrive\Pictures\test-image\com-tam3.jpg").convert("RGB"))
src = img.copy(); src.thumbnail((2000, 2000))
res = model.predict(img.resize((640, 640)), conf=0.25, imgsz=640)
dets = volume_integration.extract_yolo_detections(res, class_names,
        bbox_scale=(src.width/640.0, src.height/640.0))

pipeline = FoodVolumePipeline()
result = pipeline.analyze(np.asarray(src), dets, image_path="comtam3.jpg")
v_raw = raw_holder.get("v")
print(f"raw volume (no floor): {v_raw:.1f} cm3")
# mask area
seg = result.seg_results[0]
area_cm2 = seg.mask.sum() * (result.scale_result.mm_per_pixel ** 2) / 100.0
print(f"mask area ~{area_cm2:.0f} cm2 -> raw mean height = {v_raw/area_cm2*10:.1f} mm")
