import logging
logging.basicConfig(level=logging.INFO)
import pandas
from PIL import Image, ImageOps
from ultralytics import YOLO
import volume_integration
from class_names import class_names

model = YOLO(r"./model/yolov26/best.onnx", task="detect")
img = ImageOps.exif_transpose(Image.open(r"C:\Users\huynh\OneDrive\Pictures\test-image\com-tam3.jpg").convert("RGB"))
src = img.copy(); src.thumbnail((2000, 2000))
res = model.predict(img.resize((640, 640)), conf=0.25, imgsz=640)
dets = volume_integration.extract_yolo_detections(res, class_names,
        bbox_scale=(src.width/640.0, src.height/640.0))

# Raw volume with floor disabled: monkeypatch min_mean_height_mm to 0
from volume_nutrition import VolumeNutritionEstimator, compute_volume_column_integration
import food_volume_pipeline
est = VolumeNutritionEstimator()

# run worker-less: replicate pipeline steps quickly using pipeline classes is heavy;
# instead call estimate() then compute floor effect by ratio: volume scales linearly with mean height.
import volume_integration as vi
r = vi.estimate_nutrition_volume(src, dets, "comtam3.jpg")
print("with floor:", r.estimations[0].volume_cm3, r.estimations[0].mass_g)
