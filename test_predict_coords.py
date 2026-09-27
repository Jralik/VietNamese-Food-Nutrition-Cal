from ultralytics import YOLO
from PIL import Image
import os

img_path = r'C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg'
im = Image.open(img_path)
print(f"Original image size: {im.size}")

m = YOLO('model/yolov26/best.onnx', task='detect')

# Test 1: predict with full original image (im)
try:
    print("Running m.predict(im, imgsz=640)...")
    r1 = m.predict(im, conf=0.15, imgsz=640, verbose=False)[0]
    print(f"Result with im: {len(r1.boxes)} boxes")
    print(f"r1.orig_shape: {r1.orig_shape}")
    for b in r1.boxes:
        print(f"   box xyxy: {b.xyxy.cpu().numpy()[0]}, conf: {float(b.conf[0]):.3f}")
except Exception as e:
    print(f"Error with im: {e}")

# Test 2: predict with resized 640x640
try:
    print("\nRunning m.predict(im.resize((640, 640)), imgsz=640)...")
    im640 = im.resize((640, 640))
    r2 = m.predict(im640, conf=0.15, imgsz=640, verbose=False)[0]
    print(f"Result with im640: {len(r2.boxes)} boxes")
    print(f"r2.orig_shape: {r2.orig_shape}")
    for b in r2.boxes:
        print(f"   box xyxy: {b.xyxy.cpu().numpy()[0]}, conf: {float(b.conf[0]):.3f}")
except Exception as e:
    print(f"Error with im640: {e}")
