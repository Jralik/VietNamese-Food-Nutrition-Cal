import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import json

# Load class names
from class_names import class_names

# nvhnam classes (58 classes)
# Let's read nvhnam classes from ONNX metadata or first 58 classes
s26 = ort.InferenceSession('model/yolov26/best.onnx', providers=['CPUExecutionProvider'])
s10 = ort.InferenceSession('model/yolov10/YOLOv10m_new_total_VN_5_SGD.onnx', providers=['CPUExecutionProvider'])

print(f"yolov26 input type: {s26.get_inputs()[0].type}, shape: {s26.get_inputs()[0].shape}")
print(f"yolov10 input type: {s10.get_inputs()[0].type}, shape: {s10.get_inputs()[0].shape}")

test_images = [
    r'C:\Users\huynh\OneDrive\Pictures\test-image\Pho\Pho1 (1).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\Pho\Pho1 (2).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\Pho\Pho1 (3).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\Pho\Pho1 (4).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\ComSuon\com-suon1 (1).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\ComSuon\com-suon1 (2).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\ComSuon\com-suon1 (3).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\ComSuon\com-suon1 (4).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\ComSuon\com-suon1 (5).jpg',
]

def preprocess(img_pil, target_size=(640, 640), mode='stretch'):
    img = ImageOps.exif_transpose(img_pil).convert('RGB')
    w, h = img.size
    if mode == 'stretch':
        resized = img.resize(target_size, Image.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr, 1.0, 0, 0
    else: # letterbox
        scale = min(target_size[0] / w, target_size[1] / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', target_size, (114, 114, 114))
        pad_x = (target_size[0] - nw) // 2
        pad_y = (target_size[1] - nh) // 2
        canvas.paste(resized, (pad_x, pad_y))
        arr = np.array(canvas).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr, scale, pad_x, pad_y

for p in test_images:
    fname = os.path.basename(p)
    if not os.path.exists(p):
        print(f"Skipping missing: {fname}")
        continue
    print(f"\n==================================================")
    print(f"IMAGE: {fname}")
    orig_img = Image.open(p)
    print(f"Original size: {orig_img.size}")

    for mode in ['stretch', 'letterbox']:
        inp, scale, px, py = preprocess(orig_img, mode=mode)
        inp32 = inp.astype(np.float32)
        inp16 = inp.astype(np.float16)

        # Run yolov26 (FP32)
        out26 = s26.run(None, {'images': inp32})[0] # shape (1, 300, 6): [x1, y1, x2, y2, conf, cls]
        out10 = s10.run(None, {'images': inp16})[0] # shape (1, 300, 6)

        print(f"\n--- MODE: {mode.upper()} ---")
        
        # Parse yolov26 detections (conf >= 0.15)
        boxes26 = [b for b in out26[0] if b[4] >= 0.15]
        print(f"  [yolov26m (68 classes)] found {len(boxes26)} boxes (conf >= 0.15):")
        for b in sorted(boxes26, key=lambda x: x[4], reverse=True)[:5]:
            cls_id = int(b[5])
            cname = class_names[cls_id]['name'] if (cls_id < len(class_names) and isinstance(class_names[cls_id], dict)) else (class_names[cls_id] if cls_id < len(class_names) else f"ID_{cls_id}")
            print(f"     * {cname} (ID {cls_id}): conf = {b[4]:.3f}, box = {[round(float(c), 1) for c in b[:4]]}")
        if not boxes26:
            max_conf = max(b[4] for b in out26[0]) if len(out26[0]) else 0
            best_cls = int(out26[0][np.argmax([b[4] for b in out26[0]])][5]) if len(out26[0]) else -1
            best_name = class_names[best_cls]['name'] if (0 <= best_cls < len(class_names) and isinstance(class_names[best_cls], dict)) else f"ID_{best_cls}"
            print(f"     (NO DETECTIONS >= 0.15, max_conf was {max_conf:.3f} for {best_name})")

        # Parse yolov10 detections (conf >= 0.15)
        boxes10 = [b for b in out10[0] if b[4] >= 0.15]
        print(f"  [yolov10m nvhnam (58 classes)] found {len(boxes10)} boxes (conf >= 0.15):")
        for b in sorted(boxes10, key=lambda x: x[4], reverse=True)[:5]:
            cls_id = int(b[5])
            cname = class_names[cls_id]['name'] if (cls_id < len(class_names) and isinstance(class_names[cls_id], dict)) else (class_names[cls_id] if cls_id < len(class_names) else f"ID_{cls_id}")
            print(f"     * {cname} (ID {cls_id}): conf = {b[4]:.3f}, box = {[round(float(c), 1) for c in b[:4]]}")
        if not boxes10:
            max_conf = max(b[4] for b in out10[0]) if len(out10[0]) else 0
            best_cls = int(out10[0][np.argmax([b[4] for b in out10[0]])][5]) if len(out10[0]) else -1
            best_name = class_names[best_cls]['name'] if (0 <= best_cls < len(class_names) and isinstance(class_names[best_cls], dict)) else f"ID_{best_cls}"
            print(f"     (NO DETECTIONS >= 0.15, max_conf was {max_conf:.3f} for {best_name})")
