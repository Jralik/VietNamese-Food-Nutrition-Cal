import os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from class_names import class_names

s26 = ort.InferenceSession('model/yolov26/best.onnx', providers=['CPUExecutionProvider'])
s10 = ort.InferenceSession('model/yolov10/YOLOv10m_new_total_VN_5_SGD.onnx', providers=['CPUExecutionProvider'])

test_images = [
    r'C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (2).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (3).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (1).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (2).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (3).jpg',
    r'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (4).jpg',
]

def preprocess(img_pil, target_size=(640, 640), mode='stretch'):
    img = ImageOps.exif_transpose(img_pil).convert('RGB')
    w, h = img.size
    if mode == 'stretch':
        resized = img.resize(target_size, Image.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr
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
        return arr

def get_cname(cls_id):
    if 0 <= cls_id < len(class_names):
        item = class_names[cls_id]
        if isinstance(item, dict):
            return item.get('name', str(cls_id))
        return str(item)
    return f"ID_{cls_id}"

for p in test_images:
    fname = os.path.basename(p)
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue
    print(f"\n=======================================================")
    print(f"IMAGE: {fname}")
    orig_img = Image.open(p)
    print(f"Size: {orig_img.size}")

    for mode in ['stretch', 'letterbox']:
        inp = preprocess(orig_img, mode=mode)
        inp32 = inp.astype(np.float32)
        inp16 = inp.astype(np.float16)

        out26 = s26.run(None, {'images': inp32})[0]
        out10 = s10.run(None, {'images': inp16})[0]

        print(f"\n--- MODE: {mode.upper()} ---")

        # yolov26
        boxes26 = [b for b in out26[0] if b[4] >= 0.10]
        print(f"  [yolov26m] ({len(boxes26)} boxes with conf >= 0.10):")
        for b in sorted(boxes26, key=lambda x: x[4], reverse=True)[:6]:
            cid = int(b[5])
            print(f"     * {get_cname(cid)} (ID {cid}): conf = {b[4]:.3f}, box = {[round(float(c), 1) for c in b[:4]]}")
        if not boxes26:
            max_conf = max(b[4] for b in out26[0]) if len(out26[0]) else 0
            best_cls = int(out26[0][np.argmax([b[4] for b in out26[0]])][5]) if len(out26[0]) else -1
            print(f"     (NO DETECTIONS >= 0.10! Max conf was {max_conf:.3f} for {get_cname(best_cls)})")

        # yolov10
        boxes10 = [b for b in out10[0] if b[4] >= 0.10]
        print(f"  [yolov10m nvhnam] ({len(boxes10)} boxes with conf >= 0.10):")
        for b in sorted(boxes10, key=lambda x: x[4], reverse=True)[:6]:
            cid = int(b[5])
            print(f"     * {get_cname(cid)} (ID {cid}): conf = {b[4]:.3f}, box = {[round(float(c), 1) for c in b[:4]]}")
        if not boxes10:
            max_conf = max(b[4] for b in out10[0]) if len(out10[0]) else 0
            best_cls = int(out10[0][np.argmax([b[4] for b in out10[0]])][5]) if len(out10[0]) else -1
            print(f"     (NO DETECTIONS >= 0.10! Max conf was {max_conf:.3f} for {get_cname(best_cls)})")
