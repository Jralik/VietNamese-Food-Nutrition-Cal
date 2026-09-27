import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import os

# 1. Check if NonMaxSuppression operator is in ONNX graph
m = onnx.load('model/yolov26/best.onnx')
nms_nodes = [node for node in m.graph.node if 'nms' in node.op_type.lower() or 'nonmax' in node.op_type.lower()]
print(f"Total ONNX nodes: {len(m.graph.node)}")
print(f"NMS nodes in ONNX: {[n.name + ' (' + n.op_type + ')' for n in nms_nodes]}")

# 2. Check all 300 boxes for BanhMi_Trung images
s26 = ort.InferenceSession('model/yolov26/best.onnx', providers=['CPUExecutionProvider'])

def preprocess(img_pil, mode='stretch'):
    img = ImageOps.exif_transpose(img_pil).convert('RGB')
    w, h = img.size
    if mode == 'stretch':
        resized = img.resize((640, 640), Image.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)
    else:
        scale = min(640 / w, 640 / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = Image.new('RGB', (640, 640), (114, 114, 114))
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((640 - nw) // 2, (640 - nh) // 2))
        arr = np.array(canvas).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0)

from class_names import class_names

for i in range(1, 5):
    img_path = rf'C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung ({i}).jpg'
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path)
    print(f"\n=======================================================")
    print(f"ANALYSIS: BanhMi_Trung ({i}).jpg (Size: {img.size})")

    for mode in ['stretch', 'letterbox']:
        inp = preprocess(img, mode=mode)
        out = s26.run(None, {'images': inp})[0][0] # shape (300, 6)
        
        # All unique classes detected in 300 boxes
        classes_present = {}
        for b in out:
            cid = int(b[5])
            conf = float(b[4])
            if conf > 0.001:
                if cid not in classes_present or conf > classes_present[cid]:
                    classes_present[cid] = conf
        
        print(f"\n--- Mode: {mode.upper()} ---")
        print("Top detected classes across all 300 candidate slots:")
        for cid, max_c in sorted(classes_present.items(), key=lambda x: x[1], reverse=True)[:8]:
            cname = class_names[cid]['name'] if isinstance(class_names[cid], dict) else class_names[cid]
            print(f"   * Class {cid} ({cname}): max_conf = {max_c:.4f}")
        
        # Specifically check class 56 (Trung)
        trung_boxes = [b for b in out if int(b[5]) == 56]
        print(f"   -> Slots dedicated to 'Trung' (ID 56): {len(trung_boxes)} slots. Max conf: {max([b[4] for b in trung_boxes]) if trung_boxes else 0.0:.4f}")
