# Pipeline overview (frozen configuration)

```
Image upload (EXIF upright)
   |
   |  v10b YOLO + dual-preresize (PIL stretch + letterbox, class-aware NMS)
   |  confidence threshold = 0.30 (UI slider)
   v
Detections (bbox, class, confidence)  ── recall 0.811 (60/74) @ 0.30
   |
   |  volume detections: confidence >= 0.40 (MIN_DETECTION_CONFIDENCE)
   v
Volume pipeline (separate worker, GPU)
   ├── SAM2 segmentation (box-prompted)
   ├── Depth Anything V2 (metric, indoor)
   ├── Scale recovery: ArUco marker (0.211-0.321 mm/px) + depth anchoring
   └── Volume -> mass (density_db) -> nutrition (per-100g scaling)
         |
         v
   Nutrition cards + volume note (UI)
```

Configuration references:
- Detection: model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt, device=cpu (volume worker owns the GPU)
- Volume: backend SAM2, source image 2000px thumbnail, worker daemon
- All numbers in evaluation_final/ use this frozen configuration
