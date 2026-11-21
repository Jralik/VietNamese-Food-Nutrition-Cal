# Vendored patches — ultralytics

Repo vendored toàn bộ thư mục `ultralytics/` (194 files) thay vì cài từ PyPI. Lý do: cần hỗ trợ kiến trúc **YOLO11 / YOLO26** cho việc load model YOLOv26m mà phiên bản PyPI tại thời điểm phát triển chưa có.

## Patch duy nhất

| File | Nội dung |
|---|---|
| [`vendored_ultralytics_block_yolo26_modules.patch`](vendored_ultralytics_block_yolo26_modules.patch) | +238 dòng trong `ultralytics/nn/modules/block.py` — bổ sung module hiện đại YOLO11/YOLO26 (`Attention`, …) cần để build/load YOLOv26m. Commit: `8427cc1` ("Vendored ultralytics block.py updates"), thay đổi từ `335b7cc` lên `8427cc1`. |

## Áp dụng lại trên bản ultralytics sạch

```bash
git apply patches/vendored_ultralytics_block_yolo26_modules.patch
```

Hoặc clone repo này và dùng trực tiếp thư mục `ultralytics/` đã vendored.

## Quy tắc

- Mọi thay đổi tiếp theo đối với `ultralytics/` phải đi kèm một `.patch` file trong thư mục này + dòng giải thích trong README, tránh "sửa ngầm" framework.
- Patch KHÔNG thay đổi hành vi các kiến trúc cũ (YOLOv10/v8) — chỉ thêm class mới; đã kiểm chứng bằng chuỗi test Phase 1-7 và eval frozen (không regression).
