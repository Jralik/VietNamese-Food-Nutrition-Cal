# A/B preprocess benchmark — letterbox (app hiện tại) vs stretch 640x640 (reference)

Model: `./model/yolov26/best.onnx` (end2end ONNX, output (1,300,6)). Mỗi mode chạy 1 lần predict ở conf=0.05, các ngưỡng [0.5, 0.4, 0.3, 0.2, 0.1] áp dụng local (postprocess end2end chỉ là lọc conf).

## SupCua (1).jpg — 1800x4000

| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |
|---|---|---|---|---|---|---|---|---|
| letterbox | 0 | (none) | 0.155 (Sup cua (Crab soup)) | 0 | 0 | 0 | 1 | 4928 |
| stretch | 0 | (none) | 0.169 (Sup cua (Crab soup)) | 0 | 0 | 0 | 1 | 389 |

Max conf theo class then chốt (mọi slot, conf>0.05):

| Mode | Sup cua (Crab soup) |
|---|---|
| letterbox | 0.1546 |
| stretch | 0.1686 |

## SupCua (2).jpg — 4000x1800

| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |
|---|---|---|---|---|---|---|---|---|
| letterbox | 0 | (none) | n/a | 0 | 0 | 0 | 0 | 438 |
| stretch | 0 | (none) | n/a | 0 | 0 | 0 | 0 | 381 |

Max conf theo class then chốt (mọi slot, conf>0.05):

| Mode | Sup cua (Crab soup) |
|---|---|
| letterbox | 0.0000 |
| stretch | 0.0000 |

## SupCua (3).jpg — 4000x1800

| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |
|---|---|---|---|---|---|---|---|---|
| letterbox | 0 | (none) | n/a | 0 | 0 | 0 | 0 | 412 |
| stretch | 0 | (none) | n/a | 0 | 0 | 0 | 0 | 383 |

Max conf theo class then chốt (mọi slot, conf>0.05):

| Mode | Sup cua (Crab soup) |
|---|---|
| letterbox | 0.0000 |
| stretch | 0.0000 |

## BanhMi_Trung (1).jpg — 4000x1800

| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |
|---|---|---|---|---|---|---|---|---|
| letterbox | 0 | (none) | 0.107 (Nam (Mushroom)) | 0 | 0 | 0 | 1 | 428 |
| stretch | 1 | Banh mi (Vietnamese baguette sandwich): 0.771 | 0.771 (Banh mi (Vietnamese baguette sandwich)) | 1 | 1 | 1 | 2 | 352 |

Max conf theo class then chốt (mọi slot, conf>0.05):

| Mode | Banh mi (Vietnamese baguette sandwich) | Trung (Egg) |
|---|---|---|
| letterbox | 0.0637 | 0.0000 |
| stretch | 0.7706 | 0.0000 |

## BanhMi_Trung (2).jpg — 4000x1800

| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |
|---|---|---|---|---|---|---|---|---|
| letterbox | 0 | (none) | 0.387 (Banh mi (Vietnamese baguette sandwich)) | 0 | 1 | 1 | 1 | 412 |
| stretch | 1 | Banh mi (Vietnamese baguette sandwich): 0.928 | 0.928 (Banh mi (Vietnamese baguette sandwich)) | 1 | 1 | 1 | 1 | 359 |

Max conf theo class then chốt (mọi slot, conf>0.05):

| Mode | Banh mi (Vietnamese baguette sandwich) | Trung (Egg) |
|---|---|---|
| letterbox | 0.3869 | 0.0000 |
| stretch | 0.9283 | 0.0000 |

## BanhMi_Trung (3).jpg — 4000x1800

| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |
|---|---|---|---|---|---|---|---|---|
| letterbox | 0 | (none) | n/a | 0 | 0 | 0 | 0 | 421 |
| stretch | 1 | Banh mi (Vietnamese baguette sandwich): 0.901 | 0.901 (Banh mi (Vietnamese baguette sandwich)) | 1 | 1 | 1 | 2 | 354 |

Max conf theo class then chốt (mọi slot, conf>0.05):

| Mode | Banh mi (Vietnamese baguette sandwich) | Trung (Egg) |
|---|---|---|
| letterbox | 0.0000 | 0.0000 |
| stretch | 0.9009 | 0.0000 |

## BanhMi_Trung (4).jpg — 4000x1800

| Mode | #box @0.50 | Detections @0.50 | Max conf (300 slots) | #box @0.40 | @0.30 | @0.20 | @0.10 | Thời gian (ms) |
|---|---|---|---|---|---|---|---|---|
| letterbox | 0 | (none) | n/a | 0 | 0 | 0 | 0 | 398 |
| stretch | 2 | Banh mi (Vietnamese baguette sandwich): 0.909, Banh mi (Vietnamese baguette sandwich): 0.553 | 0.909 (Banh mi (Vietnamese baguette sandwich)) | 2 | 2 | 2 | 2 | 340 |

Max conf theo class then chốt (mọi slot, conf>0.05):

| Mode | Banh mi (Vietnamese baguette sandwich) | Trung (Egg) |
|---|---|---|
| letterbox | 0.0000 | 0.0000 |
| stretch | 0.9091 | 0.0000 |

# Diagnosis 2 — model x orientation x preprocessing (raw = không exif_transpose)

| Ảnh | Model | Preprocess | #box @0.50 | Top-3 | maxSupCua | maxTrung | maxBanhMi | ms |
|---|---|---|---|---|---|---|---|---|
| **SupCua (1).jpg** raw=4000x1800 EXIF-orient=8 | | | | | | | | |
| SupCua | v26m(onnx) | raw+letterbox | 0 | - | 0.000 | 0.000 | 0.000 | 5085 |
| SupCua | v26m(onnx) | raw+stretch | 0 | - | 0.000 | 0.000 | 0.000 | 360 |
| SupCua | v10b(ref) | raw+stretch | 1 | Sup cua:0.98 | 0.977 | 0.000 | 0.000 | 6016 |
| SupCua | v10b(ref) | raw+letterbox | 1 | Sup cua:0.94 | 0.936 | 0.000 | 0.000 | 119 |

| **SupCua (2).jpg** raw=4000x1800 EXIF-orient=1 | | | | | | | | |
| SupCua | v26m(onnx) | raw+letterbox | 0 | - | 0.000 | 0.000 | 0.000 | 434 |
| SupCua | v26m(onnx) | raw+stretch | 0 | - | 0.000 | 0.000 | 0.000 | 363 |
| SupCua | v10b(ref) | raw+stretch | 1 | Sup cua:0.74 | 0.737 | 0.000 | 0.000 | 67 |
| SupCua | v10b(ref) | raw+letterbox | 1 | Sup cua:0.63 | 0.626 | 0.000 | 0.000 | 86 |

| **SupCua (3).jpg** raw=4000x1800 EXIF-orient=1 | | | | | | | | |
| SupCua | v26m(onnx) | raw+letterbox | 0 | - | 0.000 | 0.000 | 0.000 | 418 |
| SupCua | v26m(onnx) | raw+stretch | 0 | - | 0.000 | 0.000 | 0.000 | 354 |
| SupCua | v10b(ref) | raw+stretch | 1 | Sup cua:0.94 | 0.942 | 0.000 | 0.000 | 111 |
| SupCua | v10b(ref) | raw+letterbox | 1 | Sup cua:0.78 | 0.776 | 0.000 | 0.000 | 106 |

| **BanhMi_Trung (1).jpg** raw=4000x1800 EXIF-orient=3 | | | | | | | | |
| BanhMi_Trung | v26m(onnx) | raw+letterbox | 0 | Banh mi:0.23 | 0.000 | 0.000 | 0.227 | 450 |
| BanhMi_Trung | v26m(onnx) | raw+stretch | 1 | Banh mi:0.91, Banh mi:0.13 | 0.000 | 0.000 | 0.905 | 373 |
| BanhMi_Trung | v10b(ref) | raw+stretch | 2 | Banh mi:0.89, Trung:0.68, Banh mi:0.38 | 0.000 | 0.676 | 0.888 | 216 |
| BanhMi_Trung | v10b(ref) | raw+letterbox | 0 | Banh mi:0.41, Khoai tay chien:0.06, Banh mi:0.06 | 0.000 | 0.000 | 0.407 | 81 |

| **BanhMi_Trung (2).jpg** raw=4000x1800 EXIF-orient=3 | | | | | | | | |
| BanhMi_Trung | v26m(onnx) | raw+letterbox | 1 | Banh mi:0.77, Nam:0.25 | 0.000 | 0.000 | 0.771 | 476 |
| BanhMi_Trung | v26m(onnx) | raw+stretch | 2 | Banh mi:0.93, Banh mi:0.55 | 0.000 | 0.000 | 0.934 | 387 |
| BanhMi_Trung | v10b(ref) | raw+stretch | 1 | Banh mi:0.91, Trung:0.43, Banh mi:0.19 | 0.000 | 0.427 | 0.908 | 72 |
| BanhMi_Trung | v10b(ref) | raw+letterbox | 0 | Banh mi:0.36, Tom:0.15, Khoai tay chien:0.06 | 0.000 | 0.000 | 0.355 | 80 |

| **BanhMi_Trung (3).jpg** raw=4000x1800 EXIF-orient=3 | | | | | | | | |
| BanhMi_Trung | v26m(onnx) | raw+letterbox | 1 | Banh mi:0.80, Com tam:0.14 | 0.000 | 0.000 | 0.795 | 451 |
| BanhMi_Trung | v26m(onnx) | raw+stretch | 1 | Banh mi:0.85, Banh mi:0.08 | 0.000 | 0.000 | 0.853 | 352 |
| BanhMi_Trung | v10b(ref) | raw+stretch | 3 | Banh mi:0.68, Banh mi:0.58, Trung:0.50 | 0.000 | 0.502 | 0.679 | 65 |
| BanhMi_Trung | v10b(ref) | raw+letterbox | 0 | Xoi:0.32, Khoai tay chien:0.21, Xoi:0.14 | 0.000 | 0.000 | 0.056 | 88 |

| **BanhMi_Trung (4).jpg** raw=4000x1800 EXIF-orient=3 | | | | | | | | |
| BanhMi_Trung | v26m(onnx) | raw+letterbox | 0 | - | 0.000 | 0.000 | 0.000 | 433 |
| BanhMi_Trung | v26m(onnx) | raw+stretch | 1 | Banh mi:0.91, Banh mi:0.38 | 0.000 | 0.000 | 0.908 | 367 |
| BanhMi_Trung | v10b(ref) | raw+stretch | 2 | Banh mi:0.85, Banh mi:0.82, Banh mi:0.20 | 0.000 | 0.000 | 0.855 | 102 |
| BanhMi_Trung | v10b(ref) | raw+letterbox | 0 | Nam:0.23, Nam:0.09 | 0.000 | 0.000 | 0.000 | 87 |

# Diagnosis 3 — RGB vs BGR channel order (v26m ONNX)

RGB = predict(numpy RGB array, loader pass-through). BGR = predict(PIL, loader flips to BGR).

| Ảnh | Preprocess | Channel | #box @0.50 | Top-4 | maxSupCua | maxTrung | maxBanhMi |
|---|---|---|---|---|---|---|---|
| SupCua | letterbox | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| SupCua | letterbox | BGR | 0 | - | 0.000 | 0.000 | 0.000 |
| SupCua | stretch | RGB | 1 | Canh:0.62 | 0.000 | 0.000 | 0.000 |
| SupCua | stretch | BGR | 0 | - | 0.000 | 0.000 | 0.000 |

| SupCua | letterbox | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| SupCua | letterbox | BGR | 0 | - | 0.000 | 0.000 | 0.000 |
| SupCua | stretch | RGB | 0 | Canh:0.20 | 0.000 | 0.000 | 0.000 |
| SupCua | stretch | BGR | 0 | - | 0.000 | 0.000 | 0.000 |

| SupCua | letterbox | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| SupCua | letterbox | BGR | 0 | - | 0.000 | 0.000 | 0.000 |
| SupCua | stretch | RGB | 1 | Canh:0.72 | 0.000 | 0.000 | 0.000 |
| SupCua | stretch | BGR | 0 | - | 0.000 | 0.000 | 0.000 |

| BanhMi_Trung | letterbox | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | letterbox | BGR | 0 | Banh mi:0.23 | 0.000 | 0.000 | 0.227 |
| BanhMi_Trung | stretch | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | stretch | BGR | 1 | Banh mi:0.91, Banh mi:0.13 | 0.000 | 0.000 | 0.905 |

| BanhMi_Trung | letterbox | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | letterbox | BGR | 1 | Banh mi:0.77, Nam:0.25 | 0.000 | 0.000 | 0.771 |
| BanhMi_Trung | stretch | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | stretch | BGR | 2 | Banh mi:0.93, Banh mi:0.55 | 0.000 | 0.000 | 0.934 |

| BanhMi_Trung | letterbox | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | letterbox | BGR | 1 | Banh mi:0.80, Com tam:0.14 | 0.000 | 0.000 | 0.795 |
| BanhMi_Trung | stretch | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | stretch | BGR | 1 | Banh mi:0.85, Banh mi:0.08 | 0.000 | 0.000 | 0.853 |

| BanhMi_Trung | letterbox | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | letterbox | BGR | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | stretch | RGB | 0 | - | 0.000 | 0.000 | 0.000 |
| BanhMi_Trung | stretch | BGR | 1 | Banh mi:0.91, Banh mi:0.38 | 0.000 | 0.000 | 0.908 |


# T4 — PIL pre-resize qua đúng đường predict của app (model nhận RGB)

Orientation = exif_transpose (như app). Variant D = app hiện tại.

| Ảnh | Variant | #box @0.50 | Top-4 | maxSupCua | maxTrung | maxBanhMi | ms |
|---|---|---|---|---|---|---|---|
| SupCua | A stretch-BILINEAR | 0 | Sup cua:0.17 | 0.169 | 0.000 | 0.000 | 7013 |
| SupCua | B letterbox-BILINEAR | 1 | Sup cua:0.92 | 0.917 | 0.000 | 0.000 | 609 |
| SupCua | C letterbox-LANCZOS | 1 | Sup cua:0.88 | 0.879 | 0.000 | 0.000 | 640 |
| SupCua | D no-preresize(cv2) | 0 | Sup cua:0.15 | 0.155 | 0.000 | 0.000 | 716 |

| SupCua | A stretch-BILINEAR | 0 | - | 0.000 | 0.000 | 0.000 | 591 |
| SupCua | B letterbox-BILINEAR | 0 | Sup cua:0.09 | 0.094 | 0.000 | 0.000 | 611 |
| SupCua | C letterbox-LANCZOS | 0 | - | 0.000 | 0.000 | 0.000 | 635 |
| SupCua | D no-preresize(cv2) | 0 | - | 0.000 | 0.000 | 0.000 | 708 |

| SupCua | A stretch-BILINEAR | 0 | - | 0.000 | 0.000 | 0.000 | 654 |
| SupCua | B letterbox-BILINEAR | 0 | Sup cua:0.06 | 0.064 | 0.000 | 0.000 | 590 |
| SupCua | C letterbox-LANCZOS | 0 | - | 0.000 | 0.000 | 0.000 | 527 |
| SupCua | D no-preresize(cv2) | 0 | - | 0.000 | 0.000 | 0.000 | 569 |

| BanhMi_Trung | A stretch-BILINEAR | 1 | Banh mi:0.77, Banh mi:0.20 | 0.000 | 0.000 | 0.771 | 600 |
| BanhMi_Trung | B letterbox-BILINEAR | 0 | Banh mi:0.49, Banh mi:0.21, Trung:0.09 | 0.000 | 0.090 | 0.492 | 447 |
| BanhMi_Trung | C letterbox-LANCZOS | 0 | Banh mi:0.46, Banh mi:0.22, Nam:0.08 | 0.000 | 0.000 | 0.463 | 434 |
| BanhMi_Trung | D no-preresize(cv2) | 0 | Nam:0.11, Banh mi:0.06, Banh mi:0.05 | 0.000 | 0.000 | 0.064 | 536 |

| BanhMi_Trung | A stretch-BILINEAR | 1 | Banh mi:0.93 | 0.000 | 0.000 | 0.928 | 393 |
| BanhMi_Trung | B letterbox-BILINEAR | 2 | Banh mi:0.94, Banh mi:0.57 | 0.000 | 0.000 | 0.936 | 403 |
| BanhMi_Trung | C letterbox-LANCZOS | 1 | Banh mi:0.90, Banh mi:0.39, Nam:0.06 | 0.000 | 0.000 | 0.899 | 412 |
| BanhMi_Trung | D no-preresize(cv2) | 0 | Banh mi:0.39, Nam:0.09, Banh mi:0.07 | 0.000 | 0.000 | 0.387 | 466 |

| BanhMi_Trung | A stretch-BILINEAR | 1 | Banh mi:0.90, Banh mi:0.12, Ca:0.05 | 0.000 | 0.000 | 0.901 | 405 |
| BanhMi_Trung | B letterbox-BILINEAR | 1 | Banh mi:0.69, Chao long:0.13 | 0.000 | 0.000 | 0.689 | 411 |
| BanhMi_Trung | C letterbox-LANCZOS | 1 | Banh mi:0.71 | 0.000 | 0.000 | 0.705 | 414 |
| BanhMi_Trung | D no-preresize(cv2) | 0 | - | 0.000 | 0.000 | 0.000 | 486 |

| BanhMi_Trung | A stretch-BILINEAR | 2 | Banh mi:0.91, Banh mi:0.55 | 0.000 | 0.000 | 0.909 | 408 |
| BanhMi_Trung | B letterbox-BILINEAR | 2 | Banh mi:0.91, Banh mi:0.59 | 0.000 | 0.000 | 0.905 | 411 |
| BanhMi_Trung | C letterbox-LANCZOS | 1 | Banh mi:0.65, Banh mi:0.18 | 0.000 | 0.000 | 0.651 | 411 |
| BanhMi_Trung | D no-preresize(cv2) | 0 | - | 0.000 | 0.000 | 0.000 | 460 |


# Val-set evaluation (60 anh, portion-estimation_val.txt)

| Config | Recall@0.50 | Recall@0.30 | Anh 0 box @0.50 | So anh |
|---|---|---|---|---|
| D app-current | 0.037 | 0.037 | 21/54 | 54 |
| A stretch-BIL | 0.056 | 0.056 | 18/54 | 54 |
| B letterbox-BIL | 0.093 | 0.093 | 16/54 | 54 |
| AB dual-union | 0.093 | 0.093 | 12/54 | 54 |
| REF v10b-stretch | 0.111 | 0.111 | 6/54 | 54 |

## Chi tiet theo nhom mon (Recall@0.50 / so anh)

| Nhom (folder) | D app-current | A stretch-BIL | B letterbox-BIL | AB dual-union | REF v10b-stretch |
|---|---|---|---|---|---|
| BanhCuon | 0.00 (6 anh) | 0.00 (6 anh) | 0.00 (6 anh) | 0.00 (6 anh) | 0.00 (6 anh) |
| BanhMi | 0.00 (16 anh) | 0.00 (16 anh) | 0.00 (16 anh) | 0.00 (16 anh) | 0.06 (16 anh) |
| BunBoHue | 0.00 (5 anh) | 0.00 (5 anh) | 0.00 (5 anh) | 0.00 (5 anh) | 0.00 (5 anh) |
| BunRieu-CanhBun | 0.00 (7 anh) | 0.00 (7 anh) | 0.00 (7 anh) | 0.00 (7 anh) | 0.00 (7 anh) |
| ComSuon | 0.00 (5 anh) | 0.00 (5 anh) | 0.00 (5 anh) | 0.00 (5 anh) | 0.00 (5 anh) |
| Pho | 0.40 (5 anh) | 0.60 (5 anh) | 1.00 (5 anh) | 1.00 (5 anh) | 1.00 (5 anh) |
| SupCua | 0.00 (10 anh) | 0.00 (10 anh) | 0.00 (10 anh) | 0.00 (10 anh) | 0.00 (10 anh) |

## Cac anh con thieu class @0.50 (config AB dual-union)
- com-suon1 (5).jpg: thieu Com
- com-suon1 (1).jpg: thieu Com
- com-suon1 (2).jpg: thieu Com
- com-suon1 (3).jpg: thieu Com
- com-suon1 (4).jpg: thieu Com
- BunBoHue1 (1).jpg: thieu Bun, Rau
- BunBoHue1 (2).jpg: thieu Bun, Rau
- BunBoHue1 (3).jpg: thieu Bun, Rau
- BunBoHue1 (4).jpg: thieu Bun, Rau
- BunBoHue1 (5).jpg: thieu Bun, Rau
- BanhMi_Trung (4).jpg: thieu Trung
- BanhMi_Trung (1).jpg: thieu Trung
- BanhMi_Trung (2).jpg: thieu Trung
- BanhMi_Trung (3).jpg: thieu Trung
- bun-rieu1 (4).jpg: thieu Bun
- bun-rieu1 (1).jpg: thieu Bun
- bun-rieu1 (2).jpg: thieu Bun
- bun-rieu1 (3).jpg: thieu Bun
- bun-rieu (3).jpg: thieu Bun, Rau
- bun-rieu (1).jpg: thieu Bun, Rau
- bun-rieu (2).jpg: thieu Bun, Rau

# Val-set evaluation (54 anh, portion-estimation_val.txt)

| Config | Recall@0.50 | Recall@0.30 | Anh 0 box @0.50 | So anh |
|---|---|---|---|---|
| D app-current | 0.361 | 0.361 | 21/54 | 54 |
| A stretch-BIL | 0.454 | 0.454 | 18/54 | 54 |
| B letterbox-BIL | 0.481 | 0.481 | 16/54 | 54 |
| AB dual-union | 0.546 | 0.546 | 12/54 | 54 |
| REF v10b-stretch | 0.611 | 0.611 | 6/54 | 54 |

## Chi tiet theo nhom mon (Recall@0.50 / so anh)

| Nhom (folder) | D app-current | A stretch-BIL | B letterbox-BIL | AB dual-union | REF v10b-stretch |
|---|---|---|---|---|---|
| BanhCuon | 0.00 (6 anh) | 0.00 (6 anh) | 0.00 (6 anh) | 0.00 (6 anh) | 0.00 (6 anh) |
| BanhMi | 0.44 (16 anh) | 0.69 (16 anh) | 0.53 (16 anh) | 0.75 (16 anh) | 0.69 (16 anh) |
| BunBoHue | 0.50 (5 anh) | 0.50 (5 anh) | 0.50 (5 anh) | 0.50 (5 anh) | 0.50 (5 anh) |
| BunRieu-CanhBun | 0.43 (7 anh) | 0.43 (7 anh) | 0.43 (7 anh) | 0.43 (7 anh) | 0.50 (7 anh) |
| ComSuon | 1.00 (5 anh) | 1.00 (5 anh) | 1.00 (5 anh) | 1.00 (5 anh) | 1.00 (5 anh) |
| Pho | 0.40 (5 anh) | 0.60 (5 anh) | 1.00 (5 anh) | 1.00 (5 anh) | 1.00 (5 anh) |
| SupCua | 0.00 (10 anh) | 0.00 (10 anh) | 0.20 (10 anh) | 0.20 (10 anh) | 0.60 (10 anh) |

## Cac anh con thieu class @0.50 (config AB dual-union)
- BanhCuon1 (6).jpg: thieu Banh cuon
- BanhCuon1 (1).jpg: thieu Banh cuon
- BanhCuon1 (2).jpg: thieu Banh cuon
- BanhCuon1 (3).jpg: thieu Banh cuon
- BanhCuon1 (4).jpg: thieu Banh cuon
- BanhCuon1 (5).jpg: thieu Banh cuon
- BunBoHue1 (1).jpg: thieu Rau
- BunBoHue1 (2).jpg: thieu Rau
- BunBoHue1 (3).jpg: thieu Rau
- BunBoHue1 (4).jpg: thieu Rau
- BunBoHue1 (5).jpg: thieu Rau
- banh-mi250g (1).jpg: thieu Banh mi
- banh-mi250g (2).jpg: thieu Banh mi
- BanhMi_Trung (4).jpg: thieu Trung
- BanhMi_Trung (1).jpg: thieu Trung
- BanhMi_Trung (2).jpg: thieu Trung
- BanhMi_Trung (3).jpg: thieu Trung
- bun-rieu1 (4).jpg: thieu Bun rieu
- bun-rieu1 (3).jpg: thieu Bun rieu
- bun-rieu (3).jpg: thieu Bun rieu, Rau
- bun-rieu (1).jpg: thieu Rau
- bun-rieu (2).jpg: thieu Rau
- SupCua (6).jpg: thieu Sup cua
- SupCua (2).jpg: thieu Sup cua
- SupCua (3).jpg: thieu Sup cua
- SupCua (5).jpg: thieu Sup cua
- SupCua1 (4).jpg: thieu Sup cua
- SupCua1 (1).jpg: thieu Sup cua
- SupCua1 (2).jpg: thieu Sup cua
- SupCua1 (3).jpg: thieu Sup cua

# POST-FIX verification — app path (predict_with_dual_preresize, conf=0.50)

| Ảnh | #box @0.50 | Detections (class: conf) | orig_shape | ms |
|---|---|---|---|---|
| SupCua (1).jpg | 1 | Sup cua: 0.917 @[299, 1205, 1505, 2505] | (4000, 1800) | 4386 |
| SupCua (2).jpg | 0 | (none) | (1800, 4000) | 787 |
| SupCua (3).jpg | 0 | (none) | (1800, 4000) | 776 |
| BanhMi_Trung (1).jpg | 1 | Banh mi: 0.771 @[502, 78, 1344, 1035] | (1800, 4000) | 803 |
| BanhMi_Trung (2).jpg | 2 | Banh mi: 0.936 @[2826, 711, 3610, 1741]; Banh mi: 0.568 @[2297, 602, 2882, 1189] | (1800, 4000) | 860 |
| BanhMi_Trung (3).jpg | 1 | Banh mi: 0.901 @[3199, 564, 3895, 1590] | (1800, 4000) | 802 |
| BanhMi_Trung (4).jpg | 2 | Banh mi: 0.909 @[870, 109, 1725, 1377]; Banh mi: 0.587 @[1649, 690, 2272, 1321] | (1800, 4000) | 815 |
