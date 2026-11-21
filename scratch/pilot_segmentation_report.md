# Báo cáo Thực nghiệm Phân đoạn Thực phẩm (PILOT: 6 ảnh / 12 GT instances)

## 1. Kịch bản A — Oracle Box Segmentation Benchmark (Đánh giá thuần năng lực phân đoạn)

### 1.1. So sánh tổng thể: Raw Mask (Thuần mô hình) vs Post-processed Mask (Có hậu xử lý)

| Mô hình | Chế độ Mask | Micro mIoU | Macro mIoU | Micro Dice | Boundary IoU | Seg Success (IoU≥0.5) | Under-seg (<2/3 box) | Oversized (>95% box) | Spillover ngoài box |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | **Raw Mask** (Chính) | 0.9133 | 0.9022 | 0.9506 | 0.4719 | 100.0% | 33.3% | 0.0% | 0.12% |
| SAM 2.1 (Hiera-Small) | Post-processed (Erode 3×3) | 0.8885 | 0.8736 | 0.9280 | 0.4443 | 91.7% | 33.3% | 0.0% | 0.68% |
| MobileSAM (TinyViT) | **Raw Mask** (Chính) | 0.9276 | 0.9196 | 0.9610 | 0.4481 | 100.0% | 33.3% | 16.7% | 1.48% |
| MobileSAM (TinyViT) | Post-processed (Erode 3×3) | 0.9482 | 0.9445 | 0.9732 | 0.4573 | 100.0% | 33.3% | 0.0% | 0.07% |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | **Raw Mask** (Chính) | 0.9541 | 0.9534 | 0.9764 | 0.5246 | 100.0% | 33.3% | 0.0% | 0.02% |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | Post-processed (Erode 3×3) | 0.9538 | 0.9530 | 0.9762 | 0.5218 | 100.0% | 33.3% | 0.0% | 0.01% |

### 1.2. Phân tích theo Nhóm Hình thái Món ăn (Raw Mask — Micro mIoU / Boundary IoU)

| Nhóm hình thái | Số mẫu (n) | SAM 2.1 (mIoU / BIoU) | MobileSAM (mIoU / BIoU) | FoodSAM (mIoU / BIoU) |
|---|---:|---:|---:|---:|
| Món nước trong tô (`soup_bowl`) | 3 | 0.9406 / 0.3477 | 0.8438 / 0.1929 | 0.9607 / 0.4561 |
| Món khô trên đĩa (`dry_plate`) | 2 | 0.7093 / 0.1800 | 0.9527 / 0.3762 | 0.9567 / 0.4154 |
| Món rời / nguyên liệu (`discrete_item`) | 7 | 0.9599 / 0.6085 | 0.9563 / 0.5780 | 0.9505 / 0.5851 |

### 1.3. Phân tích theo từng Lớp món ăn (Raw Mask mIoU)

| Lớp món ăn | n | SAM 2.1 | MobileSAM | FoodSAM |
|---|---:|---:|---:|---:|
| Banh mi | 2 | 0.9693 | 0.9646 | 0.9390 |
| Bun bo Hue | 1 | 0.9042 | 0.7840 | 0.9392 |
| Canh | 1 | 0.9770 | 0.7717 | 0.9738 |
| Com chien ga | 1 | 0.8521 | 0.9473 | 0.9513 |
| Com tam | 1 | 0.5666 | 0.9582 | 0.9621 |
| Pho | 1 | 0.9406 | 0.9756 | 0.9692 |
| Rau | 1 | 0.9137 | 0.9005 | 0.9020 |
| Thit ga | 1 | 0.9621 | 0.9595 | 0.9595 |
| Thit kho | 1 | 0.9683 | 0.9640 | 0.9627 |
| Trung | 2 | 0.9682 | 0.9705 | 0.9755 |

## 2. Hiệu năng Tốc độ Suy luận & Tài nguyên Phần cứng (GTX 1650 4 GB — Tiến trình cô lập)

| Mô hình | Preprocess (ms) | Image Encode / AMG+SETR (ms) | Prompt Decode / Merge (ms) | Postprocess (ms) | Total Mean±Std (ms/ảnh) | Median P50 (ms) | P95 (ms) | Peak VRAM (MiB) | CPU RAM (MiB) | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | 1.3 | 308.1 | 24.5 | 53.4 | **387.2 ± 23.4** | 384.4 | 428.4 | 623.4 | 3820.2 | 0 |
| MobileSAM (TinyViT) | 16.5 | 88.0 | 18.4 | 129.7 | **252.6 ± 38.2** | 248.8 | 318.2 | 313.2 | 4411.7 | 0 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.0 | 7038.4 | 2974.9 | 25.2 | **10038.5 ± 721.3** | 10135.7 | 10638.2 | 2418.0 | 1739.8 | 0 |

## 3. Tác động của Phân đoạn lên Ước lượng Khối lượng (Oracle Box — 11 mẫu nhóm hợp lệ)

| Nguồn Mặt nạ (Mask Source) | MAE (g) | MAPE (%) | $\Delta\text{MAE}_{\text{seg}}$ so với GT Mask (g) | $\Delta\text{MAPE}_{\text{seg}}$ so với GT Mask (%) | Paired $\|m_{\text{pred}} - m_{\text{GT mask}}\|$ MAE (g) |
|---|---:|---:|---:|---:|---:|
| **1. Oracle GT Polygon Mask** (Cơ sở) | **169.87** | **101.73%** | 0.00 | 0.00% | 0.00 |
| SAM 2.1 (Hiera-Small) — Raw | 147.67 | 82.09% | -22.20 | -19.65% | 46.67 |
| SAM 2.1 (Hiera-Small) — Post-processed | 144.81 | 81.00% | -25.06 | -20.73% | 43.90 |
| MobileSAM (TinyViT) — Raw | 156.21 | 70.52% | -13.66 | -31.21% | 113.43 |
| MobileSAM (TinyViT) — Post-processed | 169.25 | 101.39% | -0.62 | -0.34% | 2.62 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) — Raw | 138.25 | 81.31% | -31.62 | -20.42% | 36.16 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) — Post-processed | 138.18 | 81.05% | -31.69 | -20.68% | 36.07 |

## 4. Kịch bản B — Đánh giá End-to-End với YOLO Box & Phân loại 4 Trường hợp (TH1–TH4)

### 4.1. Thống kê Detection & Localization (Two-Tier One-to-One Matching @ IoU≥0.5)

- **Tổng số GT instances:** `12` · **Tổng số YOLO predictions:** `12`
- **Class-Agnostic Localization:** Recall = `100.0%` (12/12) · Precision = `100.0%` (12/12)
- **Class-Aware Detection:** Recall = `91.7%` (11/12) · Precision = `91.7%` (11/12) · **F1 = `91.7%`**
- **Phân rã 4 trường hợp:**
  - **TH1 (Đúng vị trí & Đúng tên món):** `11` instances
  - **TH2 (Đúng vị trí nhưng Sai tên món):** `1` instances
  - **TH3 (Bỏ sót thức ăn — False Negative):** `0` instances
  - **TH4-A (YOLO phát hiện thừa — Detection FP):** `0` boxes · **FoodSAM False Extras:** `1` components

### 4.2. Chất lượng Phân đoạn & Tỷ lệ Thành công End-to-End trên TH1 & TH2

| Mô hình | TH1 Matched mIoU (Raw / Post) | TH2 Class-Agnostic mIoU | TH2-A Seg Success / TH2-B Seg Fail | End-to-End Success Rate (trên tổng GT) |
|---|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | 0.9088 / 0.8763 | 0.9695 | 1 / 0 | **91.7%** (11/12) |
| MobileSAM (TinyViT) | 0.9284 / 0.9352 | 0.9640 | 1 / 0 | **91.7%** (11/12) |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9531 / 0.9529 | 0.9605 | 1 / 0 | **91.7%** (11/12) |

#### Tác động của Nhận diện sai lớp (TH2) lên Khối lượng: `(Pred Class + Pred Mask)` vs `(GT Class + Pred Mask)`

| Ảnh | GT Class (Mass g) | YOLO Pred Class | Mô hình | Class-Agnostic Mask IoU | Mass với Pred Class (g) / Sai số | Mass với GT Class (g) / Sai số |
|---|---|---|---|---:|---:|---:|
| `ThitKho1 (2).jpg` | Thit kho (92.0g) | Bong cai | SAM 2.1 (Hiera-Small) | 0.9695 | 35.5g (|56.5g|) | 147.0g (|55.0g|) |
| `ThitKho1 (2).jpg` | Thit kho (92.0g) | Bong cai | MobileSAM (TinyViT) | 0.9640 | 34.1g (|57.9g|) | 147.0g (|55.0g|) |
| `ThitKho1 (2).jpg` | Thit kho (92.0g) | Bong cai | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9605 | 35.3g (|56.7g|) | 147.0g (|55.0g|) |

### 4.3. Đánh giá Khả năng Phục hồi Ngữ nghĩa của FoodSAM trên TH3 (YOLO False Negatives)

- **Số GT instances bị YOLO bỏ sót ($N_{\text{FN}}$):** `0`
- **Số instances được FoodSAM (`extras`) phục hồi thành công ($N_{\text{FN recovered}}$):** `0` (Recovery Rate / Recall = `0.0%`)
- **Số lượng `extras` phát sinh nhầm (False Recovery):** `1` (Recovery Precision = `0.0%`)

### 4.4. Sai số Khối lượng Cuối chuỗi: Matched-Only vs Full-Image (Bữa ăn hoàn chỉnh)

| Mô hình | Matched-Only MAE (g) | Matched-Only MAPE (%) | Full-Image Meal MAE (g) | Full-Image Meal MAPE (%) | Tổng khối lượng thừa từ TH4-A (g) |
|---|---:|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | 156.45 | 84.03% | **88.60** | **21.84%** | 0.0 |
| MobileSAM (TinyViT) | 179.08 | 103.01% | **169.25** | **36.48%** | 0.0 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | 147.57 | 83.59% | **177.67** | **38.66%** | 451.4 |

## 5. Chi tiết từng Mẫu trong Tập Kiểm thử (Oracle Box — Raw Mask IoU & Khối lượng)

| Ảnh | Lớp | Nhóm | GT Mass (g) | GT Mask Mass (g) | SAM 2.1 IoU / Mass (g) | MobileSAM IoU / Mass (g) | FoodSAM IoU / Mass (g) |
|---|---|---|---:|---:|---:|---:|---:|
| `Pho1 (1).jpg` | Pho | `soup_bowl` | 819.5 | 782.9 | 0.9406 / 782.9g | 0.9756 / 782.9g | 0.9692 / 782.9g |
| `BunBoHue1 (1).jpg` | Rau | `discrete_item` | 90.0 | 120.0 | 0.9137 / 120.0g | 0.9005 / 120.0g | 0.9020 / 120.0g |
| `BunBoHue1 (1).jpg` | Bun bo Hue | `soup_bowl` | 804.5 | 707.4 | 0.9042 / 707.4g | 0.7840 / 707.4g | 0.9392 / 707.4g |
| `com-suon1 (5).jpg` | Com tam | `dry_plate` | 390.5 | 290.9 | 0.5666 / 208.2g | 0.9582 / 300.3g | 0.9621 / 288.5g |
| `com-suon1 (5).jpg` | Canh | `soup_bowl` | 150.0 | 824.1 | 0.9770 / 451.4g | 0.7717 / 824.1g | 0.9738 / 451.4g |
| `ComGa2 (1).jpg` | Com chien ga | `dry_plate` | 466.1 | 130.8 | 0.8521 / 110.9g | 0.9473 / 125.8g | 0.9513 / 129.6g |
| `ComGa2 (1).jpg` | Thit ga | `discrete_item` | 106.0 | 275.9 | 0.9621 / 274.7g | 0.9595 / 281.2g | 0.9595 / 284.7g |
| `BanhMi_Trung (1).jpg` | Trung | `discrete_item` | 221.2 | 73.6 | 0.9599 / 73.6g | 0.9718 / 73.6g | 0.9799 / 73.6g |
| `BanhMi_Trung (1).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 91.1 | 0.9590 / 88.9g | 0.9546 / 88.0g | 0.9610 / 88.6g |
| `BanhMi_Trung (1).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 211.7 | 0.9796 / 208.5g | 0.9745 / 206.4g | 0.9171 / 225.8g |
| `ThitKho1 (2).jpg` | Thit kho | `discrete_item` | 92.0 | 147.0 | 0.9683 / 147.0g | 0.9640 / 147.0g | 0.9627 / 147.0g |
| `ThitKho1 (2).jpg` | Trung | `discrete_item` | 65.0 | 45.1 | 0.9765 / 44.1g | 0.9692 / 44.4g | 0.9711 / 45.0g |
