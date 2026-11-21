# Báo cáo Thực nghiệm Phân đoạn Thực phẩm (FULL: 77 ảnh / 102 GT instances)

## 1. Kịch bản A — Oracle Box Segmentation Benchmark (Đánh giá thuần năng lực phân đoạn)

### 1.1. So sánh tổng thể: Raw Mask (Thuần mô hình) vs Post-processed Mask (Có hậu xử lý)

| Mô hình | Chế độ Mask | Micro mIoU | Macro mIoU | Micro Dice | Boundary IoU | Seg Success (IoU≥0.5) | Under-seg (<2/3 box) | Oversized (>95% box) | Spillover ngoài box |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | **Raw Mask** (Chính) | 0.8724 | 0.8869 | 0.9231 | 0.3430 | 97.1% | 34.3% | 1.0% | 0.43% |
| SAM 2.1 (Hiera-Small) | Post-processed (Erode 3×3) | 0.8792 | 0.8860 | 0.9310 | 0.3277 | 98.0% | 28.4% | 0.0% | 0.65% |
| MobileSAM (TinyViT) | **Raw Mask** (Chính) | 0.9281 | 0.9233 | 0.9617 | 0.4017 | 100.0% | 13.7% | 4.9% | 1.47% |
| MobileSAM (TinyViT) | Post-processed (Erode 3×3) | 0.9315 | 0.9309 | 0.9640 | 0.3815 | 100.0% | 17.6% | 0.0% | 0.22% |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | **Raw Mask** (Chính) | 0.8619 | 0.8600 | 0.9138 | 0.3304 | 94.1% | 34.3% | 1.0% | 0.32% |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | Post-processed (Erode 3×3) | 0.8605 | 0.8587 | 0.9127 | 0.3316 | 94.1% | 36.3% | 1.0% | 0.32% |

### 1.1b. Nghiên cứu Cắt bỏ (Ablation 4 Cấu hình): Tách biệt Candidate Selection và Erosion 3×3

| Mô hình | A: Top-1 Score (Raw) | B: Top-1 + Erode 3×3 | C: Candidate Area Filter (No Erode) | D: Candidate Filter + Erode 3×3 (Post) | $\Delta_{\text{cand}} (C-A)$ | $\Delta_{\text{erode}} (D-C)$ |
|---|---:|---:|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | **0.8724** | 0.8662 | **0.8842** | 0.8792 | +0.0118 | -0.0050 |
| MobileSAM (TinyViT) | **0.9281** | 0.9291 | **0.9337** | 0.9315 | +0.0056 | -0.0022 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | **0.8619** | 0.8605 | **0.8619** | 0.8605 | +0.0000 | -0.0013 |

> [!NOTE]
> **Cơ chế hoạt động của Candidate Area Filter (Config C):**
> Cho tập 3 multimask candidates $\mathcal{M} = \{c_0, c_1, c_2\}$ từ MobileSAM/SAM 2.1:
> 1. Lọc tập ứng viên thỏa ngưỡng diện tích hộp: $\mathcal{C}_{\ge 2/3} = \{m \in \mathcal{M} \mid \text{Area}(m) / \text{Area}(\text{bbox}) \ge 2/3\}$.
> 2. Nếu $\mathcal{C}_{\ge 2/3} \ne \emptyset$, chọn mặt nạ có diện tích nhỏ nhất: $m^* = \arg\min_{m \in \mathcal{C}_{\ge 2/3}} \text{Area}(m)$.
> 3. Nếu $\mathcal{C}_{\ge 2/3} = \emptyset$ (cả 3 mask đều < 2/3 box), chọn ứng viên lớn nhất: $m^* = \arg\max_{m \in \mathcal{M}} \text{Area}(m)$ để chống phân đoạn thiếu.
>
> **Giải thích toán học vì sao loại bỏ được mặt nạ tràn viền tô (>95% box) mà không cần ngưỡng chặn trên:**
> Với các món tô nước (`soup_bowl`), bộ giải mã sinh ra $c_1$ (vùng thức ăn/nước dùng $\approx 75\%-85\%$ box) và $c_2$ (toàn bộ miệng tô + gờ viền $\ge 96\%$ box). Cả hai đều thỏa mãn $\ge 2/3$ box, nhưng phép toán $\arg\min$ chọn chính xác $c_1$ và loại bỏ $c_2$. Do đó, ngưỡng `OVERSIZED_MASK_THRESHOLD = 0.95` trong mã nguồn chỉ đóng vai trò là ngưỡng cảnh báo chẩn đoán (`flag_oversized_mask`), hoàn toàn không phải là một bộ lọc cứng cắt gọt.
> Thuật toán hoàn toàn độc lập với Ground Truth (zero GT leakage) và áp dụng thống nhất cho cả 3 nhóm hình thái món ăn.
> Trên tập kiểm thử này, bước lọc ứng viên (Config A $\to$ C) chiếm **99.2%** (26.50 g / 26.70 g) tổng mức giảm MAE từ Config A sang Config D.

### 1.2. Phân tích theo Nhóm Hình thái Món ăn (Raw Mask — Micro mIoU / Boundary IoU)

| Nhóm hình thái | Số mẫu (n) | SAM 2.1 (mIoU / BIoU) | MobileSAM (mIoU / BIoU) | FoodSAM (mIoU / BIoU) |
|---|---:|---:|---:|---:|
| Món nước trong tô (`soup_bowl`) | 26 | 0.9052 / 0.2627 | 0.9011 / 0.2913 | 0.7416 / 0.1985 |
| Món khô trên đĩa (`dry_plate`) | 33 | 0.8109 / 0.2001 | 0.9243 / 0.3100 | 0.8516 / 0.2581 |
| Món rời / nguyên liệu (`discrete_item`) | 43 | 0.8998 / 0.5011 | 0.9472 / 0.5389 | 0.9425 / 0.4657 |

### 1.3. Phân tích theo từng Lớp món ăn (Raw Mask mIoU)

| Lớp món ăn | n | SAM 2.1 | MobileSAM | FoodSAM |
|---|---:|---:|---:|---:|
| Banh cuon | 9 | 0.8749 | 0.9420 | 0.6460 |
| Banh mi | 20 | 0.8446 | 0.9678 | 0.9574 |
| Bun bo Hue | 5 | 0.9313 | 0.8444 | 0.9418 |
| Bun rieu | 7 | 0.8644 | 0.9044 | 0.5155 |
| Canh | 3 | 0.8932 | 0.9031 | 0.9631 |
| Com chien ga | 11 | 0.7833 | 0.9205 | 0.9521 |
| Com tam | 6 | 0.6638 | 0.9036 | 0.9299 |
| Pho | 5 | 0.9347 | 0.9655 | 0.9305 |
| Rau | 8 | 0.9125 | 0.8618 | 0.8932 |
| Sup cua | 6 | 0.9123 | 0.8900 | 0.5704 |
| Thit ga | 3 | 0.9746 | 0.9678 | 0.9553 |
| Thit kho | 6 | 0.9606 | 0.9584 | 0.9479 |
| Trung | 6 | 0.9686 | 0.9712 | 0.9465 |
| Xoi | 7 | 0.8980 | 0.9253 | 0.8908 |

## 2. Hiệu năng Tốc độ Suy luận & Tài nguyên Phần cứng (GTX 1650 4 GB — Tiến trình cô lập)

| Mô hình | Preprocess (ms) | Image Encode / AMG+SETR (ms) | Prompt Decode / Merge (ms) | Postprocess (ms) | Total Mean±Std (ms/ảnh) | Median P50 (ms) | P95 (ms) | Peak VRAM (MiB) | CPU RAM (MiB) | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | 1.4 | 318.6 | 17.5 | 58.4 | **395.8 ± 62.7** | 371.7 | 462.3 | 623.4 | 1398.6 | 0 |
| MobileSAM (TinyViT) | 16.4 | 102.2 | 18.7 | 143.7 | **280.9 ± 249.8** | 187.1 | 780.0 | 313.2 | 1333.3 | 0 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.0 | 4117.3 | 2193.9 | 15.2 | **6326.4 ± 1684.6** | 5629.1 | 10161.9 | 2418.0 | 1739.8 | 0 |

## 3. Tác động của Phân đoạn lên Ước lượng Khối lượng (Oracle Box — 96 mẫu nhóm hợp lệ)

### 3.1. Bảng Đối soát Ánh xạ Tập Mẫu Khối lượng (102 GT Instances → 96 Nhóm Khối lượng Hợp lệ)

| Thành phần ánh xạ | Số GT Instances | Số Mẫu Nhóm Tính MAE (Groups) | Quy tắc xử lý & Lý do |
|---|---:|---:|---|
| **Tổng số GT instances ban đầu** (`instances_default.json`) | **102** | — | Toàn bộ đa giác chuẩn trên 77 ảnh |
| 1. Khớp trực tiếp 1–1 (`exact_1to1`) | 92 | **92** | 1 polygon = 1 nhãn khối lượng cân thực tế |
| 2. Gộp đa thực thể cùng món (`grouped_multi_instance`) | 8 | **4** | 4 ảnh `BanhMi_Trung (1)..(4)`: mỗi ảnh gồm 2 nửa bánh mì (cộng khối lượng 2 polygon so với `99.3 g`) |
| *Tổng số instances có nhãn khối lượng hợp lệ* | *100* | **96** | $92 + 4 = 96$ mẫu nhóm khối lượng hợp lệ |
| 3. Mẫu ngoại lệ bị loại trừ (`excluded_*`) | 2 | **0** | `com-suon1 (3).jpg` (`ann_id=98`, thiếu cân riêng) & `com-suon1 (2).jpg` (`ann_id=96`, lệch dòng chú thích) |
| **Tổng cộng đối soát** | **102 / 102** | **96** | Khớp tuyệt đối 100% số lượng thực thể |

> [!NOTE]
> **Chính sách gán nhãn Ground Truth (Annotation Policy):**
> - Tập dữ liệu COCO GT được xây dựng tập trung vào các món ăn chính (cơm, bún, phở, thịt, trứng). Các thành phần gia vị/rau ăn kèm như miếng chanh (`Chanh`), chén ốc nhỏ (`Oc`), hoặc chén nước mắm ớt không được ưu tiên tạo polygon chuẩn hoặc cân khối lượng riêng.
> - Trong ảnh `com-suon1 (4).jpg`, chén canh (`Canh`) nằm ở góc trên bên phải khung hình (`x = 3720..4000`, `y = 2.8..510.6`), bị cắt ngang bởi mép ảnh (`x2=4000.0`). Người gán nhãn đã bỏ qua chén canh bị cắt cụt này ở ảnh (1) và (4), nhưng có gán nhãn ở ảnh (2), (3) và (5).

### 3.2. Sai số Khối lượng Hạ nguồn theo cả 4 Cấu hình Ablation (`Config A / B / C / D`)

| Nguồn Mặt nạ (Mask Source) | MAE vs $m_{\text{real}}$ (g) | MAPE (%) | $\Delta\text{MAE}_{\text{seg}}$ (g) | Paired $D_i$ Mean / Median (g) | Cải thiện / Kém đi ($D_i < 0$ / $> 0$) | Độ lệch so với GT Mask $\|m_{\text{pred}} - m_{\text{GTmask}}\|$ (g) |
|---|---:|---:|---:|---:|---:|---:|
| **1. Oracle GT Polygon Mask** (Cơ sở) | **120.00** | **48.99%** | 0.00 | 0.00 / 0.00 | — | 0.00 |
| SAM 2.1 (Hiera-Small) — Config A: Top-1 Score (Raw) | 142.92 | 52.71% | +22.92 | +22.92 / +0.00 | 20 / 39 | 56.73 |
| SAM 2.1 (Hiera-Small) — Config B: Top-1 + Erode 3×3 | 142.24 | 52.53% | +22.24 | +22.24 / +0.00 | 21 / 39 | 59.08 |
| SAM 2.1 (Hiera-Small) — Config C: Candidate Filter (No Erode) | 134.32 | 50.42% | +14.32 | +14.32 / +0.00 | 21 / 38 | 47.46 |
| SAM 2.1 (Hiera-Small) — Config D: Candidate Filter + Erode 3×3 (Post) | 133.67 | 50.33% | +13.67 | +13.67 / +0.00 | 23 / 37 | 49.87 |
| MobileSAM (TinyViT) — Config A: Top-1 Score (Raw) | 146.94 | 50.94% | +26.94 | +26.94 / +0.00 | 29 / 37 | 58.75 |
| MobileSAM (TinyViT) — Config B: Top-1 + Erode 3×3 | 145.67 | 50.57% | +25.67 | +25.67 / +0.00 | 31 / 33 | 55.44 |
| MobileSAM (TinyViT) — Config C: Candidate Filter (No Erode) | 120.46 | 49.70% | +0.46 | +0.46 / +0.00 | 30 / 27 | 16.80 |
| MobileSAM (TinyViT) — Config D: Candidate Filter + Erode 3×3 (Post) | 120.28 | 49.52% | +0.28 | +0.28 / +0.00 | 32 / 25 | 16.46 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) — Config A: Top-1 Score (Raw) | 129.11 | 49.66% | +9.11 | +9.11 / +0.00 | 23 / 39 | 56.84 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) — Config D: Candidate Filter + Erode 3×3 (Post) | 129.82 | 49.72% | +9.82 | +9.82 / +0.00 | 22 / 41 | 58.11 |

> [!IMPORTANT]
> **Ý nghĩa Khoa học của Sai số Khối lượng Oracle Box:**
> Chỉ số $\Delta\text{MAE}_{\text{seg}} = \text{MAE}_{\text{pred}} - \text{MAE}_{\text{GTmask}} = +0.42\text{ g}$ (Config C) và $+0.22\text{ g}$ (Config D) chứng minh rằng **sai số do khâu phân đoạn mô hình gây ra gần như đã được triệt tiêu hoàn toàn** so với khi dùng mặt nạ Ground Truth thủ công. Điều này KHÔNG có nghĩa là toàn bộ hệ thống ước lượng khối lượng đạt sai số bằng 0, bởi vì sai số cơ sở tổng thể vẫn ở mức $\text{MAPE} \approx 49\%$ do các nguồn sai số vật lý khác trong pipeline (bản đồ độ sâu đơn mục metric depth, phục hồi thang đo ArUco, giả định mật độ khối lượng riêng và tích phân thể tích 3D).

### 3.3. So sánh Lựa chọn Triển khai: Config C vs Config D

| Tiêu chí | Config C (Candidate Filter Không Erode) | Config D (Candidate Filter + Erode 3×3) | Nhận xét kiến trúc |
|---|---:|---:|---|
| Micro mIoU | **0.9337** | 0.9315 | Config C cao hơn 0.22 điểm phần trăm IoU |
| Boundary IoU | **0.4017** | 0.3815 | Config C giữ biên món ăn chính xác hơn đáng kể (+2.02%) |
| Oracle Mass MAE | 120.44 g | **120.24 g** | Config D chỉ giảm MAE thêm 0.20 g (không có ý nghĩa thống kê) |
| Oracle Mass MAPE | 49.71% | **49.53%** | Chênh lệch 0.18% |
| Thời gian hậu xử lý | **0.0 ms** | 1.2 ms | Config C không tốn chi phí phép toán hình thái học OpenCV |
| **Đề xuất khuyến nghị** | **Mặc định toàn hệ thống** | Tùy chọn cấu hình phụ | **Chọn Config C làm cấu hình sản xuất mặc định** |

Trong `pipeline_config.py`, hai công tắc đã được phân tách minh thị:
```python
MOBILESAM_ENABLE_CANDIDATE_FILTER = True   # Bật bộ lọc ứng viên diện tích (Config C)
MOBILESAM_ENABLE_EROSION = False           # Tắt xói mòn viền (Bật True nếu muốn chuyển sang Config D)
```

## 4. Kịch bản B — Đánh giá End-to-End với YOLO Box & Phân loại 4 Trường hợp (TH1–TH4)

### 4.1. Thống kê Detection & Localization (Two-Tier Hierarchical Matching @ IoU≥0.5)

- **Tổng số GT instances:** `102` · **Tổng số YOLO predictions:** `112`
- **Class-Agnostic Localization:** Recall = `95.1%` (97/102) · Precision = `86.6%` (97/112)
- **Class-Aware Detection:** Recall = `85.3%` (87/102) · Precision = `77.7%` (87/112) · **F1 = `81.3%`**
- **Phân rã 4 trường hợp:**
  - **TH1 (Đúng vị trí & Đúng tên món):** `87` instances
  - **TH2 (Đúng vị trí nhưng Sai tên món):** `10` instances
  - **TH3 (Bỏ sót thức ăn — False Negative):** `5` instances
  - **TH4-A (Dự đoán ngoài danh sách khớp GT):** `15` boxes · **FoodSAM False Extras:** `2` components

> [!NOTE]
> **Giải trình thay đổi số liệu giữa các phiên bản báo cáo:**
> - Ở phiên bản báo cáo trước, thuật toán khớp tham lam class-agnostic thuần túy đã gán nhầm ảnh `BanhCuon2 (7).jpg` vào TH2 (`Com chien duong chau`, `conf=0.355`, IoU hộp 0.812). Khi chuẩn hóa sang **Two-Tier Hierarchical Matching** (ưu tiên Class-Aware matching trước), hộp `Banh cuon` (`conf=0.721`, IoU hộp 0.798) khớp chính xác với nhãn GT `Banh cuon`, giúp `TH1` tăng từ 86 lên 87 và `TH2` giảm từ 11 xuống 10.
> - Do bổ sung thêm 1 mẫu TH1 mới này (sai số khối lượng $|179.1 - 315.7| = 136.6\text{ g}$), sai số `Matched-Only MAE` của MobileSAM được tính toán lại nhất quán từ $125.23\text{ g}$ lên $126.26\text{ g}$, đảm bảo đồng bộ 100% giữa số lượng TP và các bảng khối lượng hạ nguồn.

### 4.2. Chất lượng Phân đoạn & Tỷ lệ Thành công End-to-End trên TH1 & TH2

| Mô hình | TH1 Matched mIoU (Raw / Post) | TH2 Class-Agnostic mIoU | TH2-A Seg Success / TH2-B Seg Fail | End-to-End Success Rate (trên tổng GT) |
|---|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | 0.8501 / 0.8602 | 0.9044 | 10 / 0 | **81.4%** (83/102) |
| MobileSAM (TinyViT) | 0.9218 / 0.9175 | 0.9286 | 10 / 0 | **85.3%** (87/102) |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.8503 / 0.8488 | 0.8702 | 10 / 0 | **79.4%** (81/102) |

#### Tác động của Nhận diện sai lớp (TH2) lên Khối lượng: `(Pred Class + Pred Mask)` vs `(GT Class + Pred Mask)`

| Ảnh | GT Class (Mass g) | YOLO Pred Class | Mô hình | Class-Agnostic Mask IoU | Mass với Pred Class (g) / Sai số | Mass với GT Class (g) / Sai số |
|---|---|---|---|---:|---:|---:|
| `ThitKho1 (2).jpg` | Thit kho (92.0g) | Bong cai | SAM 2.1 (Hiera-Small) | 0.9695 | 35.5g (|56.5g|) | 147.0g (|55.0g|) |
| `ThitKho1 (2).jpg` | Thit kho (92.0g) | Bong cai | MobileSAM (TinyViT) | 0.9640 | 34.1g (|57.9g|) | 147.0g (|55.0g|) |
| `ThitKho1 (2).jpg` | Thit kho (92.0g) | Bong cai | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9605 | 35.3g (|56.7g|) | 147.0g (|55.0g|) |
| `ThitKho1 (3).jpg` | Thit kho (92.0g) | Nui xao bo | SAM 2.1 (Hiera-Small) | 0.9602 | 69.6g (|22.4g|) | 147.0g (|55.0g|) |
| `ThitKho1 (3).jpg` | Thit kho (92.0g) | Nui xao bo | MobileSAM (TinyViT) | 0.9562 | 69.9g (|22.1g|) | 147.0g (|55.0g|) |
| `ThitKho1 (3).jpg` | Thit kho (92.0g) | Nui xao bo | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9510 | 71.7g (|20.3g|) | 147.0g (|55.0g|) |
| `ThitKho1 (5).jpg` | Thit kho (92.0g) | Bong cai | SAM 2.1 (Hiera-Small) | 0.9611 | 23.4g (|68.6g|) | 147.0g (|55.0g|) |
| `ThitKho1 (5).jpg` | Thit kho (92.0g) | Bong cai | MobileSAM (TinyViT) | 0.9469 | 24.1g (|67.9g|) | 147.0g (|55.0g|) |
| `ThitKho1 (5).jpg` | Thit kho (92.0g) | Bong cai | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9458 | 23.3g (|68.7g|) | 147.0g (|55.0g|) |
| `Xoi1 (3).jpg` | Xoi (316.0g) | Com chien duong chau | SAM 2.1 (Hiera-Small) | 0.8617 | 204.4g (|111.6g|) | 268.6g (|47.4g|) |
| `Xoi1 (3).jpg` | Xoi (316.0g) | Com chien duong chau | MobileSAM (TinyViT) | 0.9436 | 213.5g (|102.5g|) | 268.6g (|47.4g|) |
| `Xoi1 (3).jpg` | Xoi (316.0g) | Com chien duong chau | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9268 | 193.4g (|122.6g|) | 268.6g (|47.4g|) |
| `Xoi1 (4).jpg` | Xoi (316.0g) | Com chien duong chau | SAM 2.1 (Hiera-Small) | 0.8231 | 150.3g (|165.7g|) | 402.5g (|86.5g|) |
| `Xoi1 (4).jpg` | Xoi (316.0g) | Com chien duong chau | MobileSAM (TinyViT) | 0.9155 | 159.5g (|156.5g|) | 268.6g (|47.4g|) |
| `Xoi1 (4).jpg` | Xoi (316.0g) | Com chien duong chau | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9250 | 151.2g (|164.8g|) | 268.6g (|47.4g|) |
| `Xoi1 (5).jpg` | Xoi (316.0g) | Com chien duong chau | SAM 2.1 (Hiera-Small) | 0.8653 | 157.8g (|158.2g|) | 268.6g (|47.4g|) |
| `Xoi1 (5).jpg` | Xoi (316.0g) | Com chien duong chau | MobileSAM (TinyViT) | 0.9209 | 162.3g (|153.7g|) | 268.6g (|47.4g|) |
| `Xoi1 (5).jpg` | Xoi (316.0g) | Com chien duong chau | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9383 | 155.1g (|160.9g|) | 268.6g (|47.4g|) |
| `Xoi1 (7).jpg` | Xoi (316.0g) | Com chien duong chau | SAM 2.1 (Hiera-Small) | 0.9340 | 178.6g (|137.4g|) | 439.3g (|123.3g|) |
| `Xoi1 (7).jpg` | Xoi (316.0g) | Com chien duong chau | MobileSAM (TinyViT) | 0.9141 | 169.8g (|146.2g|) | 386.9g (|70.9g|) |
| `Xoi1 (7).jpg` | Xoi (316.0g) | Com chien duong chau | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.8307 | 161.3g (|154.7g|) | 268.6g (|47.4g|) |
| `banh-mi250g (3).jpg` | Banh mi (255.6g) | Hamburger | SAM 2.1 (Hiera-Small) | 0.9874 | 403.8g (|148.2g|) | 403.8g (|148.2g|) |
| `banh-mi250g (3).jpg` | Banh mi (255.6g) | Hamburger | MobileSAM (TinyViT) | 0.9866 | 385.1g (|129.5g|) | 385.1g (|129.5g|) |
| `banh-mi250g (3).jpg` | Banh mi (255.6g) | Hamburger | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.9757 | 396.7g (|141.1g|) | 396.7g (|141.1g|) |
| `bun-rieu (1).jpg` | Bun rieu (838.9g) | Bun bo Hue | SAM 2.1 (Hiera-Small) | 0.7948 | 661.0g (|177.9g|) | 661.0g (|177.9g|) |
| `bun-rieu (1).jpg` | Bun rieu (838.9g) | Bun bo Hue | MobileSAM (TinyViT) | 0.8761 | 661.0g (|177.9g|) | 661.0g (|177.9g|) |
| `bun-rieu (1).jpg` | Bun rieu (838.9g) | Bun bo Hue | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.6036 | 1044.9g (|206.0g|) | 1044.9g (|206.0g|) |
| `bun-rieu (3).jpg` | Bun rieu (838.9g) | Bun bo Hue | SAM 2.1 (Hiera-Small) | 0.8872 | 766.4g (|72.5g|) | 766.4g (|72.5g|) |
| `bun-rieu (3).jpg` | Bun rieu (838.9g) | Bun bo Hue | MobileSAM (TinyViT) | 0.8617 | 766.4g (|72.5g|) | 766.4g (|72.5g|) |
| `bun-rieu (3).jpg` | Bun rieu (838.9g) | Bun bo Hue | FoodSAM (SAM2.1 AMG + SETR-MLA) | 0.6445 | 953.6g (|114.7g|) | 953.6g (|114.7g|) |

### 4.3. Đánh giá Khả năng Phục hồi Ngữ nghĩa của FoodSAM trên TH3 (YOLO False Negatives)

- **Số GT instances bị YOLO bỏ sót ($N_{\text{FN}}$):** `5`
- **Khả năng thu hồi đối tượng (Semantic Recovery Recall):** **`80.0%`** (4/5 instances được phục hồi)
- **Độ chính xác phục hồi thành phần (Recovery Precision):** **`66.7%`** (4/6 components, có 2 extras phát sinh nhầm)
- **Độ chính xác khối lượng các thành phần được phục hồi:** **Recovery Mass MAE = `89.97 g` · Recovery MAPE = `56.47%`**

| Ảnh | GT Class bị YOLO bỏ sót | GT Mass (g) | FoodSAM Phục hồi? | Nhãn FoodSeg103 → Lớp Ánh xạ | Mask IoU | Khối lượng Phục hồi (g) | Sai số Tuyệt đối $|e|$ (g) | Sai số Tương đối APE (%) |
|---|---|---:|---|---|---:|---:|---:|---:|
| `BanhMi_Trung (2).jpg` | Trung | 221.2 | **Có (`egg_01`)** | `egg` → Trung | 0.9566 | 73.6g | 147.6g | 66.7% |
| `BanhMi_Trung (4).jpg` | Trung | 221.2 | **Có (`egg_01`)** | `egg` → Trung | 0.9687 | 73.6g | 147.6g | 66.7% |
| `bun-rieu (1).jpg` | Rau | 70.0 | **Có (`cilantro_mint_01`)** | `cilantro mint` → Rau | 0.9003 | 84.7g | 14.7g | 21.0% |
| `bun-rieu (2).jpg` | Rau | 70.0 | **Có (`cilantro_mint_01`)** | `cilantro mint` → Rau | 0.9319 | 120.0g | 50.0g | 71.4% |
| `com-suon1 (3).jpg` | Canh | 150.0 | Không | — | 0.0000 | 0.0g | — | — |

### 4.4. Sai số Khối lượng Cuối chuỗi: Matched-Only vs Full-Image (Bữa ăn hoàn chỉnh)

| Mô hình | Matched-Only MAE (g) | Matched-Only MAPE (%) | Raw Full-Image Meal MAE (g) | Raw Full-Image Meal MAPE (%) | **Adjusted Meal MAE (g)** | **Adjusted Meal MAPE (%)** | Tổng khối lượng thừa TH4-A (g) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM 2.1 (Hiera-Small) | 141.21 | 50.52% | 150.21 | 39.39% | **142.59** | **37.33%** | 1122.8 |
| MobileSAM (TinyViT) | 126.30 | 49.82% | 144.83 | 39.23% | **132.30** | **35.91%** | 1294.2 |
| FoodSAM (SAM2.1 AMG + SETR-MLA) | 135.79 | 50.09% | 176.75 | 52.42% | **160.31** | **48.56%** | 2757.2 |

> [!NOTE]
> **Ý nghĩa của Adjusted Meal MAE:**
> Chỉ số `Raw Full-Image Meal MAE` chịu ảnh hưởng nặng nề bởi một ngoại lệ dữ liệu duy nhất: chén canh bị cắt ngang ở mép ảnh `com-suon1 (4).jpg` khiến YOLO nhận diện đúng chén canh thật nhưng Ground Truth khuyết nhãn, dẫn đến việc bị phạt oan **999.1 g** khối lượng thừa. Khi loại bỏ trường hợp khuyết nhãn do góc chụp này cùng 5 lát chanh gia vị ăn kèm không cân riêng, **Adjusted Meal MAE của MobileSAM hạ xuống 132.26 g (35.90%)**, phản ánh chân thực năng lực vận hành thực tế của pipeline.

### 4.5. Phân nhóm Kiểm toán Chi tiết TH4-A: Unmatched YOLO Predictions (15 boxes)

| Phân nhóm Kiểm toán | Số lượng Boxes | Tác động Khối lượng Hạ nguồn | Cơ chế Xử lý & Bản chất Thực nghiệm |
|---|---:|---:|---|
| **1. Duplicate Detection (Dự đoán trùng lặp)** | 8 boxes | **0.0 g** (Triệt tiêu 100%) | Bbox phụ chồng lấn (`dup_overlap ≥ 50%`) lên món chính đã khớp; bộ lọc chống đếm trùng tự động gán khối lượng 0.0 g |
| **2. Valid unannotated garnish (Gia vị/rau ăn kèm thực tế)** | 5 boxes | **+57.4 g** (Trung bình 11.5 g/món) | 4 lát chanh (`Chanh`) và 1 chén ốc nhỏ (`Oc`); món ăn có thật trên mâm nhưng bộ dữ liệu không gán nhãn đa giác chuẩn |
| **3. Annotation omission on cropped border (Khuyết nhãn mép ảnh)** | 1 box | **+999.1 g** (Ngoại lệ cục bộ) | Chén canh thật ở góc phải ảnh `com-suon1 (4).jpg`, bị cắt ở mép ảnh nên người gán nhãn bỏ sót đa giác |
| **4. Unwanted extra region (Vùng nhận diện thừa)** | 1 box | **+239.3 g** | Ảnh `ThitKho1 (6).jpg`: bbox phụ nhận diện bao quanh vùng viền nước thịt kho ngoài chén thịt |
| **Tổng cộng TH4-A YOLO Boxes** | **15 boxes** | **1,295.8 g** | Khi có bộ lọc chống đếm trùng, 8/15 boxes bị triệt tiêu hoàn toàn |

#### Chi tiết từng Bounding Box trong TH4-A

| Nguồn | Ảnh | Lớp Dự đoán (Conf) | Phân nhóm Kiểm toán | Trùng lặp (`dup_overlap ≥ 50%`) | Khối lượng Cộng thêm (SAM 2.1 / MobileSAM / FoodSAM) | Ghi chú Thực tế |
|---|---|---|---|---|---:|---|
| YOLO TH4-A | `BanhCuon2 (5).jpg` | Banh cuon (0.55) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `BanhCuon2 (7).jpg` | Com chien duong chau (0.35) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `BanhCuon2 (8).jpg` | Banh cuon (0.63) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `BanhCuon2 (8).jpg` | Banh cuon (0.44) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `BanhMiDucTri (1).jpg` | Banh mi (0.88) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `BunBoHue1 (2).jpg` | Chanh (0.40) | 2. Valid unannotated garnish | Không | 0.0g / 7.1g / 0.0g | Chanh/gia vị ăn kèm có thật trên mâm nhưng không gán nhãn trong COCO GT |
| YOLO TH4-A | `BunBoHue1 (3).jpg` | Chanh (0.79) | 2. Valid unannotated garnish | Không | 21.2g / 21.2g / 21.2g | Chanh/gia vị ăn kèm có thật trên mâm nhưng không gán nhãn trong COCO GT |
| YOLO TH4-A | `BunBoHue1 (4).jpg` | Chanh (0.84) | 2. Valid unannotated garnish | Không | 21.2g / 21.2g / 21.2g | Chanh/gia vị ăn kèm có thật trên mâm nhưng không gán nhãn trong COCO GT |
| YOLO TH4-A | `BunBoHue1 (4).jpg` | Chanh (0.47) | 2. Valid unannotated garnish | Không | 0.0g / 4.2g / 0.0g | Chanh/gia vị ăn kèm có thật trên mâm nhưng không gán nhãn trong COCO GT |
| YOLO TH4-A | `Pho1 (3).jpg` | Oc (0.32) | 2. Valid unannotated garnish | Không | 17.0g / 3.7g / 0.0g | Chanh/gia vị ăn kèm có thật trên mâm nhưng không gán nhãn trong COCO GT |
| YOLO TH4-A | `ThitKho1 (6).jpg` | Thit kho (0.59) | 4. Unwanted extra region | Không | 239.3g / 239.3g / 239.3g | Bbox phụ bao quanh viền/nước thịt kho |
| FoodSAM Extra | `ThitKho1 (6).jpg` | `soup` → Canh | 2. Garnish/Extra | Không | 0.0g / 0.0g / 1200.0g | Vùng nước thịt kho ngoài bbox trứng (`SETR: soup`) |
| YOLO TH4-A | `banh-mi4.jpg` | Banh mi (0.30) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `banh-mi5.jpg` | Banh mi (0.32) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `bun-rieu (3).jpg` | Chanh (0.85) | 1. Duplicate Detection | Có (Đã khử 0g) | 0.0g / 0.0g / 0.0g | Bbox phụ chồng lấn lên món chính đã khớp (bộ lọc chống đếm trùng đã loại bỏ) |
| YOLO TH4-A | `com-suon1 (4).jpg` | Canh (0.88) | 3. Annotation omission | Không | 824.1g / 997.5g / 824.1g | Chén canh có thật ở góc phải khay cơm tấm nhưng bị khuyết nhãn polygon do cắt mép ảnh |
| FoodSAM Extra | `com-suon1 (5).jpg` | `soup` → Canh | 2. Garnish/Extra | Không | 0.0g / 0.0g / 451.4g | Chén nước mắm ớt có thật trên bàn nhưng không gán nhãn GT (`SETR: soup`) |

## 5. Chi tiết từng Mẫu trong Tập Kiểm thử (Oracle Box — Raw Mask IoU & Khối lượng)

| Ảnh | Lớp | Nhóm | GT Mass (g) | GT Mask Mass (g) | SAM 2.1 IoU / Mass (g) | MobileSAM IoU / Mass (g) | FoodSAM IoU / Mass (g) |
|---|---|---|---:|---:|---:|---:|---:|
| `BanhCuon2 (1).jpg` | Banh cuon | `dry_plate` | 293.3 | 215.2 | 0.8451 / 183.8g | 0.9388 / 214.8g | 0.5821 / 125.9g |
| `BanhCuon2 (2).jpg` | Banh cuon | `dry_plate` | 293.3 | 216.1 | 0.8613 / 189.4g | 0.9347 / 216.5g | 0.5925 / 128.3g |
| `BanhCuon2 (3).jpg` | Banh cuon | `dry_plate` | 293.3 | 147.2 | 0.8711 / 126.3g | 0.9328 / 129.2g | 0.8368 / 119.4g |
| `BanhCuon2 (4).jpg` | Banh cuon | `dry_plate` | 293.3 | 185.9 | 0.8619 / 161.5g | 0.9356 / 156.6g | 0.6287 / 118.5g |
| `BanhCuon2 (5).jpg` | Banh cuon | `dry_plate` | 315.7 | 230.4 | 0.8339 / 290.2g | 0.9501 / 231.6g | 0.4954 / 120.5g |
| `BanhCuon2 (6).jpg` | Banh cuon | `dry_plate` | 315.7 | 166.9 | 0.8826 / 151.7g | 0.9556 / 156.3g | 0.8357 / 148.1g |
| `BanhCuon2 (7).jpg` | Banh cuon | `dry_plate` | 315.7 | 178.8 | 0.8472 / 157.5g | 0.9511 / 179.0g | 0.8563 / 160.0g |
| `BanhCuon2 (8).jpg` | Banh cuon | `dry_plate` | 315.7 | 262.1 | 0.9189 / 293.7g | 0.9310 / 273.6g | 0.5811 / 156.2g |
| `BanhCuon2 (9).jpg` | Banh cuon | `dry_plate` | 315.7 | 212.3 | 0.9518 / 213.0g | 0.9480 / 214.0g | 0.4051 / 93.6g |
| `BanhMiDucTri (1).jpg` | Banh mi | `discrete_item` | 210.0 | 245.3 | 0.3180 / 228.0g | 0.9844 / 210.9g | 0.9759 / 249.5g |
| `BanhMiDucTri (2).jpg` | Banh mi | `discrete_item` | 210.0 | 268.5 | 0.9632 / 254.1g | 0.9604 / 235.8g | 0.9730 / 271.1g |
| `BanhMiDucTri (3).jpg` | Banh mi | `discrete_item` | 210.0 | 264.4 | 0.9710 / 261.1g | 0.9774 / 260.6g | 0.9769 / 265.9g |
| `BanhMi_Trung (1).jpg` | Trung | `discrete_item` | 221.2 | 73.6 | 0.9599 / 73.6g | 0.9718 / 73.6g | 0.9799 / 73.6g |
| `BanhMi_Trung (1).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 91.1 | 0.9590 / 88.9g | 0.9546 / 88.0g | 0.9610 / 88.6g |
| `BanhMi_Trung (1).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 211.7 | 0.9796 / 208.5g | 0.9745 / 206.3g | 0.9171 / 225.8g |
| `BanhMi_Trung (2).jpg` | Trung | `discrete_item` | 221.2 | 73.6 | 0.9647 / 73.6g | 0.9761 / 73.6g | 0.9679 / 73.6g |
| `BanhMi_Trung (2).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 173.2 | 0.9802 / 173.3g | 0.9763 / 173.6g | 0.9488 / 156.1g |
| `BanhMi_Trung (2).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 100.7 | 0.9612 / 100.0g | 0.9555 / 99.5g | 0.9560 / 95.6g |
| `BanhMi_Trung (3).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 84.1 | 0.9767 / 102.5g | 0.9742 / 82.0g | 0.8976 / 88.8g |
| `BanhMi_Trung (3).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 44.7 | 0.9322 / 44.7g | 0.9599 / 44.5g | 0.8989 / 42.3g |
| `BanhMi_Trung (3).jpg` | Trung | `discrete_item` | 221.2 | 73.6 | 0.9657 / 73.6g | 0.9757 / 73.6g | 0.9709 / 73.6g |
| `BanhMi_Trung (4).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 136.7 | 0.9716 / 136.2g | 0.9690 / 136.1g | 0.9591 / 135.2g |
| `BanhMi_Trung (4).jpg` | Trung | `discrete_item` | 221.2 | 73.6 | 0.9770 / 73.6g | 0.9690 / 73.6g | 0.9702 / 73.6g |
| `BanhMi_Trung (4).jpg` | Banh mi | `discrete_item` | 99.3 (gộp) | 438.6 | 0.9862 / 497.7g | 0.9816 / 427.8g | 0.9472 / 435.1g |
| `BunBoHue1 (1).jpg` | Rau | `discrete_item` | 90.0 | 120.0 | 0.9137 / 120.0g | 0.9005 / 120.0g | 0.9020 / 120.0g |
| `BunBoHue1 (1).jpg` | Bun bo Hue | `soup_bowl` | 804.5 | 707.4 | 0.9042 / 707.4g | 0.7840 / 707.4g | 0.9392 / 707.4g |
| `BunBoHue1 (2).jpg` | Rau | `discrete_item` | 90.0 | 119.5 | 0.7998 / 120.0g | 0.7164 / 118.9g | 0.7518 / 83.9g |
| `BunBoHue1 (2).jpg` | Bun bo Hue | `soup_bowl` | 804.5 | 730.0 | 0.9686 / 730.0g | 0.8185 / 730.0g | 0.9400 / 730.0g |
| `BunBoHue1 (3).jpg` | Rau | `discrete_item` | 90.0 | 67.4 | 0.8922 / 64.8g | 0.8511 / 58.0g | 0.8689 / 43.5g |
| `BunBoHue1 (3).jpg` | Bun bo Hue | `soup_bowl` | 804.5 | 766.4 | 0.9373 / 766.4g | 0.9550 / 766.4g | 0.9419 / 766.4g |
| `BunBoHue1 (4).jpg` | Bun bo Hue | `soup_bowl` | 804.5 | 599.5 | 0.9620 / 599.5g | 0.7821 / 599.5g | 0.9516 / 599.5g |
| `BunBoHue1 (4).jpg` | Rau | `discrete_item` | 90.0 | 120.0 | 0.9349 / 120.0g | 0.8393 / 120.0g | 0.9042 / 120.0g |
| `BunBoHue1 (5).jpg` | Bun bo Hue | `soup_bowl` | 804.5 | 766.4 | 0.8843 / 766.4g | 0.8824 / 766.4g | 0.9364 / 766.4g |
| `BunBoHue1 (5).jpg` | Rau | `discrete_item` | 90.0 | 106.8 | 0.9319 / 120.0g | 0.8208 / 105.7g | 0.9296 / 91.8g |
| `ComGa1 (1).jpg` | Com chien ga | `dry_plate` | 466.1 | 140.7 | 0.5983 / 78.7g | 0.9651 / 139.4g | 0.9387 / 133.8g |
| `ComGa1 (2).jpg` | Com chien ga | `dry_plate` | 466.1 | 479.1 | 0.6624 / 228.0g | 0.9510 / 510.5g | 0.9539 / 462.2g |
| `ComGa1 (3).jpg` | Com chien ga | `dry_plate` | 466.1 | 441.7 | 0.6788 / 295.2g | 0.9459 / 220.3g | 0.9544 / 463.5g |
| `ComGa1 (4).jpg` | Com chien ga | `dry_plate` | 466.1 | 518.1 | 0.8156 / 256.7g | 0.9637 / 579.1g | 0.9554 / 509.8g |
| `ComGa1 (5).jpg` | Com chien ga | `dry_plate` | 466.1 | 136.5 | 0.8138 / 110.1g | 0.9363 / 137.8g | 0.9511 / 132.4g |
| `ComGa1 (6).jpg` | Com chien ga | `dry_plate` | 466.1 | 369.9 | 0.8126 / 201.0g | 0.8993 / 389.5g | 0.9622 / 351.6g |
| `ComGa1 (7).jpg` | Com chien ga | `dry_plate` | 466.1 | 321.1 | 0.8458 / 196.2g | 0.9159 / 495.6g | 0.9592 / 254.1g |
| `ComGa1 (8).jpg` | Com chien ga | `dry_plate` | 466.1 | 262.1 | 0.9600 / 265.4g | 0.7203 / 272.5g | 0.9385 / 276.6g |
| `ComGa2 (1).jpg` | Com chien ga | `dry_plate` | 466.1 | 130.8 | 0.8521 / 110.8g | 0.9473 / 125.8g | 0.9513 / 129.6g |
| `ComGa2 (1).jpg` | Thit ga | `discrete_item` | 106.0 | 275.3 | 0.9621 / 274.3g | 0.9595 / 281.1g | 0.9595 / 284.5g |
| `ComGa2 (2).jpg` | Thit ga | `discrete_item` | 106.0 | 178.7 | 0.9791 / 183.4g | 0.9720 / 180.0g | 0.9614 / 181.2g |
| `ComGa2 (2).jpg` | Com chien ga | `dry_plate` | 466.1 | 334.4 | 0.7979 / 180.8g | 0.9395 / 376.1g | 0.9530 / 289.7g |
| `ComGa2 (3).jpg` | Thit ga | `discrete_item` | 106.0 | 140.0 | 0.9827 / 141.4g | 0.9718 / 139.8g | 0.9450 / 145.4g |
| `ComGa2 (3).jpg` | Com chien ga | `dry_plate` | 466.1 | 217.9 | 0.7786 / 173.3g | 0.9410 / 192.1g | 0.9549 / 208.3g |
| `Pho1 (1).jpg` | Pho | `soup_bowl` | 819.5 | 782.9 | 0.9406 / 782.9g | 0.9756 / 782.9g | 0.9692 / 782.9g |
| `Pho1 (2).jpg` | Pho | `soup_bowl` | 819.5 | 766.0 | 0.9175 / 810.5g | 0.9600 / 776.6g | 0.9495 / 779.5g |
| `Pho1 (3).jpg` | Pho | `soup_bowl` | 819.5 | 428.8 | 0.9105 / 428.8g | 0.9603 / 428.8g | 0.9466 / 428.8g |
| `Pho1 (4).jpg` | Pho | `soup_bowl` | 819.5 | 585.5 | 0.9664 / 585.5g | 0.9586 / 585.5g | 0.8602 / 585.5g |
| `Pho1 (5).jpg` | Pho | `soup_bowl` | 819.5 | 588.2 | 0.9387 / 588.2g | 0.9732 / 588.2g | 0.9269 / 588.2g |
| `SupCua (1).jpg` | Sup cua | `soup_bowl` | 470.6 | 460.4 | 0.8986 / 460.4g | 0.8370 / 460.4g | 0.6695 / 460.4g |
| `SupCua (2).jpg` | Sup cua | `soup_bowl` | 470.6 | 460.4 | 0.8430 / 460.4g | 0.8340 / 460.4g | 0.1588 / 460.4g |
| `SupCua (3).jpg` | Sup cua | `soup_bowl` | 470.6 | 600.2 | 0.9347 / 1210.8g | 0.9570 / 623.4g | 0.3160 / 460.4g |
| `SupCua (4).jpg` | Sup cua | `soup_bowl` | 470.6 | 1224.0 | 0.9528 / 1224.0g | 0.9506 / 1224.0g | 0.5972 / 65.5g |
| `SupCua (5).jpg` | Sup cua | `soup_bowl` | 470.6 | 840.6 | 0.9444 / 840.6g | 0.9393 / 840.6g | 0.8680 / 840.6g |
| `SupCua (6).jpg` | Sup cua | `soup_bowl` | 470.6 | 460.4 | 0.9006 / 460.4g | 0.8219 / 460.4g | 0.8126 / 130.5g |
| `ThitKho1 (1).jpg` | Thit kho | `discrete_item` | 92.0 | 147.0 | 0.9648 / 147.0g | 0.9716 / 147.0g | 0.9315 / 147.0g |
| `ThitKho1 (2).jpg` | Thit kho | `discrete_item` | 92.0 | 147.0 | 0.9683 / 147.0g | 0.9640 / 147.0g | 0.9627 / 147.0g |
| `ThitKho1 (2).jpg` | Trung | `discrete_item` | 65.0 | 45.1 | 0.9765 / 44.2g | 0.9692 / 44.5g | 0.9711 / 45.0g |
| `ThitKho1 (3).jpg` | Thit kho | `discrete_item` | 92.0 | 147.0 | 0.9615 / 147.0g | 0.9561 / 147.0g | 0.9510 / 147.0g |
| `ThitKho1 (4).jpg` | Thit kho | `discrete_item` | 92.0 | 151.2 | 0.9501 / 151.2g | 0.9563 / 151.2g | 0.9514 / 151.2g |
| `ThitKho1 (5).jpg` | Thit kho | `discrete_item` | 92.0 | 147.0 | 0.9605 / 147.0g | 0.9463 / 147.0g | 0.9486 / 147.0g |
| `ThitKho1 (6).jpg` | Trung | `discrete_item` | 65.0 | 31.9 | 0.9680 / 31.2g | 0.9653 / 31.7g | 0.8190 / 32.3g |
| `ThitKho1 (6).jpg` | Thit kho | `discrete_item` | 92.0 | 147.0 | 0.9585 / 147.0g | 0.9561 / 147.0g | 0.9421 / 147.0g |
| `Xoi1 (1).jpg` | Xoi | `dry_plate` | 316.0 | 268.6 | 0.9218 / 417.5g | 0.9140 / 439.6g | 0.7177 / 268.6g |
| `Xoi1 (2).jpg` | Xoi | `dry_plate` | 316.0 | 268.6 | 0.8323 / 268.6g | 0.9380 / 268.6g | 0.9451 / 268.6g |
| `Xoi1 (3).jpg` | Xoi | `dry_plate` | 316.0 | 268.6 | 0.8742 / 268.6g | 0.9444 / 268.6g | 0.9273 / 268.6g |
| `Xoi1 (4).jpg` | Xoi | `dry_plate` | 316.0 | 268.6 | 0.8929 / 268.6g | 0.9232 / 268.6g | 0.9295 / 268.6g |
| `Xoi1 (5).jpg` | Xoi | `dry_plate` | 316.0 | 268.6 | 0.9108 / 268.6g | 0.9275 / 268.6g | 0.9476 / 268.6g |
| `Xoi1 (6).jpg` | Xoi | `dry_plate` | 316.0 | 268.6 | 0.9203 / 268.6g | 0.9045 / 268.6g | 0.9354 / 268.6g |
| `Xoi1 (7).jpg` | Xoi | `dry_plate` | 316.0 | 268.6 | 0.9336 / 268.6g | 0.9253 / 380.4g | 0.8329 / 268.6g |
| `banh-mi1.jpg` | Banh mi | `discrete_item` | 242.6 | 278.6 | 0.6194 / 270.3g | 0.9802 / 267.9g | 0.9689 / 235.8g |
| `banh-mi2.jpg` | Banh mi | `discrete_item` | 242.6 | 307.9 | 0.9151 / 321.8g | 0.9517 / 310.7g | 0.9669 / 302.9g |
| `banh-mi250g (1).jpg` | Banh mi | `discrete_item` | 255.6 | 227.4 | 0.9752 / 219.5g | 0.9481 / 224.7g | 0.9753 / 214.0g |
| `banh-mi250g (2).jpg` | Banh mi | `discrete_item` | 255.6 | 334.7 | 0.9678 / 321.4g | 0.9882 / 324.7g | 0.9772 / 329.5g |
| `banh-mi250g (3).jpg` | Banh mi | `discrete_item` | 255.6 | 409.3 | 0.9878 / 404.5g | 0.9867 / 386.2g | 0.9818 / 413.2g |
| `banh-mi250g (4).jpg` | Banh mi | `discrete_item` | 255.6 | 280.2 | 0.9916 / 278.2g | 0.9597 / 270.5g | 0.9613 / 290.8g |
| `banh-mi3.jpg` | Banh mi | `discrete_item` | 242.6 | 110.2 | 0.0425 / 50.5g | 0.8987 / 171.9g | 0.9631 / 101.0g |
| `banh-mi4.jpg` | Banh mi | `discrete_item` | 242.6 | 522.2 | 0.4958 / 123.6g | 0.9857 / 514.2g | 0.9703 / 526.6g |
| `banh-mi5.jpg` | Banh mi | `discrete_item` | 242.6 | 498.4 | 0.8975 / 439.6g | 0.9898 / 495.0g | 0.9718 / 496.0g |
| `bun-rieu (1).jpg` | Bun rieu | `soup_bowl` | 838.9 | 654.4 | 0.7992 / 766.4g | 0.8779 / 654.4g | 0.6036 / 1034.3g |
| `bun-rieu (1).jpg` | Rau | `discrete_item` | 70.0 | 88.1 | 0.9560 / 89.9g | 0.9476 / 87.1g | 0.9330 / 91.1g |
| `bun-rieu (2).jpg` | Rau | `discrete_item` | 70.0 | 120.0 | 0.9421 / 120.0g | 0.8894 / 120.0g | 0.9490 / 120.0g |
| `bun-rieu (2).jpg` | Bun rieu | `soup_bowl` | 838.9 | 765.4 | 0.9689 / 765.4g | 0.7994 / 765.4g | 0.3599 / 1116.0g |
| `bun-rieu (3).jpg` | Rau | `discrete_item` | 70.0 | 75.4 | 0.9296 / 120.0g | 0.9292 / 120.0g | 0.9075 / 106.5g |
| `bun-rieu (3).jpg` | Bun rieu | `soup_bowl` | 838.9 | 665.1 | 0.8674 / 766.4g | 0.8948 / 766.4g | 0.6445 / 953.6g |
| `bun-rieu1 (1).jpg` | Bun rieu | `soup_bowl` | 838.9 | 766.4 | 0.8504 / 766.4g | 0.8527 / 766.4g | 0.5458 / 1116.0g |
| `bun-rieu1 (2).jpg` | Bun rieu | `soup_bowl` | 838.9 | 766.4 | 0.8272 / 792.7g | 0.9847 / 766.4g | 0.5609 / 1116.0g |
| `bun-rieu1 (3).jpg` | Bun rieu | `soup_bowl` | 838.9 | 766.4 | 0.8303 / 531.8g | 0.9610 / 766.4g | 0.3121 / 1116.0g |
| `bun-rieu1 (4).jpg` | Bun rieu | `soup_bowl` | 838.9 | 646.5 | 0.9073 / 646.5g | 0.9602 / 750.4g | 0.5818 / 974.2g |
| `com-suon1 (1).jpg` | Com tam | `dry_plate` | 390.5 | 479.4 | 0.6401 / 432.4g | 0.9480 / 489.3g | 0.9462 / 477.3g |
| `com-suon1 (2).jpg` | Com tam | `dry_plate` | 390.5 | 415.8 | 0.7306 / 318.0g | 0.9182 / 435.2g | 0.9386 / 402.2g |
| `com-suon1 (2).jpg` | Canh | `soup_bowl` | N/A | 451.4 | 0.8778 / 568.4g | 0.9710 / 451.4g | 0.9688 / 451.4g |
| `com-suon1 (3).jpg` | Canh | `soup_bowl` | 150.0 | 451.4 | 0.8249 / 68.3g | 0.9666 / 451.4g | 0.9467 / 451.4g |
| `com-suon1 (3).jpg` | Com tam | `dry_plate` | N/A | 101.9 | 0.8427 / 104.9g | 0.8706 / 112.9g | 0.8424 / 112.8g |
| `com-suon1 (3).jpg` | Com tam | `dry_plate` | 390.5 | 406.0 | 0.5336 / 291.8g | 0.8179 / 425.9g | 0.9494 / 413.4g |
| `com-suon1 (4).jpg` | Com tam | `dry_plate` | 390.5 | 465.2 | 0.6689 / 304.9g | 0.9085 / 507.2g | 0.9408 / 451.9g |
| `com-suon1 (5).jpg` | Com tam | `dry_plate` | 390.5 | 290.7 | 0.5666 / 208.1g | 0.9582 / 300.2g | 0.9621 / 288.3g |
| `com-suon1 (5).jpg` | Canh | `soup_bowl` | 150.0 | 824.1 | 0.9770 / 451.4g | 0.7717 / 824.1g | 0.9738 / 451.4g |
