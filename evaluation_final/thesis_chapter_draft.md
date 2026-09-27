# DRAFT — Chương Kết quả và Đánh giá (khung + số liệu thật, phục vụ viết thesis)

> Tài liệu này là KHUNG NHÁP tổng hợp từ các báo cáo frozen trong evaluation_final/.
> Người viết tùy biến văn phong; các con số và mệnh đề phương pháp luận đã được
> freeze trong evaluation_summary.md — KHÔNG thay đổi khi copy vào thesis.

---

## 4.1 Thiết lập thí nghiệm (Experimental Setup)

- **Bộ dữ liệu đánh giá**: 54 ảnh (74 target khối lượng, 10 nhóm món Việt), chụp
  thực tế có marker ArUco làm chuẩn tỉ lệ; GT khối lượng đo bằng cân, diễn giải theo
  semantics đã chuẩn hóa (gram = tổng khối lượng của K box thuộc cùng class trong ảnh).
- **Cấu hình pipeline frozen**: YOLO (v10b, dual-preresize PIL stretch+letterbox,
  class-aware NMS) · ngưỡng tin cậy 0.30 · volume SAM2 (box-prompted) + Depth Anything
  V2 Metric + ArUco scale recovery · density DB 0.45 g/cm³ cho bánh mì · volume worker
  GPU riêng biệt.
- **Isolation của các đánh giá**: mọi thay đổi GT/evaluator đều được chứng minh không
  tác động đến inference — 54/54 bản ghi detection/estimation identical trước/sau
  (full_eval_records.jsonl.bak_gt_old so với bản hiện hành).

## 4.2 Kết quả Detection

Detection recall = **0.811 (60/74)**, confidence threshold = **0.30** · FP/extras: 36 ·
0-box: 0/54.

| Class | Recall | | Class | Recall |
|---|---|---|---|---|
| Bánh mì | 20/20 | | Rau | 6/8 |
| Bún riêu | 7/7 | | Phở | 5/5 |
| Cơm tấm | 6/6 | | Bún bò Huế | 5/5 |
| Súp cua | 8/10 | | Trứng | 2/4 |
| Canh | 1/3 | | Bánh cuốn | 0/6 |

(FP/extras 36 — phân tích theo class thấy tập trung ở món có nhiều phần/dụng cụ trong
khung hình; bảng đầy đủ trong detection_per_class.csv.)

## 4.3 Kết quả ước lượng khẩu phần và dinh dưỡng

| Class | Portion MAPE | Nutrition MAPE |
|---|---|---|
| Cơm tấm | **10.1%** | 10.1% |
| Bún bò Huế | **11.0%** | 11.0% |
| Bún riêu | **12.1%** | 12.1% |
| Phở | 21.9% | 21.9% |
| Rau | 33.3% | 33.3% |
| Súp cua | 61.3% | 61.3% |
| Bánh mì | 85.2% | 85.2% |
| Trứng | 66.7% | 66.7% |
| Canh | 200.9% (n=1) | 200.9% |

Tổng khối lượng/ảnh: MAE 191.3 g, MAPE 43.8% · Tổng kcal/ảnh: MAE 176.2 kcal,
MAPE 33.2%. Nutrition kế thừa sai số portion vì calo được tính từ khối lượng ước
lượng kết hợp composition lookup (nutrition_per_100g) và density DB.

## 4.4 Phân tích lỗi (Error Analysis)

Sai số xuất hiện theo **ba nhóm chính** với ba bản chất riêng biệt — pipeline không có
bias đơn hướng "luôn cao"/"luôn thấp".

### 4.4.1 Bánh mì — Over-estimation ~2.9×

GT chuẩn hóa: BanhMi_Trung = 1 full ổ + 1 nửa ổ bánh mì **thường**, tổng 99.3 g
(ước lượng pipeline: 202.5 + 85.5 = 288 g). Hai ablation độc lập:

- **Density sweep** (chỉ thay ρ của bánh mì, volume frozen): MAPE 85.2% (ρ=0.45) →
  51.2% (ρ=0.25), **đi ngang ~51% khi ρ < 0.30**.
- **Mask ablation** (chỉ thay mask, density frozen): cconly (largest connected
  component) đưa MAPE 84.8% → 61.6%; area −11% → volume −9% → mass −8% tương ứng.

**Kết luận**: hai yếu tố đều có contribution nhưng **không thể xem là các thành phần
cộng tuyến tính của sai số** (m = V×ρ nhân nhau); mask over-segmentation được quan sát
trực tiếp (crop + mask chiếm 13-70% diện tích ảnh), density 0.45 g/cm³ là yếu tố khả dĩ
— phần dư sau cả hai ablation cho thấy còn thành phần volume/thickness chưa được cô lập.

### 4.4.2 Trứng — Under-estimation ~3×

GT 221.2 g = cả đĩa trứng chiên (3-4 quả + dầu); ước lượng 73.6 g ổn định. Mask phủ
đúng đĩa (crop xác nhận) — đây là **limitation của monocular depth estimation với thức
ăn nhiều lớp, mỏng và phẳng** trên nền tương phản thấp, không phải lỗi pipeline.

### 4.4.3 Súp cua — Geometry/observation-configuration dependent

Cùng bát 470.6 g: top-down **-2%** (451 cm³ ≈ đúng thể tích chất lỏng); oblique
**+79-158%** (1188 cm³ ≈ dung tích khoang hộp — tích phân depth vào khoang trống do
depth cliff tại miềng hộp + sự mơ hồ của mặt chất lỏng trong suốt). Sai số phụ thuộc
vào **cấu hình hình học quan sát và độ ổn định của depth-anchor correction** (anchor
ratio dao động x0.33-x0.60 giữa các shot), không chỉ góc chụp theo phân loại nhị phân.

## 4.5 Ablation Studies

### 4.5.1 Density sensitivity (Experiment A)

| ρ (g/cm³) | 0.45 | 0.40 | 0.35 | 0.30 | 0.27 | 0.25 |
|---|---|---|---|---|---|---|
| Bánh mì MAPE | 85.2% | 70.8% | 60.4% | 53.8% | 51.7% | 51.2% |

Vùng sensitivity low-error: ρ = 0.25-0.30 g/cm³ — **chưa phải calibration** của density
thực; density một mình không giải thích được toàn bộ over-estimation.

### 4.5.2 Mask contribution (Experiment B)

| Variant | area | volume | mass | MAPE |
|---|---|---|---|---|
| original | 1.00 | 1.00 | 1.00 | 84.8% |
| otsu | 0.32 | 0.42 | 0.42 | 93.0% (over-filtering) |
| fixed60cc | 0.20 | 0.43 | 0.43 | 87.7% (over-filtering) |
| **cconly** | 0.89 | 0.91 | 0.92 | **61.6%** |

cconly loại blob nền rời rạc — contribution khiêm tốn nhưng thực có; các brightness
threshold variants cho thấy nguy cơ over-filtering. cconly là **proposed future
improvement**, chưa đưa vào production.

## 4.6 Capture condition exploration (không phải ablation — không thay đổi biến pipeline)

### 4.6.1 Soup container case study (Experiment C1, exploratory)

Phân loại exploratory 10 ảnh súp cua theo điều kiện chụp: sai số trải từ -2.2% đến
+160.1% và **không tách sạch theo binary top-down/tilted** (tilted có thể chính xác
-2.2%, high-angle có thể +78.6%). Trong phạm vi dữ liệu khảo sát, depth-anchor
correction ratio tương quan với sai số thể tích mạnh hơn phân loại góc đơn giản —
**tín hiệu quan sát được, chưa đủ khẳng định quan hệ nhân quả** (n hiệu dụng ≈ 4-5
cảnh độc lập). Ghi vào limitation; C2 (controlled capture) là future work.

---

## Tài liệu nguồn (evaluation_final/)

- full_pipeline_eval.md — full-pipeline 54 ảnh
- experiment_a_density.md / experiment_b_mask.md — ablation A/B
- depth_audit.md — root-cause audit 3 failure modes
- phase_d_results.md — detection sweep
- evaluation_summary.md — master index + consistency checklist
- plots/*.png, thesis_tables/*.csv — hình bảng sẵn sàng
