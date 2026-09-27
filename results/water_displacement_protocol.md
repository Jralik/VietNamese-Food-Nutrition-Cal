# Protocol — Water Displacement đo Density Database (Core Target #2)

> **Trạng thái:** protocol sẵn sàng, chờ buổi thực nghiệm cân đo. KHÔNG chặn viết Chương 4.
> **Ranh giới provenance (bắt buộc):** `measured density → new density_db version → new post-freeze evaluation → compare against frozen baseline`. Tuyệt đối KHÔNG ghi đè `density_db.py` hiện hành rồi re-run để **thay** số Chương 4.

## 1. Mục tiêu

- Hoàn thành Mục tiêu #2 [LÕI] của đề cương: density DB thực nghiệm bằng water displacement.
- Mở khóa RQ2 định lượng: có mean ± std **đo được** để phân tích đóng góp biến thiên density vào sai số khối lượng (hiện std 0.05-0.12 là assumption).

## 2. Phạm vi

- 7 nhóm đã benchmark: Pho, BunBoHue, BunRieu-CanhBun, ComSuon (cơm), BanhMi, BanhCuon, SupCua.
- **~35 mẫu là target (7 nhóm × 5 mẫu) — KHÔNG phải acceptance criterion.** 25 mẫu (5 nhóm × 5) vẫn đạt yêu cầu; không kéo thêm mẫu chỉ để đủ con số.
- Mỗi mẫu đo **2 lần lặp** để có repeatability.

## 3. Thiết bị

- Cân bếp ±1 g (tốt nhất ±0.1 g) — chính là cân đã dùng cho GT khối lượng.
- Thùng/cốc nước đủ to để chìm mẫu, vạch chia ±5 ml; hoặc bình tràn + becher thu nước.
- Giỏ lưới mảnh (chìm hoàn toàn mẫu, không giữ bọt khí).
- Nhiệt kế (ghi nhiệt độ nước để hiệu chỉnh ρ_water: 0.9982 g/ml @20°C, 0.9970 @25°C).
- Túi zip/film mỏng (cho món dễ rời/võng): cân cả bao bì, trừ khối lượng bao.

## 4. Quy trình từng mẫu (≈ 5 phút)

1. Cân mẫu: `w_food` (g). Ghi tên món + nhóm + cấu trúc (đặc/nước/nhiều khoảng khí).
2. Đo thể tích displacement:
   - Món đặc (cơm, bánh mì, bánh cuốn, thịt): đặt trong giỏ lưới, nhúng chìm hoàn toàn, đọc mực nước trước `V1` và sau `V2` → `V_disp = V2 − V1` (ml).
   - Món nước (phở, bún bò, bún riêu, súp cua): **đo riêng phần chất đặc** (bún + topping sau khi vớt, để ráo 30 s) và **phần nước dùng cân riêng** — phần nước dùng dùng khối lượng & giả định ρ ≈ ρ_water (ghi chú); không nhúng tô nước vào thùng đo.
3. Hiệu chỉnh: `ρ = w_food / (V_disp × ρ_water_correction)` — với V đo bằng ml nước ở nhiệt độ t, thể tích (cm³) = V_ml (đã là thể tích nước dịch chuyển); ρ_food = w_food / V_disp.
4. Lặp lại lần 2 với cùng mẫu (nhúng lại, lau khô nếu món hấp nước — ưu tiên dùng mẫu riêng cho lần 2 nếu món bị ướt).
5. Chụp ảnh mẫu + thùng đo cho provenance.

## 5. Số liệu sheet (CSV template — `data/density_measured_samples.csv`)

```csv
group,food_name,sample_id,repeat_id,w_food_g,v1_ml,v2_ml,water_temp_c,rho_g_cc,method,notes
ComSuon,Com tam,cs01,1,390.5,1200,1701.3,26,0.779,basket_immersion,co tam rac nuoc 30s
SupCua,Sup cua (phan dac),sc01,1,482.6,1000,1481.7,26,1.000,basket_immersion,phan dac; nuoc dung can rieng rho~1.0
```

Quy ước: `method ∈ {basket_immersion, overflow, wrapped}`; phần nước dùng ghi `notes` và rho xấp xỉ 1.0.

## 6. Xử lý & cập nhật DB

1. Tính mean ± std per nhóm (dùng cả 2 repeat; outlier chỉ loại khi có lý do đo được ghi chú).
2. Tạo **file DB mới** `density_db_measured.py` (copy cấu trúc `density_db.py`), source label `"measured_water_displacement"`, kèm commit message + ngày đo + nhiệt độ nước. KHÔNG sửa `density_db.py` frozen.
3. Chạy post-freeze evaluation riêng: `python eval_full_pipeline.py` với env override trỏ tới DB mới (hoặc flag trong script) → ghi `results/post_freeze_density_evaluation.md`: MAPE per nhóm với ρ đo được **so sánh** với frozen baseline (44.9% matched-only; per-class bảng SSOT §3.3).
4. Kỳ vọng phân tích: với nhóm có ρ đo lệch nhiều so với assumption → giải thích phần MAPE; với nhóm plateau ~51% (Experiment A) → xác nhận residual không do density.

## 7. Fallback (nếu không kịp trước hạn nộp)

- Ghi limitation §4.7: density chủ yếu literature, chưa đo water displacement.
- Chỉnh phát biểu Mục tiêu #2 trong báo cáo thành "thiết kế protocol + pilot đo" (nếu có vài mẫu) — tương tự cách đã xử lý Mục tiêu #5 (SAM2 thay FoodSAM/MobileSAM).
- Protocol + CSV template vẫn nộp kèm để thể hiện kế hoạch hoàn thiện.
