
# Experiment A — Density sensitivity (Bánh mì, offline sweep)

Baseline frozen: 16 ảnh có Banh mi volume (volume_cm3 từ pipeline đã freeze). Chỉ thay density của Bánh mì; các class khác giữ DB. GT Bánh mì: 99.3g TỔNG 2 phần (BanhMi_Trung — 1 full ổ + 1 nửa ổ bánh mì thường), 242.6/255.6/210g/ổ (bánh mì thịt).

| Density (g/cm³) | Bánh mì MAE (g) | Bánh mì MAPE | Tổng mass/ảnh MAPE | Tổng kcal/ảnh MAPE |
|---|---|---|---|---|
| 0.45 | 125.9 | 85.2% | 56.1% | 63.1% |
| 0.40 | 102.5 | 70.8% | 48.1% | 52.9% |
| 0.35 | 88.3 | 60.4% | 44.1% | 46.1% |
| 0.30 | 82.8 | 53.8% | 43.5% | 43.7% |
| 0.27 | 83.9 | 51.7% | 44.2% | 43.7% |
| 0.25 | 86.8 | 51.2% | 45.3% | 44.8% |

**Đọc kết quả (sensitivity analysis)**: ρ giảm làm MAPE Bánh mì giảm từ 85.2% (0.45) xuống 51.2% (0.25) nhưng **đi ngang ~51% khi ρ < 0.30** — tức density một mình KHÔNG giải thích được toàn bộ over-estimation. **Phần sai lệch còn lại có khả năng liên quan đến thể tích ước lượng; trong đó mask là giả thuyết có bằng chứng trực tiếp (crop), còn contribution của depth chưa được cô lập** (xem depth_audit.md, Experiment B). **Vùng sensitivity: ρ = 0.25–0.30 g/cm³** trong tập audit hiện tại — chưa phải calibration chứng minh "density thực của bánh mì thường". Wording: đây là phân tích độ nhạy theo mật độ, KHÔNG phải kết luận "density gây X×".

