
# Full-pipeline evaluation — 54 ảnh portion-estimation_val

Pipeline: v10b + dual-preresize @ conf=0.30 (app default) → volume SAM2 daemon (ArUco scale, 2000px source) → nutrition. GT: 74 targets / 54 ảnh hợp lệ.

## 0b. Experiment A — Density sensitivity (đã chạy)

Sweep ρ ∈ {0.45→0.25} trên volume frozen: MAPE Bánh mì 85.2% → 51.2% và **đi ngang ~51% khi ρ < 0.30** — density một mình không giải thích được hết over-estimation. Phần sai lệch còn lại có khả năng liên quan đến thể tích ước lượng; trong đó mask là giả thuyết có bằng chứng trực tiếp (crop), còn contribution của depth chưa được cô lập. Chi tiết: experiment_a_density.md + depth_audit.md (3 failure mode độc lập). Lưu ý phương pháp luận: Experiment A (density) và Experiment B (mask ablation cconly: 84.8% → 61.6%) là hai ablation ĐỘC LẬP trên cùng baseline — không cộng/trừ thành decomposition "density = X%, mask = Y%" vì hai yếu tố nhân nhau trong m = V×ρ.

## 1. Detection (class-level, greedy one-to-one theo counts)

- **Detection recall = 0.811 (60/74), confidence threshold = 0.30** · FP/extras: 36 · 0-box: 0/54

| Class | Recall |
|---|---|
| Banh cuon | 0/6 |
| Banh mi | 20/20 |
| Bun bo Hue | 5/5 |
| Bun rieu | 7/7 |
| Canh | 1/3 |
| Com tam | 6/6 |
| Pho | 5/5 |
| Rau | 6/8 |
| Trung | 2/4 |
| Sup cua | 8/10 |

## 2. Portion (khối lượng ước lượng vs GT)

| Class | n so sánh | GT mass (g) | MAE (g) | MAPE |
|---|---|---|---|---|
| Com tam | 5 | 1952 | 39.6 | 10.1%
| Bun bo Hue | 5 | 4022 | 88.8 | 11.0%
| Bun rieu | 6 | 5033 | 101.5 | 12.1%
| Pho | 5 | 4098 | 179.7 | 21.9%
| Rau | 4 | 360 | 30.0 | 33.3%
| Sup cua | 7 | 3324 | 289.5 | 61.3%
| Trung | 2 | 442 | 147.6 | 66.7%
| Banh mi | 16 | 3263 | 125.9 | 85.2%
| Canh | 1 | 150 | 301.4 | 200.9%

**Tổng khối lượng/bức ảnh**: MAE = 191.3 g · MAPE = 43.8% (trên 53 ảnh) · khối lượng unmatched extras cộng dồn: 7609 g

## 3. Nutrition (Calories ước lượng vs GT)

| Class | n so sánh | GT kcal | MAE (kcal) | MAPE |
|---|---|---|---|---|
| Com tam | 5 | 3222 | 65.4 | 10.1% |
| Bun bo Hue | 5 | 3620 | 79.9 | 11.0% |
| Bun rieu | 6 | 4278 | 86.3 | 12.1% |
| Pho | 5 | 3073 | 134.8 | 21.9% |
| Rau | 4 | 90 | 7.5 | 33.3% |
| Sup cua | 7 | 2327 | 202.7 | 61.3% |
| Trung | 2 | 633 | 211.1 | 66.7% |
| Banh mi | 16 | 8646 | 333.6 | 85.2% |
| Canh | 1 | 52 | 105.5 | 201.0% |

**Tổng kcal/bức ảnh**: MAE = 176.2 kcal · MAPE = 33.2% (trên 37 ảnh có đủ DB dinh dưỡng)

## 2b. Portion MATCHED-ONLY (tách khỏi detection extras)

Chỉ tính top-k box theo confidence khớp GT counts; phần thừa ghi riêng.

| Class | n | GT mass (g) | Matched MAE (g) | Matched MAPE | Extras mass (g) |
|---|---|---|---|---|---|
| Banh mi | 16 | 3263 | 123.5 | 84.1% | 128 |
| Bun bo Hue | 5 | 4022 | 88.8 | 11.0% | 0 |
| Bun rieu | 6 | 5033 | 101.5 | 12.1% | 0 |
| Canh | 1 | 150 | 301.4 | 200.9% | 0 |
| Com tam | 5 | 1952 | 23.9 | 6.1% | 0 |
| Pho | 5 | 4098 | 179.7 | 21.9% | 0 |
| Rau | 4 | 360 | 30.0 | 33.3% | 0 |
| Trung | 2 | 442 | 147.6 | 66.7% | 0 |
| Sup cua | 7 | 3324 | 289.5 | 61.3% | 0 |

**Matched-only tổng khối lượng/ảnh**: MAE = 198.2 g · MAPE = 44.9% (trên 54 ảnh)

## 3b. Nutrition MATCHED-ONLY

| Class | n | GT kcal | MAE (kcal) | MAPE |
|---|---|---|---|---|
| Banh mi | 16 | 8646 | 327.2 | 84.1% |
| Bun bo Hue | 5 | 3620 | 79.9 | 11.0% |
| Bun rieu | 6 | 4278 | 86.3 | 12.1% |
| Canh | 1 | 52 | 105.5 | 201.0% |
| Com tam | 5 | 3222 | 39.5 | 6.1% |
| Pho | 5 | 3073 | 134.8 | 21.9% |
| Rau | 4 | 90 | 7.5 | 33.3% |
| Trung | 2 | 633 | 211.1 | 66.7% |
| Sup cua | 7 | 2327 | 202.7 | 61.3% |

**Matched-only tổng kcal/ảnh**: MAE = 224.7 kcal · MAPE = 45.7% (trên 54 ảnh)

## 4. Scale source + timing

- Scale source: {'aruco_marker': 49, 'plate_heuristic': 4, '': 1}
- Volume worker: mean 2.5s · median 2.2s · max 6.7s (53 requests)

## 5. Theo nhóm món (MAPE tổng khối lượng / tổng kcal)

| Nhóm | n | MAPE mass | MAPE kcal |
|---|---|---|---|
| BanhMi | 14 | 38.0% | 40.2% |
| BunBoHue | 4 | 8.0% | 11.2% |
| BunRieu-CanhBun | 4 | 11.4% | 11.4% |
| ComSuon | 3 | 25.6% | 11.9% |
| Pho | 5 | 21.9% | 21.9% |
| SupCua | 7 | 61.3% | 61.3% |
