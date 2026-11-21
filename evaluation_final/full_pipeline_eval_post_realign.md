
# Full-pipeline evaluation — 54 ảnh portion-estimation_val

Pipeline: v10b + dual-preresize @ conf=0.30 (app default) → volume SAM2 daemon (ArUco scale, 2000px source) → nutrition. GT: 74 targets / 54 ảnh hợp lệ.

## 1. Detection (class-level, greedy one-to-one theo counts)

- **Recall: 0.770 (57/74)** · FP/extras: 31 · 0-box: 0/54

| Class | Recall |
|---|---|
| Banh cuon | 0/6 |
| Banh mi | 19/20 |
| Bun bo Hue | 5/5 |
| Bun rieu | 5/7 |
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
| Bun rieu | 5 | 4194 | 94.8 | 11.3%
| Pho | 5 | 4098 | 179.7 | 21.9%
| Rau | 4 | 360 | 30.0 | 33.3%
| Sup cua | 7 | 3324 | 289.5 | 61.3%
| Trung | 2 | 442 | 147.6 | 66.7%
| Banh mi | 15 | 3007 | 124.7 | 87.2%
| Canh | 1 | 150 | 301.4 | 200.9%

**Tổng khối lượng/bức ảnh**: MAE = 206.7 g · MAPE = 46.1% (trên 53 ảnh) · khối lượng unmatched extras cộng dồn: 5189 g

## 3. Nutrition (Calories ước lượng vs GT)

| Class | n so sánh | GT kcal | MAE (kcal) | MAPE |
|---|---|---|---|---|
| Com tam | 5 | 3222 | 65.3 | 10.1% |
| Bun bo Hue | 5 | 3620 | 79.9 | 11.0% |
| Bun rieu | 5 | 3565 | 80.6 | 11.3% |
| Pho | 5 | 3073 | 134.8 | 21.9% |
| Rau | 4 | 101 | 8.4 | 33.3% |
| Sup cua | 7 | 2327 | 202.7 | 61.3% |
| Trung | 2 | 664 | 221.4 | 66.7% |
| Banh mi | 15 | 7969 | 330.5 | 87.2% |
| Canh | 1 | 52 | 105.5 | 201.0% |

**Tổng kcal/bức ảnh**: MAE = 170.5 kcal · MAPE = 32.4% (trên 36 ảnh có đủ DB dinh dưỡng)

## 2b. Portion MATCHED-ONLY (tách khỏi detection extras)

Chỉ tính top-k box theo confidence khớp GT counts; phần thừa ghi riêng.

| Class | n | GT mass (g) | Matched MAE (g) | Matched MAPE | Extras mass (g) |
|---|---|---|---|---|---|
| Banh mi | 15 | 3007 | 122.2 | 85.9% | 128 |
| Bun bo Hue | 5 | 4022 | 88.8 | 11.0% | 0 |
| Bun rieu | 5 | 4194 | 94.8 | 11.3% | 0 |
| Canh | 1 | 150 | 301.4 | 200.9% | 0 |
| Com tam | 5 | 1952 | 23.9 | 6.1% | 0 |
| Pho | 5 | 4098 | 179.7 | 21.9% | 0 |
| Rau | 4 | 360 | 30.0 | 33.3% | 0 |
| Trung | 2 | 442 | 147.6 | 66.7% | 0 |
| Sup cua | 7 | 3324 | 289.5 | 61.3% | 0 |

**Matched-only tổng khối lượng/ảnh**: MAE = 213.4 g · MAPE = 47.1% (trên 54 ảnh)

## 3b. Nutrition MATCHED-ONLY

| Class | n | GT kcal | MAE (kcal) | MAPE |
|---|---|---|---|---|
| Banh mi | 15 | 7969 | 323.7 | 85.9% |
| Bun bo Hue | 5 | 3620 | 79.9 | 11.0% |
| Bun rieu | 5 | 3565 | 80.6 | 11.3% |
| Canh | 1 | 52 | 105.5 | 201.0% |
| Com tam | 5 | 3222 | 39.5 | 6.1% |
| Pho | 5 | 3073 | 134.8 | 21.9% |
| Rau | 4 | 101 | 8.4 | 33.3% |
| Trung | 2 | 664 | 221.4 | 66.7% |
| Sup cua | 7 | 2327 | 202.7 | 61.3% |

**Matched-only tổng kcal/ảnh**: MAE = 240.8 kcal · MAPE = 47.8% (trên 54 ảnh)

## 4. Scale source + timing

- Scale source: {'aruco_marker': 49, 'plate_heuristic': 4, '': 1}
- Volume worker: mean 2.5s · median 2.3s · max 4.6s (53 requests)

## 5. Theo nhóm món (MAPE tổng khối lượng / tổng kcal)

| Nhóm | n | MAPE mass | MAPE kcal |
|---|---|---|---|
| BanhMi | 13 | 36.7% | 38.8% |
| BunBoHue | 4 | 8.0% | 11.1% |
| BunRieu-CanhBun | 4 | 11.4% | 11.4% |
| ComSuon | 3 | 25.6% | 11.9% |
| Pho | 5 | 21.9% | 21.9% |
| SupCua | 7 | 61.3% | 61.3% |
