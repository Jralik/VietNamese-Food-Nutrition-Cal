
# Full-pipeline evaluation — 87 ảnh portion-estimation_val

Pipeline: v10b + dual-preresize @ conf=0.30 (app default) → volume SAM2 daemon (ArUco scale, 2000px source) → nutrition. GT: 112 targets / 87 ảnh hợp lệ.

## 1. Detection (class-level, greedy one-to-one theo counts)

- **Recall: 0.786 (88/112)** · FP/extras: 43 · 0-box: 0/87

| Class | Recall |
|---|---|
| Banh cuon | 9/15 |
| Banh mi | 19/20 |
| Bun bo Hue | 5/5 |
| Bun rieu | 5/7 |
| Canh | 1/3 |
| Com tam | 6/6 |
| Pho | 5/5 |
| Rau | 6/8 |
| Thit ga | 3/3 |
| Thit kho | 3/6 |
| Trung | 4/6 |
| Xoi | 3/7 |
| Com chien ga | 11/11 |
| Sup cua | 8/10 |

## 2. Portion (khối lượng ước lượng vs GT)

| Class | n so sánh | GT mass (g) | MAE (g) | MAPE |
|---|---|---|---|---|
| Com tam | 5 | 1952 | 39.6 | 10.1%
| Bun bo Hue | 5 | 4022 | 88.8 | 11.0%
| Bun rieu | 5 | 4194 | 94.8 | 11.3%
| Xoi | 3 | 948 | 64.2 | 20.3%
| Pho | 5 | 4098 | 179.7 | 21.9%
| Rau | 4 | 360 | 30.0 | 33.3%
| Banh cuon | 9 | 2752 | 132.5 | 43.5%
| Com chien ga | 11 | 5127 | 243.1 | 52.2%
| Trung | 4 | 572 | 89.1 | 56.9%
| Sup cua | 7 | 3324 | 289.5 | 61.3%
| Thit ga | 3 | 318 | 87.6 | 82.6%
| Banh mi | 15 | 3007 | 124.7 | 87.2%
| Thit kho | 3 | 276 | 144.1 | 156.6%
| Canh | 1 | 150 | 301.4 | 200.9%

**Tổng khối lượng/bức ảnh**: MAE = 194.3 g · MAPE = 50.8% (trên 86 ảnh) · khối lượng unmatched extras cộng dồn: 5999 g

## 3. Nutrition (Calories ước lượng vs GT)

| Class | n so sánh | GT kcal | MAE (kcal) | MAPE |
|---|---|---|---|---|
| Com tam | 5 | 3222 | 65.3 | 10.1% |
| Bun bo Hue | 5 | 3620 | 79.9 | 11.0% |
| Bun rieu | 5 | 3565 | 80.6 | 11.3% |
| Xoi | 3 | 1849 | 125.3 | 20.3% |
| Pho | 5 | 3073 | 134.8 | 21.9% |
| Rau | 4 | 101 | 8.4 | 33.3% |
| Banh cuon | 9 | 3440 | 165.7 | 43.5% |
| Com chien ga | 11 | 9229 | 437.6 | 52.2% |
| Trung | 4 | 859 | 133.6 | 56.9% |
| Sup cua | 7 | 2327 | 202.7 | 61.3% |
| Thit ga | 3 | 633 | 174.3 | 82.6% |
| Banh mi | 15 | 7969 | 330.5 | 87.2% |
| Thit kho | 3 | 607 | 317.0 | 156.6% |
| Canh | 1 | 52 | 105.5 | 201.0% |

**Tổng kcal/bức ảnh**: MAE = 212.7 kcal · MAPE = 39.5% (trên 62 ảnh có đủ DB dinh dưỡng)

## 2b. Portion MATCHED-ONLY (tách khỏi detection extras)

Chỉ tính top-k box theo confidence khớp GT counts; phần thừa ghi riêng.

| Class | n | GT mass (g) | Matched MAE (g) | Matched MAPE | Extras mass (g) |
|---|---|---|---|---|---|
| Banh cuon | 9 | 2752 | 137.8 | 45.2% | 160 |
| Banh mi | 15 | 3007 | 122.2 | 85.9% | 128 |
| Bun bo Hue | 5 | 4022 | 88.8 | 11.0% | 0 |
| Bun rieu | 5 | 4194 | 94.8 | 11.3% | 0 |
| Canh | 1 | 150 | 301.4 | 200.9% | 0 |
| Com tam | 5 | 1952 | 23.9 | 6.1% | 0 |
| Pho | 5 | 4098 | 179.7 | 21.9% | 0 |
| Rau | 4 | 360 | 30.0 | 33.3% | 0 |
| Thit ga | 3 | 318 | 87.6 | 82.6% | 0 |
| Thit kho | 3 | 276 | 57.4 | 62.4% | 260 |
| Trung | 4 | 572 | 89.1 | 56.9% | 0 |
| Xoi | 3 | 948 | 64.2 | 20.3% | 0 |
| Com chien ga | 11 | 5127 | 243.1 | 52.2% | 0 |
| Sup cua | 7 | 3324 | 289.5 | 61.3% | 0 |

**Matched-only tổng khối lượng/ảnh**: MAE = 196.2 g · MAPE = 49.7% (trên 87 ảnh)

## 3b. Nutrition MATCHED-ONLY

| Class | n | GT kcal | MAE (kcal) | MAPE |
|---|---|---|---|---|
| Banh cuon | 9 | 3440 | 172.2 | 45.2% |
| Banh mi | 15 | 7969 | 323.7 | 85.9% |
| Bun bo Hue | 5 | 3620 | 79.9 | 11.0% |
| Bun rieu | 5 | 3565 | 80.6 | 11.3% |
| Canh | 1 | 52 | 105.5 | 201.0% |
| Com tam | 5 | 3222 | 39.5 | 6.1% |
| Pho | 5 | 3073 | 134.8 | 21.9% |
| Rau | 4 | 101 | 8.4 | 33.3% |
| Thit ga | 3 | 633 | 174.3 | 82.6% |
| Thit kho | 3 | 607 | 126.3 | 62.4% |
| Trung | 4 | 859 | 133.6 | 56.9% |
| Xoi | 3 | 1849 | 125.3 | 20.3% |
| Com chien ga | 11 | 9229 | 437.6 | 52.2% |
| Sup cua | 7 | 2327 | 202.7 | 61.3% |

**Matched-only tổng kcal/ảnh**: MAE = 260.4 kcal · MAPE = 50.2% (trên 87 ảnh)

## 4. Scale source + timing

- Scale source: {'aruco_marker': 82, 'plate_heuristic': 4, '': 1}
- Volume worker: mean 2.4s · median 2.3s · max 4.6s (86 requests)

## 5. Theo nhóm món (MAPE tổng khối lượng / tổng kcal)

| Nhóm | n | MAPE mass | MAPE kcal |
|---|---|---|---|
| BanhCuon | 9 | 43.5% | 43.5% |
| BanhMi | 13 | 36.7% | 38.8% |
| BunBoHue | 4 | 8.0% | 11.1% |
| BunRieu-CanhBun | 4 | 11.4% | 11.4% |
| ComGa | 11 | 45.1% | 44.5% |
| ComSuon | 3 | 25.6% | 11.9% |
| Pho | 5 | 21.9% | 21.9% |
| SupCua | 7 | 61.3% | 61.3% |
| ThitKho | 3 | 101.6% | 113.4% |
| Xoi | 3 | 20.3% | 20.3% |
