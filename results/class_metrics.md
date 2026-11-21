# Class Metrics — coverage & target-matched MAPE (bảng derived)

> **Chain B (historical)** — nguồn: `data/benchmark_error_attribution_audit.json` (54 records, GT = khối lượng món mục tiêu, pre-freeze pipeline v26m ONNX).
> Bảng derived theo schema khóa trong `results/frozen_numbers.md` §0. Hai cột detection trình bày cạnh nhau — xem phần 'Hai phép đo detection' bên dưới.

## Bảng chính

| Class | N | Detected (D1) | Exact-class | Coverage (D1) | MAPE (target-matched) |
|---|---:|---:|---:|---:|---:|
| Pho | 5 | 1 | 1 | 20% | 46.1% |
| BanhCuon | 6 | 0 | 0 | 0% | N/A |
| ComSuon | 5 | 3 | 1 | 60% | 15.4% |
| BunBoHue | 5 | 5 | 5 | 100% | 19.4% |
| BanhMi | 16 | 14 | 14 | 88% | 43.3% |
| BunRieu-CanhBun | 7 | 7 | 4 | 100% | 17.3% |
| SupCua | 10 | 4 | 0 | 40% | 69.5% |
| **Tổng** | **54** | **34** | **25** | **63.0%** | — |

### Hai phép đo detection (đọc kỹ trước khi trích)

- **Detected (D1)** — taxonomy frozen của audit: *vùng món mục tiêu được phát hiện*, kể cả khi nhãn là generic/sibling. Khớp `tier_1_detection_status_counts` (D1 = 34, asserted trong `tests/test_phase4_acceptance.py`).
- **Exact-class** — chỉ đếm khi đúng chuỗi class mục tiêu xuất hiện trong `detected_classes` (phép đo nghiêm ngặt hơn, KHÔNG nằm trong taxonomy frozen).
- Chênh lệch D1 − Exact-class phát sinh từ nhãn generic/sibling: ComSuon (3 ảnh D1, 1 exact); BunRieu-CanhBun (7 ảnh D1, 4 exact); SupCua (4 ảnh D1, 0 exact) — ví dụ SupCua D1 chỉ có `Canh (Soup)`, ComSuon D1 đa số `Com (Rice)`, BunRieu có ảnh D1 mang nhãn `Bun bo Hue (Hue beef noodle soup)`.

## MAPE | D1-only (chỉ ảnh có trạng thái D1 theo audit)

| Class | n (D1) | MAPE |
|---|---:|---:|
| Pho | 1 | 4.5% |
| BanhCuon | 0 | N/A |
| ComSuon | 3 | 15.0% |
| BunBoHue | 5 | 19.4% |
| BanhMi | 14 | 38.5% |
| BunRieu-CanhBun | 7 | 17.3% |
| SupCua | 4 | 58.2% |

## MAPE | Exact-class-only (chỉ ảnh có đúng class mục tiêu)

| Class | n (exact) | MAPE |
|---|---:|---:|
| Pho | 1 | 4.5% |
| BanhCuon | 0 | N/A |
| ComSuon | 1 | 11.2% |
| BunBoHue | 5 | 19.4% |
| BanhMi | 14 | 38.5% |
| BunRieu-CanhBun | 4 | 14.9% |
| SupCua | 0 | N/A |

## Auxiliary D2 error (các nhóm Coverage = 0% theo cả hai phép đo)

- **BanhCuon** — Auxiliary D2 error: **67.5%** — 67.5% được tính từ detection nhầm lớp (không phải MAPE trên detection đúng lớp Banh cuon).

## Phân bổ trạng thái không phải D1 (theo nhóm)

- Pho: D2=1, D3=2, D4=1
- BanhCuon: D2=6
- ComSuon: D5=2
- BanhMi: D2=1, D3=1
- SupCua: D2=4, D3=2

## Ghi chú

- MAPE (target-matched) là số frozen từ `summary_comparison.p2_mape` — tính trên TOÀN BỘ ảnh nhóm (gồm cả ảnh không phải D1). Hai bảng MAPE con (D1-only / exact-only) tính từ `rel_err_p2_pct` của các subset tương ứng — phục vụ so sánh MAPE|all vs MAPE|detect-đúng.
- ⚠️ Lưu ý nhất quán: báo cáo frozen `data/benchmark_portion_report.md` nêu ví dụ D2 "Súp cua → Canh", nhưng JSON audit xếp các ảnh SupCua chỉ có `Canh (Soup)` vào **D1** (D2 của SupCua là các ảnh chỉ thấy Thit heo/Dau hu/Banh mi). Số count (34/12/5/1/2) là chuẩn theo JSON + test; ví dụ văn xuôi trong báo cáo là minh họa, không đụng vào.
- Overall Chain B: P1 = 51.6% · P2 = 42.9% (population 54 ảnh).
