# Frozen Numbers — Single Source of Truth cho Chương 4

> **Trạng thái:** SSOT publication-facing. Chương 4 CHỈ lấy số từ file này.
> Tầng trên: `scratch/evaluation_summary.md` (master evidence index của chuỗi đánh giá).
> Tầng dưới: `results/chapter4_outline.md` (mapping tiết → số/bảng/hình).
> Mọi số dưới đây đã được đối chiếu ngược với file nguồn (cột Source). Bảng derived (class_metrics) sinh bằng script, không gõ tay.

## 0. Nguyên tắc SSOT

1. **Không sửa số liệu để các báo cáo khớp nhau.** Chỉ sửa báo cáo sai provenance/label/định nghĩa. Hai số cùng đúng nhưng khác population/protocol → giữ cả hai, ghi rõ phạm vi.
2. Một run/metric chỉ có MỘT giá trị official; các giá trị khác cùng metric chỉ được nhắc với nhãn **historical**.
3. **Frozen evaluation (`18cd148`) và production pipeline (`8427cc1`+) là hai trạng thái provenance khác nhau.** Không dùng post-freeze code để tái tạo/thay thế frozen numbers; không re-run để "cập nhật" Ch4.
4. Post-freeze work (Experiment D, density đo mới, duplicate suppression) → evaluation riêng có nhãn post-freeze, so sánh với frozen baseline, không ghi đè.

## 1. Provenance

```text
18cd148
  └── Official evaluation freeze → NGUỒN SỐ CHÍNH THỨC Chương 4
      (isolation proof: 54/54 est identical trước/sau GT normalization)

8427cc1
  └── Post-freeze repository checkpoint → production code KHÔNG ĐỔI từ đây đến HEAD
      (đưa data/, tests/, scripts/, nutrition_rule_engine.py, eval_*.py vào git;
       vendored ultralytics/nn/modules/block.py +238 dòng — module YOLO11/26)

25575ee → 98a38bd
  └── Post-freeze analysis/documentation
      (cross-class duplicate suppression, Experiment D rim-plane, audit reports)
```

- **Production inference = YOLOv10b `.pt`** (`utils.py:295`, `main.py:402 load_model()`).
- **`model/yolov26/best.onnx` chỉ dùng bởi eval scripts lịch sử** (`utils.py:1670` là dead code trong app; `scripts/benchmark_portion_ground_truth.py:57`).
- README cũ ghi "YOLOv26 + mAP50 0.95" — đã đính chính: model production là YOLOv10b; YOLOv26m chỉ là nghiên cứu so sánh (RQ3, kết quả âm).

## 2. Hai chuỗi đánh giá — vai trò trong Chương 4

| | **Chain A — OFFICIAL (Ch4 chính)** | **Chain B — Historical (bằng chứng phân tích)** |
|---|---|---|
| Pipeline | v10b `.pt` + dual-preresize @ conf=0.30 → SAM2 volume daemon (ArUco scale) | `model/yolov26/best.onnx` (script `scripts/benchmark_portion_ground_truth.py:57`; docstring file ghi "YOLOv11" là stale) + DAv2 + ArUco, CUDA 12.1 |
| GT | 74 targets / 54 ảnh, semantics "G = TỔNG của K box" (user-confirmed 2026-09-27, `eval_full_pipeline.py:47-56`) | JSON `data/real_images_ground_truth.json` (sinh từ txt phiên bản cũ; GT = khối lượng món mục tiêu) |
| Matching | Class-level + greedy one-to-one theo confidence; extras ghi riêng | Dual-protocol: P1 full-image / P2 target-matched |
| Vai trò Ch4 | Số liệu định lượng chính thức (detection, portion, nutrition) | Phân rã lỗi hai tầng D1-D5/E1-E3 + case study Cơm sườn + justification Protocol 2 |
| Regression lock | `test_gt_parser.py` | `tests/test_phase4_acceptance.py` (assert 51.6/42.9/15.4) |

⚠️ Hai chuỗi KHÔNG so sánh chéo số (pipeline, GT, matching khác nhau). Mỗi bảng ghi rõ thuộc chuỗi nào.

## 3. Bảng số liệu

Schema: `Metric | Type | Value | Population | Protocol | Source | Frozen` — Type ∈ {performance, consistency}.

### 3.1 Detection trên val split (official run)

| Metric | Type | Value | Population | Protocol | Source | Frozen |
|---|---|---|---|---|---|---|
| mAP@0.50 | performance | **0.951** | val 6602 ảnh / 11582 instances | YOLOv26m `yolo26m_vietfood67_sgd`, run hoàn chỉnh cuối ("6 epochs completed in 11.010 hours", Kaggle), data=/kaggle/working/data.yaml | `yolo26-food-detection-train2.ipynb` (header `Box(P R mAP50 mAP50-95)` kề dòng `all 6602 11582 0.921 0.910 0.951 0.829`) | YES |
| mAP@0.50:0.95 | performance | **0.829** | như trên | như trên | như trên | YES |
| Precision / Recall (val) | performance | 0.921 / 0.910 | như trên | như trên | như trên | YES |
| Baseline prior-work | performance | mAP50 = 0,92 | VietFood67 + YOLOv10b | — | `docx_extracted.txt:36,117` (đề cương, [1]) | ref |

*Historical (KHÔNG dùng làm số chính): 0.949/0.826 — epoch đầu của chuỗi validation, chỉ nhắc kèm nhãn "earlier epoch/run".*

### 3.2 Detection trên 54 ảnh thật (Chain A) + RQ3 domain-shift

| Metric | Type | Value | Population | Protocol | Source | Frozen |
|---|---|---|---|---|---|---|
| Detection recall | performance | **0.811 (60/74)** | 74 targets / 54 ảnh | v10b-dual @ conf=0.30, class-level + greedy one-to-one | `scratch/phase_d_results.md:13`, `scratch/full_pipeline_eval.md:12` | YES |
| 0-box rate | performance | **0/54** | 54 ảnh | như trên | như trên | YES |
| FP/extras | performance | 36 | như trên | như trên | `scratch/phase_d_results.md:13` | YES |
| Recall v10b-dual @0.50 | performance | 0.730 (54/74), 0-box 2/54 | 74 targets / 54 ảnh | như trên, @0.50 | `scratch/phase_d_results.md:11` | YES |
| Recall v26m-dual @0.30 | performance | 0.541 (40/74), 0-box 4/54 | 74 targets / 54 ảnh | v26m-dual @0.30 | `scratch/phase_d_results.md:10` | YES |
| Recall v10b-dual+raw @0.30 | performance | 0.824 (61/74) | như trên | ablation đã tài liệu hóa (raw branch bị loại: raw-only boxes = position FP) | `scratch/phase_d_results.md:16` | YES |
| Latency detection | performance | 1270 ms/ảnh (v10b) · 839 ms/ảnh (v26m) | CPU runtime, inference một lần @0.05 | `scratch/phase_d_results.md:4,8-16` | YES |

**Bảng domain-shift (RQ3) — so sánh val vs ảnh thật:**

| Config | val mAP@0.50 | Recall@0.50 ảnh thật (pooled target-level) | 0-box @0.50 | Recall@0.30 pooled |
|---|---|---|---|---|
| YOLOv26m-dual | 0.951 (official val run) | 0.514 (38/74) | 12/54 | 0.541 (40/74) |
| YOLOv10b-dual | (baseline tái lập đề cương: 0,92 [1]) | 0.730 (54/74) | 2/54 | **0.811 (60/74)** |
| YOLOv10b-dual+raw | — | 0.743 (55/74) | 2/54 | 0.824 (61/74) |

Nguồn: `scratch/phase_d_results.md` + notebook. Kết luận RQ3 (âm): **YOLO26m không cải thiện YOLOv10b trên ảnh deployment dù mAP val cao hơn — domain shift giữa split random và điều kiện chụp thực tế.**

*Historical, aggregation khác (không so sánh trực tiếp với bảng trên): `scratch/model_compare.md` — Recall@0.50 0.546 (v26m) vs 0.759 (v10b), 12/54 vs 2/54 ảnh 0-box, tổng hợp theo ảnh/class; giữ làm tham chiếu historical.*

### 3.3 Portion — Chain A (OFFICIAL)

| Metric | Type | Value | Population | Protocol | Source | Frozen |
|---|---|---|---|---|---|---|
| Mass MAPE (matched-only) | performance | **44.9%** | 54 ảnh | matched-only (top-k box khớp GT counts), extras ghi riêng | `scratch/full_pipeline_eval.md:75` | YES |
| Mass MAE (matched-only) | performance | 198.2 g | 54 ảnh | như trên | như trên | YES |
| Mass MAPE (total-per-image) | performance | 43.8% | **53 ảnh** (1 ảnh không có tổng) | cộng dồn mọi box/ảnh | `scratch/full_pipeline_eval.md:41` | YES |
| Unmatched extras mass | performance | 7609 g tổng cộng dồn | 54 ảnh | Chain A matching | `scratch/full_pipeline_eval.md:41` | YES |

**Per-class mass MAPE (matched-only, Chain A)** — nguồn `scratch/full_pipeline_eval.md:63-73`:

| Class | n | GT mass (g) | Matched MAE (g) | Matched MAPE | Extras (g) |
|---|---|---|---|---|---|
| Com tam | 5 | 1952 | 23.9 | 6.1% | 0 |
| Bun bo Hue | 5 | 4022 | 88.8 | 11.0% | 0 |
| Bun rieu | 6 | 5033 | 101.5 | 12.1% | 0 |
| Pho | 5 | 4098 | 179.7 | 21.9% | 0 |
| Rau | 4 | 360 | 30.0 | 33.3% | 0 |
| Sup cua | 7 | 3324 | 289.5 | 61.3% | 0 |
| Trung | 2 | 442 | 147.6 | 66.7% | 0 |
| Banh mi | 16 | 3263 | 123.5 | 84.1% | 128 |
| Canh | 1 | 150 | 301.4 | 200.9% | 0 |

Per-group (theo nhóm ảnh, `scratch/full_pipeline_eval.md:100-107`): BanhMi 38.0% (n=14) · BunBoHue 8.0% (4) · BunRieu 11.4% (4) · ComSuon 25.6% (3) · Pho 21.9% (5) · SupCua 61.3% (7).

Scale source (Chain A): `aruco_marker` 49 · `plate_heuristic` 4 · không xác định 1 (53 requests) → 5/54 ảnh không có marker vẫn được ước lượng (liên quan limitation §1.6.4, xem chapter4_outline).

### 3.4 Portion — Chain B (historical, dual-protocol + taxonomy)

| Metric | Type | Value | Population | Protocol | Source | Frozen |
|---|---|---|---|---|---|---|
| Mass MAPE — Protocol 1 (full-image) | performance | 51.6% | 54 ảnh | P1 = Σ mọi box | `data/benchmark_portion_report.md:26` + audit JSON metadata | YES |
| Mass MAPE — Protocol 2 (target-matched) | performance | 42.9% | 54 ảnh | P2 = chỉ món mục tiêu | như trên | YES |
| ComSuon P1 → P2 | performance | 127.1% → 15.4% | 5 ảnh ComSuon, GT = đĩa cơm sườn (390.5 g) | dual-protocol | `data/benchmark_error_attribution_audit.json` summary_comparison | YES |
| BanhCuon (CẢ HAI protocol) | performance | 67.5% | 6 ảnh, target-class 0/6 — tính trên detection nhầm lớp `Cha` (D2) | dual-protocol | `data/benchmark_portion_report.md:24` | YES |
| Latency pipeline | performance | 34.63 s/ảnh | CUDA 12.1, SAM2 + DAv2 + ArUco | full pipeline | `data/benchmark_portion_report.md:7` | YES |

**Taxonomy D1-D5 (Tầng 1, tương hỗ, tổng 54):** D1 Correct 34 (63.0%) · D2 Misclassified 12 (22.2%) · D3 Missed/Partial 5 (9.3%) · D4 Fragmented 1 (1.9%) · D5 Duplicate 2 (3.7%).
**Taxonomy E1-E3 (Tầng 2, đa nhãn):** E1 Side-dish interference 10 (18.5%) · E2 Target-association mismatch 2 (3.7%) · E3 Geometric/Depth 6 (11.1%).
Nguồn: `data/benchmark_portion_report.md:74-82` + `data/benchmark_error_attribution_audit.json` tier counts; asserted trong `tests/test_phase4_acceptance.py:136-146`.

### 3.5 Nutrition — Chain A

| Metric | Type | Value | Population | Protocol | Source | Frozen |
|---|---|---|---|---|---|---|
| Kcal MAPE (matched-only) | performance | 45.7% | 54 ảnh | matched-only | `scratch/full_pipeline_eval.md:91` | YES |
| Kcal MAPE (total-per-image) | performance | 33.2% | 37 ảnh có đủ DB dinh dưỡng | cộng dồn mọi box | `scratch/full_pipeline_eval.md:57` | YES |
| Lan truyền dinh dưỡng tuyến tính | consistency | \|N̂−N\| = c_N·\|m̂−m\| khớp trên các ảnh có GT dương (c_N = N_100g/100) | các ảnh có mass + kcal | hệ số thành phần cố định | `data/benchmark_portion_report.md:88-90`, `tests/test_phase4_acceptance.py:89-111` | YES |

### 3.6 Experiments post-freeze (nhãn post-freeze — KHÔNG cộng/trừ vào Ch4)

| Metric | Type | Value | Nguồn |
|---|---|---|---|
| Exp A — density sensitivity | consistency | MAPE Bánh mì 85.2% → 51.2%, plateau ~51% khi ρ<0.30 — density có contribution, KHÔNG giải thích hết | `scratch/experiment_a_density.md` |
| Exp B — mask ablation (cconly) | consistency | 84.8% → 61.6%; mask có contribution, partial | `scratch/experiment_b_mask.md` |
| Exp C1 — capture geometry | consistency | Sai số phụ thuộc cấu hình hình học quan sát (tilted có thể −2.2%, top-down có thể +79%) — exploratory, n nhỏ | `scratch/experiment_c1_supcua.md` |
| Exp D — rim-plane | consistency | NEGATIVE result: không sửa được Súp cua oblique; root cause = depth đọc cavity thay vì mặt chất lỏng | `scratch/experiment_d_results.md` |

Hai ablation A/B ĐỘC LẬP — không cộng/trừ thành decomposition "density = X%, mask = Y%" (m = V×ρ nhân nhau).

## 4. Consistency notes (các chỗ dễ bị bắt lỗi — đã phân loại theo rule §0)

1. **0.949/0.826 vs 0.951/0.829**: hai validation epoch khác nhau CÙNG notebook. Official = run hoàn chỉnh cuối (0.951/0.829). 0.949/0.826 = historical.
2. **42.9% vs 44.9%**: hai chuỗi khác nhau (Chain B dual-protocol vs Chain A matched-only), pipeline + GT + matching khác nhau. Không phải mâu thuẫn — Ch4 phải nêu cả hai với nhãn chuỗi.
3. **0.546/0.759 (model_compare) vs 0.514/0.730 (phase_d @0.50)**: cùng metric tên giống nhau nhưng aggregation khác (per-image/class vs pooled target-level). Giữ cả hai, không so sánh chéo, ghi định nghĩa theo file nguồn.
4. **43.8% (53 ảnh) vs 44.9% (54 ảnh)**: hai population khác nhau trong cùng Chain A (total-per-image thiếu 1 ảnh vs matched-only đủ 54). Ghi rõ population khi trích.
5. **README cũ "YOLOv26, mAP50 0.95"**: đã đính chính — production là YOLOv10b; 0.951 là mAP@0.50 của YOLOv26m trên val split (official run), không phải mô tả hệ thống shipped.
6. **Docstring `benchmark_portion_ground_truth.py` ghi "YOLOv11"**: stale — code thực tế load `model/yolov26/best.onnx` (:57). Đã ghi chú, không sửa file historical.
7. **`evaluation_summary.md:56` "Portion 6-13% (bún/phở/cơm)"**: wording imprecise (Phở = 21.9%) — đã sửa label thành "(cơm 6.1%, bún bò 11.0%, bún riêu 12.1%; phở 21.9%)" theo rule chỉ sửa label/định nghĩa.
8. **Semantics của D1 (Chain B)**: D1 = "vùng món mục tiêu được phát hiện", KỂ CẢ nhãn generic/sibling (SupCua D1 chỉ có `Canh (Soup)`, ComSuon D1 đa số `Com (Rice)`, BunRieu có ảnh D1 nhãn `Bun bo Hue`). Phép đo nghiêm ngặt (exact-class string) cho 25/54 so với D1 34/54 — cả hai trình bày cạnh nhau trong `results/class_metrics.md`. Văn bản báo cáo frozen nêu ví dụ D2 "Súp cua → Canh" trong khi JSON xếp SupCua+Canh vào D1 — ví dụ minh họa imprecise, count 34/12/5/1/2 theo JSON + test là chuẩn; KHÔNG sửa báo cáo frozen.
