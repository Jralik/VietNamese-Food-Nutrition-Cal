# Chapter 4 Outline — mapping tiết → nguồn số liệu frozen

> Mọi số trích từ `results/frozen_numbers.md` (SSOT). File này chỉ mapping, không chứa số mới.
> Viết theo narrative: **Pilot → phát triển hệ thống → evaluation protocol → freeze → đánh giá chính thức → error analysis → controlled ablation → limitations.**

## 4.1. Thiết kế thực nghiệm & Ground-Truth protocol
- Pipeline đánh giá: v10b + dual-preresize @0.30 → SAM2 volume daemon (ArUco scale) → nutrition. Nguồn: `scratch/full_pipeline_eval.md:4`.
- GT: 54 ảnh / 74 targets, cân thực tế, 7 nhóm món; semantics "G = TỔNG của K box" (user-confirmed 2026-09-27), `(tong)` không chia K, `?` count-only. Nguồn: `eval_full_pipeline.py:47-56`, `test_gt_parser.py`.
- Hai protocol đối soát (P1 full-image / P2 target-matched) — lý do: side-dish interference. Nguồn: `data/benchmark_portion_report.md:12-15`.
- Bảng: phân bổ 54 ảnh theo nhóm (Pho 5, BanhCuon 6, ComSuon 5, BunBoHue 5, BanhMi 16, BunRieu-CanhBun 7, SupCua 10) — `data/real_images_ground_truth.json`.

## 4.2. Huấn luyện & đánh giá detector trên val split
- Official run: YOLOv26m `yolo26m_vietfood67_sgd` — P 0.921 / R 0.910 / mAP@0.50 0.951 / mAP@0.50:0.95 0.829 (val 6602 ảnh/11582 instances). Nguồn: SSOT §3.1, notebook.
- Baseline prior-work: VietFood67 + YOLOv10b mAP50 = 0,92 (đề cương [1]) → kết quả tiệm cận baseline, KHÔNG tuyên bố vượt SOTA.
- Yêu cầu §2.8 đề cương: báo cả mAP@0.5 + mAP@0.5:0.95 — đã đáp ứng (điểm khác với README cũ chỉ badge mAP50).
- Ghi chú trung thực: 0.949/0.826 là earlier epoch — historical.

## 4.3. Đánh giá toàn hệ thống trên 54 ảnh thật (Chain A — OFFICIAL)
- Detection: recall 0.811 (60/74) @0.30, 0-box 0/54, FP 36; per-class recall @0.30 (Banh mi 20/20, Rau 6/8, Banh cuon 0/6…). Nguồn: `scratch/phase_d_results.md`, `scratch/full_pipeline_eval.md:12-25`.
- Portion: matched-only mass MAPE **44.9%** (54 ảnh), MAE 198.2 g; per-class bảng (Com tam 6.1% → Canh 200.9%); total-per-image 43.8% (53 ảnh); unmatched extras 7609 g. Nguồn: SSOT §3.3.
- Scale source: aruco 49 / plate_heuristic 4 / không xác định 1 — tiền đề cho limitation §4.8.
- Nutrition: kcal matched-only 45.7% (54 ảnh); total-per-image 33.2% (37 ảnh đủ DB). Nguồn: SSOT §3.5.

## 4.4. RQ3 — Domain shift giữa val split và deployment (kết quả ÂM — điểm nhấn)
- Bảng domain-shift: SSOT §3.2. YOLO26m val mAP@0.50 0.951 nhưng Recall@0.50 thật 0.514 (pooled) / 0-box 12/54; YOLOv10b 0.730 / 2/54 @0.50, 0.811 / 0/54 @0.30.
- Thông điệp: mAP trên split random KHÔNG dự báo hiệu năng ảnh thật; production giữ YOLOv10b.
- Historical tham chiếu: `scratch/model_compare.md` (0.546 vs 0.759, aggregation khác — ghi rõ).
- Threshold sweep: 3 config × 3 threshold (0.30/0.40/0.50) — `scratch/phase_d_results.md`.

## 4.5. Phân rã lỗi hai tầng (Chain B — bằng chứng causal)
- Taxonomy D1-D5 tương hỗ (D1 63.0%, D2 22.2%, D3 9.3%, D4 1.9%, D5 3.7%) + E1-E3 đa nhãn (E1 18.5%, E2 3.7%, E3 11.1%). Nguồn: SSOT §3.4, `data/benchmark_error_attribution_audit.json`.
- Case study 1 — Cơm sườn: P1 127.1% → P2 15.4% (side-dish interference; GT chỉ cân đĩa cơm). Nguồn: `data/benchmark_portion_report.md:36-46`.
- Case study 2 — Phở: nút thắt ở detector (nhầm Bun mam, fragmentation Canh/Bun). Nguồn: `data/benchmark_portion_report.md:48-57`.
- Case study 3 — Bánh cuốn: target-class 0/6 ở MỌI config (5 preprocess configs + 3 model configs) → MAPE 67.5% tính trên nhầm lớp `Cha` (D2), KHÔNG phải lỗi thể tích. Trình bày: coverage 0%, MAPE N/A trong bảng chính + auxiliary D2 error. Nguồn: `results/class_metrics.md` (derived), `scratch/ab_preprocess_results.md:247,292`.
- Lan truyền dinh dưỡng tuyến tính (consistency): |N̂−N| = c_N·|m̂−m|. Nguồn: SSOT §3.5.

## 4.6. Ablation có kiểm soát (post-freeze, nhãn riêng)
- Exp A density sensitivity: 85.2% → 51.2%, plateau ~51% (ρ<0.30) — density có contribution, không giải thích hết. Hình: `scratch/exp_a_density_sweep.png`, bảng `scratch/experiment_a_density_sweep.csv`.
- Exp B mask ablation (cconly): 84.8% → 61.6%; over-filtering loại. Hình: `scratch/exp_b_mask_ablation.png`.
- Exp C1 capture geometry (exploratory): sai số phụ thuộc cấu hình hình học quan sát; C2 không bắt buộc.
- Exp D rim-plane: NEGATIVE result — root cause = depth đọc cavity (Súp cua oblique).
- Nguyên tắc: A/B độc lập, KHÔNG cộng/trừ thành decomposition (m = V×ρ nhân nhau). Nguồn: `scratch/evaluation_summary.md`.

## 4.7. Density Database — provenance & trạng thái
- 67 entry: literature 58 · usda_fdc 8 · user_measured_gt 1; std 0.05-0.12 là assumption (không có provenance đo). Nguồn: `density_db.py:37-118`.
- Water displacement (Core Target #2): CHƯA thực hiện tại thời điểm freeze → limitation + kế hoạch `results/water_displacement_protocol.md` (post-freeze evaluation riêng, không thay số Ch4).

## 4.8. Limitations (viết thẳng, trung thực)
1. GT 54 ảnh (mục tiêu đề cương ~100) — đủ cho MAPE tổng hợp, mỏng cho phân tích per-class (n=1-2 ở Trung/Canh).
2. Density chủ yếu literature; std chưa đo.
3. Mục tiêu #5 [LÕI]: MobileSAM chưa khảo sát; chưa có mIoU/Dice (SAM2 là ứng viên MỞ RỘNG được chọn).
4. 5/54 ảnh Chain A không có ArUco vẫn được ước lượng (4 plate_heuristic + 1 không xác định) — lệch chính sách §1.6.4 đề cương; các ảnh này chưa benchmark riêng.
5. DAv2 bản vits (Small, metric-indoor); monocular depth có 3 failure mode (over Bánh mì ~2.9×, under Trứng ~3×, geometry-dependent Súp cua).
6. Chain B chạy trên pipeline pre-freeze (v26m ONNX) — số dual-protocol/taxonomy là bằng chứng phân tích, không phải hiệu năng production.

## Hình/bảng sẵn dùng cho thesis
`scratch/experiment_a_density_sweep.csv`, `scratch/exp_b_mask_ablation.csv`, `scratch/supcua_geometry_c1.csv`, `scratch/detection_per_class.csv`, `scratch/exp_a_density_sweep.png`, `scratch/exp_b_mask_ablation.png`, `scratch/annotated_final/*.jpg` (case study), `data/benchmark_portion_metrics.csv`.
