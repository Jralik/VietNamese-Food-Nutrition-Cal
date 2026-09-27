# Evaluation summary — trạng thái FROZEN (2026-09-27)

Master index cho toàn bộ chuỗi đánh giá pipeline. Mọi số liệu trong các báo cáo con
đã được review nhất quán theo checklist cuối trang.

## Câu chuyện (frozen narrative)

```text
Frozen baseline (v10b + dual-preresize, slider 0.30, volume SAM2 GPU)
      │
      ├── Detection recall = 0.811 (60/74), threshold = 0.30 · FP 36 · 0-box 0/54
      │
      ├── Experiment A — Density sweep (offline, volume frozen)
      │     └── density CÓ contribution: MAPE Bánh mì 85.2% → 51.2%
      │         nhưng ĐI NGANG ~51% khi ρ < 0.30 → không giải thích hết
      │         (vùng sensitivity ρ = 0.25–0.30, chưa phải calibration)
      │
      ├── Experiment B — Mask ablation (cconly, mọi thứ khác frozen)
      │     └── mask CÓ contribution: area −11% → volume −9% → mass −8%
      │         MAPE Bánh mì 84.8% → 61.6%; over-filtering bị loại (otsu/fixed60cc/fixed80cc)
      │         nhưng KHÔNG giải thích toàn bộ residual
      │
      └── cconly mask filter → PROPOSED FUTURE IMPROVEMENT (chưa vào production)
      └── Experiment C1 — Capture/geometry exploratory ✅ FROZEN (existing data)
            └── Sai số phụ thuộc vào CẤU HÌNH HÌNH HỌC QUAN SÁT + độ ổn định của
                depth-anchor correction, KHÔNG chỉ góc chụp nghiêng (phân loại nhị phân):
                tilted có thể chính xác (-2.2%) và top-down có thể sai (+79%)
            └── C2 controlled capture → KHÔNG bắt buộc (chỉ khi dư >1-2 tuần)
```

Hai ablation A/B là **độc lập** — không cộng/trừ thành decomposition "density = X%,
mask = Y%" (m = V×ρ nhân nhau; chưa có baseline độc lập cho volume/density thật).

## Ba failure mode (không gộp)

| Case | Failure mode | Evidence |
|---|---|---|
| Bánh mì thường | Over-estimation ~2.9× | Experiment A (density sensitivity) + Experiment B (mask bleed — quan sát trực tiếp) |
| Trứng (đĩa ốp la 3-4 trứng + dầu) | Under-estimation ~3× | Mask đúng; **limitation của monocular depth estimation** với thức ăn nhiều lớp mỏng |
| Súp cua | Geometry/observation-configuration dependent (-2% top-down / +79-158% oblique) | Depth discontinuity tại biên hộp + độ ổn định anchor correction (hypothesis có audit) |

Pipeline KHÔNG có bias đơn hướng "luôn cao"/"luôn thấp" — ba failure mode là ba
bản chất lỗi riêng biệt (over / under / geometry-dependent), mỗi loại gắn với một
cơ chế ước lượng khác nhau của monocular-depth pipeline.

## Bảng evidence → điều mỗi bằng chứng chứng minh

| Evidence | Chứng minh được | KHÔNG chứng minh |
|---|---|---|
| Recall 0.811 @ 0.30 | baseline detection performance trên val set | quality tuyệt đối (chỉ 1 tập, 1 ngưỡng) |
| 0-box 0/54 | không có ảnh bị bỏ hoàn toàn | chất lượng box |
| Isolation 54/54 identical | metric change đến từ GT/evaluator, không phải model | model tốt hơn |
| Portion 6-13% (bún/phở/cơm) | volume/mass tương đối ổn cho nhóm này | chính xác trên mọi loại món |
| 61-85% (bánh mì/trứng) | tồn tại failure mode lớn | nguyên nhân (cần A/B) |
| Exp A: 85.2 → 51.2% | density có ảnh hưởng đáng kể đến mass error | 0.25-0.30 là density thực (plateau ~51%) |
| Exp B: cconly 84.8 → 61.6% | mask degradation truyền xuống area → volume → mass | mọi residual do mask (còn depth/geometry) |
| C1 | geometry/capture condition có tín hiệu liên hệ với sai số | quan hệ nhân quả (n nhỏ, exploratory) |

## Các báo cáo con (file map)

| File | Nội dung | Trạng thái |
|---|---|---|
| `phase_d_results.md` | Detection sweep 3 config × 3 threshold | frozen |
| `full_pipeline_eval.md` | Full-pipeline 54 ảnh: detection/portion/nutrition | frozen |
| `experiment_a_density.md` | Density sensitivity sweep | frozen |
| `experiment_b_mask.md` | Mask ablation per-variant + per-image | frozen |
| `depth_audit.md` | Root-cause audit (Súp cua/Trứng/Bánh mì) + A/B kết quả | frozen |
| `ab_preprocess_results.md` | Historical (pre-Experiment-A chẩn đoán) | tham chiếu |
| `full_eval_records.jsonl` (+ `.bak_gt_old`, `.bak_pre_volume`) | Dữ liệu thô + baseline backups | frozen |
| `experiment_c1_supcua.md` | C1 capture-geometry exploratory (Súp cua) | frozen |
| `experiment_a_density_sweep.csv` / `exp_b_mask_ablation.csv` / `supcua_geometry_c1.csv` / `detection_per_class.csv` | Bảng CSV cho thesis | frozen |
| `exp_a_density_sweep.png` / `exp_b_mask_ablation.png` | Hình cho thesis | frozen |

## Consistency checklist (đã review 2026-09-27)

- [x] GT: 74 targets, gram = TỔNG của K box (user-confirmed); parser unit-tested
      (`(tong)` không chia K · `?` count-only · legacy formats) — test_gt_parser.py ALL PASS
- [x] Detection threshold = 0.30 ghi cạnh mọi bảng recall (0.811 @0.30; sàn 0.40
      của volume path được ghi chú riêng khi liên quan)
- [x] Frozen inference: 54/54 est identical trước/sau GT normalization
      (isolation proof — full_eval_records.jsonl.bak_gt_old)
- [x] Density baseline = 0.45 (không đổi); ρ = 0.25-0.30 chỉ là sensitivity region
- [x] Matching rule: class-level GT + greedy one-to-one theo confidence;
      MAPE population ghi rõ trong từng báo cáo
- [x] Experiment A wording: "contribution đáng kể, không giải thích toàn bộ" —
      KHÔNG gọi 0.25-0.30 là density thực
- [x] Experiment B wording: "mask contribution confirmed, partial" — KHÔNG nói
      residual "không còn do mask"; 13-70% = mask lớn so với ảnh, KHÔNG = toàn bộ bleed
- [x] Hai ablation độc lập — không decomposition cộng-trừ
- [x] Volume worker GPU (env fix) · raw branch đã loại (vị trí FP) · cconly chưa vào production

## Bước tiếp theo (khi bạn sẵn sàng)

- **C1**: exploratory analysis trên 10 ảnh Súp cua hiện có (top-down vs tilted) — chi phí 0
- **C2**: controlled capture nếu C1 cho tín hiệu nhất quán
- Sau C: cân nhắc pipeline changes (density/mask) bằng experiment riêng có regression
