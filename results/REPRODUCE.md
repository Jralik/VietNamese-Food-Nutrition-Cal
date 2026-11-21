# REPRODUCE — Chạy lại thực nghiệm của khóa luận

> Provenance: số liệu Chương 4 frozen tại commit `18cd148` (xem `results/frozen_numbers.md` §1).
> Chạy lại các script dưới đây trên pipeline **hiện tại (production, post-freeze)** có thể cho số lệch nhẹ do pipeline đã phát triển — đó là **post-freeze evaluation**, không dùng để thay số Chương 4.

## 1. Yêu cầu môi trường & tài sản ngoài repo

- Python 3.10+, GPU CUDA 12.1 cho pipeline đầy đủ (đo frozen: **34.63 s/ảnh**); detection-only chạy CPU được (v10b: 1270 ms/ảnh).
- Env lock: `pip install -r requirements-lock.txt` (freeze từ `.venv` dự án) hoặc `requirements.txt` (chưa pin).
- **SAM 2.1**: checkpoint + config `checkpoints/sam2.1_hiera_small.pt`, `configs/sam2.1/sam2.1_hiera_small.yaml` theo `pipeline_config.py:148-149` — tải từ Meta SAM2 release, đặt đúng đường dẫn.
- **Depth Anything V2**: `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (tự tải qua HuggingFace Hub).
- **FoodSAM** (backend tuỳ chọn): repo ngoài, path trong `pipeline_config.py:22` (`FOODSAM_REPO`).
- **Camera calibration**: `Camera calibration/camera_calib.npz`.
- Ảnh Ground-Truth: bản sao trong `gt_images/` (JSON GT có `rel_path`); bản gốc OneDrive (historical provenance, giữ trong JSON `path`).

## 2. Bộ test acceptance (không cần GPU, không cần pytest)

```bash
python test_gt_parser.py                      # parser GT: semantics G = TỔNG của K box — 74 targets
python tests/test_phase1_acceptance.py        # Atwater / basis
python tests/test_phase2_acceptance.py        # mapping confidence
python tests/test_phase3_acceptance.py        # NIN canonical DB
python tests/test_phase4_acceptance.py        # P4-A..D: frozen benchmark 51.6/42.9/15.4 + taxonomy
python tests/test_phase5_acceptance.py
python tests/test_phase6_cross_class_suppression.py
python tests/test_phase7_e2e_duplicate_regression.py
```

Lưu ý interpreter: trên máy phát triển dùng `.venv\Scripts\python.exe` (python hệ thống thiếu `av`).

## 3. Chuỗi đánh giá (detector)

```bash
python eval_val_detection.py      # recall trên val txt: 3 config × threshold sweep (phase D)
python eval_phase_d.py            # bảng phase_d_results (v26m/v10b/v10b+raw @0.30/0.40/0.50)
python eval_model_compare.py      # model_compare.md (v26m ONNX vs v10b .pt, 54 ảnh)
python benchmark_ab_preprocess.py # 5 config preprocess (stretch/letterbox/dual-union/…)
```

## 4. Pipeline đầy đủ (cần GPU + checkpoint SAM2)

```bash
python eval_full_pipeline.py      # Chain A: 54 ảnh, matched-only MAPE, nutrition, scale source
python scripts/benchmark_portion_ground_truth.py   # Chain B: dual-protocol P1/P2 + audit JSON (v26m ONNX)
python scripts/make_class_metrics_table.py         # bảng derived coverage/MAPE (chỉ đọc audit JSON, không cần GPU)
python summarize_full_eval.py     # tổng hợp full_eval_records.jsonl → báo cáo
```

## 5. Ablation post-freeze

```bash
python experiment_a_density.py    # density sensitivity sweep (85.2 → 51.2%, plateau ~51%)
python experiment_b_mask.py       # mask ablation cconly (84.8 → 61.6%)
python experiment_d_rimcap.py     # rim-plane ablation (NEGATIVE result — Experiment D)
```

## 6. Thứ tự sản phẩm → file kết quả

| Lệnh | Sản phẩm frozen |
|---|---|
| `eval_phase_d.py` | `scratch/phase_d_results.md` |
| `eval_full_pipeline.py` | `scratch/full_pipeline_eval.md` + `scratch/full_eval_records.jsonl` |
| `scripts/benchmark_portion_ground_truth.py` | `data/benchmark_portion_report.md` + `data/benchmark_portion_results.json` + audit JSON |
| `experiment_a/b/d` | `scratch/experiment_*.md` + CSV/PNG cho thesis |
| `scripts/make_class_metrics_table.py` | `results/class_metrics.md` |

## 7. Quy tắc khi chạy lại

1. **Không ghi đè** các báo cáo frozen trong `scratch/` và `data/` — kết quả chạy lại ghi ra thư mục/nhãn *post-freeze* riêng.
2. Không "cập nhật" số Chương 4 bằng kết quả mới; so sánh với frozen baseline thay vào đó.
3. Demo Streamlit Cloud = UI + detection + nutrition + chatbot; KHÔNG pipeline nặng.
