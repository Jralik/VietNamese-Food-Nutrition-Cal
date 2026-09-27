# 🇻🇳 Vietnamese Food Detection & Portion Estimation (VietNamese-Food-Nutrition-Cal)

[![Live Demo](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://vietnamese-food-nutrition-cal.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Production-YOLOv10b-green.svg)](https://github.com/Jralik/VietNamese-Food-Nutrition-Cal)
[![mAP@0.50](https://img.shields.io/badge/mAP@0.50_/_mAP@0.50:0.95-0.951_/_0.829-brightgreen.svg)](results/frozen_numbers.md)

Khóa luận: **hệ thống nhận diện món ăn Việt Nam và ước lượng khối lượng khẩu phần** bằng pipeline thị giác máy tính 3D — ArUco marker + SAM 2 (segmentation) + Depth Anything V2 (metric depth) + Density Database — kế tiếp bước detection và tầng tính toán dinh dưỡng, kèm **hệ thống phân rã lỗi hai tầng** (D1-D5 / E1-E3) trên bộ Ground-Truth 54 phần ăn cân thực tế.

> 📌 **Số liệu chính thức của luận văn** được đóng băng và truy ngược tại [`results/frozen_numbers.md`](results/frozen_numbers.md). Hướng dẫn chạy lại thực nghiệm: [`results/REPRODUCE.md`](results/REPRODUCE.md).

## 🔬 Kết quả frozen (tóm tắt)

| Metric | Giá trị | Ghi chú |
|---|---|---|
| mAP@0.50 / mAP@0.50:0.95 (val split) | 0.951 / 0.829 | YOLOv26m, 6602 ảnh / 11582 instances — baseline VietFood67 [1] là 0,92 |
| Detection recall trên 54 ảnh thật | **0.811** (60/74 targets) @ conf 0.30, 0-box 0/54 | production model **YOLOv10b** |
| Mass MAPE (matched-only, 54 ảnh thật) | **44.9%** (MAE 198.2 g) | Chain A — chuỗi đánh giá chính thức |
| Mass MAPE dual-protocol | 51.6% (P1 full-image) / 42.9% (P2 target-matched) | Chain B — benchmark historical, phục vụ phân rã lỗi |
| Kcal MAPE (matched-only) | 45.7% (54 ảnh) | tổng-per-image 33.2% trên 37 ảnh đủ DB |
| Latency pipeline đầy đủ | 34.63 s/ảnh (CUDA 12.1) | SAM2 + DAv2 + ArUco, đo local |

**Kết quả nghiên cứu đáng chú ý (RQ3, âm):** YOLOv26m đạt mAP@0.50 = 0.951 trên val split nhưng Recall@0.50 = 0.514 trên ảnh thật — **thua YOLOv10b (0.730)** dù mAP cao hơn. Đây là minh chứng domain shift giữa split random và điều kiện chụp thực tế; production giữ YOLOv10b.

## 🧩 Kiến trúc

1. **Detection**: YOLOv10b (production, `model/yolov10/`) — VietFood67, 68 món; YOLOv26m chỉ dùng trong nghiên cứu so sánh.
2. **Scale recovery**: ArUco marker (30 mm) / checkerboard; plate heuristic chỉ là fallback có cảnh báo.
3. **Segmentation**: SAM 2.1 (`SEGMENTATION_BACKEND=sam2`); FoodSAM là backend thay thế.
4. **Depth**: Depth Anything V2 **Metric-Indoor-Small** (`DEPTH_ENCODER=vits`).
5. **Volume → Mass**: tích phân thể tích theo phương pháp từng dạng món + Density Database (67 entry, nhãn provenance từng giá trị).
6. **Nutrition Layer**: NIN canonical mapping (68 món), tính Calories/Protein/Fat/Saturates/Sugar/Salt + traffic-light, RDA theo BMI/TDEE (Mifflin-St Jeor).
7. **Advisory**: chatbot đa LLM (Gemini/Cerebras/OpenRouter/Cloudflare) đọc Structured Facts JSON; kèm RAG kiến thức (Qdrant).

> ⚠️ **Disclaimer y tế**: Hệ thống chỉ hỗ trợ tham khảo, KHÔNG thay thế chẩn đoán/ tư vấn của bác sĩ hay chuyên gia dinh dưỡng. Khối lượng ước lượng có sai số MAPE ≈ 45% trên tập đánh giá thực tế và thay đổi mạnh theo món (6% – 200%); không dùng con số này cho chế độ ăn trị liệu.

## 🌐 Live Demo

👉 [https://vietnamese-food-nutrition-cal.streamlit.app/](https://vietnamese-food-nutrition-cal.streamlit.app/)

**Lưu ý phạm vi demo**: bản cloud chỉ gồm giao diện + detection + nutrition + chatbot (CPU). Pipeline ước lượng khối lượng (SAM2 + depth + ArUco) cần GPU + checkpoint ngoài repo — không chạy được trên Streamlit Cloud; xem [`results/REPRODUCE.md`](results/REPRODUCE.md).

## 🛠️ Cài đặt & chạy cục bộ

```bash
git clone https://github.com/Jralik/VietNamese-Food-Nutrition-Cal.git
cd VietNamese-Food-Nutrition-Cal
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt        # bản lock: requirements-lock.txt
streamlit run main.py                  # http://localhost:8501
```

Chạy lại bộ test/eval: [`results/REPRODUCE.md`](results/REPRODUCE.md).

## 📁 Cấu trúc thư mục

```
VietNamese-Food-Nutrition-Cal/
├── main.py                  # Giao diện Streamlit: image/video/webcam/IP + chatbot
├── utils.py                 # Detection, nutrition, LLM clients
├── class_names.py           # DB 68 món Việt Nam + dinh dưỡng (NIN)
├── nutrition_rule_engine.py # Rule-based health engine (BMI/TDEE/RDA)
├── ui_components.py         # Component dùng chung (badge, traffic-light…)
├── density_db.py            # Density Database 67 entry + provenance
├── volume_nutrition.py      # Volume → mass → nutrition estimation core
├── volume_integration.py / volume_worker.py / food_volume_pipeline.py
├── foodsam_segmenter.py / scale_recovery.py / depth_estimation.py
├── pipeline_config.py       # Cấu hình backend (SAM2/depth/marker)
├── ultralytics/             # Vendored (block.py có patch YOLO11/26 — xem patches/)
├── rag/                     # RAG kiến thức (Qdrant) + knowledge_docs/
├── eval_*.py                # Chuỗi đánh giá (val detection, full pipeline, model compare)
├── scripts/                 # Benchmark + audit + bảng derived
├── tests/                   # Acceptance tests (python tests/test_phase*.py)
├── data/                    # GT JSON, benchmark results, mapping NIN
├── gt_images/               # Bản sao bằng chứng 54 ảnh Ground-Truth
├── results/                 # SSOT frozen numbers, outline Ch4, REPRODUCE
├── scratch/                 # Báo cáo frozen của chuỗi đánh giá + dữ liệu thô
├── evaluation_final/        # Bản export báo cáo frozen cho thesis
├── model/                   # Trọng số (không commit — MANIFEST.md ghi SHA256)
└── portion-estimation_val.txt  # GT gốc 54 ảnh (nguồn parse chính thức)
```

## 📖 Đọc thêm

- [`results/frozen_numbers.md`](results/frozen_numbers.md) — SSOT số liệu + provenance chain.
- [`results/chapter4_outline.md`](results/chapter4_outline.md) — mapping Chương 4 → bằng chứng.
- [`results/deviation_from_proposal.md`](results/deviation_from_proposal.md) — sai lệch so với đề cương và lý do.
