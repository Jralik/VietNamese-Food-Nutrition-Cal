# Sai lệch giữa Đề cương và Sản phẩm thực tế

> Ghi nhận trung thực các điểm sản phẩm khác phạm vi/ kỳ vọng ban đầu của đề cương (`docx_extracted.txt`), kèm lý do. Mỗi dòng kèm cách trình bày lại trong báo cáo/luận văn. Trung thực ở đây được tính điểm, không trừ điểm.

## 1. LLM Advisory: local Ollama → Cloud API

| | Nội dung |
|---|---|
| Đề cương | Qwen2.5-7B-Instruct chạy suy luận cục bộ qua Ollama; "không dùng API trả phí bên thứ ba" (`docx_extracted.txt:59,70`) — bản thân mục đã gắn [MỞ RỘNG] với điều kiện "nếu tiến độ cho phép" |
| Thực tế | 4 nhà cung cấp cloud: Gemini / Cerebras / OpenRouter / Cloudflare (`main.py:143`) — API key người dùng tự nhập, free tier |
| Lý do | GPU cục bộ phải ưu tiên cho pipeline CV (SAM2 + Depth Anything); suy luận Ollama local cạnh tranh tài nguyên và chậm hơn. Chatbot truyền Structured Facts JSON nên số liệu dinh dưỡng không phụ thuộc model |
| Điều chỉnh | Ghi rõ trong §5: LLM advisory là thành phần MỞ RỘNG, chạy cloud API theo cấu hình người dùng; góp phần tách bạch "số liệu đo được" khỏi "lời văn của LLM" |

## 2. RAG + Qdrant (ngoài phạm vi đề cương)

| | Nội dung |
|---|---|
| Đề cương | Không có thành phần RAG/vector-DB (0 match "Qdrant"/"RAG") |
| Thực tế | `rag/` + Qdrant Cloud (free tier) + `rag/knowledge_docs/` gồm tài liệu y khoa |
| Lý do | Neo lời khuyên chatbot vào tài liệu tham khảo thay vì suy luận tự do; giảm hallucination |
| Điều chỉnh | Trình bày là mở rộng hỗ trợ advisory; số liệu dinh dưỡng chính vẫn từ Nutrition Layer, không từ RAG |

## 3. Ground-Truth: ~100 phần ăn → 54 ảnh

| | Nội dung |
|---|---|
| Đề cương | "~100 phần ăn cân thực tế" (`docx_extracted.txt:50,80`) — [LÕI] |
| Thực tế | 54 ảnh / 74 targets, 7 nhóm món, 100% cân đo thực tế |
| Lý do | Thời gian cân đo + quy trình đo đạc 2 protocol; ưu tiên chất lượng metadata (nhiều thành phần/ảnh) hơn số lượng |
| Điều chỉnh | §4 nêu rõ population từng bảng; hạn chế: đủ cho MAPE tổng hợp + taxonomy, mỏng cho phân tích per-class (Trung n=2, Canh n=1) |

## 4. Segmentation: FoodSAM/MobileSAM → SAM 2.1

| | Nội dung |
|---|---|
| Đề cương | Mục tiêu #5 [LÕI]: khảo sát, thực nghiệm và lựa chọn giữa **FoodSAM, MobileSAM**; SAM 2 là ứng viên thứ ba [MỞ RỘNG] (`docx_extracted.txt:52,57`) |
| Thực tế | `SEGMENTATION_BACKEND = "sam2"` (`pipeline_config.py:17`); FoodSAM có integration (`foodsam_segmenter.py`); **MobileSAM 0 tham chiếu trong code dự án**; **chưa có mIoU/Dice** |
| Lý do | SAM 2.1 tích hợp sẵn trong ultralytics vendored, chất lượng mask tốt hơn; thời lượng không đủ cho khảo sát 3 ứng viên với metric chuẩn |
| Điều chỉnh | §4.7 + §5: ghi mục tiêu #5 đạt "khảo sát 2/2+ ứng viên thực tế dùng (SAM2, FoodSAM-integration)", MobileSAM chưa khảo sát; thiếu mIoU/Dice là limitation → RQ2 ở tầng segmentation chưa định lượng được |

## 5. Depth Anything V2: encoder Small (vits)

| | Nội dung |
|---|---|
| Đề cương | "Depth Anything V2 Metric" — không chỉ định encoder size (`docx_extracted.txt:48,61`) |
| Thực tế | `DEPTH_ENCODER="vits"` + `DEPTH_USE_METRIC=True` (bản Metric-Indoor-**Small**, `pipeline_config.py:160-167`) |
| Lý do | Tốc độ GPU + RAM; checkpoint Metric được giữ đúng như cam kết |
| Điều chỉnh | §3 ghi rõ encoder Small và bàn luận ảnh hưởng sai số so với Large; 3 failure mode (Bánh mì over ~2.9×, Trứng under ~3×, Súp cua geometry-dependent) gắn với limitation của monocular depth |

## 6. Chính sách fallback §1.6.4 (không có ArUco)

| | Nội dung |
|---|---|
| Đề cương | "không phát hiện ArUco → yêu cầu chụp lại, không ước lượng trên dữ liệu thiếu" (`docx_extracted.txt:313,370`) |
| Thực tế | Fallback `plate_heuristic` / `bbox_fallback` vẫn sinh gam/calo kèm cảnh báo; confidence note ghi rõ cấu hình không benchmark ("CHƯA được kiểm chứng"); Chain A: 5/54 ảnh không marker (4 plate_heuristic + 1 không xác định) |
| Lý do | UX: chặn cứng làm app mất tính năng với ảnh chụp thường ngày; các ảnh này KHÔNG nằm trong benchmark số liệu chính |
| Điều chỉnh | §4.3 nêu 5/54; §5.7 ghi chính sách hiện tại = fallback + cảnh báo; việc chặn cứng/checkbox xác nhận là việc còn mở nếu yêu cầu |

## 7. Density Database: water displacement chưa đo

| | Nội dung |
|---|---|
| Đề cương | Mục tiêu #2 [LÕI]: density DB thực nghiệm bằng **water displacement** (`docx_extracted.txt:49`) |
| Thực tế | 67 entry: literature 58 · usda_fdc 8 · user_measured_gt 1; std 0.05-0.12 là assumption; protocol đo đã chuẩn bị (`results/water_displacement_protocol.md`) |
| Lý do | Ưu tiên chuỗi đánh giá + ablation trước; phép đo cần buổi thực nghiệm riêng |
| Điều chỉnh | §4.7 limitation; nếu đo xong trước hạn nộp → density_db version mới + **post-freeze evaluation riêng**, so sánh với frozen baseline, KHÔNG thay số Chương 4 |

## 8. Chỉ số §2.8 chưa phủ đủ

| | Nội dung |
|---|---|
| Đề cương | §2.8 yêu cầu Detection (mAP@0.5, mAP@0.5:0.95, P/R/F1) · Segmentation (IoU/mIoU/Dice) · Depth (MAE/RMSE/AbsRel) · Volume/Mass (MAE/MAPE) |
| Thực tế | Detection: mAP@0.50/0.50:0.95 + P/R đã có (val), recall/FP/0-box trên ảnh thật; Volume/Mass: MAE/MAPE đủ 2 chuỗi; **Segmentation & Depth metrics chưa đo** (không có GT segmentation/depth) |
| Điều chỉnh | §2.8 trình bày những gì đo được + lý do phần còn lại (thiếu GT annotation, ngoài ngân sách thời gian); F1 tính được từ P/R nếu cần |
