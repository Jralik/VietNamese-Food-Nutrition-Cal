# Báo cáo 4 Thực nghiệm Bổ sung cho Bảo vệ Khóa luận (Tập 418 Ảnh / 55 Lớp Món ăn Việt Nam)

> [!NOTE]
> Báo cáo này tổng hợp kết quả định lượng từ **4 thực nghiệm chuyên sâu** được chạy trực tiếp trên tập dữ liệu kiểm thử mở rộng (`418` ảnh COCO đã đối chiếu, `552` thực thể, `500` nhóm khối lượng, `409` bữa ăn `Full-Meal` đầy đủ dinh dưỡng, `55` lớp món ăn Việt Nam) cùng cấu hình sản phẩm cuối cùng (**MobileSAM Config C + Depth Anything V2 Small + ArUco Depth-Anchored + Cải tiến `A1–C6`**).
>
> **Dữ liệu nguồn & mã tái lập:**
> - Thực nghiệm 1–3: [run_thesis_extra_experiments_1_3.py](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_thesis_extra_experiments_1_3.py) → [extra_experiments_1_3.json](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/bench_cache_v2/extra_experiments_1_3.json)
> - Thực nghiệm 4: [run_e2e_runtime_profile.py](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_e2e_runtime_profile.py) → [e2e_runtime_profile_v2.json](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/bench_cache_v2/e2e_runtime_profile_v2.json)
> - Báo cáo benchmark lõi 418 ảnh trước đó: [benchmark_v2_418_report.md](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/results/benchmark_v2_418_report.md)

---

## Tóm tắt Kết quả Chính (Executive Summary)

| Thực nghiệm | Quy mô đánh giá | Chỉ số cốt lõi (Production) | Ý nghĩa bảo vệ khóa luận |
| :--- | :--- | :--- | :--- |
| **1. Dinh dưỡng (Calories & Macros)** | `409` bữa `Full-Meal` / `500` nhóm món | **Calories `MAE = 104.0 kcal` (`MAPE = 29.16%`, `r = 0.856`)**; Protein `29.23%`; Fat `30.18%`; Carbs `29.13%` (Oracle MobileSAM C) | Lõi 3D + bảng dinh dưỡng đạt sai số `~29%–30%` trên toàn bộ 4 chỉ số dinh dưỡng; tách bạch rõ nút cổ chai `TH2` của YOLO khi chạy E2E. |
| **2. Đối chứng ArUco (Calibration Ablation)** | `402` ảnh có ArUco / `483` nhóm món | **`MAPE`: `78.95%` (Không ArUco) → `32.95%` (ArUco) → `30.21%` (ArUco + C6)**; `r`: `0.655 → 0.934` | Giảm **`-48.74` điểm % MAPE**, cắt hơn `51%` MAE và `53%` RMSE; chứng minh hiệu chuẩn vật lý là bắt buộc. |
| **3. Ổn định Đa góc chụp (Multi-View CV%)** | `95` phiên chụp / `428` ảnh (`≥ 3` góc/món) | **`Median CV = 21.04%`**, `P75 CV = 31.67%`, **`Mean Max–Min = 92.5 g`** (thấp nhất cả 3 cấu hình) | Bù phối cảnh `C6` giảm `Median CV` `-4.76` điểm %, `P75 CV` `-8.54` điểm % và biên độ gram `-26.5 g` so với không có `C6`. |
| **4. Độ trễ & VRAM Toàn tuyến (GTX 1650)** | `20` ảnh thực tế (+ `2` warm-up) trên GPU `4 GB` | **`P50 = 2.78 s`**, `Mean = 3.11 s`, **`Peak VRAM = 569.4 MiB` (`13.9%` ngân sách 4 GB)** | Chạy gọn toàn tuyến trên GPU phổ thông 4 GB (dư `86.1%` VRAM); mở lộ trình `< 1.8 s/ảnh` khi đưa YOLO lên CUDA. |

---

## Thực nghiệm 1: Độ chính xác Ước lượng Năng lượng (Calories) và Đa lượng (Protein, Fat, Carbs)

### 1.1. Thiết lập đánh giá
Từ khối lượng thực tế (`GT mass`) và khối lượng dự đoán (`Pred mass`), hệ thống quy đổi sang **Calories (`kcal`), Protein (`g`), Fat (`g`), Carbs (`g`)** thông qua bảng dinh dưỡng chuẩn hóa của pipeline (`nutrition_data.py`). Đánh giá được thực hiện song song ở hai mức độ:
1. **Cấp nhóm món (`Mass Group`)**: `500` nhóm món (Oracle MobileSAM Config C) và `437` nhóm khớp detection (End-to-End YOLO + MobileSAM Config C).
2. **Cấp trọn bữa ăn (`Full-Meal`)**: `409` bữa ăn đầy đủ thông tin khối lượng & dinh dưỡng (cộng tổng toàn bộ các món trên mâm/đĩa trong cùng một ảnh).

> [!TIP]
> Đối với các chỉ số có giá trị thực tế bằng `0` ở một số món thuần đạm hoặc thuần tinh bột (ví dụ: thịt nạc có `GT Carbs = 0 g` hoặc cơm trắng có `GT Fat = 0 g`), phép tính `MAPE` chỉ lấy trên các mẫu có `GT > 0` (`N = 492`/`406` cho Fat và `N = 445`/`367` cho Carbs) để tránh chia cho 0.

### 1.2. Kết quả cấp Trọn bữa ăn (`Full-Meal`, `N = 409`)

| Chỉ số dinh dưỡng | Chế độ đánh giá | Số mẫu (`N`) | MAE ↓ | RMSE ↓ | MAPE (%) ↓ | MPE (%) | Pearson `r` ↑ | `R²` ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calories (kcal)** | **Oracle MobileSAM C** | 409 | **104.0 kcal** | **152.4 kcal** | **29.16%** | -11.27% | **0.856** | **0.732** |
| | End-to-End (YOLO + MS-C) | 409 | 140.1 kcal | 198.4 kcal | 49.87% | -2.49% | 0.762 | 0.580 |
| **Protein (g)** | **Oracle MobileSAM C** | 409 | **5.46 g** | **8.28 g** | **29.23%** | -11.20% | **0.871** | **0.759** |
| | End-to-End (YOLO + MS-C) | 409 | 8.06 g | 12.72 g | 53.20% | -3.35% | 0.694 | 0.481 |
| **Fat (g)** | **Oracle MobileSAM C** | 406 | **4.19 g** | **6.49 g** | **30.18%** | -11.69% | **0.845** | **0.714** |
| | End-to-End (YOLO + MS-C) | 406 | 6.29 g | 10.02 g | 68.50% | +9.36% | 0.642 | 0.412 |
| **Carbs (g)** | **Oracle MobileSAM C** | 367 | **12.77 g** | **19.87 g** | **29.13%** | -11.80% | **0.871** | **0.758** |
| | End-to-End (YOLO + MS-C) | 367 | 20.11 g | 31.80 g | 112.04% | +60.27% | 0.684 | 0.468 |

### 1.3. Kết quả cấp Nhóm món (`Mass Groups`)

| Chỉ số dinh dưỡng | Chế độ đánh giá | Số mẫu (`N`) | MAE ↓ | RMSE ↓ | MAPE (%) ↓ | MPE (%) | Pearson `r` ↑ | `R²` ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calories (kcal)** | **Oracle MobileSAM C** | 500 | **88.8 kcal** | **135.4 kcal** | **30.71%** | -8.85% | **0.868** | **0.754** |
| | End-to-End (Matched) | 437 | 114.5 kcal | 165.7 kcal | 41.74% | -4.72% | 0.806 | 0.649 |
| **Protein (g)** | **Oracle MobileSAM C** | 500 | **4.66 g** | **7.37 g** | **30.63%** | -8.84% | **0.875** | **0.766** |
| | End-to-End (Matched) | 437 | 6.35 g | 10.43 g | 44.90% | -3.82% | 0.748 | 0.559 |
| **Fat (g)** | **Oracle MobileSAM C** | 492 | **3.56 g** | **5.81 g** | **31.10%** | -8.82% | **0.864** | **0.746** |
| | End-to-End (Matched) | 432 | 4.95 g | 8.09 g | 67.95% | +17.49% | 0.727 | 0.529 |
| **Carbs (g)** | **Oracle MobileSAM C** | 445 | **10.88 g** | **17.85 g** | **30.68%** | -9.48% | **0.889** | **0.790** |
| | End-to-End (Matched) | 397 | 15.99 g | 25.72 g | 84.50% | +39.40% | 0.765 | 0.586 |

### 1.4. Phân tích chuyên sâu
- **Độ chính xác lõi dinh dưỡng vượt mục tiêu (`< 30%` ở mức bữa ăn):** Với nhãn lớp chính xác (Oracle MobileSAM C), cả 4 chỉ số dinh dưỡng cấp `Full-Meal` đều ổn định quanh **`29.1%–30.2%` MAPE** với hệ số tương quan rất cao **`r = 0.845–0.871`**, sai số tuyệt đối chỉ **`104.0 kcal`**, **`5.46 g` Protein**, **`4.19 g` Fat** và **`12.77 g` Carbs** mỗi bữa ăn.
- **Cơ chế khuếch đại sai số Carbs/Fat trong chế độ End-to-End:** Trên E2E, `MAE` tuyệt đối tăng khá vừa phải (`+36.1 kcal`, `+2.60 g` Protein, `+2.10 g` Fat, `+7.34 g` Carbs), nhưng **`MAPE Carbs` nhảy vọt lên `112.04%`** (`MPE = +60.27%`). Nguyên nhân trực tiếp là các lỗi phân loại nhầm (`TH2`, `18.8%`) của `YOLOv10b`: khi một món giàu đạm/ít tinh bột (như `Thit nuong`, `Thit ga`, `Dau hu`) bị nhận nhầm thành món giàu tinh bột (`Com tam`, `Xoi`, `Banh mi`), mẫu số `GT Carbs` rất nhỏ trong khi lượng Carbs dự đoán tăng đột biến, khuếch đại tỷ lệ sai số tương đối. Phát hiện này cung cấp bằng chứng định lượng sắc nét rằng **tái huấn luyện `YOLOv10b` để giảm `TH2` sẽ mang lại mức cải thiện lớn nhất cho độ chính xác dinh dưỡng toàn hệ thống**.

---

## Thực nghiệm 2: Đối chứng Hiệu chuẩn Tỷ lệ ArUco (`ArUco Calibration Ablation`)

### 2.1. Thiết lập đối chứng
Trên đúng **`402` ảnh có thẻ ArUco phát hiện thành công (`483` nhóm món)**, ta giữ nguyên hoàn toàn mặt nạ `MobileSAM Config C` và bản đồ độ sâu `Depth Anything V2`, chỉ thay đổi duy nhất cơ chế khôi phục tỷ lệ vật lý (metric scale):
1. **Không có ArUco (`plate_heuristic`)**: Cố tình tắt thẻ ArUco, hệ thống quay về ước lượng đường kính vật chứa theo giả định hình học mặc định.
2. **Có ArUco, không `C6` (`aruco_no_c6`)**: Khôi phục tỷ lệ `cm/px` và neo chiều sâu từ thẻ ArUco `4×4 cm`, nhưng chưa hiệu chỉnh tỷ lệ khoảng cách phối cảnh `Z_food / Z_marker`.
3. **Có ArUco + `C6` (`aruco_with_c6` — Production)**: Đầy đủ hiệu chuẩn ArUco kết hợp bù phối cảnh chiều sâu `C6` (`scale_food = scale_marker × (Z_food / Z_marker)^α`).

### 2.2. Kết quả tổng thể (`N = 483` nhóm món)

| Cấu hình hiệu chuẩn | MAE (g) ↓ | RMSE (g) ↓ | MAPE (%) ↓ | MPE (%) | Bias (%) | Pearson `r` ↑ | `R²` ↑ | Chênh lệch MAPE so với Không ArUco |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Không ArUco (`plate_heuristic`) | 129.38 | 196.08 | 78.95% | +48.08% | +7.28% | 0.655 | 0.429 | — (Baseline) |
| 2. Có ArUco, không C6 (`aruco_no_c6`) | 71.41 | 113.99 | 32.95% | -0.86% | -6.63% | 0.889 | 0.790 | **-46.00 điểm %** |
| **3. Có ArUco + C6 (Production)** | **62.90** | **92.17** | **30.21%** | -9.68% | -10.60% | **0.934** | **0.872** | **-48.74 điểm % (`-51.4%` MAE)** |

### 2.3. Phân rã theo 6 nhóm loại vật chứa (`Per Dish Type`)

| Nhóm vật chứa (`dish_type`) | `N` | Không ArUco: MAPE (MAE, `r`) | ArUco không C6: MAPE (MAE, `r`) | **ArUco + C6: MAPE (MAE, `r`)** | Tác động của `C6` (so với ArUco không C6) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tô nước (`soup_bowl`)** | 92 | 45.26% (218.0 g, `r=0.090`) | 19.56% (116.6 g, `r=0.731`) | **18.58% (96.8 g, `r=0.846`)** | `-0.98%` MAPE, **`-42.5 g` RMSE** |
| **Chén cơm (`rice_bowl`)** | 32 | 68.72% (101.8 g, `r=0.075`) | 29.20% (46.6 g, `r=0.832`) | **22.52% (39.4 g, `r=0.865`)** | **`-6.68%` MAPE**, `-7.1 g` MAE |
| **Đĩa chính (`flat_plate`)** | 143 | 64.33% (126.5 g, `r=0.481`) | 38.23% (100.3 g, `r=0.628`) | **32.91% (85.1 g, `r=0.687`)** | **`-5.32%` MAPE**, `-15.1 g` MAE |
| **Đĩa nhỏ (`small_plate`)** | 143 | 78.68% (70.5 g, `r=-0.098`) | 34.80% (38.6 g, `r=0.705`) | **33.84% (38.9 g, `r=0.739`)** | `-0.96%` MAPE, `+0.034` `r` |
| **Rau lá (`side_vegetables`)** | 40 | 71.26% (54.3 g, `r=0.333`) | 45.38% (37.8 g, `r=0.268`) | **38.90% (32.4 g, `r=0.493`)** | **`-6.48%` MAPE**, `+0.224` `r` |
| **Món nhỏ (`ingredient_small`)** | 33 | 256.68% (267.6 g, `r=0.456`) | **27.89% (27.6 g, `r=0.852`)** | 32.16% (36.2 g, `r=0.802`) | `+4.27%` MAPE (nằm sát mặt bàn) |

![Tương quan Calories Full-Meal và Đối chứng ArUco](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/results/thesis_figures_v2/fig6_calorie_and_aruco_ablation_v2.png)

### 2.4. Phân tích chuyên sâu
- **Tại sao phương pháp không có vật tham chiếu (`plate_heuristic`) thất bại hoàn toàn trong thực tế?** Khi tắt ArUco, hệ thống buộc phải gán đường kính vật chứa mặc định; kết quả là `r` gần như bằng `0` ở các nhóm tô nước (`0.090`), chén cơm (`0.075`) và thậm chí âm ở đĩa nhỏ (`-0.098`) — nghĩa là hệ thống **không phân biệt được khẩu phần lớn hay nhỏ**, đồng thời làm các món nhỏ (`ingredient_small`) bị phóng đại sai số lên **`256.68%`** (`MPE = +237.4%`).
- **Giá trị cộng hưởng của `ArUco + C6`:** Hiệu chuẩn ArUco giúp phục hồi tức thì thước đo vật lý (`MAPE` giảm từ `78.95%` xuống `32.95%`, `R²` tăng từ `0.429` lên `0.790`). Việc bổ sung công thức bù phối cảnh chiều sâu `C6` tiếp tục cải thiện **5/6 nhóm vật chứa** (mạnh nhất ở `rice_bowl`, `side_vegetables`, `flat_plate` và `soup_bowl`), đưa tổng thể xuống **`30.21%` MAPE**, **`62.90 g` MAE**, **`92.17 g` RMSE** và **`R² = 0.872`**. Ngoại lệ duy nhất là `ingredient_small` (`+4.27%`) do các món nhỏ dẹt nằm ngang cao độ với thẻ marker nên tỷ lệ `Z_food / Z_marker` dễ chịu nhiễu depth biên — một quan sát trung thực và có thể tối ưu bằng cách bỏ qua `C6` cho nhóm vật thể phẳng sát bàn trong phiên bản tương lai.

---

## Thực nghiệm 3: Độ ổn định Đa góc chụp (`Multi-View Repeatability / CV%`)

### 3.1. Thiết lập đánh giá
Trong bộ dữ liệu thực tế, cùng một phần ăn vật lý được chụp liên tiếp từ nhiều góc độ và khoảng cách khác nhau (từ chụp thẳng `90°` đến chụp nghiêng `30°–60°`). Ta gom toàn bộ các phiên chụp có **`≥ 3` góc chụp cùng một phần ăn** → thu được **`95` phiên chụp độc lập với tổng cộng `428` ảnh**. Với mỗi phiên chụp, tính:
- **Hệ số biến thiên `CV (%) = (Std / Mean) × 100%`** giữa các góc chụp của cùng một phần ăn (càng nhỏ càng ổn định).
- **Biên độ dao động tuyệt đối `Max − Min (g)`** giữa góc dự đoán cao nhất và thấp nhất.

### 3.2. Kết quả định lượng (`95` phiên chụp / `428` góc chụp)

| Cấu hình hiệu chuẩn | Mean CV (%) ↓ | Median CV (%) ↓ | P75 CV (%) ↓ | Mean `Max − Min` (g) ↓ | MAPE thực tế (Exp 2) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Không ArUco (`plate_heuristic`) | 16.91%* | 12.32%* | 24.23%* | 104.1 g | 78.95% *(sai lệch cố định)* |
| 2. Có ArUco, không C6 (`aruco_no_c6`) | 26.17% | 25.80% | 40.21% | 119.0 g | 32.95% |
| **3. Có ArUco + C6 (Production)** | **23.40%** | **21.04%** | **31.67%** | **92.5 g** | **30.21%** |

*\*Lưu ý: `plate_heuristic` có `CV%` tương đối thấp chỉ vì luôn khóa mọi góc chụp về cùng một giả định đường kính đĩa cố định ("ổn định một cách sai lầm" — low variance, extreme bias với `MAPE = 78.95%`).*

![Độ ổn định đa góc chụp trên 95 phiên chụp](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/results/thesis_figures_v2/fig7_multiview_consistency_v2.png)

### 3.3. Phân tích chuyên sâu
- **Tác dụng kép của `C6` (Chính xác hơn VÀ Ổn định hơn):** Khi chụp ảnh thực tế ở các góc nghiêng khác nhau, thẻ ArUco đặt cạnh đĩa/tô sẽ gần hoặc xa camera hơn tâm món ăn, khiến hệ số `cm/px` trên mặt phẳng marker bị trôi theo góc chụp (`aruco_no_c6` có `P75 CV = 40.21%`, `Mean Max–Min = 119.0 g`).
- Khi kích hoạt `C6` (`Z_food / Z_marker`), sự trôi phối cảnh giữa các góc chụp được triệt tiêu rõ rệt: **`Median CV` giảm `-4.76` điểm %** (`25.80% → 21.04%`), **`P75 CV` giảm mạnh `-8.54` điểm %** (`40.21% → 31.67%`), và **biên độ gram `Mean Max–Min` hạ xuống chỉ còn `92.5 g`** — **thấp nhất trong cả 3 cấu hình** (vượt qua cả `plate_heuristic` `104.1 g`).

---

## Thực nghiệm 4: Phân rã Độ trễ Toàn tuyến & Mức tiêu thụ VRAM trên GPU Phổ thông (`GTX 1650 — 4 GB`)

### 4.1. Cấu hình đo đạc thực tế
- **Phần cứng kiểm thử:** `NVIDIA GeForce GTX 1650` (**4 GB / 4096 MiB VRAM**), Windows, chạy toàn bộ pipeline sản phẩm (`food_volume_pipeline.py`) với **MobileSAM Config C** và **Depth Anything V2 Small** được giữ **nạp đồng thời trên GPU**.
- **Quy trình đo:** Chạy `2` ảnh làm nóng (`warm-up`) để ổn định CUDA kernel & bộ đệm, sau đó đo thời gian thực (`wall-clock` đồng bộ `torch.cuda.synchronize()`) và đỉnh bộ nhớ (`torch.cuda.max_memory_allocated()`) trên **`20` ảnh thực tế** đại diện đầy đủ các nhóm món đơn và mâm nhiều món.

### 4.2. Phân rã độ trễ theo từng giai đoạn (`N = 20` ảnh)

| Giai đoạn trong Pipeline | Thiết bị chạy | Mean (ms) | Median `P50` (ms) | `P95` (ms) | Tỷ trọng Mean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. YOLOv10b (Dual-PreResize 640×640)** | CPU | 1,126.0 | 1,121.6 | 1,171.2 | **36.2%** |
| **2. Depth Anything V2 Small** | GPU (`GTX 1650`) | 900.0 | 1,000.9 | 1,114.9 | **29.0%** |
| **3. MobileSAM Config C (`TinyViT`)** | GPU (`GTX 1650`) | 508.6 | **170.2** | 1,746.5 | **16.4%** |
| **4. ArUco Scale + Depth Anchoring** | CPU (`OpenCV`) | 367.6 | 283.2 | 503.3 | **11.8%** |
| **5. Volume (`A1–C6`) + Nutrition** | CPU (`NumPy/SciPy`) | 107.8 | **102.2** | 162.4 | **3.5%** |
| *Chi phí phụ trợ (I/O ảnh, tiền/hậu xử lý)* | CPU | *96.7* | — | — | *3.1%* |
| **Tổng toàn tuyến End-to-End** | **Hybrid CPU+GPU** | **3,106.7 ms (~3.11 s)** | **2,778.1 ms (~2.78 s)** | **4,550.9 ms (~4.55 s)** | **100.0%** |

### 4.3. Tài nguyên bộ nhớ hệ thống

| Chỉ số tài nguyên | Giá trị đo thực tế | Ghi chú triển khai |
| :--- | :---: | :--- |
| **Peak GPU VRAM (MobileSAM + Depth V2 đồng thời)** | **`569.4 MiB`** | Chỉ chiếm **`13.9%`** giới hạn `4096 MiB` của GTX 1650 (dư `3,526.6 MiB` = `86.1%`). |
| **Mean CPU RAM (Process RSS)** | **`5,560.3 MiB`** (`~5.43 GiB`) | Phù hợp hoàn toàn trên máy trạm / laptop phổ thông `16 GB RAM`. |

![Phân rã độ trễ toàn tuyến và Peak VRAM trên GTX 1650](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/results/thesis_figures_v2/fig8_e2e_runtime_breakdown_v2.png)

### 4.4. Nhận định kỹ thuật & Hướng tối ưu tốc độ
1. **Tính khả thi triển khai cực cao trên phần cứng phổ thông:** Toàn bộ hệ thống 3 mô hình học sâu (`YOLOv10b` + `MobileSAM Config C` + `Depth Anything V2 Small`) cùng lõi hình học 3D vận hành mượt mà trong **`2.78 s` trung vị (`P50`)** với chỉ **`569.4 MiB` Peak VRAM**, hoàn toàn không xảy ra tràn bộ nhớ (`OOM`) ngay cả trên GPU laptop đời cũ 4 GB.
2. **Giải thích chênh lệch `P50` vs `P95` ở MobileSAM:** Với ảnh 1–2 món ăn phổ biến, `MobileSAM Config C` chỉ tốn **`170.2 ms`** (`P50`); mức `P95 = 1,746.5 ms` xuất hiện ở các mâm đa món (`5–8` thực thể/ảnh) do giải mã tuần tự từng prompt. Phần hình học 3D (`Volume A1–C6 + Nutrition`) sau khi tối ưu subsample biên (`≤ 4000` điểm) chạy cực nhẹ chỉ **`102.2 ms` (`3.5%` tổng thời gian)**.
3. **Nút cổ chai tốc độ rõ ràng & dễ mở khóa:** Giai đoạn tốn thời gian nhất là **`YOLOv10b` chạy trên CPU (`1,126.0 ms`, chiếm `36.2%` toàn tuyến)** trong khi GPU vẫn còn trống tới **`3.5 GB` VRAM**. Việc chuyển `YOLOv10b` sang CUDA FP16 hoặc TensorRT có thể giảm thời gian detect từ `~1,126 ms` xuống `< 80 ms`, kéo tổng độ trễ End-to-End xuống **`~1.7–1.9 giây/ảnh`**.

---

## Kết luận Tổng hợp Cho Bảo vệ Khóa luận

Bốn thực nghiệm bổ sung trên tập 418 ảnh đã khép kín đầy đủ vòng lặp kiểm chứng học thuật cho toàn bộ hệ thống:
1. **Về mục tiêu ứng dụng dinh dưỡng (Exp 1):** Khi phân loại đúng món, hệ thống ước lượng **Calories, Protein, Fat và Carbs cấp bữa ăn với `MAPE = 29.1%–30.2%` và `r = 0.85–0.87`**, đáp ứng ngưỡng thực tế cho bài toán theo dõi dinh dưỡng từ ảnh đơn sắc 2D.
2. **Về đóng góp thuật toán cốt lõi (Exp 2 & Exp 3):** Hiệu chuẩn vật lý `ArUco + Depth Anchoring + Bù phối cảnh C6` cắt **`-48.74` điểm % MAPE** so với không dùng ArUco (`78.95% → 30.21%`), nâng `R²` từ `0.429` lên `0.872`, đồng thời giảm mạnh biến thiên đa góc chụp xuống **`Median CV = 21.04%`** và **`Mean Max–Min = 92.5 g`**.
3. **Về tính thực tiễn kỹ thuật (Exp 4):** Hệ thống chạy toàn tuyến **`2.78 s` (`P50`)** chỉ tốn **`569.4 MiB` Peak VRAM** trên GPU 4 GB phổ thông, sẵn sàng đóng gói dịch vụ Web/Edge.
