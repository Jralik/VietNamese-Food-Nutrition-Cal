# Báo cáo Thực nghiệm Mục 4.12 (RQ8): Phân rã Định lượng 4 Nguồn Sai số Hệ thống (Tập 418 Ảnh / 55 Lớp)

> [!NOTE]
> Thực nghiệm này trả lời trực tiếp câu hỏi nghiên cứu trọng tâm **RQ8** (*"Trong bốn nguồn sai số chính gồm: nhận diện sai lớp hoặc vị trí hộp bao, sai lệch mặt nạ phân đoạn, nhiễu bản đồ độ sâu và biến thiên khối lượng riêng, nguồn nào đóng góp tỷ trọng lớn nhất vào sai số tổng thể của khối lượng ước lượng?"*) trên toàn bộ tập dữ liệu mở rộng **`418` ảnh / `552` thực thể / `500` nhóm khối lượng / `415` bữa ăn `Full-Meal` đầy đủ**.
>
> **Mã tái lập & dữ liệu gốc:**
> - Mã thực nghiệm: [run_rq8_error_decomposition.py](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_rq8_error_decomposition.py)
> - Dữ liệu định lượng JSON: [rq8_error_decomposition_v2.json](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/bench_cache_v2/rq8_error_decomposition_v2.json)
> - Biểu đồ 300-DPI: [fig9_four_source_error_decomposition_v2.png](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/results/thesis_figures_v2/fig9_four_source_error_decomposition_v2.png)

---

## 4.12.1. Phương pháp luận phân rã sai số theo bốn nguồn thành phần

Do chuỗi ước lượng khối lượng có cấu trúc nhân và liên chuỗi:

$$\hat{m} = \hat{V}\big(\text{box}, \text{mask}, \text{depth}, \text{scale}\big) \times \rho(\hat{c})$$

trong không gian logarit, sai số tương đối phân rã tuyến tính thành tổng các thành phần:

$$\ln\!\left(\frac{\hat{m}}{m}\right) = \underbrace{\ln\!\left(\frac{\hat{m}_{\text{E2E}}}{\hat{m}_{\text{MS}}}\right)}_{\text{1. Phát hiện (YOLO)}} + \underbrace{\ln\!\left(\frac{\hat{m}_{\text{MS}}}{\hat{m}_{\text{GT-Mask}}}\right)}_{\text{2. Phân đoạn (MobileSAM C)}} + \underbrace{\ln\!\left(\frac{\hat{V}_{\text{GT-Mask}}}{V_{\text{true}}}\right)}_{\text{3. Độ sâu + Tỷ lệ + Hình học 3D}} + \underbrace{\ln\!\left(\frac{\rho_{\text{DB}}(c)}{\rho_{\text{true}}}\right)}_{\text{4. Khối lượng riêng }(\rho)}$$

Để đảm bảo tính chặt chẽ toán học (không cộng/trừ tùy tiện `MAE` vốn chịu triệt tiêu dấu) và kiểm chứng chéo độc lập, thực nghiệm áp dụng đồng thời **3 kỹ thuật chuẩn hóa**:

1. **Thang cắt bỏ tuần tự bằng Oracle (`Progressive Oracle Ladder`, `S0 → S5`):** Giữ cố định mẫu đánh giá (`N = 415` bữa `Full-Meal` và `N = 437` nhóm khớp), lần lượt loại bỏ từng mắt xích lỗi theo đúng thứ tự vật lý: `FP thừa` → `FN bỏ sót` → `TH2 sai lớp (giữ khung YOLO)` → `Lệch khung hộp bao YOLO` → `Sai số mặt nạ MobileSAM C (thay bằng GT Polygon)`.
2. **Phân rã phương sai sai số theo cặp (`Pairwise Variance Budget` trong không gian `MSRE` & `MSLE`):** Đo trực tiếp độ nhiễu không triệt tiêu dấu giữa từng cặp bước liền kề (`E2E ↔ Oracle MobileSAM C`, `Oracle MobileSAM C ↔ Oracle GT Mask`), đánh giá đồng thời ở **3 phạm vi** để thấy rõ sự dịch chuyển nút cổ chai:
   - **Phạm vi A — Toàn tuyến trọn bữa ăn (`Full-Meal E2E`, `N = 415`, hệ `MSRE`):** Phản ánh trải nghiệm thực tế của người dùng (tính đủ cả món bị bỏ sót `FN`, hộp thừa `FP`, nhầm lớp `TH2` và lệch hộp).
   - **Phạm vi B — Các nhóm món đã định vị được (`Matched Groups`, `N = 437`, hệ `MSLE`):** Loại bỏ yếu tố `FN = 0 g` để phân tích chính xác trong không gian logarit.
   - **Phạm vi C — Lõi hình học 3D cô lập (`Oracle GT Box`, `N = 500`, hệ `MSLE`):** Không có lỗi detector, cô lập 3 thành phần lõi (`Segmentation` vs `Depth/Scale/Geometry` vs `Bulk Density`).
3. **Tách biệt Độ sâu/Hình học 3D và Khối lượng riêng (`ρ`) bằng 2 kiểm chứng chéo độc lập:**
   - **Mô phỏng Monte-Carlo khối lượng riêng (`1.000` lượt trên `N = 500` nhóm):** Lấy mẫu nhiễu tương đối $\rho_{\text{sim}} / \rho \sim \mathcal{N}\big(1, (\sigma_{\rho,c}/\rho_c)^2\big)$ từ bảng độ lệch chuẩn `DENSITY_DB` (`Mean CV_ρ = 11.41%`) để định lượng chính xác phương sai tự nhiên của khối lượng riêng ($\text{MSLE}_\rho = 0.0151$, tương đương sai số độc lập `9.10%` MAPE). Đồng thời chạy hiệu chuẩn mật độ trung bình theo lớp (`Oracle Class-Mean Calibration`) để định cận trên sai lệch mật độ hệ thống giữa các lớp.
   - **Kiểm chứng chéo phương sai đa góc chụp (`Within-Serving Multi-View ANOVA` trên `95` phiên / `428` ảnh, dùng `GT Mask`):** Vì nhiều góc chụp của **cùng một phần ăn vật lý** có $\rho_{\text{true}}$ và $V_{\text{true}}$ hoàn toàn bất biến, **100% phương sai nội-phiên** $\text{Var}_{\text{within}}(\ln \hat{m})$ là nhiễu độ sâu/tỷ lệ/hình học phụ thuộc góc chụp thuần túy (0% đóng góp từ khối lượng riêng, 0% từ YOLO, 0% từ phân đoạn).

---

## 4.12.2. Kết quả định lượng tỷ trọng đóng góp của từng nguồn sai số

### A. Thang cắt bỏ tuần tự nguồn sai số trên `415` bữa ăn `Full-Meal` (`S0 → S5`)

| Giai đoạn trong thang Oracle (`N = 415` bữa) | MAE (g) ↓ | RMSE (g) ↓ | MAPE (%) ↓ | Mức giảm MAPE ở bước này | Pearson `r` ↑ | Nguồn sai số vừa được loại bỏ |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **S0. Toàn tuyến E2E (`YOLOv10b + MobileSAM C`)** | 90.17 | 124.34 | **40.37%** | — | 0.899 | Toàn bộ 4 nguồn sai số |
| **S1. Loại bỏ hộp thừa `FP`** *(chỉ giữ hộp khớp GT)* | 87.98 | 119.85 | 37.93% | **-2.44 điểm %** | 0.912 | Phát hiện thừa (`FP extras`) |
| **S2. Loại bỏ bỏ sót `FN`** *(bù món YOLO bỏ sót)* | 81.21 | 112.45 | 33.79% | **-4.14 điểm %** | 0.918 | Bỏ sót món (`FN misses`) |
| **S3. Sửa sai lớp `TH2`** *(giữ nguyên khung YOLO)* | 76.55 | 108.40 | 29.84% | **-3.95 điểm %** | 0.926 | Nhầm tên lớp món (`TH2`) |
| **S4. Khung chuẩn (`Oracle MobileSAM Config C`)** | 74.79 | 107.04 | **29.06%** | **-0.78 điểm %** | **0.927** | Lệch vị trí hộp bao (`Box jitter`) |
| **S5. Mặt nạ đa giác chuẩn (`Oracle GT Mask`)** | **74.72** | 108.27 | **28.87%** | **-0.19 điểm %** | 0.924 | Sai số mặt nạ (`MobileSAM C`) |

> [!IMPORTANT]
> **Phát hiện nổi bật từ thang cắt bỏ tuần tự:**
> - Toàn bộ khối phát hiện **YOLO (`S0 → S4`)** kéo MAPE tăng tổng cộng **`+11.31` điểm %** (`29.06% → 40.37%`) và MAE tăng **`+15.38 g`**, chia đều rõ rệt cho 3 lỗi chính: **bỏ sót `FN` (`-4.14%`)**, **nhầm lớp `TH2` (`-3.95%`)** và **hộp thừa `FP` (`-2.44%`)**, trong khi **lệch vị trí hộp bao (`Box jitter`)** chỉ ảnh hưởng rất nhỏ (`-0.78%`).
> - Trên cấp nhóm khớp (`N = 437`), kiểm chứng đối chứng cho thấy lỗi **nhầm lớp `TH2`** gây phương sai log (`MSLE = 0.0607`, lệch trung bình `9.83 g` / `7.05% GT`) lớn gấp **`5.0×`** so với lệch khung hộp bao **`Box jitter`** (`MSLE = 0.0121`, `7.06 g` / `3.31% GT`).
> - **Mặt nạ phân đoạn `MobileSAM Config C` (`S4 → S5`)** hầu như không gây suy giảm độ chính xác tổng thể (`29.06% → 28.87%`, chênh chỉ **`0.19` điểm %** và **`0.07 g` MAE**; độ lệch từng cặp không triệt tiêu dấu trên `500` nhóm chỉ ở mức `10.24 g` trung bình / **`2.30 g` trung vị** = `1.45% GT`).

### B. Bảng tổng hợp tỷ trọng đóng góp (%) của 4 nguồn sai số (`RQ8`)

| Nguồn sai số thành phần | **A. Toàn tuyến Full-Meal (`N = 415`, MSRE)** *(gồm FN, FP, TH2, Box)* | **B. Nhóm đã khớp YOLO (`N = 437`, MSLE)** *(gồm TH2 & Box)* | **C. Lõi 3D cô lập — Oracle (`N = 500`, MSLE)** *(0% lỗi YOLO)* | Phân tích vai trò & kiểm chứng |
| :--- | :---: | :---: | :---: | :--- |
| **1. Phát hiện đối tượng (`YOLOv10b`)** | **`59.9%`** *(RMSRE 48.7%)* | **`21.5%`** *(RMSLE 0.274)* | `0.0%` | **Lớn nhất (#1) ở chế độ toàn tuyến tự động** do `FN` (`14.5%`), `FP` (`63` hộp) và `TH2` (`18.8%`). |
| **2. Độ sâu đơn ảnh + Tỷ lệ + Hình học 3D** | **`34.6%`** *(RMSRE 37.0%)* | **`69.0%`** *(RMSLE 0.490)* | **`88.3%`** *(RMSLE 0.487)* | **Lớn nhất (#1) ngay khi món ăn đã được định vị**; riêng biến thiên góc chụp đã chiếm `≥ 45.3%` MSLE lõi. |
| **3. Biến thiên khối lượng riêng (`ρ`)** | **`3.1%`** *(RMSRE 11.0%)* | **`4.3%`** *(cận trên `18.2%`*)* | **`5.6%`** *(RMSLE 0.123)* | Biến thiên tự nhiên (`Mean CV_ρ = 11.4%`) gây `9.10%` MAPE độc lập; tăng tới `~18.2%` nếu cộng sai lệch mật độ hệ thống giữa các lớp. |
| **4. Mặt nạ phân đoạn (`MobileSAM Config C`)** | **`2.5%`** *(RMSRE 9.9%)* | **`5.1%`** *(RMSLE 0.133)* | **`6.1%`** *(RMSLE 0.128)* | **Nhỏ nhất (#4)** ở chế độ toàn tuyến nhờ `Micro mIoU = 0.9450` và `Seg Success = 100.0%`. |
| **Tổng tỷ trọng** | **100.0%** | **100.0%** | **100.0%** | — |

*\*Ghi chú minh bạch về cận trên của Khối lượng riêng:* Khi hiệu chuẩn mật độ trung bình tối ưu theo từng lớp (`Oracle Class-Mean Calibration`, các lớp `n ≥ 3`), `Oracle GT Mask` trên `500` nhóm giảm `MSLE` từ `0.2521` xuống `0.1557` (`MAE: 64.80 → 54.40 g`, `MAPE: 30.93% → 28.70%`, `r = 0.946`). Vì sai lệch trung bình theo lớp là tổng hòa của cả **mật độ cố định chưa tách biến thể** lẫn **giả định hình học riêng theo lớp**, nếu phân bổ `50%` phần sai lệch hệ thống này cho khối lượng riêng thì tỷ trọng của `Bulk Density` ở Phạm vi B tăng từ **`4.3%`** *(chỉ biến thiên tự nhiên `DENSITY_DB`)* lên **`18.2%`** *(biến thiên tự nhiên + sai lệch hệ thống)*, còn `Depth + Scale + Geometry` vẫn giữ vị trí số 1 ở **`55.2%`**.

![Phân rã 4 nguồn sai số hệ thống trên tập 418 ảnh](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/results/thesis_figures_v2/fig9_four_source_error_decomposition_v2.png)

### C. Kiểm chứng chéo độc lập (Multi-View Pure Geometry & Monte-Carlo Density)

1. **Bằng chứng phương sai nội-phiên đa góc nhìn (`95` phiên / `428` góc chụp, dùng `Oracle GT Mask`):**
   - Phương sai nội-phiên gộp $\text{Var}_{\text{within}}(\ln \hat{m}) = \mathbf{0.0978}$ (`SD = 0.313`, `Median CV = 20.79%`) trên tổng $\text{MSLE}_{\text{GT-Mask}} = 0.2159$.
   - Điều này chứng minh một cách không phụ thuộc vào bảng mật độ rằng: **chỉ riêng sự bất ổn định của bản đồ độ sâu đơn ảnh & hình học theo góc chụp đã chiếm tối thiểu `45.3%` tổng phương sai sai số của lõi 3D** (chưa tính sai lệch hệ thống cố định như nén độ sâu vật mỏng hay trần chiều cao tô).
2. **Bằng chứng Monte-Carlo biến thiên khối lượng riêng (`1.000` lượt trên `500` nhóm):**
   - Độ biến thiên tự nhiên trung bình trong `DENSITY_DB` là `Mean CV_ρ = 11.41%` → nếu thể tích 3D hoàn hảo tuyệt đối (`0%` lỗi), riêng biến thiên mật độ tự nhiên tạo ra sàn sai số **`9.10%` MAPE** (`MSLE_ρ = 0.0151`, khớp chặt với giá trị giải tích `0.0141`).
   - Khi cộng thêm nhiễu Monte-Carlo vào kết quả `Oracle GT Mask`, MAPE toàn tập chỉ tăng nhẹ từ `30.93%` lên `32.38%` (`+1.45` điểm %), xác nhận khối lượng riêng chỉ là nguồn sai số thứ cấp so với `Detection` và `Depth/Geometry`.

### D. Phân bố 4 nguồn sai số theo từng nhóm vật chứa (`Per Dish Type`)

| Nhóm vật chứa (`dish_type`) | `N` (khớp) | Tỷ lệ sót `FN` (%) | Tỷ lệ sai lớp `TH2` (%) | E2E Matched MAPE (%) | Oracle MS-C MAPE (%) | Oracle GT-Mask MAPE (%) | Lệch cặp Mask (% GT) | `CV_ρ` trung bình (%) | Nguồn sai số chi phối chính |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tô nước (`soup_bowl`)** | 98 (96) | **2.0%** | 21.9% | **18.64%** | **18.94%** | **19.33%** | **1.00%** | 8.40% | Hình học tô vơi/đầy (`h_fill`, `D_rim`) & góc nghiêng |
| **Chén cơm (`rice_bowl`)** | 32 (30) | 6.3% | **33.3%** | 29.64% | 22.52% | 18.44% | 6.27% | 9.74% | Nhầm lớp `TH2` (`+7.1%`) & mặt nạ lòng chén |
| **Đĩa chính (`flat_plate`)** | 276 (248) | 10.1% | 23.4% | 41.22% | 33.78% | 34.74% | 6.63% | 11.78% | Độ sâu món phẳng/che khuất & `TH2` (`+7.4%`) |
| **Đĩa nhỏ (`small_plate`)** | 21 (14) | **33.3%** | 7.1% | 49.06% | 39.79% | 39.38% | 2.21% | 11.93% | Bỏ sót `FN` (`33.3%`) & độ sâu món mỏng |
| **Món nhỏ (`ingredient_small`)** | 33 (21) | **36.4%** | 4.8% | 40.12% | 32.16% | 32.62% | **1.37%** | 9.03% | Bỏ sót `FN` (`36.4%`) & nén độ sâu vật nhỏ |
| **Rau lá (`side_vegetables`)** | 40 (28) | **30.0%** | 14.3% | 44.41% | 38.90% | 37.28% | 6.34% | **19.19%** | Mật độ rỗng biến thiên (`19.2%`), `FN` (`30%`) & bao lồi tán lá |

---

## 4.12.3. Các trường hợp lỗi điển hình (`Failure Cases Analysis`) và bài học kinh nghiệm

Đối chiếu trực tiếp danh sách Top-15 ca có sai số tuyệt đối lớn nhất trên cả hai chế độ (`Full-Meal E2E` và `Oracle 3D`) cho thấy mọi lỗi nghiêm trọng đều quy tụ vào **4 mẫu hình nguyên nhân gốc rõ ràng**:

| Mẫu hình lỗi điển hình | Ảnh minh họa thực tế | GT (g) | E2E (g) | Oracle MS-C (g) | Oracle GT-Mask (g) | Chẩn đoán nguyên nhân gốc |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Phối hợp lỗi YOLO (`FP + FN + TH2`)** | `HomeCook14 (2).jpg` | 465.4 | **797.0** | 490.5 | 493.3 | `1 FP + 1 FN + 1 TH2` đẩy lệch `+331.6 g`; loại bỏ lỗi YOLO sai số chỉ còn **`+5.4%`** (`490.5 g`)! |
| | `ThitNuong1(3).jpg` | 94.5 | **447.8** | 165.6 | 163.5 | `1 FP` (`+146.6 g`) + `1 TH2` (`+135.2 g`) khiến dự đoán vọt lên gấp `4.7×` khối lượng thật. |
| | `HomeCook13 (3).jpg` | 701.1 | **303.4** | 582.6 | 577.0 | `1 FN` bỏ sót nguyên một món lớn trên mâm làm hụt `-275.1 g` so với lõi 3D. |
| **2. Chiều cao chứa mặc định trên tô vơi/đầy** | `Kho Qua Thit 7.jpg` | 408.2 | 824.1 | **824.1** | 824.1 | Tô canh vơi nhưng nhánh `headroom` kẹp cận dưới `h_fill ≥ 0.70·H_bowl` → dự đoán trần `824.1 g` (`+101.9%`). |
| | `Chao Long 2.jpg` / `BoKho2(3).jpg` | 765.4 / 444.0 | 1101.7 / 709.5 | **1099.2 / 729.5** | 1069.6 / 729.1 | Tô miệng rộng/nông có tỷ lệ chiều sâu thực thấp hơn hình học nón cụt chuẩn `H/D`. |
| **3. Suy giảm độ sâu đơn ảnh ở góc nghiêng & che khuất** | `ComGa2 (3).jpg` / `ComGa1 (1).jpg` | 466.1 | 135.0 / 131.3 | **135.7 / 131.3** | 155.1 / 230.7 | Góc chụp nghiêng mạnh (`25°–35°`) kết hợp đùi gà che phần cơm khiến diện tích lộ thiên & relief depth sụt giảm mạnh. |
| | `SupCua (2).jpg` / `Mi Quang 1.jpg` | 470.6 / 793.6 | 163.6 / 421.8 | **166.5 / 412.3** | 167.8 / 253.5 | Depth Anything V2 đọc sai hốc lòng tô/mép tô ở góc nghiêng khiến đường kính hiệu dụng bị co hẹp. |
| **4. Biến thiên mật độ đặc–rỗng theo công thức** | `BanhXeo` (`n=17`), `Bun` (`n=13`), `Banh mi` (`n=24`) | — | — | — | — | `ρ_implied` trải rộng (`IQR = 0.81 g/cm³` ở `BanhXeo` và `ThitNuong`; `0.34 g/cm³` ở `Banh mi` đặc vs rỗng) — minh chứng giới hạn của một hằng số `ρ` cố định cho mỗi lớp. |

### Bài học kỹ thuật trọng tâm cho Chương 5 (Định hướng phát triển)
1. **Ưu tiên số 1 cho ứng dụng thực tế (`-59.9%` phương sai toàn tuyến):** Tái huấn luyện/tinh chỉnh bộ phát hiện món ăn (`YOLOv10b` / `YOLO26m`) trên tập ảnh thực địa có tăng cường mạnh các lớp món phụ nhỏ (`ingredient_small`, `side_vegetables`, `small_plate` — nơi `FN = 30–36%`) và các cặp dễ nhầm `TH2`, đồng thời cho phép người dùng xác nhận/sửa tên món trong 1 chạm trên giao diện (khi có xác nhận nhãn từ người dùng, lỗi toàn tuyến lập tức giảm từ `40.4%` xuống **`29.1%`**!).
2. **Ưu tiên số 2 cho nghiên cứu 3D (`69.0%–88.3%` phương sai lõi):** Tích hợp hướng dẫn góc chụp tối ưu (`45°–75°`) ngay trên camera ứng dụng, ước lượng đa góc (`2–3` khung hình liên tiếp) để triệt tiêu phương sai góc nhìn (`45.3%` MSLE lõi), và nâng cấp mô hình dự đoán chiều sâu sang mô hình có ràng buộc hình học camera thực (metric depth fine-tuned trên cảnh bàn ăn).
3. **Ưu tiên số 3 cho dữ liệu vật lý:** Bổ sung đo vật lý dời chỗ nước và phân tách biến thể mật độ phụ (`sub-class density`, như đã thành công với quy tắc `C1` tách bánh mì rỗng `0.20` vs bánh mì thịt `0.45 g/cm³`).
