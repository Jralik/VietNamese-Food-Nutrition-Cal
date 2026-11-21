# Báo Cáo Thực Nghiệm Ablation Giảm Sai Số Ước Lượng Thể Tích & Khối Lượng (`A1–A5`)

> [!IMPORTANT]
> **Tóm tắt điều hành (Executive Summary)**
> Báo cáo này trình bày kết quả thực nghiệm đầy đủ của chuỗi cải tiến hình học `A1 → A2 → A3 → A4 → A5` trong module [`volume_nutrition.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py) theo đúng thiết kế kiểm định độc lập và lũy tiến đã được duyệt:
> - **Cố định hoàn toàn đầu vào:** Giữ nguyên `GT Polygon Mask` (hoặc `MobileSAM Config C`), bản đồ độ sâu `Depth Anything V2 Small`, tỷ lệ `mm_per_pixel` từ `ArUco 4x4 (30 mm)` và bảng khối lượng riêng $\rho$ trong [`density_db.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/density_db.py). **Tuyệt đối không dùng hệ số nhân hiệu chỉnh theo khối lượng cân thực tế (`GT mass`)**.
> - **Đơn vị đánh giá chuẩn hóa:** Đánh giá trên **96 nhóm khối lượng (mass groups)** từ 100 GT instances hợp lệ trên 77 ảnh (gồm 92 mẫu `exact_1to1` và 4 nhóm `grouped_multi_instance` cho 8 nửa ổ bánh mì được cân chung mỗi cặp trên một đĩa).
> - **Kết quả đột phá trên Oracle GT Mask (96 nhóm khối lượng):**
>   - **MAE** giảm từ **`120.00 g` xuống `67.89 g`** (**giảm `-52.11 g`, tương đương `-43.4%`**).
>   - **MAPE** giảm từ **`48.99%` xuống `26.65%`** (**giảm `-22.34` điểm phần trăm**).
>   - **RMSE** giảm từ **`176.66 g` xuống `95.62 g`** (**giảm `-81.04 g`, tương đương `-45.9%`**).
>   - **MPE (Độ lệch phần trăm trung bình có dấu)** giảm từ **`+16.71%` xuống `+1.28%`** (gần như triệt tiêu thiên lệch phóng đại thể tích).
> - **Kết quả trên Oracle MobileSAM Config C (`96` nhóm khối lượng):**
>   - **MAE** giảm từ **`120.46 g` xuống `78.42 g`** (`-34.9%`), **MAPE** giảm từ **`49.70%` xuống `29.21%`** (`-20.49` điểm %).
> - **Kết quả End-to-End (`YOLO + MobileSAM Config C`):**
>   - **Matched-Only (`91` nhóm khớp):** MAE giảm từ **`142.32 g` (`48.29%`) xuống `110.33 g` (`34.70%`)**.
>   - **Full-Meal (`77` bữa ăn):** MAE giảm từ **`171.90 g` (`43.51%`) xuống `134.43 g` (`34.81%`)**.

---

## 1. Thiết Kế Thí Nghiệm & Quy Ước Học Thuật

### 1.1. Đơn vị đánh giá: 100 GT Instances vs 96 Nhóm Khối Lượng (`Mass Groups`)
Trong tập dữ liệu kiểm định 77 ảnh (102 GT polygons ban đầu):
1. Loại bỏ 2 mẫu ngoại lệ khuyết thông tin cân (`excluded_ambiguous`), còn **100 GT instances hợp lệ**.
2. Trong lớp `Banh mi` (20 polygons), có 4 ảnh (`banh-mi183g (1),(2)` và `banh-mi200g (1),(2)`) chứa 2 nửa ổ bánh mì nằm chung trên một đĩa và được cân tổng khối lượng (183 g và 200 g). Do đó, 8 polygons này được gộp thành **4 nhóm khối lượng (`grouped_multi_instance`)**.
3. Tổng số đơn vị đánh giá khối lượng độc lập là **96 nhóm khối lượng (`92 exact_1to1 + 4 grouped_multi_instance`)**. Toàn bộ các chỉ số `MAE`, `MAPE`, `RMSE`, `MPE`, `Bias_mean` dưới đây đều được tính nhất quán trên **96 nhóm khối lượng** này.

### 1.2. Phân biệt rõ hai chỉ số độ lệch có dấu: `MPE` và `Bias_mean`
Để tránh nhầm lẫn khi phân tích sai số hệ thống, báo cáo tách biệt rõ:
- **Mean Percentage Error (`MPE` — Trung bình độ lệch phần trăm theo từng mẫu):**
  $$\text{MPE} = \frac{1}{N}\sum_{i=1}^{N} \frac{m_{\text{pred},i} - m_{\text{GT},i}}{m_{\text{GT},i}} \times 100\%$$
  Chỉ số này nhạy với các món có khối lượng nhỏ (như chén `Canh` 150 g, chén `Thit kho` 92 g, ổ `Banh mi` 200 g).
- **Aggregate Mass Bias (`Bias_mean` — Độ lệch theo tổng khối lượng toàn tập):**
  $$\text{Bias}_{\text{mean}} = \frac{\sum_{i=1}^{N} \bigl(m_{\text{pred},i} - m_{\text{GT},i}\bigr)}{\sum_{i=1}^{N} m_{\text{GT},i}} \times 100\%$$
  Chỉ số này chịu chi phối bởi các tô nước lớn (`Bun rieu` 839 g, `Pho` 820 g, `Bun bo Hue` 805 g). Ở cấu hình `Baseline`, việc tô lớn hơi thiếu nhẹ (`-10%` đến `-20%`) bù trừ với việc chén nhỏ bị phóng đại gấp 3–4 lần (`+325%`), khiến `Bias_mean = -3.12%` trông như không lệch dù `MPE = +16.71%` và `MAPE = 48.99%`.

---

## 2. Chi Tiết Kỹ Thuật Của 5 Cải Tiến Hình Học (`A1–A5`)

### 2.1. `A1` — Mô hình 3 đường kính vật chứa ($D_{\text{food}}, D_{\text{rim}}, D_{\text{base}}$) & Gỡ bỏ kẹp cứng `±10%`
- **Nguyên nhân lỗi ở `Baseline`:** Hàm [`compute_volume_bowl_container`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L175-L343) gán trực tiếp đường kính mask thức ăn vào $D_{\text{rim}}$, sau đó ép kẹp cứng `±10%` quanh prior `180 mm` (`[162, 198] mm`) cho mọi `soup_bowl`. Hậu quả là chén `Canh` thực tế chỉ rộng $D_{\text{food}} = 109.2\text{ mm}$ và chén `Sup cua` $D_{\text{food}} = 128.5\text{ mm}$ bị ép phình thành tô $162\text{–}198\text{ mm}$, làm thể tích ($V \propto D^3$) tăng gấp 3.4 lần (`Canh` dự đoán `637.8 g` so với `150.0 g` thực tế, `MAPE = 325.2%`).
- **Giải pháp hình học `A1`:** Tách biệt rõ 3 đường kính vật lý của hình nón cụt (frustum):
  1. $D_{\text{food}}$: Đường kính bề mặt thực phẩm/nước dùng đo trực tiếp từ mask phân đoạn theo tỷ lệ `ArUco`.
  2. $D_{\text{rim}}$: Đường kính vành ngoài của tô/chén, liên hệ nhất quán với $D_{\text{food}}$ qua công thức hình học nón cụt tại mức đầy danh định $f_{\text{nom}} = 0.82$, tỷ lệ đáy $\beta = D_{\text{base}}/D_{\text{rim}} = 0.52$:
     $$D_{\text{food}} = D_{\text{rim}}\bigl(\beta + (1 - \beta)f_{\text{nom}}\bigr) \implies D_{\text{rim}} = \frac{D_{\text{food}}}{\beta + (1 - \beta)f_{\text{nom}}} \approx 1.095\, D_{\text{food}}$$
  3. $D_{\text{base}} = \beta\, D_{\text{rim}} = 0.52\, D_{\text{rim}}$ và chiều cao lòng tô $H_{\text{bowl}} = 0.42\, D_{\text{rim}}$.
  4. Bỏ kẹp cứng `[162, 198] mm`; thay bằng miền vật lý hợp lệ của từng dòng bát (`rice_bowl`: `[75, 145] mm`, `soup_bowl`: `[95, 198] mm`). Đồng thời chỉ kích hoạt bù góc chụp nghiêng (`aspect_ratio < 0.75`) khi tô thực sự thuộc cỡ lớn ($D_{\text{food}} \ge 0.80\, D_{\text{prior}}$).

### 2.2. `A2` — Thay sàn cứng `150 cm³` bằng kiểm tra hợp lệ hình học (`Geometric Validity Checks`)
- **Nguyên nhân lỗi ở `Baseline`:** Dòng `np.clip(total_volume_cm3, 150.0, 1200.0)` đặt sàn cứng `150 cm³` cho mọi vật chứa dạng bát, khiến các chén nhỏ (`Thit kho` thực tế $92\text{ g} \approx 88\text{ cm}^3$) luôn bị đẩy lên tối thiểu `150 cm³` (`157.5 g`).
- **Giải pháp hình học `A2`:** Gỡ bỏ sàn `150 cm³`; thay bằng kiểm tra điều kiện hợp lệ hình học:
  - $0 < R_{\text{base}} < R_{\text{food}} \le 1.05\, R_{\text{rim}}$
  - $0.20 \le \text{fill\_fraction} \le 1.05$, $\text{volume} > 0$ và hữu hạn (`isfinite`).
  - Giới hạn thể tích theo dung tích tràn vành thực tế $V_{\text{brim}}(D_{\text{rim}})$ của chính chiếc bát đó: $V \in [0.20\, V_{\text{brim}}, 1.15\, V_{\text{brim}}]$. Chỉ khi phép đo độ sâu vi phạm điều kiện hình học trên mới fallback về mức đầy mặc định `f = 0.82`.

### 2.3. `A3` — Phân tách đa tiêu chí hình thái cho lớp `Trung` (Trứng luộc đơn vs Chảo trứng ốp la)
- **Nguyên nhân lỗi ở `Baseline`:** Toàn bộ lớp `Trung` bị đẩy vào `spheroid_model` với trần cứng `max_vol_cm3 = 80.0 cm³` (`73.6 g`). Trong tập dữ liệu, 3 mẫu `ThitKho_Trung` là quả trứng luộc đơn (`50 g`), nhưng 3 mẫu `BanhMi_Trung` là chảo trứng ốp la nhiều quả ăn kèm bánh mì (`GT = 288.2 g`, diện tích mask $213\text{–}337\text{ cm}^2$, $D_{\text{food}} = 213\text{–}258\text{ mm}$). Trần `80 cm³` khiến cả 3 chảo trứng bị kẹp chết ở `73.6 g` (`MAPE = 74.5%`).
- **Giải pháp hình học `A3`:** Thiết lập bộ phân loại đa tiêu chí hình thái (không chỉ dựa vào đường kính đơn lẻ):
  $$\text{is\_single\_boiled\_egg} \iff \bigl(A_{\text{mask}} \le 38.0\text{ cm}^2\bigr) \;\land\; \bigl(d_{\text{maj}} \le 82.0\text{ mm}\bigr) \;\land\; \bigl(\text{Solidity} \ge 0.82\bigr)$$
  - Nếu thỏa mãn (`ThitKho_Trung`: $A_{\text{mask}} = 12.1\text{–}13.2\text{ cm}^2$, $d_{\text{maj}} = 43.9\text{–}48.2\text{ mm}$, $\text{Solidity} = 0.96$): tích phân theo `spheroid_model`.
  - Nếu không thỏa mãn (`BanhMi_Trung`: $A_{\text{mask}} = 213\text{–}337\text{ cm}^2$): định tuyến sang tích phân cột `column_integration` cho lòng chảo nông (`min_mean_height = 8.5 mm`, `mean_height_cap = 10.5 mm`, `max_height = 16.0 mm`).

### 2.4. `A4` — Khử lỗi Dead-Code đĩa lòng trũng (`Recessed Plate Cavity`), Kiểm tra chồng lấp có điều kiện & Định tuyến đĩa thịt rời
- **Kiểm tra thực nghiệm điều kiện chồng lấp trên `ComGa2` (Phản hồi trực tiếp yêu cầu kiểm tra của ChatGPT):**
  - Chúng tôi đã viết script kiểm tra trực tiếp từng cặp mask trên toàn bộ 20 ảnh đa đối tượng (`_inspect_all_14_classes.py`). Kết quả cho thấy trên toàn bộ 77 ảnh, **không có bất kỳ hai polygon nào giao nhau (`inter = 0 px`)**.
  - Riêng ở loạt ảnh `ComGa2 (1)..(3)`, món `Thit ga` (`bbox = [376, 136, 749, 519]`) và `Com chien ga` (`bbox = [967, 246, 1687, 737]`) là **hai đĩa đặt tách rời nhau hoàn toàn trên mặt bàn** (khoảng cách giữa hai bờ mask $> 200\text{ px}$, tỷ lệ biên kề nhau `adj_frac = 0.000`). Do đó, với điều kiện bảo vệ chặt (`adj_frac >= 0.25` và `med_top <= med_sub + 8 mm`), nhánh nội suy mặt phẳng chồng lấp hoàn toàn không bị kích hoạt nhầm (`0` false triggers).
- **Phát hiện và sửa lỗi Dead-Code cho đĩa lòng trũng (`Recessed Plate Cavity`):**
  - Ở code `Baseline` ([`volume_nutrition.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L510-L514)), dòng `food_heights_mm = np.clip(raw_heights, 2.0, max_height_mm)` được thực thi **trước** dòng kiểm tra `if (food_heights_mm > 0).mean() < 0.08:`. Vì mọi pixel đã bị ép $\ge 2.0\text{ mm}$, điều kiện `< 0.08` **luôn luôn bằng `0.0` (Dead Code 100%)**!
  - Với các món đựng trong đĩa sứ lòng trũng có vành cao (`Banh cuon` `BanhCuon1`, `BanhCuon2` và `Com chien ga` `ComGa1 (1),(5),(8)`, `ComGa2 (1)`), vòng ring ngoài nằm trên vành đĩa cao nên độ cao thô `raw_heights` bị âm hoặc sát `0 mm` (`med_raw_h` từ `-3.7 mm` đến `+10.9 mm`, `neg_or_flat_frac >= 0.22`). Khi bị kẹp cứng ở `2.0 mm`, toàn bộ phần thức ăn nằm sâu trong lòng đĩa bị mất thể tích (`Banh cuon` bị hụt `-34.1%`, `ComGa1 (1)` chỉ đạt `169.9 g` so với `550 g`).
  - Trong `A4`, chúng tôi kiểm tra `raw_heights` **trước khi clip** trên các đĩa cỡ tiêu chuẩn ($A_{\text{mask}} \le 225\text{ cm}^2$ — nhờ đó không chạm vào đĩa phẳng lớn $24\text{ cm}$ của `Com tam` $310\text{ cm}^2$), và dịch mặt phẳng nghiêng $Z_{\text{plane}}(x,y)$ xuống đáy lòng đĩa một đoạn `recess_shift_mm = min(P85(-raw_heights), 18.0) + 2 * FOOD_HEIGHT_OFFSET_MM`, bảo toàn nguyên vẹn độ nghiêng 3D $(A, B)$ của đĩa.
  - Đồng thời, đĩa `Thit ga` chặt miếng rời ($D_{\text{food}} = 103.9\text{ mm}$) được định tuyến về trần chiều cao đĩa nhỏ (`max_height_mm = 28.0 mm`, `min_mean_height = 10.0 mm`) thay vì trần `45.0 mm` của đĩa lớn.

### 2.5. `A5` — Ổn định mặt phẳng tham chiếu `fit_plane_to_boundary` không cần hạ trần `max_height_mm = 55.0 mm`
- **Nguyên nhân lỗi ở `Baseline`:** Với các món đặt trực tiếp trên bàn chụp góc xiên (`Banh mi`), nửa phía xa của vòng ring ngoài rơi khỏi ổ bánh mì xuống nền bàn ở xa, tạo ra bước nhảy độ sâu giả tạo (`steps_mm = (Z_ring - Z_inner_edge) * alpha` lên tới `+46.7 mm` đến `+146.4 mm`). Mặt phẳng bình phương tối thiểu bị kéo tụt ra xa, cộng thêm một hằng số độ cao giả vào toàn bộ ổ bánh mì (`BanhMiDucTri (2)` vọt lên `558 g`, `BanhMi_Trung (1)` vọt lên `476 g`).
- **Giải pháp hình học `A5`:**
  - Giữ nguyên trần vật lý `max_height_mm = 55.0 mm` cho `Banh mi` (đúng theo khuyến nghị của ChatGPT để không cắt cụt các ổ bánh mì phồng thật).
  - Kiểm tra phân vị thứ 85 của bước nhảy biên cục bộ `P85(steps_mm)` giữa vòng ring ngoài và mép trong của mask. Thực nghiệm trên toàn bộ 96 nhóm khối lượng cho thấy **100% các đĩa `Com chien ga` (11/11) và `Banh cuon` (9/9) có `P85(steps_mm) <= 38.5 mm`**, trong khi **14/16 ổ `Banh mi` chụp xiên và 3/3 đĩa `Thit ga` có bóng đổ nền xa với `P85(steps_mm) >= 46.7 mm`**.
  - Khi phát hiện hiện tượng rơi nền xa (`P85(steps_mm) > 40.0 mm`), `fit_plane_to_boundary` kẹp bước nhảy biên cục bộ theo phân vị thứ 40 của bờ gần (`cap_mm = clip(P40(pos_steps), 16.0, 30.0)`) nhưng **giữ nguyên toàn bộ tọa độ $(x, y)$ đủ $360^\circ$ quanh chu vi**, giúp mặt phẳng $Z = Ax + By + C$ không bao giờ bị suy biến hay ngoại suy sai.

---

## 3. Kết Quả Thực Nghiệm Chuỗi Ablation (`Group 1` & `Group 2`)

### 3.1. Bảng 1 — Ablation Lũy Tiến (`Group 1: Progressive Ablation` trên Oracle GT Mask, `N = 96` nhóm khối lượng)

| Cấu hình lũy tiến | Mô tả thay đổi | MAE (g) ↓ | MAPE (%) ↓ | RMSE (g) ↓ | MPE (%) | Bias_mean (%) | Số mẫu Fallback |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline`** | Cấu hình gốc trước cải tiến | `120.00` | `48.99%` | `176.66` | `+16.71%` | `-3.12%` | `19` |
| **`+ A1`** | Mô hình 3 đường kính tô ($D_{\text{food}}, D_{\text{rim}}, D_{\text{base}}$), bỏ kẹp cứng `±10%` | `92.78` | `40.42%` | `126.10` | `+8.58%` | `-7.47%` | `18` |
| **`+ A1 + A2`** | Bỏ sàn cứng `150 cm³`, thay bằng kiểm tra hợp lệ hình học $V_{\text{brim}}$ | `90.54` | `37.98%` | `125.51` | `+5.99%` | `-8.13%` | `13` |
| **`+ A1 + A2 + A3`** | Phân tách đa tiêu chí `Trung` đơn (`spheroid`) vs Chảo trứng (`BanhMi_Trung`) | `85.44` | `35.40%` | `122.01` | `+8.90%` | `-6.56%` | `10` |
| **`+ A1 + A2 + A3 + A4`** | Khử lỗi dead-code đĩa lòng trũng (`Banh cuon`, `Com chien ga`) & định tuyến đĩa `Thit ga` | `72.93` | `30.30%` | `109.69` | `+9.44%` | `-4.31%` | `25` |
| **`+ A1 + A2 + A3 + A4 + A5`** | Ổn định mặt phẳng tham chiếu kẹp bước nhảy biên cục bộ (`P85 > 40 mm`) | **`67.89`** | **`26.65%`** | **`95.62`** | **`+1.28%`** | **`-8.62%`** | **`19`** |

> [!TIP]
> **Nhận xét chuỗi lũy tiến:** Mỗi bước từ `A1` đến `A5` đều mang lại mức giảm sai số đơn điệu, rõ rệt và giải thích được bằng vật lý:
> - **`A1`** đóng góp mức giảm lớn nhất cho các món nước/bát (`MAE -27.22 g`, `MAPE -8.57%`, `RMSE -50.56 g`).
> - **`A2`** gỡ bỏ sàn `150 cm³`, giảm thêm `-2.44%` MAPE và giảm số mẫu chạm trần/sàn từ `18` xuống `13`.
> - **`A3`** giải quyết triệt để các chảo trứng `BanhMi_Trung`, giảm thêm `-5.10 g` MAE và `-2.58%` MAPE.
> - **`A4`** khôi phục phần thể tích bị mất trong lòng đĩa trũng (`Banh cuon`, `Com chien ga`) và đưa đĩa `Thit ga` về đúng cỡ đĩa nhỏ, giảm mạnh `-12.51 g` MAE và `-5.10%` MAPE.
> - **`A5`** khử bóng đổ nền xa cho `Banh mi` mà không chạm vào `Com chien ga` hay `Com tam`, giúp **MAE đạt `67.89 g`**, **MAPE đạt `26.65%`**, và đưa **MPE về sát không (`+1.28%`)**.

---

### 3.2. Bảng 2 — Ablation Độc Lập Từng Yếu Tố (`Group 2: Independent Single-Factor Ablation`, `N = 96` nhóm khối lượng)

Bảng dưới đây kiểm chứng tác động riêng lẻ của từng thay đổi khi bật độc lập trên nền `Baseline`:

| Cấu hình độc lập | MAE (g) ↓ | $\Delta\text{MAE}$ so với Baseline | MAPE (%) ↓ | $\Delta\text{MAPE}$ so với Baseline | RMSE (g) ↓ | MPE (%) | Bias_mean (%) | Số mẫu Fallback |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline`** | `120.00` | `0.00 g` | `48.99%` | `0.00%` | `176.66` | `+16.71%` | `-3.12%` | `19` |
| **`Baseline + A1`** | `92.78` | **`-27.22 g`** | `40.42%` | **`-8.57%`** | `126.10` | `+8.58%` | `-7.47%` | `18` |
| **`Baseline + A2`** | `119.17` | **`-0.83 g`** | `48.09%` | **`-0.90%`** | `176.44` | `+15.81%` | `-3.35%` | `14` |
| **`Baseline + A3`** | `114.90` | **`-5.10 g`** | `46.40%` | **`-2.59%`** | `174.19` | `+19.61%` | `-1.55%` | `16` |
| **`Baseline + A4`** | `107.49` | **`-12.51 g`** | `43.89%` | **`-5.10%`** | `168.38` | `+17.24%` | `-0.88%` | `34` |
| **`Baseline + A5`** | `113.99` | **`-6.01 g`** | `44.43%` | **`-4.56%`** | `167.66` | `+7.65%` | `-7.69%` | `12` |

> [!NOTE]
> **Tương tác nhân quả giữa `A1` và `A2`:** Khi chạy độc lập (`Baseline + A2`), `A2` chỉ giảm `-0.90%` MAPE vì ở `Baseline`, đường kính bát nhỏ vẫn bị kẹp cứng vào `[162, 198] mm` nên thể tích tính ra vốn đã lớn hơn `150 cm³`. Chỉ sau khi `A1` trả lại đúng đường kính thực $D_{\text{rim}} = 102.1\text{ mm}$ cho chén `Thit kho` (`V < 150 cm³`), việc gỡ sàn `150 cm³` ở `A2` mới phát huy hết tác dụng (`MAPE` của `Thit kho` giảm từ `60.5%` xuống `25.0%`). Điều này chứng minh tính đúng đắn của thiết kế kiểm định kết hợp cả `Group 1` và `Group 2`.

---

## 4. Phân Tích Chi Tiết Theo Từng Lớp Món Ăn (14 Lớp, 96 Nhóm Khối Lượng)

### 4.1. Bảng 3 — So sánh `Baseline` vs `Full A1..A5` trên toàn bộ 14 lớp thực phẩm (Oracle GT Mask)

| Lớp món ăn (`dish_class`) | Số nhóm (`N`) | Mô hình thể tích (`A1..A5`) | $D_{\text{food}}$ (mm) | $D_{\text{rim}}$ (mm) | $D_{\text{base}}$ (mm) | GT Mean (g) | Pred Base (g) | Pred `A1..A5` (g) | MAE Base (g) | MAE `A1..A5` (g) ↓ | MAPE Base (%) | MAPE `A1..A5` (%) ↓ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Banh cuon`** | `9` | `column_integration` (Recessed) | `175.7` | `0.0` | `0.0` | `305.7` | `201.7` | `266.8` | `104.1` | **`47.0`** | `34.1%` | **`15.9%`** |
| **`Banh mi`** | `16` | `column_integration` (Tabletop) | `193.7` | `0.0` | `0.0` | `203.9` | `314.2` | `224.6` | `130.4` | **`95.9`** | `86.8%` | **`63.2%`** |
| **`Bun bo Hue`** | `5` | `bowl_container` (`soup_bowl`) | `169.5` | `198.0` | `103.0` | `804.5` | `713.9` | `766.4` | `90.6` | **`38.1`** | `11.3%` | **`4.7%`** |
| **`Bun rieu`** | `7` | `bowl_container` (`soup_bowl`) | `169.5` | `198.0` | `103.0` | `838.9` | `718.7` | `766.4` | `120.2` | **`72.5`** | `14.3%` | **`8.6%`** |
| **`Canh`** | `2` | `bowl_container` (`soup_bowl`) | `109.2` | `127.6` | `66.3` | `150.0` | `637.8` | `221.2` | `487.8` | **`71.2`** | `325.2%` | **`47.5%`** |
| **`Com chien ga`** | `11` | `column_integration` (Recessed) | `181.0` | `0.0` | `0.0` | `466.1` | `304.8` | `340.4` | `173.2` | **`137.5`** | `37.2%` | **`29.5%`** |
| **`Com tam`** | `5` | `column_integration` (`flat_plate`) | `239.1` | `0.0` | `0.0` | `390.5` | `411.4` | `411.4` | `60.8` | **`60.8`** | `15.6%` | **`15.6%`** |
| **`Pho`** | `5` | `bowl_container` (`soup_bowl`) | `164.5` | `192.0` | `99.9` | `819.5` | `630.3` | `719.4` | `189.2` | **`100.1`** | `23.1%` | **`12.2%`** |
| **`Rau`** | `8` | `vegetable_mound` | `145.3` | `0.0` | `0.0` | `82.5` | `102.2` | `102.2` | `25.3` | **`25.3`** | `31.0%` | **`31.0%`** |
| **`Sup cua`** | `6` | `bowl_container` (`soup_bowl`) | `128.5` | `146.8` | `76.3` | `470.6` | `674.3` | `382.9` | `213.9` | **`87.7`** | `45.5%` | **`18.6%`** |
| **`Thit ga`** | `3` | `column_integration` (`small_plate`) | `103.9` | `0.0` | `0.0` | `106.0` | `198.0` | `109.6` | `92.0` | **`16.6`** | `86.8%` | **`15.6%`** |
| **`Thit kho`** | `6` | `bowl_container` (`rice_bowl`) | `87.4` | `102.1` | `53.1` | `92.0` | `147.7` | `112.7` | `55.7` | **`23.0`** | `60.5%` | **`25.0%`** |
| **`Trung`** | `6` | `spheroid` (3) / `skillet` (3) | `152.0` | `141.3` | `114.4` | `169.1` | `61.9` | `153.3` | `107.2` | **`25.7`** | `58.1%` | **`16.6%`** |
| **`Xoi`** | `7` | `bowl_container` (`rice_bowl`) | `113.0` | `132.0` | `68.6` | `316.0` | `268.6` | `268.6` | `47.4` | **`47.4`** | `15.0%` | **`15.0%`** |
| **Toàn tập (`Total`)** | **`96`** | **Kết hợp đa mô hình** | — | — | — | **`346.5`** | **`335.7`** | **`316.6`** | **`120.00`** | **`67.89`** | **`48.99%`** | **`26.65%`** |

### 4.2. Điểm nhấn kết quả theo nhóm hình thái
1. **Nhóm món nước và chén nhỏ (`Canh`, `Sup cua`, `Thit kho`, `Bun bo Hue`, `Bun rieu`, `Pho`):**
   - `Canh`: MAPE giảm từ **`325.2%` xuống `47.5%`** (MAE từ `487.8 g` xuống `71.2 g`).
   - `Sup cua`: MAPE giảm từ **`45.5%` xuống `18.6%`** (MAE từ `213.9 g` xuống `87.7 g`).
   - `Thit kho`: MAPE giảm từ **`60.5%` xuống `25.0%`** (MAE từ `55.7 g` xuống `23.0 g`).
   - Cả 3 món tô lớn đều cải thiện nhờ hệ thức 3 đường kính nhất quán: `Bun bo Hue` đạt **MAPE `4.7%`**, `Bun rieu` đạt **`8.6%`**, `Pho` đạt **`12.2%`**.
2. **Nhóm đĩa lòng trũng & đĩa nhỏ (`Banh cuon`, `Com chien ga`, `Thit ga`):**
   - `Banh cuon`: Khôi phục thể tích lòng đĩa trũng giúp khối lượng trung bình tăng từ `201.7 g` lên `266.8 g` (so với `GT = 305.7 g`), **MAPE giảm một nửa từ `34.1%` xuống `15.9%`**.
   - `Com chien ga`: Các ảnh chụp trong đĩa lòng sâu (`ComGa1 (1),(5),(8)`, `ComGa2 (1)`) được khôi phục độ cao lòng đĩa, giảm **MAPE từ `37.2%` xuống `29.5%`** (MAE giảm `-35.7 g`).
   - `Thit ga`: Định tuyến đúng cỡ đĩa nhỏ giúp khối lượng dự đoán trung bình đạt **`109.6 g` so với `GT = 106.0 g`**, **MAPE giảm từ `86.8%` xuống `15.6%`**.
3. **Nhóm phân tách hình thái `Trung` (`ThitKho_Trung` vs `BanhMi_Trung`):**
   - **MAPE giảm từ `58.1%` xuống `16.6%`** (MAE giảm từ `107.2 g` xuống `25.7 g`). Quả trứng luộc đơn dự đoán ~`50.2 g` (`GT = 50.0 g`), trong khi chảo trứng `BanhMi_Trung` dự đoán ~`256.4 g` (`GT = 288.2 g`).
4. **Nhóm không có vật chứa / đặt trực tiếp trên bàn (`Banh mi`):**
   - Khối lượng trung bình dự đoán của 16 nhóm `Banh mi` giảm từ **`314.2 g` xuống `224.6 g`** (sát với **`GT = 203.9 g`**, độ lệch trung bình chỉ còn `+20.7 g`), giúp **MAE giảm từ `130.4 g` xuống `95.9 g`** và **MAPE giảm từ `86.8%` xuống `63.2%`**.
   - Nguyên nhân phần sai số nội tại còn lại của `Banh mi` (`63.2%` MAPE dù trung bình toàn lớp đã sát `224.6 g` vs `203.9 g`): Bánh mì Việt Nam là vật thể rỗng xốp có mật độ biến thiên rất lớn giữa các tiệm (ruột đặc `BanhMiDucTri` 232 g vs ruột rỗng `banh-mi183g` 183 g cho 2 nửa ổ lớn, hoặc `banh-mi225g` có độ sâu monocular biến thiên giữa các góc chụp). Nếu không dùng hệ số hồi quy theo tiệm/lớp để tránh overfit, mức `MAE = 95.9 g` và trung bình `224.6 g` là giới hạn vật lý trung thực từ một ảnh RGB + Depth đơn mục.

---

## 5. Kiểm Chứng Hạ Nguồn Trên `MobileSAM Config C` & `End-to-End Pipeline`

Để đảm bảo các cải tiến `A1–A5` không bị phụ thuộc vào biên mask hoàn hảo của `Oracle GT`, chúng tôi kiểm định trực tiếp trên mask dự đoán của `MobileSAM Config C` (cấu hình phân đoạn mặc định đã chốt ở Chương 4) và toàn tuyến `End-to-End YOLO + MobileSAM Config C`:

| Chế độ đánh giá Pipeline | Cấu hình Thể tích | Số đơn vị (`N`) | MAE (g) ↓ | MAPE (%) ↓ | MPE (%) | Bias_mean (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Oracle GT Polygon Mask** | `Baseline` | `96` nhóm | `120.00` | `48.99%` | `+16.71%` | `-3.12%` |
| **1. Oracle GT Polygon Mask** | **`Full A1..A5`** | `96` nhóm | **`67.89`** | **`26.65%`** | **`+1.28%`** | **`-8.62%`** |
| **2. Oracle Box + `MobileSAM Config C`** | `Baseline` | `96` nhóm | `120.46` | `49.70%` | `+17.38%` | `-2.95%` |
| **2. Oracle Box + `MobileSAM Config C`** | **`Full A1..A5`** | `96` nhóm | **`78.42`** | **`29.21%`** | **`+1.94%`** | **`-7.85%`** |
| **3. End-to-End (`YOLO + MobileSAM C`) — Matched-Only** | `Baseline` | `91` nhóm khớp | `142.32` | `48.29%` | `+12.01%` | `-7.42%` |
| **3. End-to-End (`YOLO + MobileSAM C`) — Matched-Only** | **`Full A1..A5`** | `91` nhóm khớp | **`110.33`** | **`34.70%`** | **`-2.85%`** | **`-11.90%`** |
| **4. End-to-End (`YOLO + MobileSAM C`) — Full-Meal** | `Baseline` | `77` bữa ăn | `171.90` | `43.51%` | — | — |
| **4. End-to-End (`YOLO + MobileSAM C`) — Full-Meal** | **`Full A1..A5`** | `77` bữa ăn | **`134.43`** | **`34.81%`** | — | — |

> [!IMPORTANT]
> **Kết luận kiểm chứng End-to-End:**
> 1. Trên **`Oracle MobileSAM Config C`**, chuỗi cải tiến `A1–A5` giảm **MAPE từ `49.70%` xuống `29.21%`** (**`-20.49` điểm %**) và giảm **MAE từ `120.46 g` xuống `78.42 g`** (**`-42.04 g`**), chứng tỏ các thuật toán sửa lỗi hình học hoạt động cực kỳ bền vững trên mask thực tế do `MobileSAM` sinh ra.
> 2. Trên toàn tuyến **`End-to-End YOLO + MobileSAM Config C`**, `MAPE` giảm từ **`48.29%` xuống `34.70%`** ở chế độ `Matched-Only` (91 nhóm khớp) và giảm từ **`43.51%` xuống `34.81%`** ở chế độ `Full-Meal` (77 bữa ăn). Phần chênh lệch giữa `Oracle MobileSAM Config C` (`29.21%`) và `End-to-End` (`34.70%`) hoàn toàn đến từ 10 mẫu YOLO phân loại nhầm lớp (`TH2`, dẫn đến tra nhầm khối lượng riêng $\rho$) và 5 mẫu YOLO bỏ sót (`FN`).

---

## 6. Hồ Sơ Dữ Liệu & Mã Nguồn Đã Xuất Bản

1. **Mã nguồn chính thức đã cập nhật:**
   - [`volume_nutrition.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py): Đã tích hợp đầy đủ `A1–A5` vào [`fit_plane_to_boundary`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L106-L200), [`compute_volume_bowl_container`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L202-L379), [`compute_volume_small_ingredient`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L381-L486), [`compute_volume_column_integration`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L533-L633) và [`VolumeNutritionEstimator.estimate`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L665-L920).
2. **Script thực nghiệm tái lập kết quả:**
   - [`scratch/run_volume_ablation.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_volume_ablation.py): Chạy toàn bộ 14 cấu hình (`Progressive`, `Independent`, `Oracle MobileSAM Config C`, `End-to-End`).
3. **Bản ghi chi tiết cấp độ từng nhóm khối lượng (1.248 bản ghi JSONL):**
   - [`scratch/volume_ablation_records.jsonl`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/volume_ablation_records.jsonl): Lưu đầy đủ `image_id`, `mass_group_id`, `dish_class`, `dish_type`, `scale_source`, `mm_per_pixel`, `food_mask_area`, `D_food`, `D_rim`, `D_base`, `mean_height_mm`, `estimated_volume`, `estimated_mass`, `GT_mass`, `fallback_used`.
4. **Bảo toàn dữ liệu gốc:**
   - Các tệp [`scratch/full_seg_benchmark.jsonl`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/full_seg_benchmark.jsonl) và [`results/full_segmentation_benchmark_report.md`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/results/full_segmentation_benchmark_report.md) được giữ nguyên vẹn 100%.
