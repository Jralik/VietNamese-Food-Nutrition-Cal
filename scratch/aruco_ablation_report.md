# Báo Cáo Thực Nghiệm ArUco & Depth-Anchoring Ablation (5 Cấu Hình)

> [!IMPORTANT]
> **Phạm vi, Nguồn Dữ Liệu và Quy ước Học thuật:**
> - Thực nghiệm này đánh giá đóng góp thành phần của mô-đun hiệu chỉnh tỷ lệ **ArUco 30 mm** và **Neo độ sâu (`solvePnP` Depth Anchoring)** trên cùng **77 ảnh / 96 nhóm khối lượng (`Oracle GT Polygon Mask`)** đã sử dụng để phát triển chuỗi mô hình hình học `A1–A5`.
> - **Làm rõ nguồn dữ liệu của Config 1 (`Config_1_GT_Assisted_Plate_Heuristic`):** Vì toàn bộ bảng thực nghiệm `Oracle GT Polygon Mask` cố định mặt nạ phân vùng, khung giới hạn (`bbox_xyxy`) và nhãn lớp món ăn (`vietfood67_class_name`) từ `gt_mapping_verified.json` cho cả 5 cấu hình, cơ chế ước lượng tỷ lệ theo vật chứa ở Config 1 (`scaled_dets` trong `_try_dish_guided_scale` và công thức ghi đè `mm_per_px = expected_dia / max(bw, bh)` trên các lớp vật chứa) sử dụng chính khung giới hạn và nhãn lớp GT này. Do đó, Config 1 là **GT-Assisted Plate Heuristic** (cận trên của phương pháp heuristic khi biết chính xác vị trí và loại vật chứa, giúp cô lập thuần túy cơ chế tỷ lệ ngang/neo độ sâu mà không bị nhiễu bởi lỗi phát hiện của YOLO), **không phải** là heuristic hoàn toàn độc lập với ground truth.
> - Do 77 ảnh này là **tập dữ liệu kiểm định thực nghiệm (validation/evaluation benchmark)** chứ không phải tập kiểm thử độc lập (held-out test set), kết quả dưới đây dùng để chứng minh **cơ chế vật lý và đóng góp của từng biến số điều khiển**, không dùng để tuyên bố khả năng tổng quát hóa ngoài mẫu.
> - Toàn bộ 5 cổng kiểm định tự động (**Gate 1 → Gate 5**) đều đạt 100%: `Config_3b_Full_ArUco` tái lập tương đương về mặt số học trong dung sai kiểm định toàn bộ 96 nhóm khối lượng của báo cáo `A1–A5` (`MAE = 67.89 g`, `MAPE = 26.65%`, `RMSE = 95.62 g`, `MPE = +1.28%`, `Bias_mean = -8.62%`, `Fallback = 19`).

---

## 1. Kiểm Định Hình Học Marker ArUco & Phân Rã Nhiễu Mặt Phẳng (77 Ảnh)

### 1.1. Thống kê nguồn tỷ lệ ngang trên 77 ảnh (Gate 1 & Gate 2)
- **Chế độ chuẩn (Ưu tiên ArUco — Config 2, 3a, 3a-Intrinsics, 3b):**
  - `scale_source == 'aruco_marker'`: **`76/77` ảnh** (`95/96` nhóm khối lượng, `depth_anchored == True`).
  - `scale_source == 'checkerboard'`: **`0/77` ảnh** (`0/96` nhóm khối lượng).
  - `scale_source == 'plate_heuristic'` (fallback): **`1/77` ảnh** (`Pho1 (5).jpg`, `1/96` nhóm khối lượng, `depth_anchored == False`).
- **Chế độ không dùng thẻ chuẩn (`Config_1_GT_Assisted_Plate_Heuristic` — Stateless Heuristic Recovery):** Đạt tỷ lệ hợp lệ `77/77` ảnh (`0` ảnh lỗi), trong đó:
  - Nhánh khớp elip theo bounding box vật chứa (`dish_guided_ellipse`): **`77/77` ảnh**.
  - Nhánh dò đường tròn Hough (`hough_circle`): **`0/77` ảnh**.
  - Nhánh kích thước ảnh dự phòng (`image_dimension_fallback`): **`0/77` ảnh**.
- **Kiểm định cửa sổ $5\times 5$ tại tâm chiếu $(0,0,0)$ của Marker:** Trên `76` ảnh có ArUco, **`76/76` ảnh** có tâm chiếu nằm trọn trong đa giác 4 đỉnh và **`76/76` ảnh** có đủ $\ge 15/25$ pixel độ sâu hợp lệ trong cửa sổ $5\times 5$.

### 1.2. Phân phối khoảng cách, góc nghiêng và phân rã phần dư độ sâu tại Marker ($N = 76$ ảnh ArUco)

> [!NOTE]
> Để tách biệt độ nghiêng hình học 3D của mặt phẳng marker khỏi nhiễu bề mặt nội tại của mạng DA2 mà không bị phụ thuộc tuyến tính vào hệ số neo một điểm $\alpha_{\text{poly}} = t_z / d_{\text{pred,poly}}$, bảng dưới đây báo cáo thêm hệ số khớp mặt phẳng chẩn đoán độc lập $\alpha_{\text{fit}} = \text{median}(Z_{\text{plane}}(u,v) / Z_{\text{DA2,raw}}(u,v))$ và độ lệch chuẩn phần dư khớp mặt phẳng $\sigma_{\text{res,plane\_fit}} = \text{Std}(\alpha_{\text{fit}} Z_{\text{DA2,raw}} - Z_{\text{plane}})$ (chỉ dùng để chẩn đoán trong Mục 1.2, hoàn toàn không thay đổi $\alpha_{\text{poly}}$ của Config 3a–3b).

| Đại lượng kiểm định | Đơn vị | Mean ± Std | Min | P10 | P25 | Median (P50) | P75 | P90 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Độ sâu tâm marker theo trục quang học $t_z$ (`tvec_z`) | mm | `418.7 ± 85.1` | `246.8` | `326.1` | `351.9` | **`404.0`** | `470.7` | `526.6` | `662.7` |
| Khoảng cách Euclidean $\|\mathbf{t}\|_2$ | mm | `434.8 ± 86.3` | `258.9` | `343.0` | `372.6` | **`413.4`** | `484.6` | `551.3` | `680.6` |
| Độ sâu DA2 thô trung vị đa giác $d_{\text{pred,poly}}$ | mm | `958.3 ± 158.5` | `544.1` | `744.4` | `867.1` | **`992.0`** | `1067.3` | `1121.1` | `1297.9` |
| Độ sâu DA2 thô cửa sổ $5\times 5$ tâm chiếu $d_{\text{pred,center}}$ | mm | `957.8 ± 159.4` | `541.5` | `742.3` | `860.3` | **`990.7`** | `1067.9` | `1119.4` | `1304.7` |
| Góc nghiêng pháp tuyến marker $\theta_{\text{tilt}}$ | độ ($^\circ$) | `17.6 ± 10.7` | `2.0` | `3.2` | `7.2` | **`17.4`** | `25.8` | `32.5` | `36.7` |
| Độ lệch chuẩn thô trong marker $\sigma_{\text{marker,raw}}$ | mm (raw) | `11.20 ± 4.78` | `2.72` | `4.77` | `7.36` | **`11.16`** | `14.70` | `17.02` | `24.30` |
| Độ lệch chuẩn hình học do nghiêng $\sigma_{\text{plane}} = \text{Std}(Z_{\text{plane}})$ | mm (thực) | `2.60 ± 1.53` | `0.31` | `0.50` | `1.10` | **`2.61`** | `3.83` | `4.69` | `5.25` |
| Độ lệch chuẩn phần dư khớp mặt phẳng $\sigma_{\text{res,plane\_fit}} = \text{Std}(\alpha_{\text{fit}} Z_{\text{DA2,raw}} - Z_{\text{plane}})$ | mm (thực) | `3.40 ± 1.64` | `1.30` | `1.78` | `2.37` | **`2.90`** | `3.95` | `5.95` | `9.62` |
| Tỷ lệ nhiễu không phẳng tương đối $\sigma_{\text{res,plane\_fit}} / t_z$ | % | `0.825% ± 0.366%` | `0.265%` | `0.403%` | `0.537%` | **`0.776%`** | `1.052%` | `1.284%` | `2.106%` |
| Độ lệch chuẩn phần dư bản đồ neo $\sigma_{\text{res,anc\_poly}} = \text{Std}(\alpha_{\text{poly}} Z_{\text{DA2,raw}} - Z_{\text{plane}})$ | mm (thực) | `3.40 ± 1.64` | `1.30` | `1.79` | `2.36` | **`2.90`** | `3.95` | `5.95` | `9.61` |
| Hệ số neo đa giác $\alpha_{\text{poly}} = t_z / d_{\text{pred,poly}}$ | tỷ số | `0.4451 ± 0.1004` | `0.3151` | `0.3421` | `0.3791` | **`0.4117`** | `0.4938` | `0.6024` | `0.7359` |
| Hệ số neo tâm chiếu $\alpha_{\text{center}} = t_z / d_{\text{pred,center}}$ | tỷ số | `0.4454 ± 0.1003` | `0.3152` | `0.3427` | `0.3790` | **`0.4115`** | `0.4939` | `0.6030` | `0.7341` |
| Hệ số khớp mặt phẳng chẩn đoán $\alpha_{\text{fit}} = \text{median}(Z_{\text{plane}} / Z_{\text{DA2,raw}})$ | tỷ số | `0.4450 ± 0.1004` | `0.3153` | `0.3418` | `0.3795` | **`0.4111`** | `0.4939` | `0.6023` | `0.7360` |
| Độ chênh tuyệt đối $\Delta\alpha = |\alpha_{\text{poly}} - \alpha_{\text{center}}|$ | tỷ số | `0.00124 ± 0.00110` | `0.00001` | `0.00023` | `0.00042` | **`0.00085`** | `0.00178` | `0.00275` | `0.00540` |
| Độ chênh tương đối $\Delta\alpha / \alpha_{\text{center}}$ | % | `0.272% ± 0.226%` | `0.001%` | `0.052%` | `0.099%` | **`0.221%`** | `0.385%` | `0.562%` | `1.152%` |
| Độ chênh so với khớp mặt phẳng $|\alpha_{\text{poly}} - \alpha_{\text{fit}}|$ | tỷ số | `0.00037 ± 0.00037` | `0.00001` | `0.00005` | `0.00011` | **`0.00025`** | `0.00050` | `0.00098` | `0.00149` |

---

## 2. Bảng Tổng Hợp Kết Quả 5 Cấu Hình (`N = 96` Nhóm Khối Lượng, Oracle GT Mask)

| Cấu hình | Nguồn `mm_per_px` | Bản đồ độ sâu | `Intrinsics Area` | `relief_scale` | MAE (g) ↓ | MAPE (%) ↓ | Median APE (%) ↓ | RMSE (g) ↓ | MPE (%) | Bias_mean (%) | Số Fallback | Median `mm_per_px` | Median `scale_anchor` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Config_1_GT_Assisted_Plate_Heuristic` | GT-Assisted Heur (77) | `raw_depth_mm` | `False` | `1.0` | `150.61` | `77.14%` | `31.36%` | `217.21` | `+55.45%` | `+7.52%` | `18` | `0.3014` | `1.0000` |
| `Config_2_ArUco_XY_Only` | ArUco (76)+Heur (1) | `raw_depth_mm` | `False` | `1.0` | `83.63` | `34.67%` | `20.66%` | `115.78` | `+9.24%` | `-5.46%` | `22` | `0.2776` | `1.0000` |
| `Config_3a_ArUco_XY_Depth_Anchor` | ArUco (76)+Heur (1) | `anchored_depth_mm` | `False` | `1.0` | `76.48` | `30.47%` | `17.42%` | `107.41` | `+6.15%` | `-5.85%` | `19` | `0.2776` | `0.4151` |
| `Config_3a_Intrinsics` | ArUco (76)+Heur (1) | `anchored_depth_mm` | `True` | `1.0` | `76.13` | `30.36%` | `17.42%` | `106.87` | `+6.05%` | `-5.94%` | `19` | `0.2776` | `0.4151` |
| **`Config_3b_Full_ArUco`** | ArUco (76)+Heur (1) | `anchored_depth_mm` | `True` | `0.79` (76) / `1.0` (1) | **`67.89`** | **`26.65%`** | `15.48%` | **`95.62`** | `+1.28%` | `-8.62%` | `19` | `0.2776` | `0.4151` |

---

## 3. Phân Rã Đóng Góp Từng Bước Chuyển Đơn Biến (Stepwise Attribution)

| Bước chuyển cấu hình | Thành phần thay đổi | $\Delta\text{MAE}$ (g) | $\Delta\text{MAPE}$ (điểm %) | $\Delta\text{RMSE}$ (g) | $\Delta\text{MPE}$ (điểm %) | Số nhóm giảm lỗi / tăng lỗi / giữ nguyên |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Config_1_GT_Assisted_Plate_Heuristic` $\rightarrow$ `Config_2_ArUco_XY_Only` | Chuyển cơ chế tỷ lệ ngang & $D_{\text{rim}}$ từ GT-Assisted Heuristic sang ArUco XY | `-66.98 g` | `-42.47%` | `-101.42 g` | `-46.20%` | `59` giảm / `28` tăng / `9` bằng |
| `Config_2_ArUco_XY_Only` $\rightarrow$ `Config_3a_ArUco_XY_Depth_Anchor` | Bật neo độ sâu dọc `scale_anchor = tvec_z / d_pred` (`raw` $\rightarrow$ `anchored`) | `-7.16 g` | `-4.21%` | `-8.38 g` | `-3.09%` | `35` giảm / `22` tăng / `39` bằng |
| `Config_3a_ArUco_XY_Depth_Anchor` $\rightarrow$ `Config_3a_Intrinsics` | Bật công thức diện tích phần nhô miệng bát `Z²/(fx·fy)` | `-0.35 g` | `-0.11%` | `-0.54 g` | `-0.11%` | `2` giảm / `0` tăng / `94` bằng |
| `Config_3a_Intrinsics` $\rightarrow$ `Config_3b_Full_ArUco` | Áp dụng hệ số hiệu chỉnh thực nghiệm `relief_scale = 0.79` (76 ảnh ArUco, overlap `2/96`) | `-8.24 g` | `-3.71%` | `-11.25 g` | `-4.77%` | `34` giảm / `22` tăng / `40` bằng |
| `Config_1_GT_Assisted_Plate_Heuristic` $\rightarrow$ `Config_3a_Intrinsics` | **Tổng tác động các thành phần ArUco (XY + neo độ sâu) & intrinsics trước `relief_scale = 0.79`** | `-74.48 g` | `-46.79%` | `-110.33 g` | `-49.40%` | `63` giảm / `27` tăng / `6` bằng |
| `Config_1_GT_Assisted_Plate_Heuristic` $\rightarrow$ `Config_3b_Full_ArUco` | **Tổng hợp toàn bộ pipeline hiện hành (ArUco XY + Neo độ sâu + Intrinsics + `relief_scale = 0.79`)** | `-82.72 g` | `-50.50%` | `-121.59 g` | `-54.17%` | `71` giảm / `21` tăng / `4` bằng |

---

## 4. Phân Tích Chi Tiết Theo Từng Lớp Món Ăn (14 Lớp, 96 Nhóm Khối Lượng)

| Lớp món ăn (`dish_class`) | `N` | GT Mean (g) | MAE C1 (g) | MAE C2 (g) | MAE C3a (g) | MAE C3a-Intr (g) | MAE C3b (g) | MAPE C1 (%) | MAPE C2 (%) | MAPE C3a (%) | MAPE C3a-Intr (%) | MAPE C3b (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Banh cuon`** | `9` | `305.7` | `28.5` | `87.7` | `49.4` | `49.4` | **`47.0`** | `9.3%` | `28.7%` | `16.5%` | `16.5%` | **`15.9%`** |
| **`Banh mi`** | `16` | `203.9` | `239.1` | `121.6` | `109.7` | `109.7` | **`95.9`** | `205.0%` | `85.2%` | `74.1%` | `74.1%` | **`63.2%`** |
| **`Bun bo Hue`** | `5` | `804.5` | `228.7` | `38.1` | `38.1` | `38.1` | **`38.1`** | `28.4%` | `4.7%` | `4.7%` | `4.7%` | **`4.7%`** |
| **`Bun rieu`** | `7` | `838.9` | `263.1` | `72.5` | `72.5` | `72.5` | **`72.5`** | `31.4%` | `8.6%` | `8.6%` | `8.6%` | **`8.6%`** |
| **`Canh`** | `2` | `150.0` | `469.1` | `71.2` | `71.2` | `71.2` | **`71.2`** | `312.7%` | `47.5%` | `47.5%` | `47.5%` | **`47.5%`** |
| **`Com chien ga`** | `11` | `466.1` | `93.0` | `156.8` | `171.1` | `171.1` | **`137.5`** | `20.0%` | `33.6%` | `36.7%` | `36.7%` | **`29.5%`** |
| **`Com tam`** | `5` | `390.5` | `80.3` | `57.7` | `51.3` | `51.3` | **`60.8`** | `20.6%` | `14.8%` | `13.1%` | `13.1%` | **`15.6%`** |
| **`Pho`** | `5` | `819.5` | `231.3` | `102.2` | `102.2` | `102.2` | **`100.1`** | `28.2%` | `12.5%` | `12.5%` | `12.5%` | **`12.2%`** |
| **`Rau`** | `8` | `82.5` | `37.5` | `37.5` | `30.2` | `30.2` | **`25.3`** | `47.6%` | `47.6%` | `38.2%` | `38.2%` | **`31.0%`** |
| **`Sup cua`** | `6` | `470.6` | `162.6` | `116.9` | `99.1` | `99.0` | **`87.7`** | `34.5%` | `24.8%` | `21.1%` | `21.0%` | **`18.6%`** |
| **`Thit ga`** | `3` | `106.0` | `380.9` | `16.5` | `16.4` | `16.4` | **`16.6`** | `359.3%` | `15.6%` | `15.4%` | `15.4%` | **`15.6%`** |
| **`Thit kho`** | `6` | `92.0` | `87.8` | `23.0` | `23.0` | `23.0` | **`23.0`** | `95.4%` | `25.0%` | `25.0%` | `25.0%` | **`25.0%`** |
| **`Trung`** | `6` | `169.1` | `60.5` | `37.8` | `27.2` | `27.2` | **`25.7`** | `28.8%` | `22.1%` | `17.3%` | `17.3%` | **`16.6%`** |
| **`Xoi`** | `7` | `316.0` | `80.6` | `73.4` | `66.5` | `61.7` | **`47.4`** | `25.5%` | `23.2%` | `21.0%` | `19.5%` | **`15.0%`** |
| **Toàn tập (`Total`)** | **`96`** | **`346.5`** | `150.61` | `83.63` | `76.48` | `76.13` | **`67.89`** | `77.14%` | `34.67%` | `30.47%` | `30.36%` | **`26.65%`** |

### 4.1. Phân tích lớp món ăn cải thiện và bị tác động ngược theo từng bước chuyển
- **Bước 1 (`Config 1 → Config 2`: GT-Assisted Heuristic → ArUco XY):**
  - *Cải thiện (`11` lớp):* `Canh` (MAE `469.1 → 71.2 g`, MAPE `312.7% → 47.5%`), `Thit ga` (MAE `380.9 → 16.5 g`, MAPE `359.3% → 15.6%`), `Bun bo Hue` (MAE `228.7 → 38.1 g`, MAPE `28.4% → 4.7%`), `Bun rieu` (MAE `263.1 → 72.5 g`, MAPE `31.4% → 8.6%`), `Pho` (MAE `231.3 → 102.2 g`, MAPE `28.2% → 12.5%`), `Banh mi` (MAE `239.1 → 121.6 g`, MAPE `205.0% → 85.2%`), `Thit kho` (MAE `87.8 → 23.0 g`, MAPE `95.4% → 25.0%`), `Sup cua` (MAE `162.6 → 116.9 g`, MAPE `34.5% → 24.8%`), `Trung` (MAE `60.5 → 37.8 g`, MAPE `28.8% → 22.1%`), `Com tam` (MAE `80.3 → 57.7 g`, MAPE `20.6% → 14.8%`), `Xoi` (MAE `80.6 → 73.4 g`, MAPE `25.5% → 23.2%`).
  - *Tác động ngược (`2` lớp):* `Com chien ga` (MAE `93.0 → 156.8 g`, MAPE `20.0% → 33.6%`), `Banh cuon` (MAE `28.5 → 87.7 g`, MAPE `9.3% → 28.7%`).
- **Bước 2 (`Config 2 → Config 3a`: `raw_depth_mm` → `anchored_depth_mm`):**
  - *Cải thiện (`8` lớp):* `Banh cuon` (MAE `87.7 → 49.4 g`, MAPE `28.7% → 16.5%`), `Sup cua` (MAE `116.9 → 99.1 g`, MAPE `24.8% → 21.1%`), `Banh mi` (MAE `121.6 → 109.7 g`, MAPE `85.2% → 74.1%`), `Trung` (MAE `37.8 → 27.2 g`, MAPE `22.1% → 17.3%`), `Rau` (MAE `37.5 → 30.2 g`, MAPE `47.6% → 38.2%`), `Xoi` (MAE `73.4 → 66.5 g`, MAPE `23.2% → 21.0%`), `Com tam` (MAE `57.7 → 51.3 g`, MAPE `14.8% → 13.1%`), `Thit ga` (MAE `16.5 → 16.4 g`, MAPE `15.6% → 15.4%`).
  - *Tác động ngược (`1` lớp):* `Com chien ga` (MAE `156.8 → 171.1 g`, MAPE `33.6% → 36.7%`).
- **Bước 3 (`Config 3a → Config 3a-Intrinsics`: `mm_per_px²` → `Z²/(fx·fy)`):**
  - *Cải thiện (`2` lớp):* `Xoi` (MAE `66.5 → 61.7 g`, MAPE `21.0% → 19.5%`), `Sup cua` (MAE `99.1 → 99.0 g`, MAPE `21.1% → 21.0%`).
  - *Tác động ngược (`0` lớp):* không có (`0` lớp tăng lỗi).
- **Bước 4 (`Config 3a-Intrinsics → Config 3b`: `relief_scale = 1.0 → 0.79`):**
  - *Cải thiện (`8` lớp):* `Com chien ga` (MAE `171.1 → 137.5 g`, MAPE `36.7% → 29.5%`), `Xoi` (MAE `61.7 → 47.4 g`, MAPE `19.5% → 15.0%`), `Banh mi` (MAE `109.7 → 95.9 g`, MAPE `74.1% → 63.2%`), `Sup cua` (MAE `99.0 → 87.7 g`, MAPE `21.0% → 18.6%`), `Rau` (MAE `30.2 → 25.3 g`, MAPE `38.2% → 31.0%`), `Banh cuon` (MAE `49.4 → 47.0 g`, MAPE `16.5% → 15.9%`), `Pho` (MAE `102.2 → 100.1 g`, MAPE `12.5% → 12.2%`), `Trung` (MAE `27.2 → 25.7 g`, MAPE `17.3% → 16.6%`).
  - *Tác động ngược (`2` lớp):* `Com tam` (MAE `51.3 → 60.8 g`, MAPE `13.1% → 15.6%`), `Thit ga` (MAE `16.4 → 16.6 g`, MAPE `15.4% → 15.6%`).

### 4.2. Ghi chú giới hạn tập dữ liệu kiểm định (`77` ảnh / `96` nhóm khối lượng)
- Toàn bộ kết quả trong báo cáo này được đo trên **tập dữ liệu kiểm định 77 ảnh / 96 nhóm khối lượng** (cùng tập đã dùng để thiết kế chuỗi `A1–A5`, **không phải tập kiểm thử độc lập hoàn toàn / held-out test set**). Giá trị của thực nghiệm là chứng minh **cơ chế vật lý và đóng góp từng thành phần** trong điều kiện `Oracle GT Polygon Mask`.
- Riêng bước `Config 1 → Config 2` cho thấy 2 lớp đĩa phẳng (`Banh cuon` và `Com chien ga`) tăng lỗi khi chuyển sang `ArUco XY` trên nền `raw_depth_mm`, trước khi hồi phục mạnh ở `Config 3a` (`Banh cuon`: `87.7 → 49.4 g`) và `Config 3b` (`Com chien ga`: `171.1 → 137.5 g`): nguyên nhân là ở `Config 1`, `mm_per_pixel` heuristic lớn hơn thực tế vô tình bù trừ cho độ sâu chưa neo.

---

## 5. Truy Vết Nguồn Gốc `relief_scale = 0.79`, Kiểm Định Độ Nhạy & Phân Tích Lỗi Còn Lại (`Config 3b`)

### 5.1. Truy vết nguồn gốc hằng số `relief_scale = 0.79`
- **Bản chất:** `relief_scale = 0.79` là **hệ số hiệu chỉnh thực nghiệm từ giai đoạn phát triển (`31/08/2026`, commit `3905bc1`) có chồng lấn `2/96` nhóm với tập đánh giá**, không phải hằng số được tối ưu (sweep) trên 96 nhóm và cũng **không phải** hệ số từ một tập calibration độc lập.
- **Lịch sử & chênh lệch mốc tham chiếu:** Tại commit `3905bc1`, hệ số được chỉnh từ `0.62 → 0.79` trên hai ảnh `banh-mi1.jpg` và `banh-mi2.jpg` hướng tới mốc dev `~500 cm³ / 225 g` (`ρ = 0.45 g/cm³`). Khi đóng băng benchmark ở commit `18cd148` (`27/09/2026`), hai ảnh này có GT cân thực tế là **`242.6 g`** (chênh lệch **`+7.82%`** so với `225 g`), và `0.79` được giữ nguyên không đổi qua toàn bộ chuỗi `A1–A5` và ArUco ablation.

### 5.2. Kiểm định độ nhạy loại bỏ mẫu chồng lấn (`Config 3a-Intrinsics → Config 3b`)

| Tập con kiểm định | `N` | MAE C3a-Intr → C3b (g) | $\Delta\text{MAE}$ (g) | MAPE C3a-Intr → C3b (%) | $\Delta\text{MAPE}$ | MPE C3a-Intr → C3b (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Toàn tập benchmark | `96` | `76.13 → 67.89` | `-8.24` | `30.36% → 26.65%` | `-3.71%` | `+6.05% → +1.28%` |
| **Trường hợp A:** Loại bỏ 2 ảnh calibration (`banh-mi1/2`) | `94` | `76.74 → 68.06` | **`-8.68`** | `30.59% → 26.69%` | **`-3.90%`** | `+6.35% → +1.52%` |
| **Trường hợp B:** Loại bỏ toàn bộ lớp `Banh mi` (`13` lớp khác) | `80` | `69.41 → 62.29` | **`-7.12`** | `21.61% → 19.34%` | **`-2.27%`** | `-1.71% → -4.79%` |
| Riêng 2 ảnh calibration (`banh-mi1.jpg`, `banh-mi2.jpg`) | `2` | `47.55 → 60.05` | `+12.50` | `19.60% → 24.75%` | `+5.15%` | `-8.22% → -9.91%` |

- Trên chính `banh-mi1.jpg` (`270.2 → 278.6 g`, GT `242.6 g`) và `banh-mi2.jpg` (`175.1 → 158.5 g`, GT `242.6 g`), `0.79` làm tổng sai số tuyệt đối tăng nhẹ **`+25.00 g`**; ngược lại, **`72.1%`** mức giảm `Σ|AE|` (`-569.70 g / -790.60 g`) đến từ `80` nhóm ngoài `Banh mi`.
- **Lưu ý diễn giải:** MPE trên `N = 80` nhóm ngoài `Banh mi` dịch chuyển từ `-1.71% → -4.79%`, cho thấy `0.79` làm giảm sai lệch tổng thể nhưng tăng nhẹ xu hướng ước lượng thiếu ở một số món phẳng/trũng, chưa phải hằng số tối ưu đồng đều cho mọi lớp món ăn.

### 5.3. Tổng hợp lỗi còn lại trên `Config_3b_Full_ArUco` (`N = 96`)

| Hạng | Nhóm khối lượng (`mass_group_id`) | Lớp (`dish_class`) | GT (g) | Dự đoán C3b (g) | Sai số tuyệt đối (g) | APE (%) | Nguồn scale | Fallback |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `ComGa2 (1).jpg::ann_43` | `Com chien ga` | `466.1` | `170.1` | `296.0` | `63.5%` | `aruco_marker` | `True` |
| `2` | `ComGa1 (5).jpg::ann_39` | `Com chien ga` | `466.1` | `186.0` | `280.1` | `60.1%` | `aruco_marker` | `True` |
| `3` | `SupCua (5).jpg::ann_58` | `Sup cua` | `470.6` | `201.8` | `268.8` | `57.1%` | `aruco_marker` | `False` |
| `4` | `ComGa2 (3).jpg::ann_48` | `Com chien ga` | `466.1` | `217.9` | `248.2` | `53.3%` | `aruco_marker` | `False` |
| `5` | `BanhMi_Trung (4).jpg::group_22_24` | `Banh mi` | `99.3` | `336.8` | `237.5` | `239.2%` | `aruco_marker` | `False` |
| `6` | `Pho1 (5).jpg::ann_53` | `Pho` | `819.5` | `588.2` | `231.3` | `28.2%` | `plate_heuristic` | `False` |
| `7` | `ComGa1 (1).jpg::ann_35` | `Com chien ga` | `466.1` | `245.5` | `220.6` | `47.3%` | `aruco_marker` | `True` |
| `8` | `BanhMi_Trung (1).jpg::group_14_15` | `Banh mi` | `99.3` | `308.7` | `209.4` | `210.9%` | `aruco_marker` | `False` |

- **Các cụm sai số chính cần phân tích sâu ở bước kế tiếp:** (1) **`Banh mi` (`MAPE = 63.2%`, `N = 16`)**: đặc biệt `BanhMi_Trung` (GT `99.3 g` là 2 nửa ổ bánh mì rỗng ruột áp chảo nhưng dùng chung mật độ `0.45 g/cm³` của bánh mì thịt và chịu ảnh hưởng góc nghiêng); (2) **`Com chien ga` (`MAE = 137.5 g`, `MPE = -27.0%`)**: các ảnh chụp đĩa sứ trũng bị ước lượng thiếu (`ComGa2 (1)`, `ComGa1 (5)`, `ComGa2 (3)`); (3) **`Sup cua (5)` (`AE = 268.8 g`)**: góc nghiêng mạnh trên chất lỏng; (4) **`Pho1 (5)` (`AE = 231.3 g`)**: mẫu duy nhất không phát hiện được ArUco (`plate_heuristic`).
