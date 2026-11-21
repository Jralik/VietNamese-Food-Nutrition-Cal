# Báo cáo Phân tích Sai số (`Error Analysis`) — Benchmark v1 (`Config_3b_Full_ArUco`)

> **Phạm vi & Nguyên tắc đóng băng Benchmark v1:**
> - Toàn bộ số liệu trong tài liệu này được truy vết ở chế độ **chỉ đọc** từ kết quả đã đóng băng của **Benchmark v1** (`77` ảnh, `96` nhóm khối lượng thuộc `20` khẩu phần cân đo gốc): [`aruco_ablation_records.jsonl`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/aruco_ablation_records.jsonl), [`aruco_ablation_report.md`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/aruco_ablation_report.md), [`gt_mapping_verified.json`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/gt_mapping_verified.json) và cache `depth_ctx_aruco_ablation`.
> - Không sửa mã nguồn thuật toán, không thay đổi Ground Truth, không tinh chỉnh `relief_scale = 0.79` hay các ngưỡng hình học trong giai đoạn phân tích lỗi.

---

## 1. Phân bố sai số tổng thể và sự tập trung theo lớp (`Config_3b_Full_ArUco`, `N = 96`)

| Chỉ số tổng hợp | Giá trị | Diễn giải phân bố |
| :--- | ---: | :--- |
| `MAE` / Tổng AE | `67.89 g` / `6517.60 g` | Trung bình trên 96 nhóm (GT Mean = `346.5 g`) |
| `MAPE` / `Median APE` | `26.65%` / `15.48%` | Phân bố lệch phải mạnh: nửa tốt nhất có `APE ≤ 15.48%` |
| `RMSE` | `95.62 g` | Phản ánh tác động của nhóm đuôi ngoại lai nặng |
| `MPE` / `Bias_mean` | `+1.28%` / `-8.62%` | `35` ca dự đoán dư vs `61` ca dự đoán thiếu |
| Tỷ lệ đạt `APE ≤ 15%` / `≤ 25%` | `48/96 (50.0%)` / `59/96 (61.5%)` | Hơn một nửa số nhóm đã đạt sai số khá ổn định |
| Cụm ngoại lai (`AE ≥ 100 g`) | **`19/96 (19.8%)`** | Đóng góp **`3459.6 g` = `53.1%` tổng AE toàn tập** |

### Đóng góp sai số tuyệt đối theo 14 lớp (`Config_3b_Full_ArUco` so với `Config_3a_Intrinsics`)

| Lớp (`n`) | Tổng AE C3b (`g`) | Tỉ trọng AE | `MAE` C3b (`g`) | `MPE` (`3aI -> 3b`) | Nhóm phân loại lỗi chính |
| :--- | ---: | ---: | ---: | :---: | :--- |
| `Banh mi` (`16`) | 1534.2 | **23.5%** | 95.89 | `+44.8% -> +31.6%` | Hệ thống (mật độ `BanhMi_Trung`) + Cảnh chụp/Ngưỡng (12 ổ thường) |
| `Com chien ga` (`11`) | 1512.7 | **23.2%** | 137.52 | `-20.1% -> -27.0%` | Cảnh chụp (diện tích hẹp/góc nghiêng + che khuất `ComGa2`) + Đảo ngưỡng |
| `Sup cua` (`6`) | 526.4 | **8.1%** | 87.73 | `-21.0% -> -18.6%` | Cảnh chụp (`Z_dish > Z_marker` + góc nghiêng ở `SupCua (4,5)`) |
| `Bun rieu` (`7`) | 507.5 | **7.8%** | 72.50 | `-8.6% -> -8.6%` | Hệ thống (trần đường kính tô `D_rim ≤ 198 mm`) |
| `Pho` (`5`) | 500.7 | **7.7%** | 100.14 | `-12.5% -> -12.2%` | Cảnh chụp (`Pho1 (5)` mất ArUco, `Pho1 (3)` `Z_dish > Z_marker`) + Trần `198 mm` |
| `Banh cuon` (`9`) | 423.4 | **6.5%** | 47.04 | `-7.1% -> -13.3%` | Tương tác tham số (`0.79` làm âm thêm nhánh `recess_shift`) |
| `Xoi` (`7`) | 331.8 | **5.1%** | 47.40 | `-6.2% -> -15.0%` | Hệ thống (tắt A1/A2 ép đĩa xôi vào khuôn `≤ 132 mm`) |
| `Com tam` (`5`) | 304.1 | **4.7%** | 60.82 | `+2.0% -> +5.4%` | Hệ thống (`area > 225 cm²` bỏ qua A4) + Phản ứng ngược của `0.79` |
| `Rau` (`8`) | 202.4 | **3.1%** | 25.30 | `+36.9% -> +24.8%` | Hệ thống (bao lồi tán lá + trần `400 cm³`) |
| `Bun bo Hue` (`5`) | 190.5 | **2.9%** | 38.10 | `-4.7% -> -4.7%` | Hệ thống (trần đường kính tô `D_rim ≤ 198 mm`) |
| `Trung` (`6`) | 153.9 | **2.4%** | 25.65 | `-13.5% -> -11.6%` | Cảnh chụp + Kẹp bề dày `spheroid` / chảo trứng |
| `Canh` (`2`) | 142.4 | **2.2%** | 71.20 | `+47.5% -> +47.5%` | Hệ thống (cận dưới `h_fill ≥ 0.70 * H_bowl`) |
| `Thit kho` (`6`) | 137.9 | **2.1%** | 22.98 | `+22.5% -> +22.5%` | Hệ thống (`h_fill ≥ 0.70 * H_bowl`) + `D_food` ở `ThitKho1 (4)` |
| `Thit ga` (`3`) | 49.7 | **0.8%** | 16.57 | `+3.2% -> +3.4%` | Chạm trần bão hòa A5 (`0.65 * 28 = 18.2 mm`) |

* **Hai lớp đứng đầu (`Banh mi` + `Com chien ga`) chiếm `(1534.2 + 1512.7) / 6517.6 = 46.75%`** tổng sai số tuyệt đối; 8 lớp đầu chiếm `86.6%`.

---

## 2. Phân loại nguyên nhân gốc theo hai trục: Lỗi hệ thống vs Lỗi phụ thuộc cảnh chụp

### A. Nhóm I — Lỗi có tính hệ thống (bắt nguồn từ giả định mô hình hoặc ràng buộc cấu hình)

1. **Một mật độ cố định `ρ = 0.45 g/cm³` áp cho mọi biến thể bánh mì (`BanhMi_Trung`):**
   - Trong [`density_db.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/density_db.py#L122-L123), `Banh mi` dùng `ρ = 0.45 g/cm³` (hiệu chỉnh trên ổ bánh mì đầy `~225 g`).
   - Kết quả kiểm tra **Check 3** cho thấy 4 mẫu `BanhMi_Trung` (2 nửa bánh mì không ăn kèm chảo trứng, đều `GT = 99.3 g`) có thể tích ước lượng `286.3–748.3 cm³`, ứng với mật độ suy diễn `ρ_implied = GT / V_3b` chỉ là **`0.133, 0.145, 0.182, 0.347 g/cm³`** (trung bình `0.202 g/cm³`). Nhân thể tích vỏ ngoài rỗng ruột với `0.45 g/cm³` gây ước lượng dư `128.8–336.8 g` (`MAE = 155.72 g`, `MPE = +156.8%`, tổng `622.9 g` AE — riêng 4 mẫu này chiếm **`40.6%` tổng AE của lớp `Banh mi`**).
   - Ngược lại, trên **12 ổ bánh mì nguyên ổ** (`banh-mi1..5`, `banh-mi250g`, `BanhMiDucTri`), `ρ_implied` trải rộng từ `0.269` (`banh-mi4.jpg`, `area = 319.0 cm²`) tới `0.908–0.967` (`BanhMiDucTri (1)`, `banh-mi250g (1)`), chứng tỏ ở 12 ổ này sai số chủ yếu nằm ở phép ước lượng thể tích $V$ (diện tích mask và mặt phẳng tham chiếu), không phải mật độ `0.45 g/cm³` (minh chứng: `banh-mi250g (4)` có `ρ_implied = 0.458`).

2. **Trần đường kính tô `D_rim ≤ 198.0 mm` gây bão hòa ở các tô phở/bún lớn:**
   - Trong [`eval_bowl_volume_aruco_ablation`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_aruco_ablation.py#L227), `D_rim_mm` bị kẹp trong `[85.0, 198.0] mm`.
   - **Check 1** xác nhận toàn bộ `5/5` mẫu `Bun bo Hue`, `7/7` mẫu `Bun rieu`, và `3/5` mẫu `Pho` (`Pho1 (1, 2, 4)`) — tổng `15/38` bát hợp lệ — đều chạm trần `D_rim = 198.0 mm`. Khi đồng thời vượt ngưỡng headroom tối đa, cả `12/12` mẫu `Bun bo Hue` và `Bun rieu` hội tụ về đúng một hằng số `824.1 cm³` (`766.4 g`), tạo sai số âm cố định `-4.7%` ở `Bun bo Hue` và `-8.6%` (`MAE = 72.50 g`, `7.8%` tổng AE) ở `Bun rieu`.

3. **Cận dưới chiều cao chứa `h_fill ≥ 0.70 * H_bowl` gây ước lượng dư trên tô vơi (`Canh`, `Thit kho`):**
   - Trong [`eval_bowl_volume_aruco_ablation`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_aruco_ablation.py#L236-L245) và [`pipeline_config.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/pipeline_config.py#L311-L315):
     - Nếu `delta_z > 4.0 mm`, `headroom = clip(delta_z, 4.0, 0.30 * H_bowl)` nên `h_fill = H_bowl - headroom ≥ 0.70 * H_bowl` (nằm trong `(0.70*H, H - 4.0)` khi `4.0 < delta_z < 0.30*H`, và kẹp đúng `0.70 * H` khi `delta_z ≥ 0.30 * H`).
     - Nếu `delta_z ≤ 4.0 mm`, pipeline rẽ sang nhánh `else: h_fill = 0.70 * H_bowl` (`BOWL_NOISE_FILL_RATIO = 0.70`).
   - Do đó mô hình tô **không thể biểu diễn mực thức ăn thấp hơn `70%` chiều cao tô**:
     - **Check 1** trên `38` tô hợp lệ: `17/38` rơi vào nhánh `dz ≤ 4.0 -> f = 0.70` (`Thit kho 6/6`, `Xoi 7/7`, `Sup cua 3/6`, `Canh 1/2`), `18/38` bị kẹp trần headroom `dz ≥ 0.30*H -> f = 0.70` (`Bun bo Hue 5/5`, `Bun rieu 7/7`, `Pho 4/5`, `Canh 1/2`, `Sup cua 1/6`), chỉ `3/38` nằm trong vùng nội suy `4.0 < dz < 0.30*H -> 0.70 < f < 1.0` (`Pho1 (2)` `f=0.706`, `SupCua (3)` `f=0.817`, `SupCua (4)` `f=0.838`).
     - Hệ quả trực tiếp: trên `Canh` (`com-suon1 (5)`), dù độ sâu đo được `dz = +30.6 mm` (`~58%` chiều cao `52.9 mm` là khoảng trống), hệ thống vẫn ép `h_fill = 37.0 mm` (`0.70*H`), cho `245.6 g` vs GT `150.0 g` (`+63.7%`). Tương tự `Thit kho` (`6/6` mẫu `dz ≤ 1.1 mm`) đều bị gán `70%` dung tích chén (`MPE = +22.5%`), riêng `ThitKho1 (4)` đo `D_food = 113.3 mm` bị khuếch đại lên `170.1 g` vs GT `92.0 g` (`+84.9%`).

4. **Ngoại lệ cấu hình của `Xoi` và giới hạn bao lồi + trần `400 cm³` của `Rau`:**
   - **`Xoi` (`7/7` mẫu dự đoán giống hệt nhau `268.6 g` vs GT `316.0 g`, `-15.0%`):** [`pipeline_config.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/pipeline_config.py#L243) xếp `Xoi` vào `rice_bowl` (`120 mm`), còn [`run_volume_ablation.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_volume_ablation.py#L632-L633) tắt cả `f_a1` và `f_a2` cho `Xoi`, khiến toàn bộ 7 ảnh đều bị kẹp về `D_rim = 132.0 mm` bất kể kích thước thật đo bằng ArUco (`137.2–206.0 mm`).
   - **`Rau` (`n = 8`, `MAE = 25.30 g`, `MPE = +24.8%`):** [`compute_volume_side_vegetables`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L513-L529) tích phân bề mặt bao ngoài của tán lá (`16.1–50.4 mm` trên `86–183 cm²`) tạo thể tích chưa kẹp `224.6–687.6 cm³`, đẩy `3/8` mẫu chạm trần cứng `400.0 cm³` (`120.0 g` vs GT `70–90 g`).
   - *Ghi nhận lỗi log metadata (không ảnh hưởng kết quả số):* Tại [`run_aruco_ablation.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_aruco_ablation.py#L575) (và dòng `666` của [`run_volume_ablation.py`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_volume_ablation.py#L666)), nhánh `side_vegetables` ghi cứng `mean_height_mm: 22.0` vào bản ghi trả về trong khi `vol_cm3` dùng trực tiếp chiều cao thực do `compute_volume_side_vegetables` tính. Cần giữ nguyên mã v1 và chỉ sửa khi nâng cấp harness cho v2.

---

### B. Nhóm II — Lỗi phụ thuộc cảnh chụp (góc nghiêng, vị trí marker, che khuất, nền & ánh sáng)

5. **Đo đường kính tô theo tỷ lệ `mm/px` tại mặt phẳng marker thay vì mặt phẳng tô (`Z_dish > Z_marker`):**
   - Trong [`eval_bowl_volume_aruco_ablation`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_aruco_ablation.py#L148-L152) và [`compute_volume_bowl_container`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/volume_nutrition.py#L238-L257), `D_food_measured_mm = max(bw_px, bh_px) * mm_per_pixel` sử dụng tỷ lệ phẳng tại thẻ ArUco (`Z_marker`), ngay cả khi bật `use_depth_intrinsics_for_area = True`.
   - Khi ảnh chụp nghiêng và thẻ ArUco đặt ở tiền cảnh gần camera hơn miệng tô (`Z_dish > Z_marker`), mô hình camera pinhole `s_x(Z) = Z / f_x` cho thấy tỷ lệ tại marker nhỏ hơn thực tế tại tô:
     - **`SupCua (5)` (`tilt = 34.7°`, Top-3 AE = `268.8 g`, `-57.1%`):** `z_marker = 246.8 mm` (`aruco_px = 0.1601`) trong khi `z_box = 321.1 mm` (**xa hơn `1.30×`**, tương ứng `z_box / fx = 0.2100 mm/px`). Kết hợp mép tô trước che một phần súp ở góc nghiêng (`ar = 0.65`), `D_food` đo chỉ còn `105.3 mm -> D_rim = 123.1 mm` (so với `163.0 mm` ở góc thẳng `SupCua (1)` chỉ lệch `-0.3%`). Do thể tích nón cụt phụ thuộc xấp xỉ bậc ba theo kích thước, co đường kính xuống `0.755×` góp phần làm thể tích giảm còn `197.8 cm³` vs `459.8 cm³`.
     - Trường hợp tương tự: **`SupCua (4)` (`tilt = 33.8°`, `z_box/z_m = 1.12`, `ar = 0.68`, `-32.7%`)** và **`Pho1 (3)` (`tilt = 32.0°`, `z_box/z_m = 1.18`, `D_rim = 186.1 mm`, `-20.7%`)**.

6. **Tổ hợp diện tích mask hẹp, góc chụp nghiêng và che khuất chưa được bù ở `Com chien ga` (`23.2%` tổng AE):**
   - **Check 2** đối chiếu diện tích mask GT (`A_rice`), diện tích `Thit ga` (`A_ck`), tỷ lệ tiếp giáp `adj_frac`, góc nghiêng và kịch bản giả định bù đủ phần `A_ck`:
     - Trên `ComGa2 (1..3)` (`MAE = 225.30 g`, `MPE = -48.3%`), nhánh bù diện tích bị che khuất trong [`evaluate_scene_instances_aruco`](file:///d:/Document/Year4_Semester1/KhoaLuanTotNghiep/FoodDetector-2%20-%20V3/scratch/run_aruco_ablation.py#L634) **không kích hoạt lần nào (`adj_frac = 0.000` trên cả 3 ảnh)**. Tuy nhiên ngay cả khi cộng thêm toàn bộ diện tích `Thit ga`: `ComGa2 (2)` (`tilt = 6.6°`, `191.3 + 63.4 = 254.7 cm²`) tiến sát GT (`452.1 g` vs `466.1 g`), còn **`ComGa2 (1)` (`tilt = 35.3°`, `100.7 + 75.3 = 176.0 cm²`) vẫn hụt tới `-169 g` (`297.0 g` vs `466.1 g`)**, và `ComGa2 (3)` vẫn hụt `-181 g` (`285.2 g`) do tụt chiều cao (`15.7 mm`).
     - Hơn nữa, trên **`ComGa1 (1)` (`tilt = 24.7°`)** và **`ComGa1 (5)` (`tilt = 29.6°`)** — vốn **không có nhãn `Thit ga` riêng** — `A_rice` vẫn chỉ đo được `119.9 cm²` và `109.7 cm²` (chỉ bằng một nửa diện tích các góc chụp phẳng `ComGa1 (2,3,4,6,8)`: `199.8–253.0 cm²`).
   - Như vậy, nguyên nhân của `Com chien ga` không phải thuần túy do che khuất, mà là **tổ hợp của diện tích mask lộ thiên bị thu hẹp trên ảnh nghiêng `24.7°–35.3°`, phần che khuất dưới `Thit ga` không được kích hoạt bù (`ComGa2`), và tụt chiều cao trên `ComGa2 (3)`**.

7. **Kẹp biên A5 quá gắt trên nền sáng/gần và 1 ca mất marker (`Pho1 (5)`):**
   - **Kẹp A5 trên bánh mì:** Trên `BanhMiDucTri (1)`, `banh-mi250g (1, 3)`, `banh-mi2`, nhánh `p85(steps_mm) > 40.0` trong `fit_plane_stabilized` kích hoạt (`step85 = 43.2–153.6 mm`), giới hạn vòng ngoài về `local_mask_z + 11.5 / 0.79` và kéo mặt phẳng tham chiếu sát lên thân bánh mì, ép `mean_height_mm` xuống sàn `15.0–19.0 mm` (`-34.7%` tới `-53.4%`). Ngược lại `banh-mi4.jpg` và `banh-mi5.jpg` có diện tích polygon `319.0` và `303.1 cm²` (gần gấp đôi `banh-mi1..3`: `172.5–188.6 cm²` của cùng một ổ), đẩy khối lượng lên `406.2 g` và `298.1 g`.
   - **Phân loại chính xác 19/96 ca `fallback_used = True` (Check 5):** Không có ca `fallback_used = True` nào trong `Config_3b` do mất ArUco: `14/19` là nhánh **A4 `recess_shift`** (`9 Banh cuon`, `4 Com chien ga`, `1 banh-mi3.jpg`) và `5/19` là nhánh **A5 `saturation_cap`** (`3 Thit ga` ở `18.2 mm`, `1 BanhMiDucTri (2)` ở `35.8 mm`, `1 BanhMi_Trung (1) Trung` ở `10.5 mm`). Trường hợp mất ArUco duy nhất thực sự là **`Pho1 (5).jpg` (`scale_source == 'plate_heuristic'`, `fallback_used = False`)**, dự đoán `588.2 g` vs `819.5 g` (`AE = 231.3 g`, Top-6 lỗi lớn nhất).

---

## 3. Phân tích tương tác phi tuyến giữa `relief_scale = 0.79` và các ngưỡng rời rạc

Mặc dù `0.79` giúp giảm AE tổng trên nhóm không phải bánh mì (`N = 80`: `MAE 69.41 -> 62.29 g`, đóng góp `72.1%` tổng mức giảm AE từ `C3a_Intrinsics -> C3b`), việc nhân `raw_heights` với `0.79` trước các phép kẹp và rẽ nhánh gây ra các hành vi phi tuyến trên một số lớp:

1. **Nghịch lý `Com tam` (`5/5` đĩa tăng khối lượng khi giảm hệ số nhân `1.0 -> 0.79`, `MAE 51.3 -> 60.8 g`, Check 4):**
   - Do `area_cm2 = 267–454 > 225.0 cm²`, A4 không bật `recess_shift`; `54%–91%` pixel âm bị kẹp ở sàn `2.0 mm` khiến sàn `min_mean_height_mm = 15.0 mm` kích hoạt (`h_final = clip(h_clip1 * (15.0 / mean_clip1), 2.0, 45.0)`).
   - Khi nhân `0.79`, các pixel đang ở sàn `2.0 mm` giữ nguyên còn phần đỉnh dương thu nhỏ (`mean_clip1` giảm trên cả `5/5` đĩa, ví dụ `com-suon1 (1): 3.87 -> 3.41 mm`, `com-suon1 (3): 8.99 -> 7.33 mm`). Hệ số khuếch đại `15.0 / mean_clip1` tăng lên (`3.872 -> 4.404`), nhưng phần đỉnh đã bị nén so với sàn nên **tỷ lệ pixel vượt trần `45.0 mm` bị cắt cụt giảm xuống ở cả `5/5` đĩa** (`9.0%->8.1%`, `4.4%->3.5%`, `12.5%->11.8%`, `9.5%->8.7%`, `10.7%->9.0%`). Do ít bị mất thể tích trên trần `45 mm` hơn, chiều cao trung bình cuối lại **tăng lên gần `15.0 mm` hơn** trên tất cả 5 đĩa (`12.65->13.28`, `13.76->14.23`, `13.56->13.77`, `13.97->14.18`, `14.16->14.45 mm`).
2. **Đảo trạng thái ngưỡng rời rạc trên `Com chien ga`, `banh-mi3` và `Xoi1 (1)`:**
   - *Ngưỡng A5 `p85(steps_mm) > 40.0`:* Ở `ComGa1 (7)`, `ComGa2 (2)`, `ComGa2 (3)`, `step85` chuyển từ `49.5, 42.9, 42.0` (`clmp=True` ở `1.0`) xuống `39.1, 33.9, 33.2` (`clmp=False` ở `0.79`). Nhả kẹp vòng ngoài kéo mặt phẳng tham chiếu lên cao, làm `ComGa2 (2)` tăng ngược `236.8 -> 334.4 g`, còn `ComGa2 (3)` bị đẩy `med_raw` từ `9.7 -> 14.4 mm` vượt quá `11.5 mm` nên **tắt `recess_shift`**, tụt từ `317.9 -> 217.9 g` (`AE: 148.2 -> 248.2 g`).
   - *Ngưỡng A4 `med_raw ≤ 11.5 mm`:* Ngược lại, `0.79` kéo `med_raw` qua mốc `11.5 mm` ở `ComGa1 (1)` (`14.5 -> 11.46 mm`, `+90.1 g`), `ComGa2 (1)` (`12.2 -> 9.6 mm`, `+39.3 g`) và `banh-mi3` (`14.2 -> 11.2 mm`, `+62.5 g`).
   - *Ngưỡng `BOWL_DELTA_Z_NOISE_MM = 4.0 mm` ở `Xoi1 (1)`:* `delta_z` đổi từ `+4.6 mm` (`1.0`, có `above_rim_volume` -> `463.6 g`) xuống `+3.6 mm` (`0.79`, rẽ nhánh `0.70*H` -> `268.6 g`).
3. **Kéo sâu sai số âm ở `Banh cuon` (`MPE -7.1% -> -13.3%`):** 7/9 đĩa có `recess_shift` bị thu nhỏ theo `0.79` làm `(1), (2), (4)` tăng sai số tuyệt đối (`+16.2 .. +20.3 g`), trong khi `(3)` và `(9)` tăng nhẹ (`+8.5`, `+13.1 g`) vì `recess_shift_mm` chạm trần `28.0 mm` còn phần `raw_h` âm bị co về gần `0`.

---

## 4. Hàm ý cho Benchmark v2 (~154 ảnh)

- **Chống rò rỉ dữ liệu đa góc chụp:** Vì `77` ảnh / `96` nhóm trong Benchmark v1 thực chất chỉ bắt nguồn từ **`20` khẩu phần cân đo gốc**, việc tách tập validation/test ở Benchmark v2 bắt buộc phải nhóm theo **`serving_id` (khẩu phần/lần cân)** chứ không tách ngẫu nhiên theo `image_id`.
- **Tránh chọn tham số trên chính tập đánh giá cuối:** Giữ nguyên các ngưỡng và `0.79` khi chạy đối chiếu `Benchmark v1` vs `dữ liệu mới` vs `Benchmark v2`. Mọi thay đổi (ví dụ phân tách mật độ bánh mì đặc/rỗng, nội suy `Z_dish / Z_marker`, nới trần `198 mm` hoặc xử lý `Xoi`) cần thử nghiệm trên tập validation độc lập trước khi kiểm định trên test.

