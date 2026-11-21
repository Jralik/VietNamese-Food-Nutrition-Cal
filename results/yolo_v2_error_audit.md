# Kiểm Toán Sâu Lỗi Nhận Diện YOLOv10b (104 TH2, 80 FN, 63 FP) Trên 418 Ảnh

> [!IMPORTANT]
> **Tác động định lượng của lỗi phân loại YOLO (`TH2`) lên ước lượng khối lượng hạ nguồn:**
> - **Nhóm `TH1` (YOLO nhận đúng lớp, `N=333` mẫu 1-1):** MAPE khối lượng = **`32.64%`** (gần sát `Oracle MobileSAM C = 30.70%`).
> - **Nhóm `TH2` (YOLO bắt đúng khung nhưng nhầm lớp, `N=95` mẫu 1-1):** MAPE khối lượng tăng vọt lên **`42.69%`** (chênh **`+10.05` điểm %**) do tra nhầm mô hình hình học (`bowl` vs `plate`) và mật độ $\rho$.
> - **Tổng hợp Detection (552 GT boxes):** `TH1 = 368 (66.7%)` | `TH2 = 104 (18.8%)` | `FN = 80 (14.5%)` | `FP = 63`.

---

## 1. Top 20 Cặp Nhầm Lớp (`TH2`: GT → YOLO Predicted)

| Hạng | Lớp thực tế (GT) | Lớp YOLO dự đoán | Số ca (`TH2`) |
| :---: | :--- | :--- | :---: |
| `1` | **`BanhCanh`** | `Bun bo Hue` | `8` |
| `2` | **`BanhBeo`** | `Xoi` | `8` |
| `3` | **`ThitNuong`** | `Com tam` | `7` |
| `4` | **`HuTieu`** | `Pho` | `6` |
| `5` | **`BunChaCa`** | `Bun rieu` | `5` |
| `6` | **`Thit ga`** | `Ca phao` | `4` |
| `7` | **`Thit ga`** | `Nam` | `4` |
| `8` | **`Thit ga`** | `Com tam` | `4` |
| `9` | **`Xoi`** | `Com chien duong chau` | `4` |
| `10` | **`Tom`** | `Bong cai` | `4` |
| `11` | **`Com tam`** | `Com chien ga` | `3` |
| `12` | **`Muc`** | `Ca phao` | `3` |
| `13` | **`Nam`** | `Bun` | `3` |
| `14` | **`ThitNuong`** | `Thit bo` | `3` |
| `15` | **`Thit ga`** | `Xoi` | `3` |
| `16` | **`BunDau`** | `Dau hu` | `2` |
| `17` | **`GoiCuon`** | `Bo la lot` | `2` |
| `18` | **`Com`** | `Com tam` | `2` |
| `19` | **`ThitNuong`** | `Heo quay` | `2` |
| `20` | **`ThitNuong`** | `Thit ga` | `2` |

## 2. Top 20 Lớp Bị Bỏ Sót (`FN`: Missed Detections)

| Lớp thực tế (GT) | Số ca bỏ sót (`FN`) | Tổng GT của lớp | Tỉ lệ bỏ sót (%) |
| :--- | :---: | :---: | :---: |
| **`Rau`** | `10` | `33` | `30.3%` |
| **`KhoaiTayChien`** | `10` | `11` | `90.9%` |
| **`ThitNuong`** | `8` | `25` | `32.0%` |
| **`Ca`** | `8` | `15` | `53.3%` |
| **`CaChua`** | `7` | `15` | `46.7%` |
| **`BunDau`** | `5` | `12` | `41.7%` |
| **`Thit bo`** | `4` | `8` | `50.0%` |
| **`Muc`** | `4` | `10` | `40.0%` |
| **`Trung`** | `3` | `14` | `21.4%` |
| **`DuaLeo`** | `3` | `9` | `33.3%` |
| **`Thit ga`** | `3` | `22` | `13.6%` |
| **`BanhTrang`** | `2` | `8` | `25.0%` |
| **`BongCai`** | `2` | `4` | `50.0%` |
| **`ChaGio`** | `2` | `11` | `18.2%` |
| **`Com`** | `2` | `16` | `12.5%` |
| **`Cha`** | `2` | `12` | `16.7%` |
| **`BanhBeo`** | `1` | `9` | `11.1%` |
| **`BoLaLot`** | `1` | `11` | `9.1%` |
| **`GoiCuon`** | `1` | `11` | `9.1%` |
| **`KhoQuaThit`** | `1` | `5` | `20.0%` |

## 3. Top 15 Lớp Dương Tính Giả (`FP`: Extra Boxes)

| Lớp YOLO dự đoán dư (`FP`) | Số lượng |
| :--- | :---: |
| `Ca chua` | `15` |
| `Com tam` | `7` |
| `Chanh` | `6` |
| `Bun dau` | `5` |
| `Banh cuon` | `3` |
| `Banh mi` | `3` |
| `Thit nuong` | `3` |
| `Thit kho` | `3` |
| `Heo quay` | `2` |
| `Ca` | `2` |
| `Bun` | `2` |
| `Com` | `2` |
| `Com chien duong chau` | `1` |
| `Banh xeo` | `1` |
| `Trung` | `1` |
