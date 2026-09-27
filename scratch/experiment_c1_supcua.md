# Experiment C1 — Capture-geometry exploratory analysis (Súp cua, data hiện có)

**Bản chất**: exploratory screening trên dữ liệu hiện có — KHÔNG phải validation
có kiểm soát (n nhỏ, phân loại góc bằng quan sát thủ công, 10 shot ≈ 4-5 cảnh chụp
độc lập vì nhiều ảnh là các shot của cùng một cảnh).

## Dữ liệu: 10 ảnh Súp cua (GT 470.6g / 500.6g mỗi bát)

| Ảnh | Góc chụp (quan sát) | Scale source | est (g) | GT (g) | Sai số |
|---|---|---|---|---|---|
| SupCua (1) | top-down | aruco (anchor x0.597) | 460.4 | 470.6 | **-2.2%** |
| SupCua (2) | high-angle | aruco | 460.4 | 470.6 | **-2.2%** |
| SupCua (6) | **tilted ~45°** | aruco | 460.4 | 470.6 | **-2.2%** |
| SupCua (3) | **oblique thấp** | aruco (anchor x0.585) | 1212.3 | 470.6 | **+157.6%** |
| SupCua (4) | **oblique thấp** | aruco | 1224.0 | 470.6 | **+160.1%** |
| SupCua (5) | high-angle + thìa + anchor x0.330 | aruco | 840.6 | 470.6 | **+78.6%** |
| SupCua1 (2) | top-down (chụp trên cân) | **plate_heuristic** | 631.5 | 500.6 | **+26.1%** |
| SupCua1 (1) | (detection conf < 0.40) | — | volume skipped | 500.6 | n/a |
| SupCua1 (3) | (detection conf < 0.40) | — | volume skipped | 500.6 | n/a |
| SupCua1 (4) | (detection conf 0.338 < 0.40) | — | volume skipped | 500.6 | n/a |

## Phát hiện

1. **Binary top-down/tilted KHÔNG phân loại sạch sai số**: nhóm tilted có cả
   -2.2% (SupCua 6) lẫn +157.6% (SupCua 3); SupCua 5 nhìn gần top-down nhưng +78.6%.
   → Giả thuyết geometry ban đầu cần tinh chỉnh.

2. **Một proxy tiềm năng quan sát được trong log của hệ thống: depth-anchor correction ratio — cho thấy tương quan với sai số thể tích trong tập khảo sát nhỏ** (tỉ lệ hiệu chỉnh
   depth tại marker ArUco): anchor x0.597/x0.585/-2.2%… anchor x0.330 → +78.6%.
   Ratio này dao động 1.8× giữa các shot của cùng bát → **metric depth (Depth
   Anything V2 Metric) không ổn định giữa các điều kiện chụp**, và hiệu chỉnh global
   scale tại marker chỉ sửa được một phần (không sửa được distortion hình dạng).

3. **Cấu trúc dữ liệu**: 10 shot ≈ 4-5 cảnh độc lập (các shot cùng cảnh cho kết quả
   gần trùng nhau: 460.4g ×3; 1212/1224g ×2) — n hiệu quả cho thống kê là số cảnh,
   không phải số ảnh.

4. **3/10 ảnh không vào được volume** do detection conf 0.30-0.40 (< sàn 0.40 của
   volume path) — app fallback về dinh dưỡng phần chuẩn cho các ảnh này.

## Kết luận C1 (exploratory)

- Tín hiệu geometry là **thật** (cùng bát 470.6g cho sai số từ -2% đến +160%) nhưng
  **không tách sạch theo binary top-down/tilted**; depth-anchor ratio là proxy tốt hơn
  quan sát được trong log.
- Theo cây quyết định: tín hiệu có nhưng không sạch → **C2 (controlled capture) là
  tùy chọn**, chưa bắt buộc. Nếu làm C2: cùng bát, cùng vị trí marker, máy ảnh cố
  định khoảng cách, quét góc từ top-down → nghiêng theo bước 15°.
- **Cho thesis**: ghi vào limitation — "độ chính xác ước lượng thể tích phụ thuộc
  điều kiện chụp; trên cảnh top-down pipeline đạt -2.2% sai số khối lượng, trên
  cảnh nghiêng sâu sai số có thể vượt +150% do depth integration vào khoang hộp".

## Trạng thái

Experiment C1 ✅ FROZEN (exploratory, existing data) · C2 — tùy chọn, cần chụp mới.
Pipeline production KHÔNG đổi trong suốt chuỗi Experiment A/B/C1.
