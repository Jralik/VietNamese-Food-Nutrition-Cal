# Depth audit — nguyên nhân sai số khẩu phần (Súp cua / Trứng / Bánh mì)

Audit trên chính mask overlay + depth map mà volume worker sinh ra
(scratch/depth_audit/), kèm số nội bộ của từng request. Bối cảnh: GT đã chuẩn hóa
(74 targets, gram = tổng của K box), pipeline frozen (v10b + dual-preresize @0.30).

## 1. Súp cua — Geometry-dependent (cùng bát 470.6g, 3 trường hợp)

| Ảnh | Góc chụp | Anchor scale | Volume | Mass | Sai số |
|---|---|---|---|---|---|
| (1) | **top-down** (mask chỉ phủ mặt nước) | x0.597 | 451 cm³ | 460.4g | **-2%** |
| (3) | **nghiêng** (thấy cả thành + đáy hộp) | x0.585 | 1188 cm³ | 1212.3g | **+158%** |
| (5) | **nghiêng mạnh** (anchor x0.330) | x0.330 | 824 cm³ | 840.6g | **+79%** |

**Cơ chế (root-cause hypothesis có audit, chưa có GT 3D độc lập)**: thể tích được tích
phân = (chiều sâu bảng nền − chiều sâu pixel) × diện tích pixel trên mặt nạ.
- **Top-down**: Depth Anything "nhìn xuyên" chất lỏng trong suốt → depth mặt chất lỏng ≈
  mặt bàn → chiều cao tích phân nhỏ → 451 cm³ ≈ đúng 460 cm³ chất lỏng thật.
- **Nghiêng**: mặt nạ bao cả thành hộp nhìn thấy + depth cliff ở miệng hộp → tích phân
  cho **thể tích khoang hộp trống** (~1200 cm³ ≈ dung tích bát 13×9cm), không phải thể
  tích chất lỏng (~460 cm³).
- Anchor ratio dao động mạnh theo góc (x0.330–x0.597) → hiệu chỉnh depth toàn cục
  không ổn định trên ảnh nghiêng.

**Kết luận**: sai số Súp cua KHÔNG phải lỗi detection hay scale — là **giới hạn của
depth metric trên chất lỏng trong suốt + góc chụp nghiêng** (thể tích khoang ≠ thể tích
thực ăn). Wording: giữ ở mức hypothesis cho đến khi có GT 3D độc lập.

## 2. Trứng — Under-estimation ~3× (GT đã xác nhận: cả đĩa)

**User xác nhận (2026-09-27)**: GT "1 - Trung 221.2g" = **cả đĩa trứng chiên (3-4 quả +
dầu)** — một khẩu phần món, không phải 1 quả.

- Mask phủ đúng toàn bộ đĩa trứng (crop xác nhận) — không phải lỗi mask.
- Volume 80 cm³ trên một lớp trứng mỏng: Depth Anything phân biệt kém chiều dày của
  thức ăn **nhiều lớp, mỏng và phẳng** trên đĩa trắng (contrast thấp) → mass thấp ~3×.
- Đây là **failure mode riêng**: under-estimation do geometry nhiều lớp mỏng — không
  cùng bản chất với over-estimation của Bánh mì.

## 3. Bánh mì — Over-estimation ~2.9× trên bánh mì thường (GT đã xác nhận)

**User xác nhận (2026-09-27)**: BanhMi_Trung có 2 detection = **1 full ổ + 1 nửa ổ bánh
mì bình thường, TỔNG 99.3g** (~50g/phần); các bánh mì >200g (banh-mi1 242.6g,
banh-mi250g 255.6g, DucTri 210g) là **bánh mì thịt/chả lụa** (1 ổ/đĩa).

| Case | est | GT | sai số |
|---|---|---|---|
| banh-mi1 (bánh mì thịt, 1 ổ) | 232.0g | 242.6g | **-4.4%** ✓ |
| BM(1) full ổ (bánh mì thường) | 202.5g | ~66g (2/3 của 99.3) | ~+207% |
| BM(1) nửa ổ (bánh mì thường) | 85.5g | ~33g | ~+159% |

**Phân rã giả thuyết (decomposition hypothesis — chưa phải kết luận):**
- **Mask bleed — quan sát trực tiếp**: crop riêng (instance_id) cho thấy tint mask tràn
  xuống vùng chảo đen ở cả 2 box.
- Effective thickness proxy (V/area — **effective thickness proxy**, không phải độ dày
  thực): full ổ 450cm³ trên hình chiếu ~25×10cm → proxy ~1.8cm, THẤP hơn chiều cao
  thị giác ~4-5cm → depth KHÔNG phóng đại độ dày tương ứng.
- Nghi vấn: **mask area inflation** + **density 0.45 g/cm³** (giá trị của bánh mì thịt)
  áp dụng cho bánh mì không nhân (~0.25-0.3). Combined effect ~2.9× khớp quan sát —
  phân rã chưa chắc duy nhất (xem Experiment A/B).

## 4. Experiment A — Density sensitivity ✅ ĐÃ CHẠY

volume_cm3 per-item đã ghi vào bản ghi evaluation → sweep ρ ∈ {0.45→0.25} tính offline
(mass = V × ρ), không chạy lại pipeline. Chỉ thay density của Bánh mì; class khác giữ DB.

| ρ (g/cm³) | 0.45 | 0.40 | 0.35 | 0.30 | 0.27 | 0.25 |
|---|---|---|---|---|---|---|
| Bánh mì MAPE | 85.2% | 70.8% | 60.4% | 53.8% | 51.7% | 51.2% |
| Tổng kcal/ảnh MAPE | 63.1% | 52.9% | 46.1% | 43.7% | 43.7% | 44.8% |

**Kết luận sensitivity**: MAPE Bánh mì giảm nhanh tới ρ≈0.30 rồi **đi ngang ~51%** —
density một mình KHÔNG giải thích được toàn bộ over-estimation. **Phần sai lệch còn lại
có khả năng liên quan đến thể tích ước lượng; trong đó mask là giả thuyết có bằng chứng
trực tiếp (crop), còn contribution của depth chưa được cô lập** → Experiment B.
**Vùng sensitivity: ρ = 0.25–0.30 g/cm³** — chưa phải calibration "density thực".

## 5. Experiment B — Mask ablation ✅ ĐÃ CHẠY (variant sweep)

**Phát hiện định lượng — mask rất lớn so với toàn ảnh**: mask SAM2 gốc (box-prompted,
lift từ 640×640) chiếm **13-70% diện tích ảnh** (banh-mi3: 1.27M/1.8M px = 70%). Lưu ý
wording: con số này chứng minh **mask rất lớn so với toàn ảnh**, KHÔNG tự động đồng nghĩa
"toàn bộ 13-70% là bleed" — phần bleed được evidences bởi quan sát trực quan (crop/tint)
và ablation `cconly`; kết hợp lại cho thấy over-segmentation/background inclusion có
đóng góp đáng kể trong một số trường hợp.

**Thiết kế**: 1 lần `analyze`/ảnh (original) + 5 variant mask được tính lại với
**captured kwargs** (cùng depth/scale/K/intrinsics/density 0.45) — pipeline không sửa.

| Variant | area ratio (median) | volume ratio | mass ratio | Bánh mì MAPE |
|---|---|---|---|---|
| original | 1.00 | 1.00 | 1.00 | 84.8% |
| otsu (iter-1) | 0.32 | 0.42 | 0.42 | 93.0% |
| fixed60 | 0.45 | 0.61 | 0.61 | 80.9% |
| fixed60cc | 0.20 | 0.43 | 0.43 | 87.7% |
| fixed80cc | 0.20 | 0.36 | 0.36 | 89.2% |
| **cconly** | **0.89** | **0.91** | **0.92** | **61.6%** |

**Verdict Experiment B: PASS — mask contribution confirmed, partial, not sufficient to
explain the residual error.** Sau khi loại bỏ thành phần nền rời rạc bằng `cconly`,
MAPE vẫn còn ~61.6%: mask over-segmentation là **một nguồn sai số có đóng góp nhưng
không giải thích toàn bộ**; các yếu tố khác — đặc biệt hình học quan sát và tích phân
depth trên ảnh nghiêng — cần được đánh giá riêng trong Experiment C. `cconly` CHƯA được
đưa vào production (ablation, không phải patch).

## 5b. Hai ablation là ĐỘC LẬP — không cộng/trừ thành decomposition

Experiment A (density sweep: MAPE 85.2% → 51.2%) và Experiment B (mask ablation:
84.8% → 61.6%) là **hai ablation độc lập trên cùng baseline** — vì hai yếu tố nằm trong
phép nhân m = V×ρ. KHÔNG trình bày dạng decomposition "density = X%, mask = Y%" — chưa
có baseline độc lập cho volume thật và density thật.

## 6. Experiment C — Capture protocol (NEXT, chưa chạy)

Trả lời câu hỏi "độ ổn định của volume estimation thay đổi như thế nào theo góc chụp
đối với các món có hình học/depth khó?" — KHÔNG phải tối ưu model. Hai điều kiện tối
thiểu: **A. Top-down · B. Inclined/tilted**, giữ nguyên pipeline/model/configuration
(không đổi SAM2 prompt, density, Depth model, integration, resolution).

- **C1 — Existing-data exploratory analysis**: phân loại 10 ảnh Súp cua trong val set
  (đã có cả top-down lẫn tilted) theo điều kiện chụp, so sánh MAPE per condition. Bản
  chất: **exploratory screening** trên dữ liệu hiện có — chưa phải validation mạnh.
- **C2 — Controlled new capture**: chỉ làm nếu C1 cho thấy tín hiệu geometry nhất quán.

Tín hiệu mạnh đã có: Súp cua top-down **-2%** vs tilted **+79-158%**.

## 7. Khuyến nghị cho thesis

1. Ghi rõ **ràng buộc góc chụp** cho chất lỏng trong hộp trong suốt — -2% top-down
   chứng minh pipeline đúng khi điều kiện thỏa; +79-158% nghiêng là limitation.
2. Plane-fit miệng hộp để giới hạn tích phân depth (hướng cải tiến, chưa thực hiện).
3. Density: giữ 0.45 làm baseline; ρ = 0.25-0.30 là vùng sensitivity — mọi thay đổi
   thực tế cần experiment riêng (baseline → modified → regression).
4. Trứng nhiều lớp mỏng: ghi chú giới hạn depth; cân nhắc GT theo cả món (đã áp dụng).
