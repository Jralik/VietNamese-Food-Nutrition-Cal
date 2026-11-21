# Nguồn đối chiếu dữ liệu dinh dưỡng — Báo cáo cross-check

Ngày chạy: 2026-09-28 22:19

## Nguồn uy tín sử dụng

| Nguồn | Kênh | Cơ sở |
|---|---|---|
| Viện Dinh dưỡng Quốc gia — Bảng TPTP VN 2017 | viendinhduong.vn (tool "Giá trị dinh dưỡng món ăn" + "thực phẩm") | per_serving → per_100g chuẩn hóa |
| Viện NC Dinh dưỡng TP.HCM — Thực đơn gia đình | thucdongiadinh.vn (tool "Tra cứu thành phần dinh dưỡng thực phẩm") | per_100g (BTP2007/2000) |
| USDA FoodData Central | fdc.nal.usda.gov | per_100g (SR Legacy / Foundation) |

## 1. Đối chiếu NIN ↔ TDGD (68 lớp)

- Khớp comparable với bảng TDGD (score ≥ 0.5, cùng loại món/nguyên liệu): **40/68**
- Lớp lệch lớn (> 25% ở ít nhất 1 chất): **18** (chi tiết bên dưới)

### Bảng chi tiết (chỉ các lớp khớp)

| Lớp | TDGD match | kcal NIN/TDGD | Δ% kcal | Protein NIN/TDGD | Δ% | Fat NIN/TDGD | Δ% | Carb NIN/TDGD | Δ% |
|---|---|---|---|---|---|---|---|---|---|
| Cha gio (Spring rolls) | Chả giò chiên | 245/245 | +0% | 19/11 | +75% | 18/12 | +44% | 200/22 | +825% |
| Thit nuong (Grilled meat) | Thịt xiên nướng | 185/322 | -42% | 23/26 | -13% | 8/24 | -65% | 4/1 | +687% |
| Xoi (Sticky rice) | Xôi gấc | 210/231 | -9% | 5/3 | +63% | 6/1 | +644% | 45/53 | -14% |
| Mi (Egg noodles) | Mì Udon | 350/75 | +367% | 11/3 | +323% | 1/0 | +200% | 74/15 | +390% |
| Chao long (Pork organ congee) | Cháo lòng | 62/69 | -10% | 6/1 | +305% | 2/4 | -53% | 5/6 | -14% |
| Cha (Vietnamese pork roll) | Giò lụa | 205/205 | +0% | 17/17 | +0% | 15/15 | +0% | 2/0 | +260% |
| Bun dau (Vermicelli with tofu) | Bún đậu | 156/151 | +3% | 11/3 | +252% | 6/6 | -6% | 15/20 | -27% |
| Banh beo (Vietnamese savory steamed rice cake) | Bánh bèo | 118/118 | +0% | 5/2 | +176% | 4/4 | +7% | 14/18 | -21% |
| Khoai tay chien (French fries) | Khoai tây chiên | 466/260 | +79% | 2/3 | -44% | 10/14 | -30% | 17/31 | -46% |
| Pho (Vietnamese noodle soup) | Phở bò chín | 100/73 | +36% | 7/5 | +33% | 1/1 | +65% | 15/11 | +32% |
| Banh cuon (Rolled rice pancake) | Bánh cuốn | 129/129 | -0% | 5/3 | +63% | 5/5 | +2% | 15/17 | -13% |
| Bun cha ca (Fish cake noodle soup) | Bún chả | 141/136 | +4% | 5/6 | -21% | 7/4 | +52% | 15/18 | -14% |
| Bun cha (Grilled pork with vermicelli) | Bún chả | 115/136 | -15% | 6/6 | -6% | 2/4 | -44% | 20/18 | +12% |
| Bun rieu (Crab noodle soup) | Bún cua | 61/53 | +15% | 1/2 | -38% | 1/1 | -11% | 12/9 | +34% |
| Hamburger | Hamburger McDONALD'S | 188/264 | -29% | 12/13 | -5% | 7/10 | -35% | 20/30 | -34% |
| Banh chung (Square sticky rice cake) | Bánh chưng | 204/181 | +13% | 5/4 | +9% | 6/4 | +33% | 34/32 | +7% |
| Com chien duong chau (Yangzhou fried rice) | Cơm rang | 208/198 | +5% | 7/6 | +10% | 7/10 | -24% | 29/22 | +32% |
| Ca phao (Pickled eggplant) | Cà pháo | 20/27 | -26% | 1/2 | -13% | —/— | — | 4/5 | -29% |
| Thit heo (Pork) | Thịt heo (lợn), nạc vai | 126/149 | -15% | 18/21 | -15% | 6/7 | -15% | —/0 | — |
| Bun (Rice vermicelli) | Bún tươi | 116/112 | +4% | 2/2 | +0% | 1/— | — | 25/26 | -3% |
| Banh trang (Rice paper) | Bánh đa nem/ Bánh tráng | 335/335 | +0% | 4/4 | +0% | 0/0 | +0% | 79/79 | +0% |
| Bong cai (Cauliflower) | Súp lơ trắng | 34/34 | +0% | 2/2 | +0% | 0/0 | +0% | 6/6 | +0% |
| Ca (Fish) | Cá chép | 96/96 | +0% | 16/16 | +0% | 4/4 | +0% | —/— | — |
| Ca chua (Tomato) | Cà chua | 24/24 | +0% | 1/1 | +0% | 0/0 | +0% | 5/5 | +0% |
| Ca rot (Carrot) | Cà rốt  | 44/44 | +0% | 2/2 | +0% | 0/0 | +0% | 9/9 | +0% |
| Chanh (Lime) | Chanh | 29/29 | +0% | 1/1 | +0% | 0/0 | +0% | 6/6 | +0% |
| Cu kieu (Pickled scallion head) | Kiệu muối | 29/29 | +0% | 1/1 | +0% | —/— | — | 6/6 | +0% |
| Cua (Crab) | Cua đồng | 87/87 | +0% | 12/12 | +0% | 3/3 | +0% | 2/2 | +0% |
| Dau hu (Tofu) | Đậu phụ (đậu hũ) | 97/97 | +0% | 11/11 | +0% | 5/5 | +0% | 1/1 | +0% |
| Dua chua (Pickled vegetables) | Dưa cải bẹ | 25/25 | +0% | 2/2 | +0% | —/— | — | 4/4 | +0% |
| Dua leo (Cucumber) | Dưa chuột/ Dưa leo | 18/18 | +0% | 1/1 | +0% | 0/0 | +0% | 4/4 | +0% |
| Muc (Squid) | Mực tươi | 73/73 | +0% | 16/16 | +0% | 1/1 | +0% | —/— | — |
| Nam (Mushroom) | Nấm mỡ (nấm tây) | 37/37 | +0% | 4/4 | +0% | 0/0 | +0% | 4/4 | +0% |
| Oc (Snails) | Ốc nhồi | 84/84 | +0% | 12/12 | +0% | 1/1 | +0% | 8/8 | +0% |
| Ot chuong (Bell pepper) | Ớt xanh to | 31/31 | +0% | 1/1 | +0% | 0/0 | +0% | 6/6 | +0% |
| Rau (Vegetables) | Rau muống | 28/28 | +0% | 3/3 | +0% | 0/0 | +0% | 3/3 | +0% |
| Thit bo (Beef) | Thịt bò loại 1 | 118/118 | +0% | 21/21 | +0% | 4/4 | +0% | —/— | — |
| Thit ga (Chicken) | Thịt gà ta | 199/199 | +0% | 20/20 | +0% | 13/13 | +0% | —/— | — |
| Tom (Shrimp) | Tôm đồng | 90/90 | +0% | 18/18 | +0% | 2/2 | +0% | —/— | — |
| Trung (Egg) | Trứng gà ta (35g/quả) | 150/150 | +0% | 13/13 | +0% | 10/10 | +0% | 1/1 | +0% |

## 2. Kiểm chứng USDA FoodData Central (commodity khớp)

_(Chưa chạy được: API key DEMO_KEY của USDA bị giới hạn tần suất (HTTP 429). Chạy lại với key riêng: `set FDC_API_KEY=<key> && python scripts/crosscheck_sources.py` — script sẽ đối chiếu 8 commodity gắn nhãn usda_fdc và ghi fdcId vào báo cáo.)_

## 3. Kiểm chứng vật lý mật độ (mô hình Choi–Okos 1986)

Mật độ hữu hiệu khi bày món (what the volume pipeline measures) phải **không vượt quá** mật độ vật liệu tính từ thành phần per-100g (mô hình thành phần song song Choi–Okos).

### Vi phạm / cận hạn

Không có vi phạm — mọi giá trị DENSITY_DB ≤ mật độ vật liệu.

Các lớp có mật độ DB sát trần vật liệu (≤ 0.02 g/cm³): Banh chung (Square sticky rice cake) (1.15/1.157), Canh (Soup) (1.0/1.018), Cha (Vietnamese pork roll) (1.05/1.048), Pho mai (Cheese) (1.05/1.046)

## 4. Kiểm tra projection class_names.py ↔ mapping SSOT

Theo thiết kế (P3-F), class_names.py **giữ nguyên 7 trường macro** (Calories/Protein/Fat/Carbs/Saturates/Sugar/Salt) từ bộ dữ liệu gốc và chỉ đồng bộ 5 vi chất từ mapping SSOT — chênh lệch macro là chủ ý, không phải lỗi.

- Vi chất (Sodium/Calcium/Iron/Zinc/Cholesterol) khớp 100% giữa class_names.py và mapping SSOT.
- Macro giữ nguyên theo thiết kế: 161 cặp lệch (top lệch nhất, ratio class_names/mapping):
  - Canh (Soup) · Carbs: 3.5 vs 0.85 (ratio 4.12)
  - Bun rieu (Crab noodle soup) · Fat: 3.2 vs 0.8 (ratio 4.0)
  - Nui xao bo (Stir-fried macaroni with beef) · Fat: 6 vs 1.51 (ratio 3.97)
  - Bun rieu (Crab noodle soup) · Protein: 4.8 vs 1.311 (ratio 3.66)
  - Banh xeo (Vietnamese sizzling pancake) · Fat: 9.5 vs 2.9 (ratio 3.28)
  - Mi Quang (Quang-style noodles) · Fat: 4.8 vs 1.711 (ratio 2.81)
  - Salad (Salad) · Protein: 1.5 vs 0.55 (ratio 2.73)
  - Khoai tay chien (French fries) · Carbs: 41 vs 17.0 (ratio 2.41)
  - Mi (Egg noodles) · Fat: 2.1 vs 0.9 (ratio 2.33)
  - Khoai tay chien (French fries) · Protein: 3.4 vs 1.6 (ratio 2.12)

## 5. Audit ĐẦU VÀO TÍNH TOÁN DINH DƯỠNG (class_names per-100g × khối lượng ước lượng)

Ứng dụng tính dinh dưỡng khẩu phần = `mass_g × per-100g / 100`, với per-100g lấy từ `class_names.nutrition_per_100g` (cả pipeline volume lẫn luồng demo). Mục này so trực tiếp GIÁ TRỊ TÍNH TOÁN với bảng TDGD (Bảng TPTP VN). Verdict: **match** (≤15%), **variant** (15–40% — khác biệt công thức giữa hai bảng), **review** (>40% — cần xem), **no_tdgd_reference** (TDGD không có dòng tương đương).

Tổng kết: **match**: 27, **no_tdgd_reference**: 28, **review**: 5, **variant**: 8 trên 68 lớp.

### Các lớp verdict `review` (lệch > 40%)

| Lớp | TDGD match | kcal tính/TDGD | Δ% kcal | Protein | Δ% | Fat | Δ% | Carb | Δ% |
|---|---|---|---|---|---|---|---|---|---|
| Bun dau (Vermicelli with tofu) | Bún đậu | 150/151.0 | -1% | 6.5/3.0 | +117% | 7.5/6.4 | +17% | 15/20.3 | -26% |
| Bun rieu (Crab noodle soup) | Bún cua | 85/53.0 | +60% | 4.8/2.1 | +129% | 3.2/0.9 | +256% | 9.2/9.1 | +1% |
| Cha (Vietnamese pork roll) | Giò lụa | 180/205.0 | -12% | 14/16.9 | -17% | 12/15.0 | -20% | 3/0.5 | +500% |
| Mi (Egg noodles) | Mì Udon | 138/75.0 | +84% | 4.5/2.6 | +73% | 2.1/0.3 | +600% | 25/15.2 | +64% |
| Chao long (Pork organ congee) | Cháo lòng | 80/69.0 | +16% | 4.5/1.4 | +221% | 2.8/4.4 | -36% | 9.5/6.0 | +58% |

### Nhất quán nội bộ per-serving = per-100g × khẩu phần

- `nutrition` (per-serving) = `nutrition_per_100g` × serving_size/100 khớp trong dung sai làm tròn cho toàn bộ 68 lớp — luồng demo dùng đúng dữ liệu đã audit.
