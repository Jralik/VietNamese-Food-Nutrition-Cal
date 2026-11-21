# Báo Cáo Phân Tích Ma Trận Ánh Xạ 68 Lớp VietFood68 Sang CSDL Viện Dinh Dưỡng

## 1. Phân Bố Chiến Lược Ánh Xạ (Mapping Strategy Distribution)

| Chiến Lược Ánh Xạ | Số Lượng Lớp | Tỷ Lệ % | Đặc Điểm Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **Direct (Trực tiếp)** | 30 | 44.1% | Ánh xạ 1-1 với thực phẩm thô hoặc món chuẩn VDD có độ tương đồng ngữ nghĩa tuyệt đối |
| **Approximate (Xấp xỉ)** | 19 | 27.9% | Ánh xạ sang món ăn nấu chín có công thức đại diện chính thức trong CSDL VDD |
| **Component-based (Thành phần)** | 16 | 23.5% | Cung cấp mô hình tri thức dinh dưỡng theo thành phần cho món phức hợp; FoodSAM Deep Mode hỗ trợ thị giác phân rã topping |
| **Unmapped (Duy trì Fallback)** | 3 | 4.4% | Lớp phi thực phẩm (Con người) hoặc đặc sản vùng miền chưa có trong CSDL VDD |
| **Tổng cộng** | **68** | **100.0%** | Toàn bộ 68 lớp đều có provenance và căn cứ phân loại minh bạch |

---

## 2. Bảng Danh Mục Chi Tiết 68 Lớp

| ID | Tên Lớp (VietFood68) | Chiến Lược | Độ Tin Cậy | Nguồn Đối Sánh NIN | Khẩu Phần Tham Chiếu | Nguồn Gốc Khẩu Phần |
| :-: | :--- | :---: | :---: | :--- | :-: | :--- |
| 00 | **Banh canh (Vietnamese thick noodle soup)** | `approximate` | `high` | Bánh canh thịt heo | 450g | Bảng TPTP VN 2017 (Tô bánh canh tiêu chuẩn) |
| 01 | **Banh chung (Square sticky rice cake)** | `approximate` | `high` | Bánh chưng cỡ vừa | 200g | Bảng TPTP VN 2017 (Khẩu phần bánh chưng) |
| 02 | **Banh cuon (Rolled rice pancake)** | `approximate` | `high` | Bánh cuốn thịt | 150g | Bảng TPTP VN 2017 (Đĩa bánh cuốn tiêu chuẩn) |
| 03 | **Banh khot (Mini savory pancakes)** | `unmapped` | `none` | *(Giữ Fallback)* | 150g | Y văn dinh dưỡng món ăn Việt Nam (Literature fallback) |
| 04 | **Banh mi (Vietnamese baguette sandwich)** | `approximate` | `high` | Bánh mỳ pate trứng | 200g | Bảng TPTP VN 2017 (Ổ bánh mì kẹp thịt/trứng) |
| 05 | **Banh trang (Rice paper)** | `direct` | `high` | Bánh đa nem | 100g | Bảng TPTP VN 2017 (100g phần ăn được) |
| 06 | **Banh trang tron (Rice paper salad)** | `component_based` | `high` | Bánh tráng trộn | 150g | Bảng TPTP VN 2017 (Suất bánh tráng trộn ăn vặt) |
| 07 | **Banh xeo (Vietnamese sizzling pancake)** | `component_based` | `high` | Bánh xèo | 250g | Bảng TPTP VN 2017 (Cái bánh xèo cỡ vừa) |
| 08 | **Bo kho (Beef stew)** | `approximate` | `high` | Bánh mỳ sốt vang | 300g | Bảng TPTP VN 2017 (Tô bò kho bánh mì) |
| 09 | **Bo la lot (Grilled beef wrapped in betel leaves)** | `approximate` | `high` | Cơm suất (sườn, đậu, chả lá lốt) | 150g | Bảng TPTP VN 2017 (Đĩa bò lá lốt nướng) |
| 10 | **Bong cai (Cauliflower)** | `direct` | `high` | Súp lơ trắng, tươi | 100g | Bảng TPTP VN 2017 (100g phần ăn được) |
| 11 | **Bun (Rice vermicelli)** | `direct` | `high` | Bún, tươi | 150g | Bảng TPTP VN 2017 (Bát bún rối tiêu chuẩn) |
| 12 | **Bun bo Hue (Hue beef noodle soup)** | `component_based` | `high` | Bún bò giò heo (Huế) | 500g | Bảng TPTP VN 2017 (Tô bún bò Huế tiêu chuẩn) |
| 13 | **Bun cha (Grilled pork with vermicelli)** | `component_based` | `high` | Bún chả | 450g | Bảng TPTP VN 2017 (Suất bún chả Hà Nội tiêu chuẩn) |
| 14 | **Bun dau (Vermicelli with tofu)** | `component_based` | `high` | Bún đậu mắm tôm | 450g | Bảng TPTP VN 2017 (Mẹt bún đậu thập cẩm) |
| 15 | **Bun mam (Fermented fish noodle soup)** | `component_based` | `high` | Bún mắm | 500g | Bảng TPTP VN 2017 (Tô bún mắm Nam Bộ) |
| 16 | **Bun rieu (Crab noodle soup)** | `component_based` | `high` | Bún riêu cua | 450g | Bảng TPTP VN 2017 (Tô bún riêu cua đậu) |
| 17 | **Ca (Fish)** | `direct` | `high` | Cá chép, tươi | 100g | Bảng TPTP VN 2017 (100g cá tươi ăn được) |
| 18 | **Ca chua (Tomato)** | `direct` | `high` | Quả cà chua, tươi | 100g | Bảng TPTP VN 2017 (100g cà chua tươi) |
| 19 | **Ca phao (Pickled eggplant)** | `direct` | `high` | Cà pháo, muối nén | 50g | Bảng TPTP VN 2017 (Đĩa cà pháo muối) |
| 20 | **Ca rot (Carrot)** | `direct` | `high` | Củ cà rốt, tươi | 100g | Bảng TPTP VN 2017 (100g cà rốt tươi) |
| 21 | **Canh (Soup)** | `approximate` | `high` | Canh rau ngót nấu thịt | 200g | Bảng TPTP VN 2017 (Bát canh rau thịt nạc tiêu chuẩn) |
| 22 | **Cha (Vietnamese pork roll)** | `direct` | `high` | Giò lụa, chín | 50g | Bảng TPTP VN 2017 (Khoanh giò lụa/chả quế) |
| 23 | **Cha gio (Spring rolls)** | `approximate` | `high` | Nem rán / Chả giò chiên | 120g | Bảng TPTP VN 2017 (Đĩa nem rán 3-4 cái) |
| 24 | **Chanh (Lime)** | `direct` | `high` | Chanh, tươi | 50g | Bảng TPTP VN 2017 (Quả chanh tươi) |
| 25 | **Com (Rice)** | `direct` | `high` | Cơm tẻ miệng bát | 150g | Bảng TPTP VN 2017 (Bát cơm tẻ nấu chín) |
| 26 | **Com tam (Broken rice)** | `component_based` | `high` | Cơm sườn | 450g | Bảng TPTP VN 2017 (Đĩa cơm tấm sườn bì chả) |
| 27 | **Con nguoi (Human)** | `unmapped` | `none` | *(Giữ Fallback)* | 0g | Không áp dụng (Non-food class) |
| 28 | **Cu kieu (Pickled scallion head)** | `direct` | `high` | Kiệu, muối | 30g | Bảng TPTP VN 2017 (Đĩa củ kiệu ngâm chua ngọt) |
| 29 | **Cua (Crab)** | `direct` | `high` | Cua đồng, tươi | 100g | Bảng TPTP VN 2017 (100g thịt cua ăn được) |
| 30 | **Dau hu (Tofu)** | `direct` | `high` | Đậu phụ, sống | 100g | Bảng TPTP VN 2017 (Bìa đậu phụ trắng) |
| 31 | **Dua chua (Pickled vegetables)** | `direct` | `high` | Dưa cải bẹ | 50g | Bảng TPTP VN 2017 (Đĩa dưa cải chua) |
| 32 | **Dua leo (Cucumber)** | `direct` | `high` | Dưa chuột, tươi | 100g | Bảng TPTP VN 2017 (100g dưa leo tươi) |
| 33 | **Goi cuon (Fresh spring rolls)** | `component_based` | `high` | Bánh tráng trộn | 180g | Bảng TPTP VN 2017 (Đĩa 3 cuốn gỏi cuốn tôm thịt) |
| 34 | **Hamburger** | `approximate` | `high` | Hamburger lợn | 200g | Bảng TPTP VN 2017 & USDA FDC (Cái bánh hamburger) |
| 35 | **Heo quay (Roast pork)** | `approximate` | `high` | Thịt lợn quay | 150g | Bảng TPTP VN 2017 (Đĩa thịt heo quay da giòn) |
| 36 | **Hu tieu (Clear rice noodle soup)** | `component_based` | `high` | Hủ tiếu Nam vang nước | 450g | Bảng TPTP VN 2017 (Tô hủ tiếu Nam Vang) |
| 37 | **Kho qua thit (Stuffed bitter melon soup)** | `approximate` | `high` | Canh khổ qua (mướp đắng) nhồi thịt | 250g | Bảng TPTP VN 2017 (Tô canh khổ qua nhồi thịt) |
| 38 | **Khoai tay chien (French fries)** | `direct` | `high` | Khoai tây chiên | 100g | Bảng TPTP VN 2017 (Đĩa khoai tây chiên giòn) |
| 39 | **Lau (Hotpot)** | `component_based` | `high` | Lẩu thập cẩm | 500g | Bảng TPTP VN 2017 (Nồi lẩu thập cẩm chia suất) |
| 40 | **Long heo (Pork offal)** | `direct` | `high` | Lòng lợn luộc | 100g | Bảng TPTP VN 2017 (Đĩa lòng lợn luộc) |
| 41 | **Mi (Egg noodles)** | `direct` | `high` | Mỳ sợi, khô | 100g | Bảng TPTP VN 2017 (100g mì sợi luộc) |
| 42 | **Muc (Squid)** | `direct` | `high` | Mực, tươi | 100g | Bảng TPTP VN 2017 (100g mực tươi ăn được) |
| 43 | **Nam (Mushroom)** | `direct` | `high` | Nấm hương, tươi | 100g | Bảng TPTP VN 2017 (100g nấm tươi) |
| 44 | **Oc (Snails)** | `direct` | `high` | Ốc nhồi, tươi | 100g | Bảng TPTP VN 2017 (100g thịt ốc ăn được) |
| 45 | **Ot chuong (Bell pepper)** | `direct` | `high` | Ớt xanh to, tươi | 100g | Bảng TPTP VN 2017 (100g ớt chuông tươi) |
| 46 | **Pho (Vietnamese noodle soup)** | `approximate` | `high` | Phở bò chín | 500g | Bảng TPTP VN 2017 (Tô phở bò chín truyền thống) |
| 47 | **Pho mai (Cheese)** | `direct` | `high` | Phô mai tam giác | 30g | Bảng TPTP VN 2017 (Miếng phô mai lát/tam giác) |
| 48 | **Rau (Vegetables)** | `direct` | `high` | Rau muống,  tươi | 100g | Bảng TPTP VN 2017 (100g rau xanh tươi) |
| 49 | **Salad (Salad)** | `component_based` | `high` | Salad rau củ | 150g | Bảng TPTP VN 2017 (Đĩa salad trộn dầu giấm) |
| 50 | **Thit bo (Beef)** | `direct` | `high` | Thịt bò, loại I, tươi | 100g | Bảng TPTP VN 2017 (100g thịt bò nạc tươi) |
| 51 | **Thit ga (Chicken)** | `direct` | `high` | Thịt gà ta, tươi | 100g | Bảng TPTP VN 2017 (100g thịt gà ta ăn được) |
| 52 | **Thit heo (Pork)** | `direct` | `high` | Thịt lợn nạc vai, luộc | 100g | Bảng TPTP VN 2017 (100g thịt lợn nạc) |
| 53 | **Thit kho (Braised pork)** | `approximate` | `high` | Thịt lợn ba chỉ kho tàu | 150g | Bảng TPTP VN 2017 (Đĩa thịt kho tàu) |
| 54 | **Thit nuong (Grilled meat)** | `approximate` | `high` | Thịt lợn xiên nướng | 120g | Bảng TPTP VN 2017 (Đĩa thịt lợn xiên nướng) |
| 55 | **Tom (Shrimp)** | `direct` | `high` | Tôm đồng, tươi | 100g | Bảng TPTP VN 2017 (100g tôm tươi ăn được) |
| 56 | **Trung (Egg)** | `direct` | `high` | Quả trứng gà, tươi | 55g | Bảng TPTP VN 2017 (1 quả trứng gà trung bình) |
| 57 | **Xoi (Sticky rice)** | `direct` | `high` | Xôi trắng miệng bát | 150g | Bảng TPTP VN 2017 (Gói xôi trắng) |
| 58 | **Banh beo (Vietnamese savory steamed rice cake)** | `approximate` | `high` | Bánh bèo | 150g | Bảng TPTP VN 2017 (Đĩa/khay 5 chén bánh bèo) |
| 59 | **Cao lau (Cao lau noodles)** | `unmapped` | `none` | *(Giữ Fallback)* | 350g | Y văn dinh dưỡng món ăn Việt Nam (Literature fallback) |
| 60 | **Mi Quang (Quang-style noodles)** | `component_based` | `high` | Mỳ Quảng | 450g | Bảng TPTP VN 2017 (Tô mì Quảng tôm thịt) |
| 61 | **Com chien duong chau (Yangzhou fried rice)** | `approximate` | `high` | Cơm rang thập cẩm | 250g | Bảng TPTP VN 2017 (Đĩa cơm rang thập cẩm) |
| 62 | **Bun cha ca (Fish cake noodle soup)** | `approximate` | `high` | Bún chả cá | 450g | Bảng TPTP VN 2017 (Tô bún chả cá Đà Nẵng / Quy Nhơn) |
| 63 | **Com chien ga (Fried rice with chicken)** | `approximate` | `high` | Cơm gà | 250g | Bảng TPTP VN 2017 (Đĩa cơm rang gà / cơm gà) |
| 64 | **Chao long (Pork organ congee)** | `component_based` | `high` | Cháo lòng | 400g | Bảng TPTP VN 2017 (Tô cháo lòng tiêu chuẩn) |
| 65 | **Nom hoa chuoi (Banana blossom salad)** | `component_based` | `high` | Nộm hoa chuối thịt gà | 200g | Bảng TPTP VN 2017 (Đĩa nộm hoa chuối tai heo) |
| 66 | **Nui xao bo (Stir-fried macaroni with beef)** | `component_based` | `high` | Nui, luộc | 300g | Bảng TPTP VN 2017 (Đĩa nui xào thịt bò cà chua) |
| 67 | **Sup cua (Crab soup)** | `approximate` | `high` | Súp ngô cua | 200g | Bảng TPTP VN 2017 (Bát súp cua bắp) |