# Bảng Kiểm Kê Ánh Xạ 68 Lớp VietFood68 Sang CSDL Viện Dinh Dưỡng (VietFood68 Inventory Checklist)

## 1. Tóm Tắt Phân Bố Chiến Lược Ánh Xạ

| Chiến Lược Ánh Xạ | Số Lớp | Tỷ Lệ % | Vai Trò Trong Pipeline Khóa Luận |
| :--- | :---: | :---: | :--- |
| **Direct (Trực tiếp)** | 30 | 44.1% | Thực phẩm thô hoặc món đơn chất, ánh xạ 1-1 với CSDL NIN Thành phần thực phẩm |
| **Approximate (Xấp xỉ)** | 19 | 27.9% | Món nấu chín có công thức đại diện chính thức trong CSDL NIN Món ăn |
| **Component-based (Thành phần)** | 16 | 23.5% | Món phức hợp nhiều topping, hỗ trợ phân rã thành phần qua FoodSAM (Deep Mode) |
| **Unmapped (Duy trì Fallback)** | 3 | 4.4% | Lớp phi thực phẩm hoặc món đặc sản chưa có trong NIN, duy trì fallback literature/USDA |
| **Tổng cộng** | **68** | **100.0%** | Toàn bộ 68 lớp được kiểm toán đầy đủ căn cứ và provenance minh bạch |

---

## 2. Bảng Danh Mục Kiểm Kê Chi Tiết 68 Lớp

| ID | Tên Lớp (VietFood68) | Chiến Lược | Độ Tin Cậy Ánh Xạ | Đối Sánh CSDL NIN | Khẩu Phần | Nguồn Khẩu Phần | Căn Cứ Quyết Định |
| :-: | :--- | :---: | :---: | :--- | :-: | :--- | :--- |
| 00 | **Banh canh (Vietnamese thick noodle soup)** | `approximate` | `high` | Bánh canh thịt heo | 450g | Bảng TPTP VN 2017 (Tô bánh canh tiêu chuẩn) | Món ăn tương đồng trực tiếp trong CSDL Món ăn VDD (Bánh canh thịt heo) |
| 01 | **Banh chung (Square sticky rice cake)** | `approximate` | `high` | Bánh chưng cỡ vừa | 200g | Bảng TPTP VN 2017 (Khẩu phần bánh chưng) | Khẩu phần bánh chưng truyền thống có trong CSDL Món ăn VDD |
| 02 | **Banh cuon (Rolled rice pancake)** | `approximate` | `high` | Bánh cuốn thịt | 150g | Bảng TPTP VN 2017 (Đĩa bánh cuốn tiêu chuẩn) | Món bánh cuốn nhân thịt có trong CSDL Món ăn VDD |
| 03 | **Banh khot (Mini savory pancakes)** | `unmapped` | `none` | *(Không có NIN / Fallback)* | 150g | Y văn dinh dưỡng món ăn Việt Nam (Literature fallback) | Đặc sản Vũng Tàu / Nam Bộ, CSDL VDD quốc gia chưa có bản ghi tương ứng. Duy trì fallback literature đã kiểm toán Atwater. |
| 04 | **Banh mi (Vietnamese baguette sandwich)** | `approximate` | `high` | Bánh mỳ pate trứng | 200g | Bảng TPTP VN 2017 (Ổ bánh mì kẹp thịt/trứng) | Món bánh mì kẹp nhân tiêu chuẩn trong CSDL Món ăn VDD |
| 05 | **Banh trang (Rice paper)** | `direct` | `high` | Bánh đa nem | 100g | Bảng TPTP VN 2017 (100g phần ăn được) | Nguyên liệu bánh tráng khô trong CSDL thực phẩm thô NIN (Mã TPTP VN 1029) |
| 06 | **Banh trang tron (Rice paper salad)** | `component_based` | `high` | Bánh tráng trộn | 150g | Bảng TPTP VN 2017 (Suất bánh tráng trộn ăn vặt) | Món ăn vặt phối hợp nhiều topping; Fast Mode ánh xạ đĩa bánh tráng trộn tổng hợp, Deep Mode phân rã topping qua FoodSAM |
| 07 | **Banh xeo (Vietnamese sizzling pancake)** | `component_based` | `high` | Bánh xèo | 250g | Bảng TPTP VN 2017 (Cái bánh xèo cỡ vừa) | Món bánh có vỏ bột và nhân tôm thịt giá đỗ tách biệt; Deep Mode FoodSAM phân rã vỏ bánh và nhân |
| 08 | **Bo kho (Beef stew)** | `approximate` | `high` | Bánh mỳ sốt vang | 300g | Bảng TPTP VN 2017 (Tô bò kho bánh mì) | Món bò sốt vang / bò kho có trong CSDL Món ăn VDD |
| 09 | **Bo la lot (Grilled beef wrapped in betel leaves)** | `approximate` | `high` | Cơm suất (sườn, đậu, chả lá lốt) | 150g | Bảng TPTP VN 2017 (Đĩa bò lá lốt nướng) | Món chả lá lốt nướng / cơm suất chả lá lốt có trong CSDL Món ăn VDD |
| 10 | **Bong cai (Cauliflower)** | `direct` | `high` | Súp lơ trắng, tươi | 100g | Bảng TPTP VN 2017 (100g phần ăn được) | Súp lơ trắng trong CSDL thực phẩm thô NIN (Mã TPTP VN 4030) |
| 11 | **Bun (Rice vermicelli)** | `direct` | `high` | Bún, tươi | 150g | Bảng TPTP VN 2017 (Bát bún rối tiêu chuẩn) | Bún tươi trong CSDL thực phẩm thô NIN (Mã TPTP VN 1012) |
| 12 | **Bun bo Hue (Hue beef noodle soup)** | `component_based` | `high` | Bún bò giò heo (Huế) | 500g | Bảng TPTP VN 2017 (Tô bún bò Huế tiêu chuẩn) | Món nước phức hợp nhiều topping (thịt bắp bò, giò heo, chả cua, huyết, sợi bún); Deep Mode phân rã các topping |
| 13 | **Bun cha (Grilled pork with vermicelli)** | `component_based` | `high` | Bún chả | 450g | Bảng TPTP VN 2017 (Suất bún chả Hà Nội tiêu chuẩn) | Món ăn kinh điển phân tách giữa bún đĩa riêng, chả nướng trong bát nước chấm và rau sống; Deep Mode FoodSAM phân rã |
| 14 | **Bun dau (Vermicelli with tofu)** | `component_based` | `high` | Bún đậu mắm tôm | 450g | Bảng TPTP VN 2017 (Mẹt bún đậu thập cẩm) | Mẹt bún đậu phân tách rõ rệt từng thành phần (bún lá, đậu rán, thịt luộc, chả cốm, mắm tôm); bài toán thực nghiệm cốt lõi của FoodSAM |
| 15 | **Bun mam (Fermented fish noodle soup)** | `component_based` | `high` | Bún mắm | 500g | Bảng TPTP VN 2017 (Tô bún mắm Nam Bộ) | Món bún mắm miền Tây nhiều topping nổi (thịt quay, tôm, mực, cá lóc); Deep Mode phân rã các topping |
| 16 | **Bun rieu (Crab noodle soup)** | `component_based` | `high` | Bún riêu cua | 450g | Bảng TPTP VN 2017 (Tô bún riêu cua đậu) | Món bún có riêu cua, đậu rán, cà chua, huyết phân tách rõ; Deep Mode phân rã topping |
| 17 | **Ca (Fish)** | `direct` | `high` | Cá chép, tươi | 100g | Bảng TPTP VN 2017 (100g cá tươi ăn được) | Cá nước ngọt / cá chép trong CSDL thực phẩm thô NIN (Mã TPTP VN 5035) |
| 18 | **Ca chua (Tomato)** | `direct` | `high` | Quả cà chua, tươi | 100g | Bảng TPTP VN 2017 (100g cà chua tươi) | Cà chua trong CSDL thực phẩm thô NIN (Mã TPTP VN 4004) |
| 19 | **Ca phao (Pickled eggplant)** | `direct` | `high` | Cà pháo, muối nén | 50g | Bảng TPTP VN 2017 (Đĩa cà pháo muối) | Cà pháo muối trong CSDL thực phẩm thô NIN (Mã TPTP VN 4016) |
| 20 | **Ca rot (Carrot)** | `direct` | `high` | Củ cà rốt, tươi | 100g | Bảng TPTP VN 2017 (100g cà rốt tươi) | Cà rốt trong CSDL thực phẩm thô NIN (Mã TPTP VN 4006) |
| 21 | **Canh (Soup)** | `approximate` | `high` | Canh rau ngót nấu thịt | 200g | Bảng TPTP VN 2017 (Bát canh rau thịt nạc tiêu chuẩn) | Canh rau ngót nấu thịt nạc đại diện cho canh gia đình Việt trong CSDL Món ăn VDD |
| 22 | **Cha (Vietnamese pork roll)** | `direct` | `high` | Giò lụa, chín | 50g | Bảng TPTP VN 2017 (Khoanh giò lụa/chả quế) | Giò lụa trong CSDL thực phẩm thô NIN (Mã TPTP VN 5092) |
| 23 | **Cha gio (Spring rolls)** | `approximate` | `high` | Nem rán / Chả giò chiên | 120g | Bảng TPTP VN 2017 (Đĩa nem rán 3-4 cái) | Nem rán nhân thịt / chả giò chiên trong CSDL Món ăn VDD |
| 24 | **Chanh (Lime)** | `direct` | `high` | Chanh, tươi | 50g | Bảng TPTP VN 2017 (Quả chanh tươi) | Quả chanh trong CSDL thực phẩm thô NIN (Mã TPTP VN 4140) |
| 25 | **Com (Rice)** | `direct` | `high` | Cơm tẻ miệng bát | 150g | Bảng TPTP VN 2017 (Bát cơm tẻ nấu chín) | Cơm tẻ nấu chín trong CSDL thực phẩm thô NIN (Mã TPTP VN 1001) |
| 26 | **Com tam (Broken rice)** | `component_based` | `high` | Cơm sườn | 450g | Bảng TPTP VN 2017 (Đĩa cơm tấm sườn bì chả) | Đĩa cơm tấm sườn bì chả gồm cơm tấm, sườn nướng, bì heo, chả trứng; bài toán thực nghiệm cốt lõi của FoodSAM |
| 27 | **Con nguoi (Human)** | `unmapped` | `none` | *(Không có NIN / Fallback)* | 0g | Không áp dụng (Non-food class) | Lớp phi thực phẩm (Non-food background class), gán giá trị dinh dưỡng 0 |
| 28 | **Cu kieu (Pickled scallion head)** | `direct` | `high` | Kiệu, muối | 30g | Bảng TPTP VN 2017 (Đĩa củ kiệu ngâm chua ngọt) | Củ kiệu muối trong CSDL thực phẩm thô NIN (Mã TPTP VN 4022) |
| 29 | **Cua (Crab)** | `direct` | `high` | Cua đồng, tươi | 100g | Bảng TPTP VN 2017 (100g thịt cua ăn được) | Cua biển / cua đồng trong CSDL thực phẩm thô NIN (Mã TPTP VN 5056) |
| 30 | **Dau hu (Tofu)** | `direct` | `high` | Đậu phụ, sống | 100g | Bảng TPTP VN 2017 (Bìa đậu phụ trắng) | Đậu phụ trắng trong CSDL thực phẩm thô NIN (Mã TPTP VN 3005) |
| 31 | **Dua chua (Pickled vegetables)** | `direct` | `high` | Dưa cải bẹ | 50g | Bảng TPTP VN 2017 (Đĩa dưa cải chua) | Dưa cải muối chua trong CSDL thực phẩm thô NIN (Mã TPTP VN 4021) |
| 32 | **Dua leo (Cucumber)** | `direct` | `high` | Dưa chuột, tươi | 100g | Bảng TPTP VN 2017 (100g dưa leo tươi) | Dưa chuột tươi trong CSDL thực phẩm thô NIN (Mã TPTP VN 4026) |
| 33 | **Goi cuon (Fresh spring rolls)** | `component_based` | `high` | Bánh tráng trộn | 180g | Bảng TPTP VN 2017 (Đĩa 3 cuốn gỏi cuốn tôm thịt) | Gỏi cuốn gồm vỏ bánh tráng cuốn tôm, thịt luộc, bún và rau thơm tách biệt; Deep Mode phân rã topping |
| 34 | **Hamburger** | `approximate` | `high` | Hamburger lợn | 200g | Bảng TPTP VN 2017 & USDA FDC (Cái bánh hamburger) | Hamburger lợn / bánh mỳ kẹp thịt nướng trong CSDL Món ăn VDD |
| 35 | **Heo quay (Roast pork)** | `approximate` | `high` | Thịt lợn quay | 150g | Bảng TPTP VN 2017 (Đĩa thịt heo quay da giòn) | Món thịt lợn quay trong CSDL Món ăn VDD (phương ngữ Bắc: Thịt lợn quay) |
| 36 | **Hu tieu (Clear rice noodle soup)** | `component_based` | `high` | Hủ tiếu Nam vang nước | 450g | Bảng TPTP VN 2017 (Tô hủ tiếu Nam Vang) | Tô hủ tiếu Nam Vang gồm sợi hủ tiếu, tôm, thịt xá xíu, gan, trứng cút; Deep Mode phân rã topping |
| 37 | **Kho qua thit (Stuffed bitter melon soup)** | `approximate` | `high` | Canh khổ qua (mướp đắng) nhồi thịt | 250g | Bảng TPTP VN 2017 (Tô canh khổ qua nhồi thịt) | Canh mướp đắng (khổ qua) nhồi thịt trong CSDL Món ăn VDD (phương ngữ Bắc: Mướp đắng nhồi thịt) |
| 38 | **Khoai tay chien (French fries)** | `direct` | `high` | Khoai tây chiên | 100g | Bảng TPTP VN 2017 (Đĩa khoai tây chiên giòn) | Khoai tây chiên giòn trong CSDL Món ăn VDD |
| 39 | **Lau (Hotpot)** | `component_based` | `high` | Lẩu thập cẩm | 500g | Bảng TPTP VN 2017 (Nồi lẩu thập cẩm chia suất) | Nồi lẩu gồm nước dùng và các đĩa nhúng (thịt bò, nấm, rau, mì); Deep Mode phân rã các đĩa nhúng |
| 40 | **Long heo (Pork offal)** | `direct` | `high` | Lòng lợn luộc | 100g | Bảng TPTP VN 2017 (Đĩa lòng lợn luộc) | Lòng lợn luộc trong CSDL Món ăn VDD (phương ngữ Bắc: Lòng lợn luộc) |
| 41 | **Mi (Egg noodles)** | `direct` | `high` | Mỳ sợi, khô | 100g | Bảng TPTP VN 2017 (100g mì sợi luộc) | Mỳ sợi khô/nấu chín trong CSDL thực phẩm thô NIN (Mã TPTP VN 1019) |
| 42 | **Muc (Squid)** | `direct` | `high` | Mực, tươi | 100g | Bảng TPTP VN 2017 (100g mực tươi ăn được) | Mực tươi trong CSDL thực phẩm thô NIN (Mã TPTP VN 5064) |
| 43 | **Nam (Mushroom)** | `direct` | `high` | Nấm hương, tươi | 100g | Bảng TPTP VN 2017 (100g nấm tươi) | Nấm rơm tươi trong CSDL thực phẩm thô NIN (Mã TPTP VN 3088) |
| 44 | **Oc (Snails)** | `direct` | `high` | Ốc nhồi, tươi | 100g | Bảng TPTP VN 2017 (100g thịt ốc ăn được) | Ốc nhồi / ốc vặn trong CSDL thực phẩm thô NIN (Mã TPTP VN 5070) |
| 45 | **Ot chuong (Bell pepper)** | `direct` | `high` | Ớt xanh to, tươi | 100g | Bảng TPTP VN 2017 (100g ớt chuông tươi) | Ớt xanh to / ớt đỏ to (Ớt chuông ngọt) trong CSDL thực phẩm thô NIN (Mã TPTP VN 4085-4087) |
| 46 | **Pho (Vietnamese noodle soup)** | `approximate` | `high` | Phở bò chín | 500g | Bảng TPTP VN 2017 (Tô phở bò chín truyền thống) | Món phở bò chín truyền thống trong CSDL Món ăn VDD |
| 47 | **Pho mai (Cheese)** | `direct` | `high` | Phô mai tam giác | 30g | Bảng TPTP VN 2017 (Miếng phô mai lát/tam giác) | Pho mát trong CSDL thực phẩm thô NIN (Mã TPTP VN 6023) |
| 48 | **Rau (Vegetables)** | `direct` | `high` | Rau muống,  tươi | 100g | Bảng TPTP VN 2017 (100g rau xanh tươi) | Rau muống / rau ăn lá tổng hợp trong CSDL thực phẩm thô NIN (Mã TPTP VN 4023) |
| 49 | **Salad (Salad)** | `component_based` | `high` | Salad rau củ | 150g | Bảng TPTP VN 2017 (Đĩa salad trộn dầu giấm) | Đĩa rau trộn gồm xà lách, cà chua, dưa chuột và sốt dầu giấm; Deep Mode phân rã các loại rau |
| 50 | **Thit bo (Beef)** | `direct` | `high` | Thịt bò, loại I, tươi | 100g | Bảng TPTP VN 2017 (100g thịt bò nạc tươi) | Thịt bò loại 1 trong CSDL thực phẩm thô NIN (Mã TPTP VN 5001) |
| 51 | **Thit ga (Chicken)** | `direct` | `high` | Thịt gà ta, tươi | 100g | Bảng TPTP VN 2017 (100g thịt gà ta ăn được) | Thịt gà ta trong CSDL thực phẩm thô NIN (Mã TPTP VN 5024) |
| 52 | **Thit heo (Pork)** | `direct` | `high` | Thịt lợn nạc vai, luộc | 100g | Bảng TPTP VN 2017 (100g thịt lợn nạc) | Thịt lợn nạc trong CSDL thực phẩm thô NIN (Mã TPTP VN 5013) |
| 53 | **Thit kho (Braised pork)** | `approximate` | `high` | Thịt lợn ba chỉ kho tàu | 150g | Bảng TPTP VN 2017 (Đĩa thịt kho tàu) | Thịt lợn ba chỉ kho tàu trong CSDL Món ăn VDD (phương ngữ Bắc: Thịt lợn kho tàu) |
| 54 | **Thit nuong (Grilled meat)** | `approximate` | `high` | Thịt lợn xiên nướng | 120g | Bảng TPTP VN 2017 (Đĩa thịt lợn xiên nướng) | Thịt lợn xiên nướng trong CSDL Món ăn VDD |
| 55 | **Tom (Shrimp)** | `direct` | `high` | Tôm đồng, tươi | 100g | Bảng TPTP VN 2017 (100g tôm tươi ăn được) | Tôm đồng / tôm biển trong CSDL thực phẩm thô NIN (Mã TPTP VN 5049) |
| 56 | **Trung (Egg)** | `direct` | `high` | Quả trứng gà, tươi | 55g | Bảng TPTP VN 2017 (1 quả trứng gà trung bình) | Trứng gà toàn phần trong CSDL thực phẩm thô NIN (Mã TPTP VN 6001) |
| 57 | **Xoi (Sticky rice)** | `direct` | `high` | Xôi trắng miệng bát | 150g | Bảng TPTP VN 2017 (Gói xôi trắng) | Xôi trắng trong CSDL Món ăn VDD (món chế biến từ gạo nếp) |
| 58 | **Banh beo (Vietnamese savory steamed rice cake)** | `approximate` | `high` | Bánh bèo | 150g | Bảng TPTP VN 2017 (Đĩa/khay 5 chén bánh bèo) | Bánh bèo trong CSDL Món ăn VDD |
| 59 | **Cao lau (Cao lau noodles)** | `unmapped` | `none` | *(Không có NIN / Fallback)* | 350g | Y văn dinh dưỡng món ăn Việt Nam (Literature fallback) | Đặc sản Hội An (sợi mì tro tràm và xá xíu đặc thù), CSDL VDD quốc gia chưa có bản ghi tương ứng. Duy trì fallback literature đã kiểm toán Atwater. |
| 60 | **Mi Quang (Quang-style noodles)** | `component_based` | `high` | Mỳ Quảng | 450g | Bảng TPTP VN 2017 (Tô mì Quảng tôm thịt) | Tô mì Quảng gồm sợi mì nghệ, tôm thịt ram, đậu phộng rang, bánh tráng nướng; Deep Mode phân rã topping |
| 61 | **Com chien duong chau (Yangzhou fried rice)** | `approximate` | `high` | Cơm rang thập cẩm | 250g | Bảng TPTP VN 2017 (Đĩa cơm rang thập cẩm) | Món cơm rang thập cẩm trong CSDL Món ăn VDD |
| 62 | **Bun cha ca (Fish cake noodle soup)** | `approximate` | `high` | Bún chả cá | 450g | Bảng TPTP VN 2017 (Tô bún chả cá Đà Nẵng / Quy Nhơn) | Món bún chả cá có bản ghi chính thức trong CSDL Món ăn VDD |
| 63 | **Com chien ga (Fried rice with chicken)** | `approximate` | `high` | Cơm gà | 250g | Bảng TPTP VN 2017 (Đĩa cơm rang gà / cơm gà) | Món cơm gà trong CSDL Món ăn VDD |
| 64 | **Chao long (Pork organ congee)** | `component_based` | `high` | Cháo lòng | 400g | Bảng TPTP VN 2017 (Tô cháo lòng tiêu chuẩn) | Tô cháo lòng gồm phần cháo gạo và đĩa lòng dồi gan luộc; Deep Mode FoodSAM phân rã topping phủ |
| 65 | **Nom hoa chuoi (Banana blossom salad)** | `component_based` | `high` | Nộm hoa chuối thịt gà | 200g | Bảng TPTP VN 2017 (Đĩa nộm hoa chuối tai heo) | Đĩa nộm trộn giữa bắp hoa chuối thái sợi, tai heo / thịt gà xé, đậu phộng rang; Deep Mode phân rã topping |
| 66 | **Nui xao bo (Stir-fried macaroni with beef)** | `component_based` | `high` | Nui, luộc | 300g | Bảng TPTP VN 2017 (Đĩa nui xào thịt bò cà chua) | CSDL NIN không có món nui xào bò nguyên đĩa; phân rã thành nui luộc + thịt bò xào dầu |
| 67 | **Sup cua (Crab soup)** | `approximate` | `high` | Súp ngô cua | 200g | Bảng TPTP VN 2017 (Bát súp cua bắp) | Món súp ngô cua trong CSDL Món ăn VDD |