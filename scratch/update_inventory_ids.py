import re

with open("scripts/inventory_vietfood68.py", "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    # Banh trang
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb1d"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cacc"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb1d", "reason": "Bánh tráng - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cacc", "reason": "Bánh đa nem (Bánh tráng) - Bảng TPTP VN 2017"}',
    
    # Bong cai
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cba7"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb7d"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cba7", "reason": "Súp lơ (Bông cải) - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb7d", "reason": "Súp lơ trắng, tươi - Bảng TPTP VN 2017"}',
    
    # Bun
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb0c"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cad6"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb0c", "reason": "Bún tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cad6", "reason": "Bún tươi - Bảng TPTP VN 2017"}',
    
    # Ca
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc20"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc54"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc20", "reason": "Cá chép tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cc54", "reason": "Cá chép, tươi - Bảng TPTP VN 2017"}',
    
    # Ca chua
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb8c"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb1f"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb8c", "reason": "Cà chua tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb1f", "reason": "Quả cà chua, tươi - Bảng TPTP VN 2017"}',
    
    # Ca phao
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb98"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb8b"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb98", "reason": "Cà pháo muối nén - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb8b", "reason": "Cà pháo, muối nén - Bảng TPTP VN 2017"}',
    
    # Ca rot
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb8e"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb21"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb8e", "reason": "Cà rốt tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb21", "reason": "Củ cà rốt, tươi - Bảng TPTP VN 2017"}',
    
    # Cha
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cbde"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc30"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cbde", "reason": "Giò lụa - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cc30", "reason": "Giò lụa, chín - Bảng TPTP VN 2017"}',
    
    # Chanh
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc15"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cba4"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc15", "reason": "Quả chanh tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cba4", "reason": "Chanh, tươi - Bảng TPTP VN 2017"}',
    
    # Com
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb00"': '"primary_match_id": "nin_dish_68f8aadc094e90b19f042a82"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb00", "reason": "Cơm tẻ nấu chín - Bảng TPTP VN 2017"}': '{"id": "nin_dish_68f8aadc094e90b19f042a82", "reason": "Cơm tẻ miệng bát - CSDL Món ăn VDD"}',
    
    # Cu kieu
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb9e"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb93"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb9e", "reason": "Củ kiệu muối - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb93", "reason": "Kiệu, muối - Bảng TPTP VN 2017"}',
    
    # Cua
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc35"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc73"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc35", "reason": "Cua đồng / Cua biển - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cc73", "reason": "Cua đồng, tươi - Bảng TPTP VN 2017"}',
    
    # Dau hu
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb3a"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb11"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb3a", "reason": "Đậu phụ trắng - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb11", "reason": "Đậu phụ, sống - Bảng TPTP VN 2017"}',
    
    # Dua chua
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb9d"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb8e"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb9d", "reason": "Dưa cải muối chua - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb8e", "reason": "Dưa cải bẹ (muối dưa) - Bảng TPTP VN 2017"}',
    
    # Dua leo
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cba3"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb35"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cba3", "reason": "Dưa chuột tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb35", "reason": "Dưa chuột, tươi - Bảng TPTP VN 2017"}',
    
    # Mi
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb13"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cad8"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb13", "reason": "Mỳ sợi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cad8", "reason": "Mỳ sợi, khô - Bảng TPTP VN 2017"}',
    
    # Muc
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc3b"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc79"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc3b", "reason": "Mực tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cc79", "reason": "Mực, tươi - Bảng TPTP VN 2017"}',
    
    # Nam
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb85"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb99"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cb85", "reason": "Nấm rơm tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb99", "reason": "Nấm hương, tươi - Bảng TPTP VN 2017"}',
    
    # Oc
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc41"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc7c"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc41", "reason": "Ốc nhồi thịt - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cc7c", "reason": "Ốc nhồi, tươi - Bảng TPTP VN 2017"}',
    
    # Pho mai
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc91"': '"primary_match_id": "nin_dish_69512eb4a0cfb0866108cb79"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc91", "reason": "Pho mát - Bảng TPTP VN 2017"}': '{"id": "nin_dish_69512eb4a0cfb0866108cb79", "reason": "Phô mai tam giác - CSDL Món ăn VDD"}',
    
    # Rau
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cba0"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cb6d"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cba0", "reason": "Rau muống tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cb6d", "reason": "Rau muống, tươi - Bảng TPTP VN 2017"}',
    
    # Thit bo
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cbb8"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cbee"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cbb8", "reason": "Thịt bò loại 1 - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cbee", "reason": "Thịt bò, loại I, tươi - Bảng TPTP VN 2017"}',
    
    # Thit ga
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cbe2"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cbf8"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cbe2", "reason": "Thịt gà ta - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cbf8", "reason": "Thịt gà ta, tươi - Bảng TPTP VN 2017"}',
    
    # Thit heo
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cbcc"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cde4"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cbcc", "reason": "Thịt lợn nạc - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cde4", "reason": "Thịt lợn nạc vai, luộc - Bảng TPTP VN 2017"}',
    
    # Tom
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc2e"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc85"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc2e", "reason": "Tôm đồng tươi - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cc85", "reason": "Tôm đồng, tươi - Bảng TPTP VN 2017"}',
    
    # Trung
    '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cc7c"': '"primary_match_id": "nin_ing_6877a6b660d6c84e9bd5cbce"',
    '{"id": "nin_ing_6877a6b660d6c84e9bd5cc7c", "reason": "Trứng gà toàn phần - Bảng TPTP VN 2017"}': '{"id": "nin_ing_6877a6b660d6c84e9bd5cbce", "reason": "Quả trứng gà, tươi - Bảng TPTP VN 2017"}',
}

for k, v in replacements.items():
    if k in content:
        content = content.replace(k, v)
        print("Replaced:", k[:50])
    else:
        print("NOT FOUND:", k[:50])

with open("scripts/inventory_vietfood68.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated scripts/inventory_vietfood68.py")
