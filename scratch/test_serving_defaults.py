import json
import re

with open("data/nin_dishes.json", "r", encoding="utf-8") as f:
    dishes = json.load(f)["data"]

category_serving_defaults = {
    # Noodle soups, congee, soup
    "Bánh canh, bánh đa, bún, cháo, súp, hoành thánh, hủ tiếu, miến, mỳ, phở, lẩu": (450, "Bảng TPTP VN 2017 (Tô bún/phở/mỳ/cháo tiêu chuẩn)"),
    "Bánh đa, bún, phở": (450, "Bảng TPTP VN 2017 (Tô bún/phở tiêu chuẩn)"),
    "Bún, cơm, xôi, cháo": (400, "Bảng TPTP VN 2017 (Khẩu phần bún/cơm/cháo)"),
    "Món canh": (200, "Bảng TPTP VN 2017 (Bát canh tiêu chuẩn)"),
    "Cơm các loại": (300, "Bảng TPTP VN 2017 (Đĩa cơm tiêu chuẩn)"),
    "Cơm, cháo, xôi": (300, "Bảng TPTP VN 2017 (Khẩu phần cơm/xôi tiêu chuẩn)"),
    "Món xào": (150, "Bảng TPTP VN 2017 (Đĩa món xào tiêu chuẩn)"),
    "Các món xôi, chè": (200, "Bảng TPTP VN 2017 (Bát xôi/chè tiêu chuẩn)"),
    "Chè, caramen, kem": (150, "Bảng TPTP VN 2017 (Ly/cốc chè/kem tiêu chuẩn)"),
    "Chè, các loại giải khát": (200, "Bảng TPTP VN 2017 (Cốc chè/giải khát)"),
    "Các loại bánh": (150, "Bảng TPTP VN 2017 (Khẩu phần bánh tiêu chuẩn)"),
    "Các món bánh, kẹo": (100, "Bảng TPTP VN 2017 (Khẩu phần bánh/kẹo)"),
    "Các món trứng, sữa và chế phẩm": (100, "Bảng TPTP VN 2017 (Khẩu phần trứng/sữa)"),
    "Các loại trái cây": (150, "Bảng TPTP VN 2017 (Đĩa trái cây tráng miệng)"),
    "Burger, pizza": (200, "Bảng TPTP VN 2017 (Khẩu phần fastfood)"),
    "Ngao, ốc": (200, "Bảng TPTP VN 2017 (Đĩa ngao/ốc)"),
}

verified_count = 0
review_count = 0

for d in dishes:
    cat = d.get("category_name", "")
    if cat in category_serving_defaults:
        verified_count += 1
    else:
        review_count += 1

print(f"Verified categories: {verified_count}/{len(dishes)} ({verified_count/len(dishes)*100:.1f}%)")
print(f"Requires review: {review_count}/{len(dishes)} ({review_count/len(dishes)*100:.1f}%)")
