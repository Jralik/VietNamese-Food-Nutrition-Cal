import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from class_names import class_names

with open("data/nin_nutrition_canonical.json", "r", encoding="utf-8") as f:
    canonical = json.load(f)["records"]

print(f"Loaded {len(canonical)} canonical records.")

# Expert curated dialect mapping dictionary for 68 classes:
CLASS_SYNONYMS = {
    "Kho qua thit (Stuffed bitter melon soup)": ["khổ qua", "mướp đắng nhồi thịt", "canh khổ qua"],
    "Heo quay (Roast pork)": ["thịt lợn quay", "thịt quay"],
    "Long heo (Pork offal)": ["lòng lợn luộc", "lòng lợn"],
    "Mi Quang (Quang-style noodles)": ["mỳ quảng", "mì quảng"],
    "Cao lau (Cao lau noodles)": ["cao lầu", "cao lau"],
    "Bo la lot (Grilled beef wrapped in betel leaves)": ["chả lá lốt", "lá lốt"],
    "Nui xao bo (Stir-fried macaroni with beef)": ["nui xào", "nui"],
    "Com tam (Broken rice)": ["cơm tấm", "sườn", "cơm sườn"],
    "Banh canh (Vietnamese thick noodle soup)": ["bánh canh"],
    "Banh chung (Square sticky rice cake)": ["bánh chưng"],
    "Banh cuon (Rolled rice pancake)": ["bánh cuốn"],
    "Banh khot (Mini savory pancakes)": ["bánh khọt"],
    "Banh mi (Vietnamese baguette sandwich)": ["bánh mì", "bánh mỳ"],
    "Banh trang (Rice paper)": ["bánh tráng"],
    "Banh trang tron (Rice paper salad)": ["bánh tráng trộn"],
    "Banh xeo (Vietnamese sizzling pancake)": ["bánh xèo"],
    "Bo kho (Beef stew)": ["bò kho"],
    "Bun (Rice vermicelli)": ["bún tươi", "bún"],
    "Bun bo Hue (Hue beef noodle soup)": ["bún bò huế", "bún bò"],
    "Bun cha (Grilled pork with vermicelli)": ["bún chả"],
    "Bun dau (Vermicelli with tofu)": ["bún đậu"],
    "Bun mam (Fermented fish noodle soup)": ["bún mắm"],
    "Bun rieu (Crab noodle soup)": ["bún riêu"],
    "Bun cha ca (Fish cake noodle soup)": ["bún chả cá"],
    "Canh (Soup)": ["canh rau", "canh cải", "canh"],
    "Chao long (Pork organ congee)": ["cháo lòng"],
    "Cha (Vietnamese pork roll)": ["giò lụa", "chả lụa", "chả quế"],
    "Cha gio (Spring rolls)": ["nem rán", "chả giò"],
    "Chanh (Lime)": ["chanh"],
    "Com (Rice)": ["cơm tẻ", "cơm trắng", "cơm"],
    "Com chien duong chau (Yangzhou fried rice)": ["cơm rang", "cơm chiên"],
    "Com chien ga (Fried rice with chicken)": ["cơm gà", "cơm chiên gà"],
    "Cu kieu (Pickled scallion head)": ["kiệu muối", "củ kiệu"],
    "Cua (Crab)": ["cua"],
    "Dau hu (Tofu)": ["đậu phụ", "đậu hũ"],
    "Dua chua (Pickled vegetables)": ["dưa cải chua", "dưa chua", "dưa muối"],
    "Dua leo (Cucumber)": ["dưa chuột", "dưa leo"],
    "Goi cuon (Fresh spring rolls)": ["gỏi cuốn", "nem cuốn"],
    "Hamburger": ["hamburger bò", "hamburger"],
    "Hu tieu (Clear rice noodle soup)": ["hủ tiếu", "hủ tíu"],
    "Khoai tay chien (French fries)": ["khoai tây chiên"],
    "Lau (Hotpot)": ["lẩu"],
    "Mi (Egg noodles)": ["mỳ trứng", "mì trứng", "mỳ"],
    "Muc (Squid)": ["mực"],
    "Nam (Mushroom)": ["nấm rơm", "nấm hương", "nấm"],
    "Nom hoa chuoi (Banana blossom salad)": ["nộm hoa chuối"],
    "Oc (Snails)": ["ốc"],
    "Ot chuong (Bell pepper)": ["ớt chuông", "ớt ngọt"],
    "Pho (Vietnamese noodle soup)": ["phở bò", "phở tái", "phở"],
    "Pho mai (Cheese)": ["phô mai", "pho mai"],
    "Rau (Vegetables)": ["rau muống", "rau cải", "rau"],
    "Salad (Salad)": ["salad", "xà lách"],
    "Sup cua (Crab soup)": ["súp cua", "súp"],
    "Thit bo (Beef)": ["thịt bò"],
    "Thit ga (Chicken)": ["thịt gà"],
    "Thit heo (Pork)": ["thịt lợn", "thịt heo"],
    "Thit kho (Braised pork)": ["thịt kho tàu", "thịt kho"],
    "Thit nuong (Grilled meat)": ["thịt nướng", "thịt lợn nướng"],
    "Tom (Shrimp)": ["tôm"],
    "Trung (Egg)": ["trứng gà", "trứng"],
    "Xoi (Sticky rice)": ["xôi nếp", "xôi"],
    "Banh beo (Vietnamese savory steamed rice cake)": ["bánh bèo"],
    "Ca (Fish)": ["cá chép", "cá quả", "cá"],
    "Ca chua (Tomato)": ["cà chua"],
    "Ca phao (Pickled eggplant)": ["cà pháo"],
    "Ca rot (Carrot)": ["cà rốt"],
    "Bong cai (Cauliflower)": ["súp lơ", "bông cải"],
    "Con nguoi (Human)": []
}

results = []
for idx, c in enumerate(class_names):
    name = c["name"]
    terms = CLASS_SYNONYMS.get(name, [])
    
    matches = []
    for term in terms:
        t_l = term.lower()
        for r in canonical:
            n_vi = r["name_vi"].lower()
            if t_l == n_vi or n_vi.startswith(t_l + " ") or (" " + t_l + " ") in (" " + n_vi + " "):
                matches.append(r)
                
    # Deduplicate matches
    seen = set()
    unique = []
    for m in matches:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)
            
    results.append({
        "class_id": idx,
        "name": name,
        "matches_count": len(unique),
        "best": unique[0]["name_vi"] if unique else "NONE"
    })

print(f"{'ID':2} | {'Class Name':45} | {'Count':5} | {'Best Match'}")
print("-" * 85)
for r in results:
    print(f"{r['class_id']:02d} | {r['name']:45} | {r['matches_count']:5} | {r['best']}")
