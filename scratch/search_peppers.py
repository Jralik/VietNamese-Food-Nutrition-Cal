import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_ingredients.json', 'r', encoding='utf-8') as f:
    raw_ing = json.load(f)

print(f"Total raw ingredients in nin_ingredients.json: {len(raw_ing)}")

found_ing = []
for r in raw_ing:
    name = r.get("TenThucPham", "")
    if "ớt" in name.lower() or "ot" in name.lower():
        found_ing.append(r)

print(f"Found {len(found_ing)} items containing 'ớt' in raw ingredients:")
for r in found_ing:
    print(f"  ID={r.get('ThucPhamId')} | Code={r.get('MaThucPham')} | Name={r.get('TenThucPham')}")

with open('data/nin_dishes.json', 'r', encoding='utf-8') as f:
    dishes = json.load(f)

found_dish = []
for r in dishes:
    name = r.get("TenMonAn", "")
    if "ớt" in name.lower() or "ot" in name.lower():
        found_dish.append(r)

print(f"\nFound {len(found_dish)} items containing 'ớt' in cooked dishes:")
for r in found_dish[:10]:
    print(f"  ID={r.get('MonAnId')} | Name={r.get('TenMonAn')}")
