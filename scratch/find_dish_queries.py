import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    records = json.load(f)['records']
rec_map = {r['id']: r for r in records}

queries = {
    "Bo kho": ["sốt vang", "bò kho"],
    "Bo la lot": ["lá lốt", "bò cuốn"],
    "Canh": ["canh rau", "rau ngót"],
    "Cha gio": ["nem rán", "chả giò"],
    "Com chien ga": ["rang gà", "cơm gà"],
    "Heo quay": ["thịt quay", "lợn quay"],
    "Kho qua thit": ["mướp đắng", "khổ qua"],
    "Khoai tay chien": ["khoai tây chiên"],
    "Long heo": ["lòng lợn", "lòng luộc"],
    "Thit kho": ["kho tàu", "thịt kho"],
    "Thit nuong": ["xiên nướng", "thịt nướng"],
}

for k, words in queries.items():
    print(f"=== {k} ===")
    found = []
    for r in records:
        for w in words:
            if w in r['name_vi'].lower():
                found.append(r)
                break
    for r in found[:3]:
        print(f"  {r['id']}: {r['name_vi']} ({r['origin_type']}, {r['canonical_basis']})")
