import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    records = json.load(f)['records']

dish_queries = [
    'Bánh canh', 'Bánh chưng', 'Bánh cuốn', 'Bánh mỳ pate', 'Bò sốt vang',
    'Bún bò Huế', 'Bún chả', 'Canh rau ngót', 'Nem rán', 'Cơm rang',
    'Khoai tây chiên', 'Phở bò chín', 'Súp ngô cua', 'Thịt lợn kho tàu',
    'Xôi trắng', 'Bánh bèo'
]

for q in dish_queries:
    matches = [r for r in records if q.lower() in r['name_vi'].lower()]
    print(f"=== Query: {q} ===")
    for m in matches[:3]:
        print(f"  ID: {m['id']} | Name: {m['name_vi']} | Basis: {m['canonical_basis']}")
