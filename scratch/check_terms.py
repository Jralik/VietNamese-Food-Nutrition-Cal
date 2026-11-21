import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    records = json.load(f)['records']

for q in ['chanh', 'tráng', 'đa nem', 'bánh phở', 'phô mai', 'pho mát', 'cơm tẻ']:
    print(f"=== {q} ===")
    matches = [r for r in records if q in r['name_vi'].lower()]
    for r in matches[:4]:
        print(f"  {r['id']}: {r['name_vi']} ({r['origin_type']}, {r['canonical_basis']})")
