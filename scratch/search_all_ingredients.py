import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    records = json.load(f)['records']

raw = [r for r in records if r['origin_type'] == 'raw_ingredient']

def find_best(keywords):
    results = []
    for r in raw:
        name = r['name_vi'].lower()
        if all(k in name for k in keywords):
            results.append(r)
    return results

queries = {
    "cà chua": ["cà chua"],
    "bánh tráng": ["tráng"],
    "chanh": ["chanh"],
    "cơm": ["cơm"],
    "gạo": ["gạo"],
    "củ kiệu": ["kiệu"],
    "cua": ["cua", "tươi"],
    "ốc": ["ốc", "tươi"],
    "pho mát": ["pho mát"],
    "phô mai": ["phô mai"],
    "thịt bò tươi": ["thịt bò", "tươi"],
    "thịt gà tươi": ["thịt gà", "tươi"],
    "tôm tươi": ["tôm", "tươi"],
    "mì sợi": ["mỳ"],
}

for label, kws in queries.items():
    print(f"=== {label} ===")
    matches = find_best(kws)
    for m in matches[:3]:
        print(f"  {m['id']}: {m['name_vi']}")
    if not matches:
        # Relax search
        print(f"  (no exact match for {kws}, searching first kw)")
        matches2 = find_best([kws[0]])
        for m in matches2[:3]:
            print(f"    {m['id']}: {m['name_vi']}")
