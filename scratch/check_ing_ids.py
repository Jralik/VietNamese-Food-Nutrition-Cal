import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    records = json.load(f)['records']

raw_records = [r for r in records if r['origin_type'] == 'raw_ingredient']

queries = [
    'bánh tráng', 'súp lơ', 'hoa lơ', 'bún', 'cá chép', 'cà chua', 'cà pháo',
    'cà rốt', 'giò lụa', 'chanh', 'cơm tẻ', 'củ kiệu', 'cua', 'đậu phụ',
    'dưa cải', 'dưa chuột', 'mì', 'mực', 'nấm', 'ốc', 'pho mát', 'phô mai',
    'rau muống', 'thịt bò', 'thịt gà', 'thịt lợn nạc', 'tôm', 'trứng gà', 'xôi'
]

for q in queries:
    matches = [r for r in raw_records if q in r['name_vi'].lower()]
    print(f"=== {q} ===")
    for m in matches[:2]:
        print(f"  {m['id']}: {m['name_vi']}")
