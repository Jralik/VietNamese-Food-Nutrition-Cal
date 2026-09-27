import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    records = json.load(f)['records']

for r in records:
    if r['origin_type'] == 'raw_ingredient' and any(w in r['name_vi'].lower() for w in ['cam', 'chanh', 'bưởi', 'quất', 'chanh leo']):
        print(f"  {r['id']}: {r['name_vi']}")
