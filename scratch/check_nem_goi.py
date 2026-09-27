import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("data/nin_nutrition_canonical.json", "r", encoding="utf-8") as f:
    canonical = json.load(f)["records"]

for term in ['gỏi', 'nem', 'bánh khọt', 'ớt chuông', 'ớt ngọt']:
    matches = [r for r in canonical if term in r['name_vi'].lower()]
    print(f"\n{term}: {len(matches)} matches")
    for m in matches[:3]:
        print(f"   [{m['origin_type']}] {m['id']} - {m['name_vi']}")
