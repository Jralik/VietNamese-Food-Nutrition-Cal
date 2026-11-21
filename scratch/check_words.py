import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("data/nin_nutrition_canonical.json", "r", encoding="utf-8") as f:
    canonical = json.load(f)["records"]

search_words = ["lá lốt", "dưa cải", "nem cuốn", "ớt", "thịt nướng", "xiên", "khoai tây", "bò kho"]

for word in search_words:
    print(f"\nSearch for: '{word}'")
    matches = [r for r in canonical if word in r["name_vi"].lower()]
    for m in matches[:3]:
        print(f"  [{m['origin_type']}] {m['id']} - {m['name_vi']}")
