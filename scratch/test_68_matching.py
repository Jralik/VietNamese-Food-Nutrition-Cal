import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from class_names import class_names

with open("data/nin_nutrition_canonical.json", "r", encoding="utf-8") as f:
    canonical = json.load(f)["records"]

print(f"Total canonical records: {len(canonical)}")

# Let's inspect some specific classes that might be tricky:
queries = [
    "Nui xao bo", "Hamburger", "Cao lau", "Mi Quang", "Kho qua thit",
    "Banh beo", "Heo quay", "Bo la lot", "Long heo", "Pho", "Com tam"
]

for q in queries:
    q_lower = q.lower()
    matches = [r for r in canonical if q_lower in r["name_vi"].lower() or q_lower in r["name_vi_ascii"].lower()]
    print(f"\nQuery '{q}': {len(matches)} matches")
    for m in matches[:3]:
        print(f"  [{m['origin_type']}] {m['id']} - {m['name_vi']} ({m['category']})")
