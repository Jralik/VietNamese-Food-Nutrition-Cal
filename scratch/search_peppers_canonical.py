import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    canon = json.load(f)["records"]

print(f"Total canonical records: {len(canon)}")

found = []
for r in canon:
    name = r["name_vi"].lower()
    if "ớt" in name or "ot" in r.get("name_vi_ascii", "").lower():
        found.append(r)

print(f"Found {len(found)} records with 'ớt':")
for r in found:
    print(f"  ID: {r['id']} | Origin: {r['origin_type']} | Name: {r['name_vi']}")

# Also search for 'ngọt', 'chuông', 'đà lạt'
print("\nSearching for 'ngọt', 'chuông', 'đà lạt':")
for r in canon:
    name = r["name_vi"].lower()
    if any(k in name for k in ["chuông", "đà lạt"]):
        print(f"  ID: {r['id']} | Origin: {r['origin_type']} | Name: {r['name_vi']}")
