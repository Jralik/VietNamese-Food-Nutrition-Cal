import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("data/nin_nutrition_canonical.json", "r", encoding="utf-8") as f:
    canonical = json.load(f)["records"]

dialects = [
    ("Kho qua thit", ["mướp đắng", "khổ qua"]),
    ("Heo quay", ["thịt lợn quay", "thịt quay", "heo quay"]),
    ("Long heo", ["lòng lợn", "lòng heo", "dồi"]),
    ("Bo la lot", ["bò cuốn lá lốt", "lá lốt", "chả lá lốt"]),
    ("Nui xao bo", ["nui"]),
    ("Mi Quang", ["mỳ quảng", "mì quảng", "quảng"]),
    ("Com tam", ["cơm tấm", "sườn"]),
    ("Cao lau", ["cao lầu", "cao lau"]),
]

for label, terms in dialects:
    print(f"\n=== Searching for {label} (terms: {terms}) ===")
    found = []
    for t in terms:
        t_l = t.lower()
        matches = [r for r in canonical if t_l in r["name_vi"].lower() or t_l in r["name_vi_ascii"].lower()]
        found.extend(matches)
    # Deduplicate by id
    seen = set()
    unique = []
    for r in found:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    print(f"Found {len(unique)} results:")
    for r in unique[:4]:
        print(f"  [{r['origin_type']}] {r['id']} - {r['name_vi']} ({r['category']})")
