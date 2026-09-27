import sys
import json
import re

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3"
sys.path.insert(0, ROOT_DIR)

from class_names import class_names

with open(r"data\nin_nutrition_canonical.json", "r", encoding="utf-8") as f:
    records = json.load(f)["records"]

rec_map = {r["id"]: r for r in records}

def clean(s):
    if not s: return ""
    return re.sub(r'[\(\)\,\-\.\/]', ' ', s.lower()).strip()

# Check each class
for idx, c in enumerate(class_names):
    name = c["name"]
    vi = name.split("(")[0].strip()
    en = name.split("(")[1].replace(")", "").strip() if "(" in name else ""
    
    vi_clean = clean(vi)
    candidates = []
    for r in records:
        r_vi = clean(r["name_vi"])
        r_ascii = clean(r.get("name_vi_ascii", ""))
        
        score = 0
        if vi_clean == r_vi or vi_clean == r_ascii:
            score = 100
        elif vi_clean in r_vi or vi_clean in r_ascii:
            score = 50
        elif any(w in r_vi for w in vi_clean.split() if len(w) > 2):
            score = 20
            
        if score > 0:
            candidates.append((score, r))
            
    candidates.sort(key=lambda x: (x[0], 1 if x[1]["canonical_basis"] == "per_100g" else 0), reverse=True)
    top = candidates[:2]
    top_str = "; ".join([f"{c[1]['id']}: {c[1]['name_vi']} ({c[1]['canonical_basis']})" for c in top]) if top else "NO MATCH"
    print(f"[{idx:02d}] {name} -> {top_str}")
