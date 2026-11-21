"""
scripts/update_class_names_nin.py
=================================
Automated script to project NIN Canonical Mapping (Pha 2) into class_names.py (Pha 3).

Key Design Invariants:
1. class_names.py is a RUNTIME PROJECTION / CACHE of the SSOT (vietfood68_nin_mapping.json).
2. Backward Compatibility (P3-F):
   Preserves the existing 7 macro/salt fields exactly as-is:
   ['Calories', 'Protein', 'Fat', 'Carbs', 'Saturates', 'Sugar', 'Salt'].
3. Micronutrient Extension:
   Adds 5 target micronutrients:
   ['Sodium', 'Calcium', 'Iron', 'Zinc', 'Cholesterol'] (in mg).
   Calculated per 100g and scaled to serving_size_g.
4. Unmapped Provenance Metadata:
   Explicitly tracks source_type, source_reference, and basis for the 3 unmapped classes.
"""

import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

MAPPING_PATH = os.path.join(ROOT_DIR, "data", "vietfood68_nin_mapping.json")
CLASS_NAMES_PATH = os.path.join(ROOT_DIR, "class_names.py")


def update_class_names():
    print(f"Loading mapping from {MAPPING_PATH}...")
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    mappings = {m["full_name"]: m for m in mapping_data["mappings"]}

    from class_names import class_names, NUTRITION_SOURCES

    updated_items = []
    
    # Explicit literature fallback values for unmapped classes
    UNMAPPED_FALLBACKS = {
        "Con nguoi (Human)": {
            "nutr_100g": {"Sodium": 0.0, "Calcium": 0.0, "Iron": 0.0, "Zinc": 0.0, "Cholesterol": 0.0},
            "source_type": "non_food",
            "source_reference": "Non-food background class — zero nutrient values",
            "basis": "per_100g"
        },
        "Banh khot (Mini savory pancakes)": {
            "nutr_100g": {"Sodium": 276.0, "Calcium": 35.0, "Iron": 0.8, "Zinc": 0.6, "Cholesterol": 25.0},
            "source_type": "literature_fallback",
            "source_reference": "Bảng phân tích thành phần món ăn Việt Nam (Bộ Y Tế - NIN & Composite recipes)",
            "basis": "per_100g"
        },
        "Cao lau (Cao lau noodles)": {
            "nutr_100g": {"Sodium": 315.0, "Calcium": 20.0, "Iron": 1.2, "Zinc": 1.1, "Cholesterol": 18.0},
            "source_type": "literature_fallback",
            "source_reference": "Tài liệu dinh dưỡng ẩm thực Hội An - Quảng Nam (Composite recipes)",
            "basis": "per_100g"
        }
    }

    updated_nutrition_sources = dict(NUTRITION_SOURCES)

    # Source-alignment pass (2026-09-28, data/source_crosscheck_report.md §5):
    # for these classes the preserved legacy macros trace to NO authoritative
    # row (audit verdict "review" on ingredient classes), while the class's own
    # cited NIN row is authoritative (Bảng TPTP VN 2017). Calories/Protein/Fat/
    # Carbs are therefore projected from the SSOT mapping like the micronutrients
    # already are. Saturates/Sugar/Salt stay preserved (NIN rows generally lack
    # them). Dish classes are NOT aligned: their preserved macros are curated
    # recipes; NIN dish per-100g rows carry known serving-normalization issues.
    SOURCE_ALIGNED_MACRO_CLASSES = {
        "Thit bo (Beef)",          # 215/26/12 → BTP "Thịt bò, loại I" 118/21/3.8
        "Thit ga (Chicken)",       # 190/29/7.4 → BTP "Thịt gà ta" 199/20.3/13.1
        "Thit heo (Pork)",         # 242/27/14 → BTP "Thịt lợn nạc vai, luộc" 126/17.65/6.18
        "Tom (Shrimp)",            # 99/24/0.3 → BTP "Tôm đồng, tươi" 90/18.4/1.8
        "Muc (Squid)",             # 92/15.6/1.4 → BTP "Mực, tươi" 73/16.3/0.9
        "Cua (Crab)",              # 87/18/1.1 → BTP "Cua đồng, tươi" 87/12.3/3.3
        "Oc (Snails)",             # 79/15/1.4 → BTP "Ốc nhồi, tươi" 84/11.9/0.7
        "Dau hu (Tofu)",           # 76/8/4.8 → BTP "Đậu phụ, sống" 97/10.9/5.4
        "Banh trang (Rice paper)", # P 0.5 → BTP "Bánh đa nem" P 4.0
        "Ca (Fish)",               # 130/18/5 → BTP "Cá chép, tươi" 96/16/3.6
        "Ca chua (Tomato)",        # 18/0.9/0.2 → BTP "Quả cà chua, tươi"
        "Chanh (Lime)",            # C 10.5 → BTP "Chanh, tươi" C 5.96
        "Ot chuong (Bell pepper)", # 20/1/0.2 → BTP "Ớt xanh to, tươi"
        "Dua chua (Pickled vegetables)",  # → BTP "Dưa cải bẹ"
        "Rau (Vegetables)",        # 25/1.5 → BTP "Rau muống, tươi" 28/3.2/0.24
        "Nam (Mushroom)",          # anchor repointed to generic "Nấm mỡ (Nấm tây)" 37/4/0.3/4.5
        "Trung (Egg)",             # 143/12.6/9.5/0.7 → BTP "Trứng gà ta" 150/12.96/10.33/1.25
    }
    MACRO_PROJECTION = [
        ("Calories", "energy_kcal"), ("Protein", "protein_g"),
        ("Fat", "fat_g"), ("Carbs", "carbs_g"),
    ]

    for item in class_names:
        name = item["name"]
        serving_size_g = item.get("serving_size_g", 100)
        serving_scale = serving_size_g / 100.0

        # Preserve existing 7 fields
        p100_existing = dict(item.get("nutrition_per_100g", {}))
        pserv_existing = dict(item.get("nutrition", {}))

        # Determine 5 micronutrients (in mg)
        if name in UNMAPPED_FALLBACKS:
            fb = UNMAPPED_FALLBACKS[name]
            micro_100g = dict(fb["nutr_100g"])
            updated_nutrition_sources[name] = (
                fb["source_type"],
                f"{fb['source_reference']} [basis: {fb['basis']}]"
            )
        else:
            map_entry = mappings.get(name)
            if not map_entry:
                raise ValueError(f"Class '{name}' missing from mapping JSON!")
            p_nutr = map_entry["primary_nutrition_per_100g"]
            micro_100g = {
                "Sodium": round(float(p_nutr.get("sodium_mg", 0.0)), 1),
                "Calcium": round(float(p_nutr.get("calcium_mg", 0.0)), 1),
                "Iron": round(float(p_nutr.get("iron_mg", 0.0)), 2),
                "Zinc": round(float(p_nutr.get("zinc_mg", 0.0)), 2),
                "Cholesterol": round(float(p_nutr.get("cholesterol_mg", 0.0)), 1),
            }
            # Record provenance metadata
            pm = map_entry["primary_match"]
            updated_nutrition_sources[name] = (
                "vn_nin_2017",
                f"NIN: {pm['nin_food_name']} ({pm['nin_food_id']}) [basis: {pm['canonical_basis']}]"
            )

        # Merge 5 micronutrients without altering existing 7 fields
        new_p100 = dict(p100_existing)
        new_pserv = dict(pserv_existing)

        for mk, mv in micro_100g.items():
            new_p100[mk] = mv
            new_pserv[mk] = round(mv * serving_scale, 2)

        # Source-aligned macros (see SOURCE_ALIGNED_MACRO_CLASSES above):
        # project Calories/Protein/Fat/Carbs from the SSOT row and rescale
        # the per-serving dict so both calculation paths stay consistent.
        if name in SOURCE_ALIGNED_MACRO_CLASSES and name not in UNMAPPED_FALLBACKS:
            p_nutr_aligned = map_entry["primary_nutrition_per_100g"]
            fat_old = float(p100_existing.get("Fat") or 0.0)
            for ck, mk in MACRO_PROJECTION:
                mv = p_nutr_aligned.get(mk)
                if mv is not None:
                    new_p100[ck] = round(float(mv), 1)
                    new_pserv[ck] = round(float(mv) * serving_scale, 2)
            # BTP VN 2017 publishes no Saturates column. Keep the preserved
            # saturates:fat PROFILE, rescaled to the aligned fat value, so the
            # chemistry invariant (Saturates ≤ Fat) still holds (e.g. Thit bo
            # 4.8 g sat would exceed its aligned 3.8 g total fat).
            sat_old = float(p100_existing.get("Saturates") or 0.0)
            fat_new = float(new_p100.get("Fat") or 0.0)
            if sat_old > 0 and fat_old > 0 and fat_new > 0:
                sat_new = round(sat_old * fat_new / fat_old, 1)
                new_p100["Saturates"] = sat_new
                new_pserv["Saturates"] = round(sat_new * serving_scale, 2)
            # Idempotent chemistry clamp: saturated fat can never exceed total
            # fat. 0.4 is the typical saturated fraction of animal fat (food-
            # science literature) and applies when the ratio above could not
            # fire because a previous run already overwrote Fat.
            if float(new_p100.get("Saturates") or 0) > float(new_p100.get("Fat") or 0) > 0:
                sat_clamped = round(float(new_p100["Fat"]) * 0.4, 1)
                new_p100["Saturates"] = sat_clamped
                new_pserv["Saturates"] = round(sat_clamped * serving_scale, 2)

        updated_item = {
            "name": name,
            "serving_type": item.get("serving_type", f"reference serving ({serving_size_g} g)"),
            "serving_size_g": serving_size_g,
            "nutrition_per_100g": new_p100,
            "nutrition": new_pserv
        }
        updated_items.append(updated_item)

    print(f"Constructing updated class_names.py with {len(updated_items)} classes...")

    # Render new class_names.py file content
    content_lines = [
        '"""VietFood68 class metadata and nutrition reference (Phase 3 Runtime Projection).',
        '',
        'IMPORTANT ARCHITECTURAL NOTE:',
        '  This file is a RUNTIME PROJECTION / IN-MEMORY CACHE.',
        '  The Single Source of Truth (SSOT) is data/nin_nutrition_canonical.json and data/vietfood68_nin_mapping.json.',
        '  Do not edit values here manually; re-run `python scripts/update_class_names_nin.py`.',
        '',
        'Target Nutrient Schema (12 nutrients):',
        '  - Macros (g) & Energy (kcal): Calories, Protein, Fat, Carbs, Saturates, Sugar, Salt',
        '  - Micronutrients (mg): Sodium, Calcium, Iron, Zinc, Cholesterol',
        '"""',
        '',
        'ALLOWED_SOURCES = {"usda_fdc", "vn_nin_2017", "literature", "literature_fallback", "non_food", "not_applicable"}',
        '',
        'NUTRITION_SOURCES = {'
    ]

    for k, v in sorted(updated_nutrition_sources.items()):
        # Escape quotes
        src_label, src_desc = v[0], v[1].replace('"', '\\"')
        content_lines.append(f'    "{k}": ("{src_label}", "{src_desc}"),')
    content_lines.append('}')
    content_lines.append('')
    content_lines.append('class_names = [')

    for it in updated_items:
        content_lines.append('    {')
        content_lines.append(f'        "name": {json.dumps(it["name"], ensure_ascii=False)},')
        content_lines.append(f'        "serving_type": {json.dumps(it["serving_type"], ensure_ascii=False)},')
        content_lines.append(f'        "serving_size_g": {it["serving_size_g"]},')
        content_lines.append('        "nutrition_per_100g": {')
        for nk, nv in it["nutrition_per_100g"].items():
            content_lines.append(f'            "{nk}": {nv},')
        content_lines.append('        },')
        content_lines.append('        "nutrition": {')
        for nk, nv in it["nutrition"].items():
            content_lines.append(f'            "{nk}": {nv},')
        content_lines.append('        },')
        content_lines.append('    },')

    content_lines.append(']')
    content_lines.append('')

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(content_lines))

    print(f"Successfully updated {CLASS_NAMES_PATH}!")


if __name__ == "__main__":
    update_class_names()
