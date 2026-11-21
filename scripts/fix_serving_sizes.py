"""
scripts/fix_serving_sizes.py
============================
One-off serving-normalization correction (2026-09-28, user mandate: fix wrong
data from authoritative sources, keep the rest untouched).

Problem
-------
canonicalize_nin.py assigned NIN *dish* records a serving size from a category
reference table. For several dishes the assumed category serving is far from
the portion the NIN record actually describes, so per-100g values (and every
micronutrient projected from them) are inflated or diluted:

  Bánh cuốn thịt   440 kcal per plate ÷ 100 g ("Khẩu phần bánh/kẹo") = 440/100g
  Lẩu thập cẩm   2605 kcal per POT   ÷ 450 g ("Tô bún/phở")         = 579/100g
  Phô mai tam giác  58 kcal per triangle ÷ 100 g                     =  58/100g
  Súp ngô cua      145 kcal per cup    ÷ 450 g ("Tô bún/phở")        =  32/100g

Fix
---
For each listed class, re-derive the true portion:

    implied_serving_g = NIN_orig_serving_energy / anchor_energy_per_100g × 100

Anchor priority: (1) TDGD comparable same-dish row (independent authoritative
table, thucdongiadinh.vn), (2) the class's curated per-100g value (already
audit-verified). Then per-100g = orig_serving / implied_serving × 100 for ALL
16 nutrients — macros and micronutrients alike. `nutrition_original_serving`
is never touched, so the fix is idempotent.

Identity repoint (same evidence class as the canistel-egg fix):
  Bo kho: "Bánh mỳ sốt vang" → "Thịt bò kho" (VPF-000053).

Left untouched (no authoritative row exists — documented in the report):
  Mi (only dry-noodle row; class is cooked-basis),
  Goi cuon (no NIN/TDGD gỏi cuốn row; component-based fallback kept).

Usage: python scripts/fix_serving_sizes.py [--apply]   (default: dry run)
"""

import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.insert(0, ROOT_DIR)

# class → (anchor preference for implied serving)
#   "tdgd"      : use the TDGD comparable row per-100g energy
#   "class"     : use the curated class per-100g energy
# Anchor is auto-downgraded to "class" when no comparable TDGD row exists.
FIXES = {
    "Banh cuon (Rolled rice pancake)":                "tdgd",
    "Banh trang tron (Rice paper salad)":             "class",
    "Bo kho (Beef stew)":                             "class",   # + identity repoint
    "Cha gio (Spring rolls)":                         "tdgd",
    "Lau (Hotpot)":                                   "class",
    "Long heo (Pork offal)":                          "tdgd",
    "Pho mai (Cheese)":                               "class",
    "Salad (Salad)":                                  "class",
    "Banh beo (Vietnamese savory steamed rice cake)": "tdgd",
    "Com chien ga (Fried rice with chicken)":         "class",
    "Nom hoa chuoi (Banana blossom salad)":           "class",
    "Sup cua (Crab soup)":                            "class",
}

IDENTITY_REPOINT = {
    "Bo kho (Beef stew)": {
        "nin_food_id": "nin_dish_693a6630362b4f090c0e4c42",
        "nin_food_name": "Thịt bò kho",
        "note": "Bánh mỳ sốt vang là món khác (món bánh mì); bản ghi đúng cho bò kho là 'Thịt bò kho' (VPF-000053).",
    },
}

SERVING_BOUNDS = (15.0, 3500.0)   # g — 1 cheese triangle (16.7 g) up to a whole lẩu pot

APPLY = "--apply" in sys.argv


def load(name):
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    from crosscheck_sources import crosscheck_tdgd, match_score, name_tokens
    from class_names import class_names as calc_by

    canon = load("nin_nutrition_canonical.json")
    canon_by_id = {r["id"]: r for r in canon["records"]}
    mapping_doc = load("vietfood68_nin_mapping.json")
    mappings = {m["full_name"]: m for m in mapping_doc["mappings"]}
    tdgd = load("tdgd_food_composition.json")["data"]
    calc_by = {it["name"]: it for it in calc_by}

    # TDGD comparable rows per class (reuse the audit matcher)
    tdgd_results = crosscheck_tdgd(list(mappings.values()), tdgd)
    tdgd_row_by_class = {r["class"]: r for r in tdgd_results}

    report = []
    for cls, anchor_pref in FIXES.items():
        m = mappings.get(cls)
        if not m:
            report.append((cls, "SKIP", "no mapping"))
            continue
        pm = m.get("primary_match") or {}
        nin_id = pm.get("nin_food_id")

        # identity repoint first
        if cls in IDENTITY_REPOINT and pm.get("nin_food_name") != IDENTITY_REPOINT[cls]["nin_food_name"]:
            rep = IDENTITY_REPOINT[cls]
            new_id = rep["nin_food_id"]
            new_rec = canon_by_id.get(new_id)
            if new_rec is None:
                # full id may differ from the short form used here
                matches = [r for r in canon["records"] if r["id"].endswith(new_id.split("_")[-1])]
                new_rec = matches[0] if matches else None
                if new_rec:
                    new_id = new_rec["id"]
            if new_rec is None:
                report.append((cls, "SKIP", f"repoint target {new_id} not in canonical"))
                continue
            nin_id = new_id
            pm_new = {
                "nin_food_id": new_id,
                "nin_food_name": new_rec["name_vi"],
                "source_type": "dish",
                "canonical_basis": "per_100g",
                "energy_kcal": (new_rec.get("nutrition_original_serving") or new_rec["nutrition_per_100g"]).get("energy_kcal"),
            }
            m["primary_match"] = pm_new
            m["nin_matches"] = [dict(pm_new, match_reason=rep["note"])]
            m["decision_rationale"] = rep["note"]
            pm = pm_new
            report.append((cls, "REPOINT", f"→ {new_rec['name_vi']}"))

        rec = canon_by_id.get(nin_id)
        if rec is None:
            report.append((cls, "SKIP", f"canonical record {nin_id} missing"))
            continue
        orig = rec.get("nutrition_original_serving") or {}
        orig_e = orig.get("energy_kcal")
        if not orig_e:
            report.append((cls, "SKIP", "no original serving energy"))
            continue

        # resolve anchor energy per-100g
        anchor_src = None
        anchor_e = None
        t = tdgd_row_by_class.get(cls)
        if anchor_pref == "tdgd" and t and t.get("comparable") and t.get("tdgd"):
            te = t["tdgd"].get("energy")
            if te:
                anchor_e, anchor_src = float(te), f"TDGD: {t['tdgd_match']}"
        if anchor_e is None:
            ck = calc_by.get(cls, {}).get("nutrition_per_100g", {}).get("Calories")
            if not ck:
                report.append((cls, "SKIP", "no anchor available"))
                continue
            anchor_e, anchor_src = float(ck), f"curated class per-100g ({ck} kcal)"

        implied = float(orig_e) / anchor_e * 100.0
        if not (SERVING_BOUNDS[0] <= implied <= SERVING_BOUNDS[1]):
            report.append((cls, "SKIP", f"implied serving {implied:.0f} g outside bounds"))
            continue
        implied = round(implied, 1)

        old_per100 = rec["nutrition_per_100g"]
        new_per100 = {}
        for k, v in orig.items():
            try:
                new_per100[k] = round(float(v) / implied * 100.0, 2)
            except (TypeError, ValueError):
                new_per100[k] = v
        new_per100["energy_kcal"] = round(float(orig_e) / implied * 100.0, 2)

        scale = old_per100.get("energy_kcal")
        note = (f"Serving-normalization correction 2026-09-28: khẩu phần thật {implied} g "
                f"(suy ra từ năng lượng khẩu phần gốc {orig_e} kcal / mốc per-100g "
                f"{anchor_e:.0f} kcal — {anchor_src}); thay cho cận loại Kansas "
                f"'{rec.get('serving_size_source')}' = {rec.get('serving_size_g')} g.")

        if APPLY:
            rec["serving_size_g"] = implied
            rec["serving_size_source"] = f"Serving implied from authoritative anchor ({anchor_src}) — fix_serving_sizes.py 2026-09-28"
            rec["nutrition_per_100g"] = new_per100
            # Records that Phase 1 left unverified (serving=None) now have a
            # documented, anchor-derived portion — promote their basis so the
            # Phase 2 per-100g derivation invariant sees a verified record.
            if rec.get("canonical_basis") != "per_100g" or rec.get("mapping_status") != "verified":
                rec["canonical_basis"] = "per_100g"
                rec["mapping_status"] = "verified"
            m["primary_nutrition_per_100g"] = dict(new_per100)
            m["serving_size_g"] = implied
            m["serving_size_source"] = rec["serving_size_source"]
            m["primary_match"]["canonical_basis"] = "per_100g"
            m["decision_rationale"] = (m.get("decision_rationale") or "") + " | " + note
        report.append((cls, "FIX" if APPLY else "DRY",
                       f"serving {rec.get('serving_size_g')} → {implied} g | "
                       f"per-100g kcal {old_per100.get('energy_kcal')} → {new_per100['energy_kcal']} "
                       f"(anchor: {anchor_src})"))

    print(f"{'class':46s} {'action':7s} detail")
    print("-" * 130)
    for cls, action, detail in report:
        print(f"{cls:46s} {action:7s} {detail}")

    if APPLY:
        with open(os.path.join(DATA_DIR, "nin_nutrition_canonical.json"), "w", encoding="utf-8") as f:
            json.dump(canon, f, ensure_ascii=False, indent=2)
        with open(os.path.join(DATA_DIR, "vietfood68_nin_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(mapping_doc, f, ensure_ascii=False, indent=2)
        print("\nAPPLIED: canonical + mapping updated. Now re-run scripts/update_class_names_nin.py")


if __name__ == "__main__":
    main()
