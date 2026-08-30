"""
Nutrition data validator.

Audits the data sources behind nutrition estimation:
  - class_names.py (single source of truth: per-100g + reference serving)
  - density_db.py  (densities + category fallbacks)
  - pipeline_config.py (CLASS_TO_DISH_TYPE mapping used by fallbacks)

Checks:
  1. Coverage — every class has per-100g nutrition, a serving size, a density,
     and a dish-type mapping (required for category fallbacks).
  2. Energy sanity — Calories vs Atwater estimate (4/4/9 kcal per P/C/F g).
  3. Serving consistency — serving nutrition equals per-100g scaled to serving_size_g.
  4. Plausibility — nutrient ranges within physiological bounds.
  5. Provenance — reports which classes resolve to fallback data instead of
     class-specific values.

Usage:
    python validate_nutrition_data.py            # human-readable report
    python validate_nutrition_data.py --strict   # non-zero exit on any issue
"""

import sys
from class_names import class_names
import density_db

ATWATER_KCAL_PER_G = {"Protein": 4.0, "Carbs": 4.0, "Fat": 9.0}
NON_FOOD_CLASSES = {"Con nguoi (Human)"}
# Source labels that indicate class-specific (high-trust) data
TRUSTED_DENSITY_SOURCES = {"density_db"}
TRUSTED_NUTRITION_SOURCES = {"class_names:per_100g", "class_names:serving_derived"}


def atwater_calories(per100: dict) -> float:
    return sum(per100.get(k, 0.0) * f for k, f in ATWATER_KCAL_PER_G.items())


def audit():
    issues, warnings = [], []
    names = [c["name"] for c in class_names]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        issues.append(f"Duplicate class names in class_names.py: {sorted(dupes)}")

    provenance = []
    for cls in class_names:
        name = cls["name"]
        if name in NON_FOOD_CLASSES:
            continue
        per100 = cls.get("nutrition_per_100g", {})
        serving = cls.get("nutrition", {})
        serving_g = cls.get("serving_size_g")

        # 1. Coverage
        if not per100 or per100.get("Calories", 0) <= 0:
            issues.append(f"[coverage] {name}: missing/zero per-100g Calories")
        if not serving_g or serving_g <= 0:
            issues.append(f"[coverage] {name}: missing/zero serving_size_g")

        # 2. Energy sanity (per 100g). Low-kcal items (<60) are high in water
        #    and fiber — the 4/4/9 rule overestimates there, so widen the band.
        kcal = per100.get("Calories", 0.0)
        atwater = atwater_calories(per100)
        if kcal > 0 and atwater > 0:
            ratio = kcal / atwater
            lo, hi = (0.55, 1.45) if kcal < 60 else (0.7, 1.3)
            if not (lo <= ratio <= hi):
                issues.append(
                    f"[energy] {name}: Calories={kcal:.0f} but macros imply "
                    f"{atwater:.0f} kcal (ratio {ratio:.2f})"
                )
            elif kcal >= 60 and not (0.85 <= ratio <= 1.15):
                warnings.append(
                    f"[energy~] {name}: Calories={kcal:.0f} vs macro-derived "
                    f"{atwater:.0f} kcal (ratio {ratio:.2f})"
                )

        # 3. Serving consistency
        if serving and serving_g and per100.get("Calories", 0) > 0:
            expected = per100["Calories"] * serving_g / 100.0
            got = serving.get("Calories", 0.0)
            if got > 0 and abs(expected - got) / expected > 0.05:
                issues.append(
                    f"[serving] {name}: serving Calories={got:.1f} but "
                    f"per-100g x {serving_g}g = {expected:.1f}"
                )

        # 4. Plausibility (per 100 g)
        bounds = {
            "Calories": (0, 950), "Protein": (0, 90), "Fat": (0, 100),
            "Carbs": (0, 100), "Saturates": (0, 60), "Sugar": (0, 100),
            "Salt": (0, 8),
        }
        for k, (lo, hi) in bounds.items():
            v = per100.get(k)
            if v is not None and not (lo <= v <= hi):
                issues.append(f"[range] {name}: {k}={v} outside ({lo}, {hi})")

        # 5. Provenance of the values the volume pipeline would actually use
        _, _, density_src = density_db.get_density_with_source(name)
        _, nutrition_src = density_db.get_nutrition_per_100g_with_source(name)
        if density_src not in TRUSTED_DENSITY_SOURCES:
            warnings.append(f"[density-fallback] {name}: {density_src}")
        if nutrition_src not in TRUSTED_NUTRITION_SOURCES:
            warnings.append(f"[nutrition-fallback] {name}: {nutrition_src}")
        provenance.append((name, density_src, nutrition_src))

        # Dish-type mapping (drives the category fallbacks)
        from pipeline_config import CLASS_TO_DISH_TYPE
        if density_db._find_best_match(name, CLASS_TO_DISH_TYPE) is None:
            issues.append(f"[dish-type] {name}: no CLASS_TO_DISH_TYPE mapping")

    return issues, warnings, provenance


def main():
    strict = "--strict" in sys.argv
    issues, warnings, provenance = audit()

    print("=" * 70)
    print("  NUTRITION DATA VALIDATION REPORT")
    print("=" * 70)
    n_food = len([c for c in class_names if c["name"] not in NON_FOOD_CLASSES])
    print(f"  Classes in class_names.py: {len(class_names)} ({n_food} food)")
    print(f"  Density DB entries:        {len(density_db.DENSITY_DB)}")
    print("-" * 70)

    if issues:
        print(f"\nISSUES ({len(issues)}):")
        for i in issues:
            print(f"  ✗ {i}")
    else:
        print("\nNo blocking issues found.")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("\nNo warnings.")

    n_class_density = sum(1 for _, d, _ in provenance if d == "density_db")
    n_class_nutrition = sum(
        1 for _, _, n in provenance if n.startswith("class_names")
    )
    print("-" * 70)
    print(f"  Provenance: {n_class_density}/{n_food} class-specific densities, "
          f"{n_class_nutrition}/{n_food} class-specific nutrition values")
    print("=" * 70)
    if strict and issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
