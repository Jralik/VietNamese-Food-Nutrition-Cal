"""
Density and Nutrition data access for the VietFood68 classes.

DATA PROVENANCE (single source of truth = class_names.py):
  - Per-100g nutrition ALWAYS comes from class_names.py (nutrition_per_100g,
    or derived from the reference serving when per-100g is absent).
  - This module only adds category-level fallbacks (keyed by dish type from
    pipeline_config.CLASS_TO_DISH_TYPE) so a lookup never silently invents
    numbers: every value carries a source label.

Density entries are (mean_density_g_per_cm3, std_density_g_per_cm3, source):
  - "usda_fdc"        — USDA FoodData Central, matching commodity
  - "vn_nin_2017"     — Bảng thành phần thực phẩm Việt Nam (Viện Dinh dưỡng, 2017)
  - "literature"      — food-science literature values for the dish category
  - "category_estimate" — engineering estimate for that dish type

Every lookup function has a *_with_source variant returning
(value, source_label); the volume pipeline uses these to flag estimates
built on fallback data instead of class-specific values.
"""

import logging
import re
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ─── Density Database (g/cm³) ──────────────────────────────────────────────
# Effective bulk density of the food as plated/served (what the volume
# pipeline measures: mask × depth), not the material density of ingredients.
DENSITY_DB: Dict[str, Tuple[float, float, str]] = {
    # Noodle soups (broth + noodles + toppings, liquid-dominated)
    "Banh canh (Vietnamese thick noodle soup)":          (0.95, 0.08, "literature"),
    "Bo kho (Beef stew)":                                (0.98, 0.10, "literature"),
    "Bun bo Hue (Hue beef noodle soup)":                 (0.93, 0.08, "literature"),
    "Bun mam (Fermented fish noodle soup)":              (0.94, 0.09, "literature"),
    "Bun rieu (Crab noodle soup)":                       (0.93, 0.08, "literature"),
    "Bun cha ca (Fish cake noodle soup)":                (0.93, 0.08, "literature"),
    "Canh (Soup)":                                       (1.00, 0.05, "literature"),
    "Chao long (Pork organ congee)":                     (1.02, 0.06, "literature"),
    "Hu tieu (Clear rice noodle soup)":                  (0.94, 0.08, "literature"),
    "Kho qua thit (Stuffed bitter melon soup)":          (0.96, 0.08, "literature"),
    "Lau (Hotpot)":                                      (0.98, 0.08, "literature"),
    "Pho (Vietnamese noodle soup)":                      (0.95, 0.08, "literature"),
    "Sup cua (Crab soup)":                               (1.02, 0.05, "literature"),

    # Dry noodles / drained pasta dishes
    "Bun (Rice vermicelli)":                             (0.68, 0.08, "literature"),
    "Bun cha (Grilled pork with vermicelli)":            (0.75, 0.10, "literature"),
    "Bun dau (Vermicelli with tofu)":                    (0.70, 0.10, "literature"),
    "Cao lau (Cao lau noodles)":                         (0.85, 0.10, "literature"),
    "Mi (Egg noodles)":                                  (0.72, 0.08, "literature"),
    "Mi Quang (Quang-style noodles)":                    (0.88, 0.10, "literature"),
    "Nui xao bo (Stir-fried macaroni with beef)":        (0.80, 0.10, "literature"),

    # Rice & sticky rice (packed cooked grain)
    "Com (Rice)":                                        (0.85, 0.08, "literature"),
    "Com tam (Broken rice)":                             (0.85, 0.08, "literature"),
    "Com chien duong chau (Yangzhou fried rice)":        (0.86, 0.08, "literature"),
    "Com chien ga (Fried rice with chicken)":            (0.86, 0.08, "literature"),
    "Xoi (Sticky rice)":                                 (1.10, 0.08, "literature"),
    "Banh chung (Square sticky rice cake)":              (1.15, 0.10, "literature"),

    # Cakes, rolls & pancakes
    "Banh beo (Vietnamese savory steamed rice cake)":    (0.92, 0.08, "literature"),
    "Banh cuon (Rolled rice pancake)":                   (0.90, 0.08, "literature"),
    "Banh khot (Mini savory pancakes)":                  (0.75, 0.10, "literature"),
    # 0.45 = measured ground truth (225 g / ~500 cm3 loaf, user-measured)
    "Banh mi (Vietnamese baguette sandwich)":            (0.45, 0.05, "user_measured_gt"),
    "Banh trang (Rice paper)":                           (0.40, 0.05, "literature"),
    "Banh trang tron (Rice paper salad)":                (0.55, 0.10, "literature"),
    "Banh xeo (Vietnamese sizzling pancake)":            (0.65, 0.10, "literature"),
    "Cha gio (Spring rolls)":                            (0.70, 0.10, "literature"),
    "Goi cuon (Fresh spring rolls)":                     (0.80, 0.10, "literature"),
    "Hamburger":                                         (0.45, 0.08, "literature"),

    # Meat & seafood
    "Bo la lot (Grilled beef wrapped in betel leaves)":  (0.90, 0.10, "literature"),
    "Ca (Fish)":                                         (0.95, 0.10, "usda_fdc"),
    "Cha (Vietnamese pork roll)":                        (1.05, 0.08, "literature"),
    "Cua (Crab)":                                        (0.85, 0.12, "usda_fdc"),
    "Heo quay (Roast pork)":                             (0.90, 0.12, "literature"),
    "Long heo (Pork offal)":                             (0.95, 0.10, "literature"),
    "Muc (Squid)":                                       (0.90, 0.10, "usda_fdc"),
    "Oc (Snails)":                                       (0.85, 0.12, "literature"),
    "Thit bo (Beef)":                                    (1.02, 0.10, "literature"),
    "Thit ga (Chicken)":                                 (0.95, 0.10, "usda_fdc"),
    "Thit heo (Pork)":                                   (0.98, 0.10, "literature"),
    "Thit kho (Braised pork)":                           (0.98, 0.12, "literature"),
    "Thit nuong (Grilled meat)":                         (0.90, 0.10, "literature"),
    "Tom (Shrimp)":                                      (0.88, 0.10, "usda_fdc"),
    "Trung (Egg)":                                       (0.92, 0.08, "usda_fdc"),

    # Vegetables & salads (loose-mound bulk density unless noted solid)
    "Bong cai (Cauliflower)":                            (0.50, 0.08, "literature"),
    "Ca phao (Pickled eggplant)":                        (0.75, 0.08, "literature"),
    "Ca rot (Carrot)":                                   (0.65, 0.08, "literature"),
    "Cu kieu (Pickled scallion head)":                   (0.70, 0.08, "literature"),
    "Dau hu (Tofu)":                                     (0.90, 0.08, "usda_fdc"),
    "Dua chua (Pickled vegetables)":                     (0.60, 0.08, "literature"),
    "Khoai tay chien (French fries)":                    (0.40, 0.08, "literature"),
    "Nam (Mushroom)":                                    (0.55, 0.08, "literature"),
    "Nom hoa chuoi (Banana blossom salad)":              (0.55, 0.08, "literature"),
    "Ot chuong (Bell pepper)":                           (0.50, 0.08, "literature"),
    "Pho mai (Cheese)":                                  (1.05, 0.05, "usda_fdc"),
    "Rau (Vegetables)":                                  (0.30, 0.06, "literature"),
    "Salad (Salad)":                                     (0.30, 0.06, "literature"),
    # Solid fruits/vegetables: near-water material density (spheroid/slice
    # models estimate the solid piece volume, not a loose mound)
    "Ca chua (Tomato)":                                  (0.90, 0.07, "literature"),
    "Chanh (Lime)":                                      (0.85, 0.07, "literature"),
    "Dua leo (Cucumber)":                                (0.80, 0.10, "literature"),
}


# ─── Category fallbacks (keyed by pipeline_config dish type) ───────────────
# Used ONLY when a class has no specific entry. Source labels on these are
# "category_estimate" so callers can flag lower data confidence.
CATEGORY_DENSITY: Dict[str, Tuple[float, float]] = {
    "soup_bowl":         (0.98, 0.08),   # broth-dominated
    "large_bowl":        (0.98, 0.08),
    "rice_bowl":         (1.00, 0.10),   # packed cooked rice
    "flat_plate":        (0.75, 0.12),   # mixed plated dishes
    "small_plate":       (0.80, 0.12),
    "side_vegetables":   (0.35, 0.08),   # loose leafy mound
    "ingredient_small":  (0.80, 0.12),   # solid pieces
    "default":           (0.85, 0.15),   # last-resort generic
}

# Category per-100g reference (kcal, g) — typical Vietnamese preparations of
# that dish type (VN National Institute of Nutrition 2017 ranges, midpoints).
CATEGORY_NUTRITION_PER_100G: Dict[str, Dict[str, float]] = {
    "soup_bowl":         {"Calories": 85,  "Protein": 5.5, "Fat": 3.2, "Carbs": 10.0, "Saturates": 0.9, "Sugar": 0.8, "Salt": 0.9},
    "large_bowl":        {"Calories": 90,  "Protein": 6.5, "Fat": 4.0, "Carbs": 7.0,  "Saturates": 1.2, "Sugar": 1.0, "Salt": 1.0},
    "rice_bowl":         {"Calories": 160, "Protein": 3.5, "Fat": 2.5, "Carbs": 30.0, "Saturates": 0.6, "Sugar": 0.5, "Salt": 0.3},
    "flat_plate":        {"Calories": 200, "Protein": 9.0, "Fat": 9.0, "Carbs": 22.0, "Saturates": 2.8, "Sugar": 2.0, "Salt": 0.8},
    "small_plate":       {"Calories": 180, "Protein": 10.0,"Fat": 8.0, "Carbs": 15.0, "Saturates": 2.5, "Sugar": 1.5, "Salt": 0.9},
    "side_vegetables":   {"Calories": 30,  "Protein": 2.0, "Fat": 0.5, "Carbs": 5.0,  "Saturates": 0.1, "Sugar": 2.0, "Salt": 0.1},
    "ingredient_small":  {"Calories": 55,  "Protein": 1.5, "Fat": 0.8, "Carbs": 10.0, "Saturates": 0.1, "Sugar": 3.0, "Salt": 0.2},
    "default":           {"Calories": 150, "Protein": 5.0, "Fat": 5.0, "Carbs": 20.0, "Saturates": 1.0, "Sugar": 1.0, "Salt": 0.5},
}

GENERIC_DENSITY = CATEGORY_DENSITY["default"]


def _get_dish_type(class_name: str) -> Optional[str]:
    """Map a class to its dish type via pipeline_config (lazy, optional)."""
    try:
        from pipeline_config import CLASS_TO_DISH_TYPE
        matched = _find_best_match(class_name, CLASS_TO_DISH_TYPE)
        if matched is not None:
            return CLASS_TO_DISH_TYPE[matched]
    except Exception as e:
        logger.debug(f"pipeline_config unavailable for dish-type lookup: {e}")
    return None


def _normalize_name(name: str) -> str:
    """Normalize food name for robust matching (ignore case, parenthetical details)."""
    # Extract prefix before '('
    prefix = name.split("(")[0].strip().lower()
    # Remove special chars
    clean = re.sub(r'[^a-z0-9]', '', prefix)
    return clean


def _find_best_match(query: str, db: dict) -> Optional[str]:
    """Find exact or fuzzy matched key in dictionary."""
    if query in db:
        return query

    norm_query = _normalize_name(query)
    for key in db.keys():
        if _normalize_name(key) == norm_query:
            return key

    # Substring / startswith match
    for key in db.keys():
        norm_key = _normalize_name(key)
        if norm_query in norm_key or norm_key in norm_query:
            return key

    return None


def get_density_with_source(class_name: str) -> Tuple[float, float, str]:
    """Return (mean_density, std_density, source) for a food class.

    Source is one of: "density_db", "category_estimate", "generic_estimate".
    """
    matched_key = _find_best_match(class_name, DENSITY_DB)
    if matched_key is not None:
        mean, std, _ = DENSITY_DB[matched_key]
        return mean, std, "density_db"

    dish_type = _get_dish_type(class_name)
    if dish_type is not None and dish_type in CATEGORY_DENSITY:
        mean, std = CATEGORY_DENSITY[dish_type]
        return mean, std, "category_estimate"

    mean, std = GENERIC_DENSITY
    return mean, std, "generic_estimate"


def get_density(class_name: str) -> Tuple[float, float]:
    """Return (mean_density_g_per_cm3, std_density_g_per_cm3) for a food class."""
    mean, std, _ = get_density_with_source(class_name)
    return mean, std


def _lookup_class_names(class_name: str) -> Optional[dict]:
    """Fetch the class_names.py entry for a class, or None."""
    try:
        from class_names import class_names
        norm_query = _normalize_name(class_name)
        for item in class_names:
            name = item.get("name", "")
            if name == class_name or _normalize_name(name) == norm_query:
                return item
    except Exception as e:
        logger.warning(f"Failed to read from class_names.py: {e}")
    return None


def get_nutrition_per_100g_with_source(class_name: str) -> Tuple[Dict[str, float], str]:
    """Return (per-100g nutrition dict, source label).

    Fallback chain (most specific first):
      1. class_names.py nutrition_per_100g          → "class_names:<source>"
         where <source> is the class's provenance label from
         class_names.NUTRITION_SOURCES (usda_fdc | vn_nin_2017 | literature).
      2. class_names.py reference serving scaled    → "class_names:serving_derived"
      3. dish-type category table                   → "category_fallback"
      4. generic mixed-dish values                  → "generic_fallback"
    """
    entry = _lookup_class_names(class_name)
    if entry is not None:
        per100 = entry.get("nutrition_per_100g")
        if per100 and per100.get("Calories", 0) > 0:
            provenance = None
            try:
                from class_names import NUTRITION_SOURCES
                hit = NUTRITION_SOURCES.get(entry.get("name", class_name))
                if hit is not None and hit[0] != "not_applicable":
                    provenance = hit[0]
            except Exception as e:
                logger.debug(f"NUTRITION_SOURCES lookup failed: {e}")
            label = f"class_names:{provenance}" if provenance else "class_names:per_100g"
            return dict(per100), label
        serving = entry.get("nutrition")
        if serving and serving.get("Calories", 0) > 0:
            serving_size = entry.get("serving_size_g", 150)
            scale = 100.0 / max(1.0, float(serving_size))
            return ({k: round(v * scale, 1) for k, v in serving.items()},
                    "class_names:serving_derived")

    dish_type = _get_dish_type(class_name)
    if dish_type is not None and dish_type in CATEGORY_NUTRITION_PER_100G:
        return dict(CATEGORY_NUTRITION_PER_100G[dish_type]), "category_fallback"

    return dict(CATEGORY_NUTRITION_PER_100G["default"]), "generic_fallback"


def get_nutrition_per_100g(class_name: str) -> Dict[str, float]:
    """Return nutrition dict per 100g for a food class (from class_names.py SSOT)."""
    per100, _ = get_nutrition_per_100g_with_source(class_name)
    return per100


def volume_to_mass(volume_cm3: float, class_name: str) -> Tuple[float, float]:
    """Convert volume (cm³) to mass (g) using density DB."""
    density_mean, density_std = get_density(class_name)
    mass_g = volume_cm3 * density_mean
    mass_std_g = volume_cm3 * density_std
    return mass_g, mass_std_g


def estimate_nutrition(mass_g: float, class_name: str) -> Dict[str, float]:
    """Scale per-100g nutrition values to the estimated mass."""
    per_100g = get_nutrition_per_100g(class_name)
    scale = mass_g / 100.0
    return {k: round(v * scale, 1) for k, v in per_100g.items()}
