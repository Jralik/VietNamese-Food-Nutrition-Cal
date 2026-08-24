"""
Density and Nutrition database for all 68 classes of VietFood dataset.

Each entry stores (mean_density_g_per_cm3, std_density_g_per_cm3, n_samples).
Includes exact matching and fuzzy normalized matching so no class is ever missed.

Nutrition per 100g is sourced from:
  1. Bảng thành phần thực phẩm Việt Nam (Viện Dinh dưỡng Quốc gia, 2017)
  2. USDA FoodData Central & class_names.py reference data
"""

import re
from typing import Tuple, Dict, Optional

# ─── Density Database (g/cm³) ──────────────────────────────────────────────
DENSITY_DB: Dict[str, Tuple[float, float, int]] = {
    # Noodles & Soups
    "Banh canh (Vietnamese thick noodle soup)":          (0.95, 0.08, 5),
    "Bun bo Hue (Hue beef noodle soup)":                 (0.93, 0.08, 5),
    "Bun mam (Fermented fish noodle soup)":              (0.94, 0.09, 5),
    "Bun rieu (Crab noodle soup)":                       (0.93, 0.08, 5),
    "Bun cha ca (Fish cake noodle soup)":                (0.93, 0.08, 5),
    "Canh (Soup)":                                       (1.00, 0.05, 5),
    "Chao long (Pork organ congee)":                     (1.02, 0.06, 5),
    "Hu tieu (Clear rice noodle soup)":                  (0.94, 0.08, 5),
    "Kho qua thit (Stuffed bitter melon soup)":          (0.96, 0.08, 5),
    "Lau (Hotpot)":                                      (0.98, 0.08, 5),
    "Pho (Vietnamese noodle soup)":                      (0.95, 0.08, 5),
    "Sup cua (Crab soup)":                               (1.02, 0.05, 5),
    "Bo kho (Beef stew)":                                (0.98, 0.10, 5),

    # Dry Noodles / Pasta
    "Bun (Rice vermicelli)":                             (0.68, 0.08, 5),
    "Bun cha (Grilled pork with vermicelli)":            (0.75, 0.10, 5),
    "Bun dau (Vermicelli with tofu)":                    (0.70, 0.10, 5),
    "Cao lau (Cao lau noodles)":                         (0.85, 0.10, 5),
    "Mi (Egg noodles)":                                  (0.72, 0.08, 5),
    "Mi Quang (Quang-style noodles)":                    (0.88, 0.10, 5),
    "Nui xao bo (Stir-fried macaroni with beef)":        (0.80, 0.10, 5),

    # Rice & Sticky Rice
    "Com (Rice)":                                        (0.85, 0.08, 5),
    "Com tam (Broken rice)":                             (0.85, 0.08, 5),
    "Com chien duong chau (Yangzhou fried rice)":        (0.86, 0.08, 5),
    "Com chien ga (Fried rice with chicken)":            (0.86, 0.08, 5),
    "Xoi (Sticky rice)":                                 (1.10, 0.08, 5),
    "Banh chung (Square sticky rice cake)":              (1.15, 0.10, 5),

    # Cakes, Rolls & Pancakes
    "Banh beo (Vietnamese savory steamed rice cake)":    (0.92, 0.08, 5),
    "Banh cuon (Rolled rice pancake)":                   (0.90, 0.08, 5),
    "Banh khot (Mini savory pancakes)":                  (0.75, 0.10, 5),
    "Banh mi (Vietnamese baguette sandwich)":            (0.35, 0.08, 5),
    "Banh trang (Rice paper)":                           (0.40, 0.05, 5),
    "Banh trang tron (Rice paper salad)":                (0.55, 0.10, 5),
    "Banh xeo (Vietnamese sizzling pancake)":            (0.65, 0.10, 5),
    "Cha gio (Spring rolls)":                            (0.70, 0.10, 5),
    "Goi cuon (Fresh spring rolls)":                     (0.80, 0.10, 5),
    "Hamburger":                                         (0.45, 0.08, 5),

    # Meat & Seafood
    "Bo la lot (Grilled beef wrapped in betel leaves)":  (0.90, 0.10, 5),
    "Ca (Fish)":                                         (0.95, 0.10, 5),
    "Cha (Vietnamese pork roll)":                        (1.05, 0.08, 5),
    "Cua (Crab)":                                        (0.85, 0.12, 5),
    "Heo quay (Roast pork)":                             (0.90, 0.12, 5),
    "Long heo (Pork offal)":                             (0.95, 0.10, 5),
    "Muc (Squid)":                                       (0.90, 0.10, 5),
    "Oc (Snails)":                                       (0.85, 0.12, 5),
    "Thit bo (Beef)":                                    (1.02, 0.10, 5),
    "Thit ga (Chicken)":                                 (0.95, 0.10, 5),
    "Thit heo (Pork)":                                   (0.98, 0.10, 5),
    "Thit kho (Braised pork)":                           (0.98, 0.12, 5),
    "Thit nuong (Grilled meat)":                         (0.90, 0.10, 5),
    "Tom (Shrimp)":                                      (0.88, 0.10, 5),
    "Trung (Egg)":                                       (0.92, 0.08, 5),

    # Vegetables & Salads
    "Bong cai (Cauliflower)":                            (0.50, 0.08, 5),
    "Ca chua (Tomato)":                                  (0.65, 0.08, 5),
    "Ca phao (Pickled eggplant)":                        (0.75, 0.08, 5),
    "Ca rot (Carrot)":                                   (0.65, 0.08, 5),
    "Chanh (Lime)":                                      (0.70, 0.05, 5),
    "Cu kieu (Pickled scallion head)":                   (0.70, 0.08, 5),
    "Dau hu (Tofu)":                                     (0.90, 0.08, 5),
    "Dua chua (Pickled vegetables)":                     (0.60, 0.08, 5),
    "Dua leo (Cucumber)":                                (0.60, 0.08, 5),
    "Khoai tay chien (French fries)":                    (0.40, 0.08, 5),
    "Nam (Mushroom)":                                    (0.55, 0.08, 5),
    "Nom hoa chuoi (Banana blossom salad)":              (0.55, 0.08, 5),
    "Ot chuong (Bell pepper)":                           (0.50, 0.08, 5),
    "Pho mai (Cheese)":                                  (1.05, 0.05, 5),
    "Rau (Vegetables)":                                  (0.30, 0.06, 5),
    "Salad (Salad)":                                     (0.30, 0.06, 5),
    "Con nguoi (Human)":                                 (0.00, 0.00, 1),
}


# ─── Nutrition per 100g ─────────────────────────────────────────────────────
# Fields: Calories (kcal), Protein (g), Fat (g), Carbs (g), Saturates (g), Sugar (g), Salt (g)
NUTRITION_PER_100G: Dict[str, Dict[str, float]] = {
    "Banh canh (Vietnamese thick noodle soup)":          {"Calories": 75,  "Protein": 3.8, "Fat": 2.5, "Carbs": 9.5,  "Saturates": 0.6, "Sugar": 0.8, "Salt": 0.6},
    "Banh chung (Square sticky rice cake)":              {"Calories": 195, "Protein": 4.5, "Fat": 3.8, "Carbs": 36.0, "Saturates": 1.2, "Sugar": 0.3, "Salt": 0.4},
    "Banh cuon (Rolled rice pancake)":                   {"Calories": 125, "Protein": 3.5, "Fat": 4.0, "Carbs": 18.0, "Saturates": 1.0, "Sugar": 0.8, "Salt": 0.5},
    "Banh khot (Mini savory pancakes)":                  {"Calories": 195, "Protein": 4.0, "Fat": 9.5, "Carbs": 23.5, "Saturates": 2.0, "Sugar": 1.8, "Salt": 0.7},
    "Banh mi (Vietnamese baguette sandwich)":            {"Calories": 265, "Protein": 8.5, "Fat": 9.8, "Carbs": 36.0, "Saturates": 3.0, "Sugar": 3.0, "Salt": 1.1},
    "Banh trang (Rice paper)":                           {"Calories": 300, "Protein": 1.0, "Fat": 0.5, "Carbs": 72.0, "Saturates": 0.1, "Sugar": 0.0, "Salt": 0.2},
    "Banh trang tron (Rice paper salad)":                {"Calories": 160, "Protein": 4.0, "Fat": 6.5, "Carbs": 21.0, "Saturates": 1.0, "Sugar": 2.0, "Salt": 0.8},
    "Banh xeo (Vietnamese sizzling pancake)":            {"Calories": 190, "Protein": 5.0, "Fat": 9.5, "Carbs": 21.0, "Saturates": 2.8, "Sugar": 1.5, "Salt": 0.7},
    "Bo kho (Beef stew)":                                {"Calories": 115, "Protein": 9.5, "Fat": 5.8, "Carbs": 6.0,  "Saturates": 2.2, "Sugar": 1.5, "Salt": 0.9},
    "Bo la lot (Grilled beef wrapped in betel leaves)":  {"Calories": 175, "Protein": 12.0,"Fat": 11.0,"Carbs": 5.0,  "Saturates": 3.5, "Sugar": 1.2, "Salt": 0.8},
    "Bong cai (Cauliflower)":                            {"Calories": 25,  "Protein": 2.0, "Fat": 0.1, "Carbs": 5.0,  "Saturates": 0.0, "Sugar": 2.0, "Salt": 0.02},
    "Bun (Rice vermicelli)":                             {"Calories": 110, "Protein": 2.0, "Fat": 0.5, "Carbs": 24.0, "Saturates": 0.1, "Sugar": 0.5, "Salt": 0.1},
    "Bun bo Hue (Hue beef noodle soup)":                 {"Calories": 90,  "Protein": 5.5, "Fat": 3.5, "Carbs": 9.0,  "Saturates": 1.0, "Sugar": 0.6, "Salt": 0.9},
    "Bun cha (Grilled pork with vermicelli)":            {"Calories": 165, "Protein": 8.5, "Fat": 7.0, "Carbs": 17.0, "Saturates": 2.0, "Sugar": 2.0, "Salt": 0.8},
    "Bun dau (Vermicelli with tofu)":                    {"Calories": 150, "Protein": 6.5, "Fat": 7.5, "Carbs": 15.0, "Saturates": 1.2, "Sugar": 1.0, "Salt": 0.8},
    "Bun mam (Fermented fish noodle soup)":              {"Calories": 92,  "Protein": 5.2, "Fat": 3.6, "Carbs": 10.0, "Saturates": 0.8, "Sugar": 1.0, "Salt": 1.2},
    "Bun rieu (Crab noodle soup)":                       {"Calories": 85,  "Protein": 4.8, "Fat": 3.2, "Carbs": 9.2,  "Saturates": 0.7, "Sugar": 0.8, "Salt": 0.8},
    "Ca (Fish)":                                         {"Calories": 130, "Protein": 18.0,"Fat": 5.0, "Carbs": 0.0,  "Saturates": 1.2, "Sugar": 0.0, "Salt": 0.3},
    "Ca chua (Tomato)":                                  {"Calories": 18,  "Protein": 0.9, "Fat": 0.2, "Carbs": 3.9,  "Saturates": 0.0, "Sugar": 2.6, "Salt": 0.01},
    "Ca phao (Pickled eggplant)":                        {"Calories": 22,  "Protein": 1.0, "Fat": 0.2, "Carbs": 4.5,  "Saturates": 0.0, "Sugar": 1.5, "Salt": 1.5},
    "Ca rot (Carrot)":                                   {"Calories": 41,  "Protein": 0.9, "Fat": 0.2, "Carbs": 9.6,  "Saturates": 0.0, "Sugar": 4.7, "Salt": 0.07},
    "Canh (Soup)":                                       {"Calories": 35,  "Protein": 2.5, "Fat": 1.2, "Carbs": 3.5,  "Saturates": 0.3, "Sugar": 1.0, "Salt": 0.6},
    "Cha (Vietnamese pork roll)":                        {"Calories": 180, "Protein": 14.0,"Fat": 12.0,"Carbs": 3.0,  "Saturates": 4.0, "Sugar": 1.0, "Salt": 1.5},
    "Cha gio (Spring rolls)":                            {"Calories": 240, "Protein": 7.5, "Fat": 14.0,"Carbs": 21.0, "Saturates": 3.0, "Sugar": 1.8, "Salt": 0.9},
    "Chanh (Lime)":                                      {"Calories": 30,  "Protein": 0.7, "Fat": 0.2, "Carbs": 10.5, "Saturates": 0.0, "Sugar": 1.7, "Salt": 0.01},
    "Com (Rice)":                                        {"Calories": 130, "Protein": 2.7, "Fat": 0.3, "Carbs": 28.0, "Saturates": 0.1, "Sugar": 0.1, "Salt": 0.01},
    "Com tam (Broken rice)":                             {"Calories": 165, "Protein": 7.0, "Fat": 5.0, "Carbs": 24.0, "Saturates": 1.5, "Sugar": 1.0, "Salt": 0.9},
    "Con nguoi (Human)":                                 {"Calories": 0,   "Protein": 0.0, "Fat": 0.0, "Carbs": 0.0,  "Saturates": 0.0, "Sugar": 0.0, "Salt": 0.0},
    "Cu kieu (Pickled scallion head)":                   {"Calories": 35,  "Protein": 1.2, "Fat": 0.1, "Carbs": 7.5,  "Saturates": 0.0, "Sugar": 4.0, "Salt": 1.2},
    "Cua (Crab)":                                        {"Calories": 87,  "Protein": 18.0,"Fat": 1.1, "Carbs": 0.0,  "Saturates": 0.2, "Sugar": 0.0, "Salt": 0.5},
    "Dau hu (Tofu)":                                     {"Calories": 76,  "Protein": 8.0, "Fat": 4.8, "Carbs": 1.9,  "Saturates": 0.7, "Sugar": 0.5, "Salt": 0.02},
    "Dua chua (Pickled vegetables)":                     {"Calories": 20,  "Protein": 1.0, "Fat": 0.2, "Carbs": 3.5,  "Saturates": 0.0, "Sugar": 1.5, "Salt": 1.5},
    "Dua leo (Cucumber)":                                {"Calories": 15,  "Protein": 0.7, "Fat": 0.1, "Carbs": 3.6,  "Saturates": 0.0, "Sugar": 1.7, "Salt": 0.01},
    "Goi cuon (Fresh spring rolls)":                     {"Calories": 115, "Protein": 5.5, "Fat": 2.5, "Carbs": 18.0, "Saturates": 0.5, "Sugar": 1.8, "Salt": 0.6},
    "Hamburger":                                         {"Calories": 250, "Protein": 13.0,"Fat": 12.0,"Carbs": 24.0, "Saturates": 4.5, "Sugar": 4.0, "Salt": 1.2},
    "Heo quay (Roast pork)":                             {"Calories": 290, "Protein": 18.0,"Fat": 23.0,"Carbs": 1.5,  "Saturates": 8.0, "Sugar": 0.5, "Salt": 1.1},
    "Hu tieu (Clear rice noodle soup)":                  {"Calories": 88,  "Protein": 5.0, "Fat": 3.0, "Carbs": 10.0, "Saturates": 0.8, "Sugar": 1.0, "Salt": 0.8},
    "Kho qua thit (Stuffed bitter melon soup)":          {"Calories": 65,  "Protein": 5.0, "Fat": 3.2, "Carbs": 4.0,  "Saturates": 1.0, "Sugar": 1.5, "Salt": 0.7},
    "Khoai tay chien (French fries)":                    {"Calories": 312, "Protein": 3.4, "Fat": 15.0,"Carbs": 41.0, "Saturates": 2.3, "Sugar": 0.3, "Salt": 0.6},
    "Lau (Hotpot)":                                      {"Calories": 90,  "Protein": 6.5, "Fat": 4.0, "Carbs": 7.0,  "Saturates": 1.2, "Sugar": 1.0, "Salt": 1.0},
    "Long heo (Pork offal)":                             {"Calories": 140, "Protein": 15.0,"Fat": 8.0, "Carbs": 1.0,  "Saturates": 2.8, "Sugar": 0.0, "Salt": 0.4},
    "Mi (Egg noodles)":                                  {"Calories": 138, "Protein": 4.5, "Fat": 2.1, "Carbs": 25.0, "Saturates": 0.5, "Sugar": 0.8, "Salt": 0.3},
    "Muc (Squid)":                                       {"Calories": 92,  "Protein": 15.6,"Fat": 1.4, "Carbs": 3.1,  "Saturates": 0.4, "Sugar": 0.0, "Salt": 0.4},
    "Nam (Mushroom)":                                    {"Calories": 28,  "Protein": 3.0, "Fat": 0.3, "Carbs": 4.5,  "Saturates": 0.0, "Sugar": 1.8, "Salt": 0.01},
    "Oc (Snails)":                                       {"Calories": 79,  "Protein": 15.0,"Fat": 1.4, "Carbs": 2.0,  "Saturates": 0.3, "Sugar": 0.0, "Salt": 0.4},
    "Ot chuong (Bell pepper)":                           {"Calories": 20,  "Protein": 1.0, "Fat": 0.2, "Carbs": 4.6,  "Saturates": 0.0, "Sugar": 4.2, "Salt": 0.02},
    "Pho (Vietnamese noodle soup)":                      {"Calories": 85,  "Protein": 5.8, "Fat": 2.6, "Carbs": 10.5, "Saturates": 0.9, "Sugar": 0.6, "Salt": 0.9},
    "Pho mai (Cheese)":                                  {"Calories": 350, "Protein": 22.0,"Fat": 28.0,"Carbs": 2.0,  "Saturates": 18.0,"Sugar": 1.0, "Salt": 1.6},
    "Rau (Vegetables)":                                  {"Calories": 25,  "Protein": 1.5, "Fat": 0.2, "Carbs": 4.5,  "Saturates": 0.0, "Sugar": 2.0, "Salt": 0.02},
    "Salad (Salad)":                                     {"Calories": 50,  "Protein": 1.5, "Fat": 3.0, "Carbs": 4.5,  "Saturates": 0.5, "Sugar": 2.0, "Salt": 0.3},
    "Thit bo (Beef)":                                    {"Calories": 215, "Protein": 24.0,"Fat": 13.0,"Carbs": 0.0,  "Saturates": 5.0, "Sugar": 0.0, "Salt": 0.4},
    "Thit ga (Chicken)":                                 {"Calories": 190, "Protein": 25.0,"Fat": 9.5, "Carbs": 0.0,  "Saturates": 2.5, "Sugar": 0.0, "Salt": 0.5},
    "Thit heo (Pork)":                                   {"Calories": 220, "Protein": 22.0,"Fat": 14.5,"Carbs": 0.0,  "Saturates": 5.2, "Sugar": 0.0, "Salt": 0.5},
    "Thit kho (Braised pork)":                           {"Calories": 220, "Protein": 15.0,"Fat": 16.0,"Carbs": 4.0,  "Saturates": 5.5, "Sugar": 3.0, "Salt": 1.5},
    "Thit nuong (Grilled meat)":                         {"Calories": 230, "Protein": 20.0,"Fat": 15.0,"Carbs": 2.5,  "Saturates": 5.0, "Sugar": 2.0, "Salt": 1.0},
    "Tom (Shrimp)":                                      {"Calories": 99,  "Protein": 21.0,"Fat": 1.0, "Carbs": 0.5,  "Saturates": 0.3, "Sugar": 0.0, "Salt": 0.4},
    "Trung (Egg)":                                       {"Calories": 145, "Protein": 12.5,"Fat": 10.0,"Carbs": 1.0,  "Saturates": 3.1, "Sugar": 0.5, "Salt": 0.3},
    "Xoi (Sticky rice)":                                 {"Calories": 195, "Protein": 4.2, "Fat": 2.5, "Carbs": 39.0, "Saturates": 0.6, "Sugar": 0.3, "Salt": 0.2},
    "Banh beo (Vietnamese savory steamed rice cake)":    {"Calories": 120, "Protein": 3.8, "Fat": 3.5, "Carbs": 18.0, "Saturates": 0.8, "Sugar": 1.2, "Salt": 0.4},
    "Cao lau (Cao lau noodles)":                         {"Calories": 135, "Protein": 6.8, "Fat": 4.5, "Carbs": 17.0, "Saturates": 1.2, "Sugar": 1.0, "Salt": 0.8},
    "Mi Quang (Quang-style noodles)":                    {"Calories": 130, "Protein": 6.0, "Fat": 4.8, "Carbs": 16.0, "Saturates": 1.2, "Sugar": 1.0, "Salt": 0.7},
    "Com chien duong chau (Yangzhou fried rice)":        {"Calories": 175, "Protein": 4.5, "Fat": 6.5, "Carbs": 25.0, "Saturates": 1.5, "Sugar": 0.8, "Salt": 0.8},
    "Bun cha ca (Fish cake noodle soup)":                {"Calories": 95,  "Protein": 5.5, "Fat": 3.2, "Carbs": 11.0, "Saturates": 0.8, "Sugar": 0.8, "Salt": 0.9},
    "Com chien ga (Fried rice with chicken)":            {"Calories": 180, "Protein": 6.5, "Fat": 6.8, "Carbs": 23.0, "Saturates": 1.6, "Sugar": 0.8, "Salt": 0.8},
    "Chao long (Pork organ congee)":                     {"Calories": 80,  "Protein": 4.5, "Fat": 2.8, "Carbs": 9.5,  "Saturates": 0.8, "Sugar": 0.5, "Salt": 0.7},
    "Nom hoa chuoi (Banana blossom salad)":              {"Calories": 65,  "Protein": 2.5, "Fat": 3.0, "Carbs": 7.0,  "Saturates": 0.5, "Sugar": 2.5, "Salt": 0.6},
    "Nui xao bo (Stir-fried macaroni with beef)":        {"Calories": 160, "Protein": 7.5, "Fat": 6.0, "Carbs": 19.0, "Saturates": 2.0, "Sugar": 1.2, "Salt": 0.7},
    "Sup cua (Crab soup)":                               {"Calories": 70,  "Protein": 4.5, "Fat": 2.0, "Carbs": 8.5,  "Saturates": 0.4, "Sugar": 0.8, "Salt": 0.6},
}


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


def get_density(class_name: str) -> Tuple[float, float]:
    """Return (mean_density_g_per_cm3, std_density_g_per_cm3) for a food class."""
    matched_key = _find_best_match(class_name, DENSITY_DB)
    if matched_key is not None:
        mean, std, _ = DENSITY_DB[matched_key]
        return mean, std
    
    # Fallback to general food density
    return 0.85, 0.15


def get_nutrition_per_100g(class_name: str) -> Dict[str, float]:
    """Return nutrition dict per 100g for a food class directly from class_names.py (SSOT)."""
    try:
        from class_names import class_names
        # Search by exact name or normalized match in class_names list
        norm_query = _normalize_name(class_name)
        for item in class_names:
            if item.get("name") == class_name or _normalize_name(item.get("name", "")) == norm_query:
                if "nutrition_per_100g" in item:
                    return dict(item["nutrition_per_100g"])
                elif "nutrition" in item:
                    # Scale serving to per 100g if serving_size_g is provided
                    serving_size = item.get("serving_size_g", 150)
                    scale = 100.0 / max(1.0, float(serving_size))
                    return {k: round(v * scale, 1) for k, v in item["nutrition"].items()}
    except Exception as e:
        logger.warning(f"Failed to read from class_names.py: {e}")

    # Fallback to local table if present
    matched_key = _find_best_match(class_name, NUTRITION_PER_100G)
    if matched_key is not None:
        return NUTRITION_PER_100G[matched_key]

    # Generic food fallback
    return {
        "Calories": 150.0,
        "Protein": 5.0,
        "Fat": 5.0,
        "Carbs": 20.0,
        "Saturates": 1.0,
        "Sugar": 1.0,
        "Salt": 0.5,
    }


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

