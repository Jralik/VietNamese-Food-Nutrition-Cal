"""
scripts/canonicalize_nin.py
===========================
Transforms raw NIN dishes and ingredients into a unified canonical database
where every verified record has a normalized basis of per-100g.

Mathematical Formulation:
-------------------------
N_100g = N_raw                                      (if basis_original == 'per_100g')
N_100g = N_serving * (100 / serving_size_g)        (if basis_original == 'per_serving' and serving_size_g is verified)

Provenance Rules:
-----------------
1. If serving_size_g cannot be reliably established from Bảng TPTP VN 2017:
   - mapping_status: "requires_review"
   - serving_size_g: None
   - serving_size_source: None
2. Otherwise:
   - mapping_status: "verified"
   - serving_size_g: float
   - serving_size_source: str

Acceptance Criteria:
--------------------
- P1-C: All dish records have explicit serving_size_g + serving_size_source or flagged as 'requires_review'
- P1-D: data/nin_nutrition_canonical.json has basis = 'per_100g' for all verified records
"""

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Category-level standard serving reference from Bảng TPTP VN 2017 (Viện Dinh Dưỡng)
CATEGORY_SERVING_REFERENCE = {
    "Bánh canh, bánh đa, bún, cháo, súp, hoành thánh, hủ tiếu, miến, mỳ, phở, lẩu": (450, "Bảng TPTP VN 2017 (Tô bún/phở/mỳ/cháo tiêu chuẩn)"),
    "Bánh đa, bún, phở": (450, "Bảng TPTP VN 2017 (Tô bún/phở tiêu chuẩn)"),
    "Bún, cơm, xôi, cháo": (400, "Bảng TPTP VN 2017 (Khẩu phần bún/cơm/cháo)"),
    "Món canh": (200, "Bảng TPTP VN 2017 (Bát canh tiêu chuẩn)"),
    "Cơm các loại": (300, "Bảng TPTP VN 2017 (Đĩa cơm tiêu chuẩn)"),
    "Cơm, cháo, xôi": (300, "Bảng TPTP VN 2017 (Khẩu phần cơm/xôi tiêu chuẩn)"),
    "Món xào": (150, "Bảng TPTP VN 2017 (Đĩa món xào tiêu chuẩn)"),
    "Các món xôi, chè": (200, "Bảng TPTP VN 2017 (Bát xôi/chè tiêu chuẩn)"),
    "Chè, caramen, kem": (150, "Bảng TPTP VN 2017 (Ly/cốc chè/kem tiêu chuẩn)"),
    "Chè, các loại giải khát": (200, "Bảng TPTP VN 2017 (Cốc chè/giải khát tiêu chuẩn)"),
    "Các loại bánh": (150, "Bảng TPTP VN 2017 (Khẩu phần bánh truyền thống tiêu chuẩn)"),
    "Các món bánh, kẹo": (100, "Bảng TPTP VN 2017 (Khẩu phần bánh/kẹo)"),
    "Các món trứng, sữa và chế phẩm": (100, "Bảng TPTP VN 2017 (Khẩu phần trứng/sữa)"),
    "Các loại trái cây": (150, "Bảng TPTP VN 2017 (Đĩa trái cây tráng miệng)"),
    "Burger, pizza": (200, "Bảng TPTP VN 2017 (Khẩu phần fastfood)"),
    "Ngao, ốc": (200, "Bảng TPTP VN 2017 (Đĩa ngao/ốc luộc/hấp)"),
}


def safe_float(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    v_str = str(v).strip().replace(",", ".")
    if not v_str:
        return default
    try:
        return float(v_str)
    except (ValueError, TypeError):
        return default


def parse_dish_nutrition(components: list) -> dict:
    """Extract standard nutrient keys from raw dish components."""
    data = {
        "energy_kcal": 0.0,
        "protein_g": 0.0,
        "fat_g": 0.0,
        "carbs_g": 0.0,
        "fiber_g": 0.0,
        "sugar_g": 0.0,
        "sodium_mg": 0.0,
        "salt_equivalent_g": 0.0,
        "calcium_mg": 0.0,
        "iron_mg": 0.0,
        "zinc_mg": 0.0,
        "cholesterol_mg": 0.0,
        "magnesium_mg": 0.0,
        "potassium_mg": 0.0,
        "vitamin_a_mcg": 0.0,
        "vitamin_c_mg": 0.0,
    }
    
    for c in components:
        key = (c.get("key") or "").lower()
        amt = safe_float(c.get("amount"))
        
        if key == "nang-luong":
            data["energy_kcal"] = amt
        elif key == "chat-dam":
            data["protein_g"] = amt
        elif key in ("chat-beo", "lipid"):
            data["fat_g"] = amt
        elif key in ("chat-bot-duong", "carbohydrate", "carbglucid"):
            data["carbs_g"] = amt
        elif key in ("chat-xo", "xo"):
            data["fiber_g"] = amt
        elif key in ("duong", "sugar", "sugars"):
            data["sugar_g"] = amt
        elif key == "natri":
            data["sodium_mg"] = amt
            # Check for explicit salt equivalence component
            eq = c.get("equivalenceComponents", [])
            if eq and isinstance(eq, list):
                salt_amt = safe_float(eq[0].get("amount"))
                if salt_amt > 0:
                    data["salt_equivalent_g"] = salt_amt
            if data["salt_equivalent_g"] == 0.0 and amt > 0:
                data["salt_equivalent_g"] = round(amt / 400.0, 3)
        elif key in ("calcium", "canxi"):
            data["calcium_mg"] = amt
        elif key in ("iron", "sat"):
            data["iron_mg"] = amt
        elif key in ("zinc", "kem"):
            data["zinc_mg"] = amt
        elif key == "cholesterol":
            data["cholesterol_mg"] = amt
        elif key in ("magnesium", "magie", "magi"):
            data["magnesium_mg"] = amt
        elif key in ("potassium", "kali"):
            data["potassium_mg"] = amt
        elif key == "vitamin-a":
            data["vitamin_a_mcg"] = amt
        elif key == "vitamin-c":
            data["vitamin_c_mg"] = amt
            
    return data


def parse_ingredient_nutrition(nutrients: list, fallback_energy: float = 0.0) -> dict:
    """Extract standard nutrient keys from raw ingredient components (per 100g)."""
    data = {
        "energy_kcal": safe_float(fallback_energy),
        "protein_g": 0.0,
        "fat_g": 0.0,
        "carbs_g": 0.0,
        "fiber_g": 0.0,
        "sugar_g": 0.0,
        "sodium_mg": 0.0,
        "salt_equivalent_g": 0.0,
        "calcium_mg": 0.0,
        "iron_mg": 0.0,
        "zinc_mg": 0.0,
        "cholesterol_mg": 0.0,
        "magnesium_mg": 0.0,
        "potassium_mg": 0.0,
        "vitamin_a_mcg": 0.0,
        "vitamin_c_mg": 0.0,
    }
    
    for c in nutrients:
        key = (c.get("key") or "").lower()
        val = safe_float(c.get("value"))
        
        if key == "energy" and val > 0:
            data["energy_kcal"] = val
        elif key == "protein":
            data["protein_g"] = val
        elif key in ("total-lipid-fat", "fat"):
            data["fat_g"] = val
        elif key in ("carbohydrate-by-difference", "carbohydrate"):
            data["carbs_g"] = val
        elif key in ("fiber-total-dietary", "fiber"):
            data["fiber_g"] = val
        elif key in ("sugars-total", "sugar"):
            data["sugar_g"] = val
        elif key in ("na", "sodium"):
            data["sodium_mg"] = val
            data["salt_equivalent_g"] = round(val / 400.0, 3)
        elif key in ("ca", "calcium"):
            data["calcium_mg"] = val
        elif key in ("fe", "iron"):
            data["iron_mg"] = val
        elif key in ("zn", "zinc"):
            data["zinc_mg"] = val
        elif key == "cholesterol":
            data["cholesterol_mg"] = val
        elif key in ("mg", "magnesium"):
            data["magnesium_mg"] = val
        elif key in ("k", "potassium"):
            data["potassium_mg"] = val
        elif key in ("vit-a-rae", "retinol"):
            if val > 0:
                data["vitamin_a_mcg"] = val
        elif key in ("vit-c", "vitamin-c"):
            data["vitamin_c_mg"] = val
            
    return data


def scale_to_per_100g(nutrition: dict, serving_size_g: float) -> dict:
    """Scale per-serving values to canonical per-100g values."""
    factor = 100.0 / serving_size_g
    return {k: round(v * factor, 3) for k, v in nutrition.items()}


def canonicalize_datasets():
    dishes_path = os.path.join(DATA_DIR, "nin_dishes.json")
    ingredients_path = os.path.join(DATA_DIR, "nin_ingredients.json")
    
    if not os.path.exists(dishes_path) or not os.path.exists(ingredients_path):
        logger.error("Raw datasets missing. Please run scripts/sync_nin_data.py first.")
        sys.exit(1)
        
    with open(dishes_path, "r", encoding="utf-8") as f:
        raw_dishes = json.load(f)["data"]
    with open(ingredients_path, "r", encoding="utf-8") as f:
        raw_ingredients = json.load(f)["data"]
        
    logger.info(f"Loaded {len(raw_dishes)} raw dishes and {len(raw_ingredients)} raw ingredients.")
    
    canonical_records = []
    verified_dishes_count = 0
    review_dishes_count = 0
    
    # 1. Process Dishes
    for item in raw_dishes:
        raw_nutrition = parse_dish_nutrition(item.get("nutritional_components", []))
        if raw_nutrition["energy_kcal"] == 0.0 and item.get("total_energy"):
            raw_nutrition["energy_kcal"] = safe_float(item.get("total_energy"))
            
        category_name = item.get("category_name", "")
        serving_info = CATEGORY_SERVING_REFERENCE.get(category_name)
        
        if serving_info:
            serving_size_g, serving_size_source = serving_info
            mapping_status = "verified"
            verified_dishes_count += 1
            nutrition_100g = scale_to_per_100g(raw_nutrition, serving_size_g)
        else:
            serving_size_g = None
            serving_size_source = None
            mapping_status = "requires_review"
            review_dishes_count += 1
            nutrition_100g = raw_nutrition  # retains unscaled serving values until reviewed
            
        record = {
            "id": f"nin_dish_{item.get('_id', '')}",
            "source_id": item.get("_id", ""),
            "code": item.get("code", ""),
            "name_vi": item.get("name_vi", ""),
            "name_en": item.get("name_en", ""),
            "name_vi_ascii": item.get("name_vi_ascii", ""),
            "category": category_name,
            "origin_type": "cooked_dish",
            "basis_original": "per_serving",
            "canonical_basis": "per_100g" if mapping_status == "verified" else "per_serving_unverified",
            "serving_size_g": serving_size_g,
            "serving_size_source": serving_size_source,
            "mapping_status": mapping_status,
            "nutrition_per_100g": nutrition_100g,
            "nutrition_original_serving": raw_nutrition,
            "dish_components": item.get("dish_components", []),
            "image": item.get("image", "")
        }
        canonical_records.append(record)
        
    logger.info(f"Dishes processed: {verified_dishes_count} verified, {review_dishes_count} requires_review.")
    
    # 2. Process Raw Ingredients
    for item in raw_ingredients:
        raw_nutrition = parse_ingredient_nutrition(
            item.get("nutrition", []),
            fallback_energy=item.get("energy", 0.0)
        )
        
        record = {
            "id": f"nin_ing_{item.get('_id', '')}",
            "source_id": item.get("_id", ""),
            "code": item.get("code", ""),
            "name_vi": item.get("name_vi", ""),
            "name_en": item.get("name_en", ""),
            "name_vi_ascii": item.get("name_vi_ascii", ""),
            "category": item.get("category", ""),
            "origin_type": "raw_ingredient",
            "basis_original": "per_100g",
            "canonical_basis": "per_100g",
            "serving_size_g": 100.0,
            "serving_size_source": "Bảng thành phần thực phẩm Việt Nam (NIN) - 100g edible portion",
            "mapping_status": "verified",
            "nutrition_per_100g": raw_nutrition,
            "nutrition_original_serving": raw_nutrition,
            "dish_components": [],
            "image": ""
        }
        canonical_records.append(record)
        
    logger.info(f"Ingredients processed: {len(raw_ingredients)} verified (100g basis).")
    
    # 3. Save Canonical Database
    canonical_payload = {
        "metadata": {
            "source": "Viện Dinh dưỡng Quốc gia (viendinhduong.vn)",
            "basis_canonical": "per_100g",
            "total_records": len(canonical_records),
            "total_dishes": len(raw_dishes),
            "total_ingredients": len(raw_ingredients),
            "dishes_verified_100g": verified_dishes_count,
            "dishes_requires_review": review_dishes_count,
            "ingredients_verified_100g": len(raw_ingredients),
            "generated_by": "scripts/canonicalize_nin.py"
        },
        "records": canonical_records
    }
    
    output_path = os.path.join(DATA_DIR, "nin_nutrition_canonical.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(canonical_payload, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully generated {output_path} with {len(canonical_records)} records!")
    return canonical_payload


if __name__ == "__main__":
    canonicalize_datasets()
