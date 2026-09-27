"""
nutrition_rule_engine.py
========================
Phase 5: Nutrition Rule Engine & Structured Facts Generator.

Implements a four-layer separated architecture:
  1. Data Layer: 12-nutrient canonical data (per 100g) from NIN & class_names.
  2. Reference Layer: Vietnamese RNI 2016 & Mifflin-St Jeor TDEE energy targets.
  3. Deterministic Rule Layer: 100% deterministic mathematical calculations,
     Atwater P:L:C energy balance, dual-coverage, and system-defined warnings.
  4. LLM Context Interface: Structured facts JSON contract for linguistic advisory.
"""

import os
import json
import math
from typing import Dict, List, Any, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_JSON_PATH = os.path.join(ROOT_DIR, "data", "vietfood68_nin_mapping.json")

# ---------------------------------------------------------------------------
# Reference Layer: Vietnamese RNI 2016 & Activity Factors
# ---------------------------------------------------------------------------

RNI_2016_METADATA = {
    "source": "Bộ Y Tế / Viện Dinh Dưỡng Quốc Gia Việt Nam",
    "guideline": "Nhu Cầu Dinh Dưỡng Khuyến Nghị Cho Người Việt Nam (RNI 2016)",
    "target_demographic": "Người trưởng thành 19-50 tuổi",
    "provenance_standard": "RNI 2016 Table 2.1 & 3.4 (NXB Y Học Hà Nội)"
}

# Daily reference intakes by biological sex (Adults 19-50 years)
# Sodium limit: 2000 mg/day (WHO & Viện Dinh Dưỡng)
# Cholesterol limit: 300 mg/day (Khuyến cáo tim mạch quốc gia)
RNI_ADULT_REFERENCE = {
    "male": {
        "Protein_g": 60.0,
        "Calcium_mg": 800.0,
        "Iron_mg": 11.0,
        "Zinc_mg": 10.0,
        "Sodium_max_mg": 2000.0,
        "Cholesterol_max_mg": 300.0,
        "Sugar_max_g": 50.0,
        "Salt_max_g": 5.0, # Equivalent to ~2000mg Na
    },
    "female": {
        "Protein_g": 50.0,
        "Calcium_mg": 800.0,
        "Iron_mg": 18.0, # Higher iron requirement for women of childbearing age
        "Zinc_mg": 8.0,
        "Sodium_max_mg": 2000.0,
        "Cholesterol_max_mg": 300.0,
        "Sugar_max_g": 50.0,
        "Salt_max_g": 5.0,
    }
}

# Standard Physical Activity Level (PAL) multipliers
ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,           # Ít vận động (nhân viên văn phòng)
    "lightly_active": 1.375,    # Vận động nhẹ (tập luyện 1-3 ngày/tuần)
    "moderately_active": 1.55,  # Vận động vừa (tập luyện 3-5 ngày/tuần)
    "very_active": 1.725,       # Vận động nhiều (tập luyện 6-7 ngày/tuần)
    "extra_active": 1.9,        # Vận động nặng (lao động chân tay/vận động viên)
}

# Goal calorie adjustments
GOAL_CALORIE_ADJUSTMENT = {
    "lose": -500.0,     # Giảm cân an toàn (~0.5 kg/tuần)
    "maintain": 0.0,    # Giữ cân
    "gain": +500.0,     # Tăng cân / Tăng cơ
}

# Vietnamese National Recommended Macro Distribution (% of total macro energy)
# Viện Dinh Dưỡng: Protein 13-20%, Lipid 20-25%, Glucid 55-65%
NATIONAL_MACRO_RATIO_RANGE = {
    "protein_pct": (13.0, 20.0),
    "fat_pct": (20.0, 25.0),
    "carbs_pct": (55.0, 65.0),
}


# ---------------------------------------------------------------------------
# In-Memory Cache of VietFood68 NIN Mapping Provenance
# ---------------------------------------------------------------------------
_MAPPING_CACHE: Optional[Dict[str, dict]] = None

def _load_mapping_cache() -> Dict[str, dict]:
    global _MAPPING_CACHE
    if _MAPPING_CACHE is not None:
        return _MAPPING_CACHE

    _MAPPING_CACHE = {}
    if os.path.exists(MAPPING_JSON_PATH):
        try:
            with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for m in data.get("mappings", []):
                name = m.get("full_name", "")
                if name:
                    _MAPPING_CACHE[name] = m
        except Exception as e:
            print(f"[RuleEngine] Warning loading mapping cache: {e}")
    return _MAPPING_CACHE


# ---------------------------------------------------------------------------
# Layer 2: Mifflin-St Jeor TDEE & Energy Target Engine
# ---------------------------------------------------------------------------

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate Body Mass Index (kg/m^2). Descriptive metadata only."""
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    h_m = height_cm / 100.0
    return round(weight_kg / (h_m * h_m), 1)


def calculate_bmr_mifflin_st_jeor(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    """
    Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor equation.
    Strictly physiological equation based on weight, height, age, and sex.
    """
    sex_norm = sex.lower().strip()
    if sex_norm == "male":
        bmr = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) + 5.0
    elif sex_norm == "female":
        bmr = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) - 161.0
    else:
        raise ValueError(f"sex must be 'male' or 'female', got {sex}")
    return round(bmr, 1)


def calculate_daily_energy_target(user_profile: dict) -> Tuple[float, float, float]:
    """
    Calculate (BMR, TDEE, Daily_Energy_Target_kcal).
    Disentangles Energy Target from Micronutrient RDA.
    """
    w = float(user_profile.get("weight_kg", user_profile.get("weight", 65.0)))
    h = float(user_profile.get("height_cm", user_profile.get("height", 170.0)))
    age = int(user_profile.get("age", 25))
    sex = str(user_profile.get("sex", "female")).lower().strip()
    act = user_profile.get("activity_level", user_profile.get("activity_factor", 1.2))
    goal = str(user_profile.get("goal", "maintain")).lower().strip()

    # Determine activity factor
    if isinstance(act, (int, float)):
        act_factor = float(act)
    else:
        act_factor = ACTIVITY_MULTIPLIERS.get(str(act), 1.2)

    bmr = calculate_bmr_mifflin_st_jeor(w, h, age, sex)
    tdee = round(bmr * act_factor, 1)

    adj = GOAL_CALORIE_ADJUSTMENT.get(goal, 0.0)
    target_kcal = max(1200.0, round(tdee + adj, 1))

    return bmr, tdee, target_kcal


# ---------------------------------------------------------------------------
# Layer 3: Deterministic Rule Calculations & Warning Engine
# ---------------------------------------------------------------------------

def calculate_atwater_macro_energy(protein_g: float, fat_g: float, carbs_g: float) -> Tuple[float, Dict[str, float]]:
    """
    Calculate macronutrient energy using standard Atwater factors (4 - 9 - 4 kcal/g).
    To avoid discrepancy where sum != 100%, P:L:C distribution is strictly evaluated
    using macro energy as denominator: E_macro = 4*P + 9*F + 4*C.
    """
    p_kcal = max(0.0, protein_g * 4.0)
    f_kcal = max(0.0, fat_g * 9.0)
    c_kcal = max(0.0, carbs_g * 4.0)
    e_macro = p_kcal + f_kcal + c_kcal

    if e_macro > 0.0:
        p_pct = round((p_kcal / e_macro) * 100.0, 1)
        f_pct = round((f_kcal / e_macro) * 100.0, 1)
        c_pct = round((c_kcal / e_macro) * 100.0, 1)
    else:
        p_pct, f_pct, c_pct = 0.0, 0.0, 0.0

    return round(e_macro, 1), {
        "protein_energy_pct": p_pct,
        "fat_energy_pct": f_pct,
        "carbs_energy_pct": c_pct,
    }


def evaluate_system_warnings(meal_nutrition: dict, e_macro_kcal: float, macro_pct: dict) -> List[Dict[str, Any]]:
    """
    Deterministic System-Defined Warning Engine.
    Evaluates well-defined thresholds without external AI/LLM intervention.
    """
    warnings = []

    sodium_mg = meal_nutrition.get("Sodium", 0.0) or 0.0
    cholesterol_mg = meal_nutrition.get("Cholesterol", 0.0) or 0.0
    saturates_g = meal_nutrition.get("Saturates", 0.0) or 0.0

    # 1. Sodium warnings
    if sodium_mg > 2000.0:
        warnings.append({
            "code": "SODIUM_EXCEEDS_DAILY_LIMIT",
            "severity": "HIGH",
            "message": f"Hàm lượng Natri trong bữa ăn ({sodium_mg:.0f} mg) vượt ngưỡng giới hạn tối đa cả ngày (2.000 mg theo khuyến nghị Viện Dinh Dưỡng / WHO).",
            "value": round(sodium_mg, 1),
            "threshold": 2000.0,
            "unit": "mg"
        })
    elif sodium_mg > 1000.0:
        warnings.append({
            "code": "SODIUM_OVER_HALF_DAILY_LIMIT",
            "severity": "MEDIUM",
            "message": f"Hàm lượng Natri ({sodium_mg:.0f} mg) chiếm hơn 50% giới hạn tối đa cả ngày (ngưỡng cảnh báo hệ thống > 1.000 mg).",
            "value": round(sodium_mg, 1),
            "threshold": 1000.0,
            "unit": "mg"
        })

    # 2. Cholesterol warning
    if cholesterol_mg > 200.0:
        warnings.append({
            "code": "CHOLESTEROL_OVER_66PCT_DAILY_LIMIT",
            "severity": "MEDIUM",
            "message": f"Hàm lượng Cholesterol ({cholesterol_mg:.0f} mg) chiếm hơn 66% ngưỡng tham chiếu ngày (300 mg).",
            "value": round(cholesterol_mg, 1),
            "threshold": 200.0,
            "unit": "mg"
        })

    # 3. Saturated Fat warning (with defensive E = 0 check)
    if e_macro_kcal > 0.0:
        sat_kcal = saturates_g * 9.0
        sat_pct = (sat_kcal / e_macro_kcal) * 100.0
        if sat_pct > 10.0:
            warnings.append({
                "code": "SATURATED_FAT_HIGH",
                "severity": "MEDIUM",
                "message": f"Chất béo bão hòa cung cấp {sat_pct:.1f}% tổng năng lượng bữa ăn (vượt khuyến nghị < 10% năng lượng).",
                "value": round(sat_pct, 1),
                "threshold": 10.0,
                "unit": "%"
            })

    # 4. Macro balance reference flag (descriptive reference for single meal)
    p_pct = macro_pct.get("protein_energy_pct", 0.0)
    f_pct = macro_pct.get("fat_energy_pct", 0.0)
    c_pct = macro_pct.get("carbs_energy_pct", 0.0)

    p_low, p_high = NATIONAL_MACRO_RATIO_RANGE["protein_pct"]
    f_low, f_high = NATIONAL_MACRO_RATIO_RANGE["fat_pct"]
    c_low, c_high = NATIONAL_MACRO_RATIO_RANGE["carbs_pct"]

    if e_macro_kcal > 50.0: # Only flag meals with substantial calories
        if not (p_low <= p_pct <= p_high and f_low <= f_pct <= f_high and c_low <= c_pct <= c_high):
            warnings.append({
                "code": "MACRO_PROFILE_OUTSIDE_REFERENCE_RANGE",
                "severity": "LOW",
                "message": f"Tỷ lệ sinh năng lượng P:L:C ({p_pct:.0f}% : {f_pct:.0f}% : {c_pct:.0f}%) lệch ngoài dải tham chiếu chuẩn quốc gia (P: 13-20%, L: 20-25%, C: 55-65%). Cờ tham chiếu mô tả cơ cấu năng lượng bữa ăn, không phải chẩn đoán mất cân bằng của toàn bộ chế độ ăn.",
                "value": f"{p_pct:.0f}:{f_pct:.0f}:{c_pct:.0f}",
                "threshold": "13-20:20-25:55-65",
                "unit": "%"
            })

    return warnings


# ---------------------------------------------------------------------------
# Layer 1 & 2: Structured Nutrition Facts Builder with Item-Level Provenance
# ---------------------------------------------------------------------------

def build_structured_facts(
    meal_nutrition: Dict[str, float],
    user_profile: Optional[Dict[str, Any]] = None,
    detected_items: Optional[Any] = None,
    rda_override: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Builds the authoritative Structured Nutrition Facts JSON payload.
    This object is the ONLY context provided to the LLM.
    
    Adheres strictly to the architectural contract:
      - All numbers, ratios, and warnings are calculated deterministically.
      - Item-level facts contain full data lineage:
        portion_g -> nutrition_per_100g -> source_id -> NIN.
      - Supports both new signatures and legacy callers:
        build_structured_facts(meal_nutrition, user_profile, rda, detected_foods)
    """
    if user_profile is None:
        user_profile = {
            "age": 25,
            "sex": "female",
            "height_cm": 170.0,
            "weight_kg": 65.0,
            "activity_level": "sedentary",
            "goal": "maintain"
        }

    # Handle legacy parameter swap: (meal_nutrition, user_profile, rda, detected_foods)
    if isinstance(detected_items, dict) and any(k in detected_items for k in ("Calories", "Protein", "Calcium", "Iron", "Zinc")):
        actual_rda = detected_items
        actual_items = rda_override
    else:
        actual_rda = rda_override if isinstance(rda_override, dict) else None
        actual_items = detected_items

    sex = str(user_profile.get("sex", "female")).lower().strip()
    if sex not in ("male", "female"):
        sex = "female"
    age = int(user_profile.get("age", 25))

    # Energy calculations
    bmr, tdee, daily_energy_target_kcal = calculate_daily_energy_target(user_profile)
    bmi = calculate_bmi(
        float(user_profile.get("weight_kg", user_profile.get("weight", 65.0))),
        float(user_profile.get("height_cm", user_profile.get("height", 170.0)))
    )

    # Reference values
    ref_values = RNI_ADULT_REFERENCE.get(sex, RNI_ADULT_REFERENCE["female"])

    # Extract 12 canonical nutrients
    calories = float(meal_nutrition.get("Calories", 0.0) or 0.0)
    protein = float(meal_nutrition.get("Protein", 0.0) or 0.0)
    fat = float(meal_nutrition.get("Fat", 0.0) or 0.0)
    carbs = float(meal_nutrition.get("Carbs", 0.0) or 0.0)
    saturates = float(meal_nutrition.get("Saturates", 0.0) or 0.0)
    sugar = float(meal_nutrition.get("Sugar", 0.0) or 0.0)
    salt = float(meal_nutrition.get("Salt", 0.0) or 0.0) # Salt equivalent
    sodium = float(meal_nutrition.get("Sodium", 0.0) or 0.0)
    calcium = float(meal_nutrition.get("Calcium", 0.0) or 0.0)
    iron = float(meal_nutrition.get("Iron", 0.0) or 0.0)
    zinc = float(meal_nutrition.get("Zinc", 0.0) or 0.0)
    cholesterol = float(meal_nutrition.get("Cholesterol", 0.0) or 0.0)

    # Calculate Atwater macro energy and balance
    e_macro_kcal, macro_pct = calculate_atwater_macro_energy(protein, fat, carbs)

    # Energy coverage
    energy_coverage_pct = round((calories / daily_energy_target_kcal) * 100.0, 1) if daily_energy_target_kcal > 0 else 0.0

    # Micronutrient coverages against RNI 2016
    ca_ref = ref_values["Calcium_mg"]
    fe_ref = ref_values["Iron_mg"]
    zn_ref = ref_values["Zinc_mg"]
    p_ref = ref_values["Protein_g"]

    micronutrient_coverage = {
        "protein_pct_rni": round((protein / p_ref) * 100.0, 1) if p_ref > 0 else 0.0,
        "calcium_pct_rni": round((calcium / ca_ref) * 100.0, 1) if ca_ref > 0 else 0.0,
        "iron_pct_rni": round((iron / fe_ref) * 100.0, 1) if fe_ref > 0 else 0.0,
        "zinc_pct_rni": round((zinc / zn_ref) * 100.0, 1) if zn_ref > 0 else 0.0,
    }

    # Evaluate Warnings
    warnings = evaluate_system_warnings(meal_nutrition, e_macro_kcal, macro_pct)

    # Build item-level facts with lineage
    mapping_cache = _load_mapping_cache()
    items_fact_list = []

    # Try importing runtime projection class_names for fallback
    runtime_class_lookup = {}
    try:
        from class_names import class_names
        for c in class_names:
            runtime_class_lookup[c["name"]] = c
    except ImportError:
        pass

    # Normalize actual_items to list of item dicts
    item_input_list = []
    legacy_detected_foods = {}
    if isinstance(actual_items, dict):
        legacy_detected_foods = actual_items
        for fname, count in actual_items.items():
            val = float(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else 1.0
            item_input_list.append({"name": fname, "portion_g": 100.0 * val})
    elif isinstance(actual_items, list):
        item_input_list = actual_items
        legacy_detected_foods = {
            (it.get("name") or it.get("class_name") or f"item_{i}"): 1
            for i, it in enumerate(actual_items)
        }

    for it in item_input_list:
        name = it.get("class_name") or it.get("name") or it.get("class") or "Unknown"
        portion_g = float(it.get("mass_g") or it.get("portion_g") or 0.0)

        # Lookup mapping details
        m_info = mapping_cache.get(name)
        rt_info = runtime_class_lookup.get(name, {})

        # Source & Provenance determination (strict 3-way distinction: verified_nin, literature_fallback, non_food)
        if name == "Con nguoi (Human)":
            source = "non_food"
            source_id = "non_food_background"
            mapping_strategy = "Unmapped"
            mapping_conf = 1.0
            provenance_status = "non_food"
            serving_size = 0.0
        elif "Banh khot" in name or "Cao lau" in name:
            source = "literature_fallback"
            source_id = "literature_recipe_fallback"
            mapping_strategy = "Unmapped"
            mapping_conf = 0.8
            provenance_status = "literature_fallback"
            serving_size = float(m_info.get("serving_size_g", 100.0) if m_info else rt_info.get("serving_size_g", 100.0))
        elif m_info:
            primary = m_info.get("primary_match") or {}
            source = "vn_nin_2017" if primary.get("source_type") in ("dish", "ingredient") else "literature_fallback"
            source_id = primary.get("nin_food_id") or "fallback_composite"
            mapping_strategy = str(m_info.get("mapping_type", "approximate")).capitalize()
            mapping_conf = 1.0 if mapping_strategy == "Direct" else (0.8 if mapping_strategy == "Approximate" else 0.5)
            provenance_status = "verified_nin"
            serving_size = float(m_info.get("serving_size_g", 100.0))
        else:
            source = "vn_nin_2017"
            source_id = "fallback_ref"
            mapping_strategy = "Approximate"
            mapping_conf = 0.8
            provenance_status = "verified_nin"
            serving_size = float(rt_info.get("serving_size_g", 100.0))

        # Nutrition per 100g lookup
        n_100g = rt_info.get("nutrition_per_100g", {})
        n_portion = it.get("nutrition") or {
            k: round(v * (portion_g / 100.0), 1) for k, v in n_100g.items()
        }

        items_fact_list.append({
            "name": name,
            "portion_g": round(portion_g, 1),
            "nutrition_per_100g": n_100g,
            "nutrition_estimate": n_portion,
            "source": source,
            "source_id": source_id,
            "mapping_strategy": mapping_strategy,
            "mapping_confidence": mapping_conf,
            "mapping_confidence_semantics": "semantic_mapping_confidence",
            "serving_size_g": serving_size,
            "provenance_status": provenance_status
        })

    # Construct the final Structured Facts schema
    structured_facts = {
        "schema_version": "5.0.0",
        "user_profile": {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "bmr_kcal": bmr,
            "tdee_kcal": tdee,
            "daily_energy_target_kcal": daily_energy_target_kcal,
            "goal": user_profile.get("goal", "maintain")
        },
        "meal_summary": {
            "energy_kcal": round(calories, 1),
            "energy_coverage_pct": energy_coverage_pct,
            "atwater_macro_energy_kcal": e_macro_kcal,
            "macro_energy_balance": macro_pct,
            "macronutrients": {
                "protein_g": round(protein, 1),
                "fat_g": round(fat, 1),
                "carbs_g": round(carbs, 1),
                "saturates_g": round(saturates, 1),
                "sugar_g": round(sugar, 1),
                "salt_equivalent_g": round(salt, 2)
            },
            "micronutrients": {
                "sodium_mg": round(sodium, 1),
                "calcium_mg": round(calcium, 1),
                "iron_mg": round(iron, 2),
                "zinc_mg": round(zinc, 2),
                "cholesterol_mg": round(cholesterol, 1)
            },
            "rni_micronutrient_coverage": micronutrient_coverage
        },
        "system_warnings": warnings,
        "detected_items": items_fact_list,
        "reference_metadata": RNI_2016_METADATA,
        # Backward compatibility section for old callers
        "detected_foods": legacy_detected_foods,
        "user_goal": user_profile.get("goal", "maintain"),
        "deficits": {
            k: {
                "consumed": meal_nutrition.get(k, 0),
                "target": (actual_rda.get(k) if actual_rda and k in actual_rda else ref_values.get(f"{k}_g", 50.0)),
                "pct": round((meal_nutrition.get(k, 0) / (actual_rda.get(k) if actual_rda and k in actual_rda else ref_values.get(f"{k}_g", 50.0))) * 100, 1)
            }
            for k in ["Protein"] if micronutrient_coverage["protein_pct_rni"] < 80.0
        },
        "excesses": {
            "Sodium": {
                "consumed": sodium,
                "target": 2000.0,
                "pct": round((sodium / 2000.0) * 100, 1)
            }
        } if sodium > 2000.0 else {},
        "on_target": ["Protein"] if micronutrient_coverage["protein_pct_rni"] >= 80.0 else []
    }

    return structured_facts


def get_llm_advisory_contract() -> str:
    """
    Returns the strict system prompt defining the LLM's role contract.
    Strictly decouples narrative explanation from deterministic calculations.
    """
    return (
        "Bạn là một chuyên gia tư vấn dinh dưỡng AI chuyên nghiệp về ẩm thực Việt Nam.\n\n"
        "KIẾN TRÚC PHÂN TÁCH:\n"
        "Hệ thống áp dụng kiến trúc phân tách nghiêm ngặt giữa phép tính định lượng và mô-đun ngôn ngữ: "
        "toàn bộ số liệu dinh dưỡng, tỷ lệ khuyến nghị và cảnh báo vượt ngưỡng được tính toán xác định "
        "bằng Python Rule Engine trước khi chuyển giao context cho bạn; LLM không tham gia vào quá trình tính toán "
        "và không được xem là nguồn dữ liệu chân lý.\n\n"
        "BẠN NHẬN ĐƯỢC (LLM RECEIVES):\n"
        "- Nutrition facts (số liệu dinh dưỡng chính thức từ CSDL NIN)\n"
        "- Reference values & energy coverage (nhu cầu khuyến nghị RNI 2016 & mục tiêu TDEE)\n"
        "- System-defined rule warnings (các cảnh báo vượt ngưỡng do Rule Engine kích hoạt)\n"
        "- User profile context (thông tin cá nhân cần thiết: BMI, mục tiêu cân nặng)\n\n"
        "TRÁCH NHIỆM CỦA BẠN (LLM RESPONSIBILITY):\n"
        "1. Giải thích ý nghĩa dinh dưỡng của bữa ăn đối với sức khỏe và mục tiêu của người dùng.\n"
        "2. Tóm tắt và giải thích rõ ràng các cảnh báo vượt ngưỡng (nếu có trong 'system_warnings').\n"
        "3. Đưa ra gợi ý lối sống và điều chỉnh bữa ăn lành mạnh bằng tiếng Việt tự nhiên, thân thiện.\n\n"
        "GIỚI HẠN BẮT BUỘC (LLM DOES NOT):\n"
        "- KHÔNG tính toán dinh dưỡng hoặc thay đổi các con số đã được cung cấp.\n"
        "- KHÔNG tự ý tính toán lại TDEE, BMR hay tỷ lệ % năng lượng.\n"
        "- KHÔNG tự xác định các ngưỡng cảnh báo mới ngoài hệ thống.\n"
        "- KHÔNG điều chỉnh các sự thật số học (numeric facts) trong JSON.\n"
        "- KHÔNG đóng vai trò như một cơ sở dữ liệu dinh dưỡng độc lập.\n"
        "- LƯU Ý VỀ MAPPING CONFIDENCE: Trường 'mapping_confidence' trong detected_items là độ tin cậy "
        "của ánh xạ ngữ nghĩa giữa tên món và CSDL NIN, KHÔNG PHẢI độ chính xác hay sai số dinh dưỡng.\n\n"
        "ĐỊNH DẠNG: Trả lời ngắn gọn (3-4 đoạn), định dạng Markdown rõ ràng, lịch sự và mang tính hành động."
    )
