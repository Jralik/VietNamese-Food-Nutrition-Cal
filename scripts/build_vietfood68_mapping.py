"""
scripts/build_vietfood68_mapping.py
===================================
Constructs the final operational mapping table from VietFood68 (68 classes)
to the Canonical NIN Nutrition Database (1,755 verified per_100g records),
directly consuming the expert-reviewed inventory checklist (data/vietfood68_inventory_checklist.json).

Pipeline Integration:
---------------------
- Fast Mode (YOLO26m + SAM 2.1):
    - Uses Direct mapping for single foods
    - Uses Approximate mapping for cooked composite dishes
    - Uses Primary Match composite reference for Component-based dishes
    - Uses Fallback literature/USDA for Unmapped dishes
- Deep Mode (YOLO26m + FoodSAM):
    - Uses Direct mapping for single foods
    - Decomposes Component-based dishes into sub-masks mapped to ingredients (nin_ingredients.json)
    - Fallback to approximate/literature if decomposition confidence is low

Outputs:
--------
- data/vietfood68_nin_mapping.json
- data/vietfood68_mapping_report.md
"""

import os
import sys
import json
import logging

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKLIST_PATH = os.path.join(DATA_DIR, "vietfood68_inventory_checklist.json")
CANONICAL_PATH = os.path.join(DATA_DIR, "nin_nutrition_canonical.json")
OUTPUT_MAPPING_PATH = os.path.join(DATA_DIR, "vietfood68_nin_mapping.json")
OUTPUT_REPORT_PATH = os.path.join(DATA_DIR, "vietfood68_mapping_report.md")


def load_canonical_lookup():
    with open(CANONICAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)["records"]
    return {r["id"]: r for r in data}


def build_mapping():
    if not os.path.exists(CHECKLIST_PATH):
        logger.error(f"Inventory checklist not found at {CHECKLIST_PATH}. Run scripts/inventory_vietfood68.py first.")
        sys.exit(1)
        
    with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
        checklist_data = json.load(f)
        
    canonical_lookup = load_canonical_lookup()
    inventory = checklist_data["inventory"]
    
    mapping_records = []
    strategy_counts = {"direct": 0, "approximate": 0, "component_based": 0, "unmapped": 0}
    
    for item in inventory:
        mtype = item["mapping_type"]
        strategy_counts[mtype] += 1
        
        # Primary reference nutrition (if mapped to NIN)
        primary_match = item.get("primary_match")
        canonical_nutr = None
        if primary_match and primary_match["nin_food_id"] in canonical_lookup:
            cand = canonical_lookup[primary_match["nin_food_id"]]
            if cand.get("canonical_basis") == "per_serving_unverified":
                # Class-level verified serving size from inventory (Bảng TPTP VN 2017 / Literature)
                serving_g = item.get("serving_size_g", 100)
                scale_ratio = 100.0 / max(1.0, float(serving_g))
                orig_nutr = cand.get("nutrition_original_serving", cand.get("nutrition_per_100g", {}))
                canonical_nutr = {k: round(v * scale_ratio, 3) for k, v in orig_nutr.items()}
                # Update primary_match basis descriptor
                primary_match = dict(primary_match)
                primary_match["canonical_basis"] = "per_100g_mapping_derived"
                primary_match["energy_kcal"] = canonical_nutr.get("energy_kcal", 0.0)
            else:
                canonical_nutr = cand["nutrition_per_100g"]
            
        # Component nutrition details (for FoodSAM)
        detailed_components = []
        if mtype == "component_based":
            for comp in item.get("components", []):
                cid = comp["source_food_id"]
                nutr = None
                if cid in canonical_lookup:
                    nutr = canonical_lookup[cid]["nutrition_per_100g"]
                detailed_components.append({
                    "source_food_id": cid,
                    "food_name": comp["food_name"],
                    "role": comp["role"],
                    "nutrition_per_100g": nutr
                })
                
        record = {
            "vietfood_class_id": item["vietfood_class_id"],
            "vietfood_class_name_vi": item["vietfood_class_name_vi"],
            "vietfood_class_name_en": item["vietfood_class_name_en"],
            "full_name": item["full_name"],
            "mapping_type": mtype,
            "mapping_confidence": item["mapping_confidence"],
            "mapping_confidence_note": item.get("mapping_confidence_note", "Mức độ tin cậy của quyết định ánh xạ phân loại ngữ nghĩa"),
            "serving_size_g": item["serving_size_g"],
            "serving_size_source": item["serving_size_source"],
            "mapping_status": item["mapping_status"],
            "decision_rationale": item["decision_rationale"],
            "primary_match": primary_match,
            "primary_nutrition_per_100g": canonical_nutr,
            "nin_matches": item.get("nin_matches", []),
            "components": detailed_components
        }
        mapping_records.append(record)
        
    payload = {
        "metadata": {
            "title": "Ma Trận Ánh Xạ VietFood68 Sang CSDL Viện Dinh Dưỡng Quốc Gia (NIN Mapping Matrix)",
            "total_classes": len(mapping_records),
            "distribution": strategy_counts,
            "provenance_standard": "Bảng thành phần thực phẩm Việt Nam (NIN) & USDA FoodData Central",
            "audit_source": "data/vietfood68_inventory_checklist.json",
            "version": "1.0.0"
        },
        "mappings": mapping_records
    }
    
    with open(OUTPUT_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Mapping saved to {OUTPUT_MAPPING_PATH}")
    
    # Generate Academic Report Markdown Table for Chapter 3/4
    report_lines = [
        "# Báo Cáo Phân Tích Ma Trận Ánh Xạ 68 Lớp VietFood68 Sang CSDL Viện Dinh Dưỡng",
        "",
        "## 1. Phân Bố Chiến Lược Ánh Xạ (Mapping Strategy Distribution)",
        "",
        "| Chiến Lược Ánh Xạ | Số Lượng Lớp | Tỷ Lệ % | Đặc Điểm Kỹ Thuật |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Direct (Trực tiếp)** | {strategy_counts['direct']} | {strategy_counts['direct']/68*100:.1f}% | Ánh xạ 1-1 với thực phẩm thô hoặc món chuẩn VDD có độ tương đồng ngữ nghĩa tuyệt đối |",
        f"| **Approximate (Xấp xỉ)** | {strategy_counts['approximate']} | {strategy_counts['approximate']/68*100:.1f}% | Ánh xạ sang món ăn nấu chín có công thức đại diện chính thức trong CSDL VDD |",
        f"| **Component-based (Thành phần)** | {strategy_counts['component_based']} | {strategy_counts['component_based']/68*100:.1f}% | Cung cấp mô hình tri thức dinh dưỡng theo thành phần cho món phức hợp; FoodSAM Deep Mode hỗ trợ thị giác phân rã topping |",
        f"| **Unmapped (Duy trì Fallback)** | {strategy_counts['unmapped']} | {strategy_counts['unmapped']/68*100:.1f}% | Lớp phi thực phẩm (Con người) hoặc đặc sản vùng miền chưa có trong CSDL VDD |",
        f"| **Tổng cộng** | **68** | **100.0%** | Toàn bộ 68 lớp đều có provenance và căn cứ phân loại minh bạch |",
        "",
        "---",
        "",
        "## 2. Bảng Danh Mục Chi Tiết 68 Lớp",
        "",
        "| ID | Tên Lớp (VietFood68) | Chiến Lược | Độ Tin Cậy | Nguồn Đối Sánh NIN | Khẩu Phần Tham Chiếu | Nguồn Gốc Khẩu Phần |",
        "| :-: | :--- | :---: | :---: | :--- | :-: | :--- |"
    ]
    
    for m in mapping_records:
        match_name = m["primary_match"]["nin_food_name"] if m["primary_match"] else "*(Giữ Fallback)*"
        report_lines.append(
            f"| {m['vietfood_class_id']:02d} | **{m['full_name']}** | `{m['mapping_type']}` | `{m['mapping_confidence']}` | {match_name} | {m['serving_size_g']}g | {m['serving_size_source']} |"
        )
        
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Academic Report saved to {OUTPUT_REPORT_PATH}")
    
    print("\n" + "="*65)
    print("VIETFOOD68 EXPERT MAPPING SUMMARY:")
    print(f"- Direct           : {strategy_counts['direct']}/68 ({strategy_counts['direct']/68*100:.1f}%)")
    print(f"- Approximate      : {strategy_counts['approximate']}/68 ({strategy_counts['approximate']/68*100:.1f}%)")
    print(f"- Component-based  : {strategy_counts['component_based']}/68 ({strategy_counts['component_based']/68*100:.1f}%)")
    print(f"- Unmapped         : {strategy_counts['unmapped']}/68 ({strategy_counts['unmapped']/68*100:.1f}%)")
    print("="*65 + "\n")
    return payload


if __name__ == "__main__":
    build_mapping()
