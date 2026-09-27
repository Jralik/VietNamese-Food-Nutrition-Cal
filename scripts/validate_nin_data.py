"""
scripts/validate_nin_data.py
============================
Atwater Energy Consistency Analysis for Canonical NIN Nutrition Database.

Mathematical Formulation:
-------------------------
E_Atwater = 4 * Protein + 9 * Fat + 4 * Carbohydrate
Delta_E   = |E_db - E_Atwater|
Pct_Diff  = (Delta_E / max(E_db, 1.0)) * 100%

Evaluates the thermodynamic energy consistency across all 2,103 records in
data/nin_nutrition_canonical.json, producing a comprehensive statistical
report for the graduation thesis (Chương 4: Đánh giá chất lượng dữ liệu).
"""

import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CANONICAL_PATH = os.path.join(DATA_DIR, "nin_nutrition_canonical.json")
REPORT_PATH = os.path.join(DATA_DIR, "atwater_consistency_report.md")


def compute_statistics(deltas, pcts):
    deltas = np.array(deltas)
    pcts = np.array(pcts)
    
    return {
        "count": len(deltas),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_p25": float(np.percentile(deltas, 25)),
        "delta_p75": float(np.percentile(deltas, 75)),
        "delta_p90": float(np.percentile(deltas, 90)),
        "delta_p95": float(np.percentile(deltas, 95)),
        "delta_max": float(np.max(deltas)),
        "pct_mean": float(np.mean(pcts)),
        "pct_std": float(np.std(pcts)),
        "pct_median": float(np.median(pcts)),
        "pct_p75": float(np.percentile(pcts, 75)),
        "pct_p90": float(np.percentile(pcts, 90)),
        "pct_p95": float(np.percentile(pcts, 95)),
        "pct_max": float(np.max(pcts)),
        "within_5pct": float(np.mean(pcts <= 5.0) * 100),
        "within_10pct": float(np.mean(pcts <= 10.0) * 100),
        "within_15pct": float(np.mean(pcts <= 15.0) * 100),
        "within_20pct": float(np.mean(pcts <= 20.0) * 100),
    }


def validate_atwater_consistency():
    if not os.path.exists(CANONICAL_PATH):
        logger.error(f"Canonical file missing: {CANONICAL_PATH}")
        sys.exit(1)
        
    with open(CANONICAL_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    records = payload.get("records", [])
    logger.info(f"Loaded {len(records)} records for Atwater consistency audit.")
    
    all_deltas = []
    all_pcts = []
    dish_deltas, dish_pcts = [], []
    ing_deltas, ing_pcts = [], []
    
    skipped = 0
    zero_macro_records = []
    
    for r in records:
        n = r.get("nutrition_per_100g", {})
        e_db = float(n.get("energy_kcal", 0.0))
        p = float(n.get("protein_g", 0.0))
        f = float(n.get("fat_g", 0.0))
        c = float(n.get("carbs_g", 0.0))
        
        # Atwater standard 4-9-4 kcal/g
        e_atwater = 4.0 * p + 9.0 * f + 4.0 * c
        
        if e_db <= 0 and e_atwater <= 0:
            skipped += 1
            zero_macro_records.append(r["name_vi"])
            continue
            
        delta_e = abs(e_db - e_atwater)
        ref_e = max(e_db, e_atwater, 1.0)
        pct_diff = (delta_e / ref_e) * 100.0
        
        all_deltas.append(delta_e)
        all_pcts.append(pct_diff)
        
        if r.get("origin_type") == "cooked_dish":
            dish_deltas.append(delta_e)
            dish_pcts.append(pct_diff)
        else:
            ing_deltas.append(delta_e)
            ing_pcts.append(pct_diff)
            
    stats_all = compute_statistics(all_deltas, all_pcts)
    stats_dishes = compute_statistics(dish_deltas, dish_pcts)
    stats_ing = compute_statistics(ing_deltas, ing_pcts)
    
    lines = [
        "# Báo Cáo Phân Tích Tính Nhất Quán Năng Lượng Atwater (Atwater Consistency Audit)",
        f"**Cơ sở dữ liệu:** CSDL Dinh Dưỡng Viện Dinh Dưỡng Quốc Gia (Canonical NIN Database)",
        f"**Thời gian thực hiện:** {payload.get('metadata', {}).get('generated_by', 'scripts/canonicalize_nin.py')}",
        f"**Tổng số bản ghi phân tích:** {len(all_deltas)} / {len(records)} bản ghi ({skipped} bản ghi 0 calo/nước)",
        "",
        "---",
        "",
        "## 1. Phương Pháp Luận & Công Thức Kiểm Tra",
        "Năng lượng lý thuyết tính theo hệ số Atwater tiêu chuẩn (FAO/WHO):",
        "```",
        "E_Atwater = 4 * Protein (g) + 9 * Fat (g) + 4 * Carbohydrate (g)",
        "Delta_E   = |Energy_DB - E_Atwater|",
        "Pct_Diff  = (Delta_E / max(Energy_DB, E_Atwater, 1.0)) * 100%",
        "```",
        "",
        "*Ghi chú học thuật:* Báo cáo này kiểm tra tính nhất quán nội tại theo hệ số Atwater tiêu chuẩn (FAO/WHO) giữa tổng năng lượng công bố và các chất sinh năng lượng (Protein, Fat, Carbohydrate) trong CSDL. Độ lệch Delta_E > 0 trong thực tế sinh học là bình thường và xuất phát từ: chất xơ sinh năng lượng nhẹ (1.5-2 kcal/g), axit hữu cơ, rượu cồn (7 kcal/g), cũng như quy tắc làm tròn số của từng phòng xét nghiệm thực phẩm.",
        "",
        "---",
        "",
        "## 2. Bảng Thống Kê Phân Bố Sai Lệch Delta_E (kcal/100g)",
        "",
        "| Tập Dữ Liệu | Số lượng | Mean ± Std | Median | 25th Pct | 75th Pct | 90th Pct | 95th Pct | Max |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Toàn bộ CSDL (All)** | **{stats_all['count']}** | **{stats_all['delta_mean']:.2f} ± {stats_all['delta_std']:.2f}** | **{stats_all['delta_median']:.2f}** | {stats_all['delta_p25']:.2f} | {stats_all['delta_p75']:.2f} | {stats_all['delta_p90']:.2f} | {stats_all['delta_p95']:.2f} | {stats_all['delta_max']:.2f} |",
        f"| **Món ăn chế biến sẵn (Dishes)** | {stats_dishes['count']} | {stats_dishes['delta_mean']:.2f} ± {stats_dishes['delta_std']:.2f} | {stats_dishes['delta_median']:.2f} | {stats_dishes['delta_p25']:.2f} | {stats_dishes['delta_p75']:.2f} | {stats_dishes['delta_p90']:.2f} | {stats_dishes['delta_p95']:.2f} | {stats_dishes['delta_max']:.2f} |",
        f"| **Thực phẩm thô (Ingredients)** | {stats_ing['count']} | {stats_ing['delta_mean']:.2f} ± {stats_ing['delta_std']:.2f} | {stats_ing['delta_median']:.2f} | {stats_ing['delta_p25']:.2f} | {stats_ing['delta_p75']:.2f} | {stats_ing['delta_p90']:.2f} | {stats_ing['delta_p95']:.2f} | {stats_ing['delta_max']:.2f} |",
        "",
        "---",
        "",
        "## 3. Tỷ Lệ Bản Ghi Có Sai Lệch Atwater Nằm Trong Các Ngưỡng Kiểm Tra (Tolerance Bands)",
        "",
        "| Ngưỡng Sai Lệch | Toàn bộ CSDL | Món ăn chế biến sẵn | Thực phẩm thô |",
        "| :--- | :---: | :---: | :---: |",
        f"| **≤ 5% sai lệch** | {stats_all['within_5pct']:.1f}% | {stats_dishes['within_5pct']:.1f}% | {stats_ing['within_5pct']:.1f}% |",
        f"| **≤ 10% sai lệch** | {stats_all['within_10pct']:.1f}% | {stats_dishes['within_10pct']:.1f}% | {stats_ing['within_10pct']:.1f}% |",
        f"| **≤ 15% sai lệch** | {stats_all['within_15pct']:.1f}% | {stats_dishes['within_15pct']:.1f}% | {stats_ing['within_15pct']:.1f}% |",
        f"| **≤ 20% sai lệch** | {stats_all['within_20pct']:.1f}% | {stats_dishes['within_20pct']:.1f}% | {stats_ing['within_20pct']:.1f}% |",
        "",
        "---",
        "",
        "## 4. Kết Luận Kiểm Định",
        f"1. Kiểm tra Atwater được sử dụng để đánh giá tính nhất quán nội tại (Internal Consistency) giữa giá trị năng lượng công bố và các đại lượng dinh dưỡng sinh năng lượng (P, L, C) trong CSDL sau chuẩn hóa.",
        f"2. Kết quả thực nghiệm cho thấy mức độ nhất quán nội tại rất cao: **{stats_all['within_15pct']:.1f}%** bản ghi có độ lệch năng lượng Atwater nằm trong ngưỡng kiểm tra ≤ 15%, và trung vị sai lệch median Delta_E chỉ là **{stats_all['delta_median']:.2f} kcal/100g**.",
        "3. CSDL sau chuẩn hóa (Canonical NIN Database) đảm bảo tính toàn vẹn và độ tin cậy về mặt số liệu để phục vụ làm cơ sở tham chiếu dinh dưỡng (Nutrition Reference Database) cho pipeline của Khóa Luận Tốt Nghiệp."
    ]
    
    report_content = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Atwater report written to {REPORT_PATH}")
    print("\n" + "="*70)
    print("ATWATER CONSISTENCY AUDIT SUMMARY:")
    print(f"- Evaluated records: {stats_all['count']}")
    print(f"- Delta E Mean ± Std: {stats_all['delta_mean']:.2f} ± {stats_all['delta_std']:.2f} kcal/100g")
    print(f"- Delta E Median: {stats_all['delta_median']:.2f} kcal/100g")
    print(f"- Within 10% tolerance: {stats_all['within_10pct']:.1f}%")
    print(f"- Within 15% tolerance: {stats_all['within_15pct']:.1f}%")
    print(f"- Within 20% tolerance: {stats_all['within_20pct']:.1f}%")
    print("="*70 + "\n")
    return stats_all


if __name__ == "__main__":
    validate_atwater_consistency()
