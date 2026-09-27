"""
scripts/audit_target_association.py
===================================
Audit 54 ground truth images to analyze:
1. Protocol 1 (Full-image aggregate): M_pred = sum(m_i for all i)
2. Protocol 2 (Target-matched): M_pred_target = m_target
3. Two-Tier Causal Error Attribution Taxonomy:
   Tier 1 — Upstream Target Detection Status:
     - D1: Correct target detected
     - D2: Misclassified (e.g. Pho -> Bun mam)
     - D3: Missed / partial detection (e.g. whole bowl missed, only noodle strand detected)
     - D4: Fragmented (dish split into multiple generic components e.g. Canh + Bun)
     - D5: Duplicate detection (same item detected multiple times due to NMS overlap)
   Tier 2 — Downstream / Evaluation Interference:
     - E1: Auxiliary side-dish interference (e.g. side soup bowl included in plate photo)
     - E2: Target-association mismatch (GT is single item, prediction includes extra table items)
     - E3: Downstream geometric/depth error (depth thickness loss on thin foods like Banh Cuon)

Attribution Protocol:
Multi-label. Each benchmark image receives exactly one primary detection status (D1..D5)
and zero or more interference factors (E1..E3). Category counts are therefore not
mutually exclusive across tiers.
"""

import os
import sys
import json
import math
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(ROOT_DIR, "data", "benchmark_portion_results.json")

def analyze():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["image_results"]
    print(f"Total benchmark images: {len(images)}")

    audit_records = []
    
    proto1_by_dish = defaultdict(list)
    proto2_by_dish = defaultdict(list)
    tier1_counts = defaultdict(int)
    tier2_counts = defaultdict(int)

    for it in images:
        folder = it["folder"]
        fname = it["filename"]
        gt_m = it["gt_mass_g"]
        pred_m_p1 = it["pred_mass_g"] # Protocol 1: full image sum
        dets = it.get("detected_items", [])
        
        t1_status = None
        t2_interferences = []

        if folder == "ComSuon":
            # For ComSuon, plate items: Com, Thit heo, Com tam, Dua leo, Ca rot
            # Side dishes: Canh (chén canh súp), Bun
            plate_items = ["Com (Rice)", "Thit heo (Pork)", "Com tam (Broken rice)", "Dua leo (Cucumber)", "Ca rot (Carrot)", "Trung cut (Quail eggs)", "Trung ga (Chicken eggs)"]
            side_items = ["Canh (Soup)", "Bun (Rice vermicelli)"]
            
            p2_mass = 0.0
            canh_count = 0
            for d in dets:
                cname = d["class"]
                if cname in plate_items:
                    p2_mass += d["mass_g"]
                elif cname in side_items:
                    if "Canh" in cname:
                        canh_count += 1
            
            if p2_mass > 0:
                t1_status = "D1_correct_target_detected"
            else:
                t1_status = "D3_missed_or_partial"
                
            if canh_count >= 1:
                t2_interferences.append("E1_auxiliary_side_dish_interference")
                t2_interferences.append("E2_target_association_mismatch")
            if canh_count > 1:
                t1_status = "D5_duplicate_detection" # Duplicate canh
                
            pred_m_p2 = p2_mass

        elif folder == "Pho":
            pho_dets = [d for d in dets if "Pho" in d["class"]]
            bun_mam_dets = [d for d in dets if "Bun mam" in d["class"]]
            bun_dets = [d for d in dets if "Bun (" in d["class"]]
            canh_dets = [d for d in dets if "Canh (" in d["class"]]
            
            if pho_dets:
                t1_status = "D1_correct_target_detected"
                pred_m_p2 = sum(d["mass_g"] for d in pho_dets)
            elif bun_mam_dets:
                t1_status = "D2_misclassified"
                pred_m_p2 = sum(d["mass_g"] for d in bun_mam_dets)
            elif canh_dets and bun_dets:
                t1_status = "D4_fragmented"
                pred_m_p2 = sum(d["mass_g"] for d in dets)
            elif bun_dets and not canh_dets:
                t1_status = "D3_missed_or_partial"
                pred_m_p2 = sum(d["mass_g"] for d in bun_dets)
            else:
                t1_status = "D3_missed_or_partial"
                pred_m_p2 = sum(d["mass_g"] for d in dets) if dets else 0.0

        elif folder == "BanhCuon":
            t2_interferences.append("E3_downstream_geometric_depth_error")
            bc_dets = [d for d in dets if "Banh cuon" in d["class"]]
            if bc_dets:
                t1_status = "D1_correct_target_detected"
                pred_m_p2 = sum(d["mass_g"] for d in bc_dets)
            else:
                pred_m_p2 = sum(d["mass_g"] for d in dets)
                t1_status = "D3_missed_or_partial" if not dets else "D2_misclassified"

        elif folder == "BanhMi":
            bm_dets = [d for d in dets if "Banh mi" in d["class"]]
            if bm_dets:
                t1_status = "D1_correct_target_detected"
                pred_m_p2 = sum(d["mass_g"] for d in bm_dets)
            else:
                pred_m_p2 = sum(d["mass_g"] for d in dets)
                t1_status = "D3_missed_or_partial" if not dets else "D2_misclassified"

        elif folder in ("BunBoHue", "BunRieu-CanhBun"):
            bowl_dets = [d for d in dets if any(t in d["class"] for t in ["Bun bo", "Bun rieu", "Canh bun"])]
            rau_dets = [d for d in dets if "Rau" in d["class"]]
            if rau_dets:
                t2_interferences.append("E1_auxiliary_side_dish_interference")
            if bowl_dets:
                t1_status = "D1_correct_target_detected"
                pred_m_p2 = sum(d["mass_g"] for d in bowl_dets)
            else:
                t1_status = "D2_misclassified"
                pred_m_p2 = sum(d["mass_g"] for d in dets)

        elif folder == "SupCua":
            if not dets:
                t1_status = "D3_missed_or_partial"
                pred_m_p2 = 0.0
            else:
                sup_dets = [d for d in dets if any(t in d["class"] for t in ["Sup cua", "Canh"])]
                if sup_dets:
                    t1_status = "D1_correct_target_detected"
                    pred_m_p2 = sum(d["mass_g"] for d in sup_dets)
                else:
                    t1_status = "D2_misclassified"
                    pred_m_p2 = sum(d["mass_g"] for d in dets)

        else:
            t1_status = "D1_correct_target_detected"
            pred_m_p2 = pred_m_p1

        # Calculate errors
        err_p1 = abs(pred_m_p1 - gt_m)
        rel_p1 = (err_p1 / gt_m) * 100.0 if gt_m > 0 else 0.0

        err_p2 = abs(pred_m_p2 - gt_m)
        rel_p2 = (err_p2 / gt_m) * 100.0 if gt_m > 0 else 0.0

        proto1_by_dish[folder].append((err_p1, rel_p1, pred_m_p1, gt_m))
        proto2_by_dish[folder].append((err_p2, rel_p2, pred_m_p2, gt_m))

        tier1_counts[t1_status] += 1
        for factor in t2_interferences:
            tier2_counts[factor] += 1

        audit_records.append({
            "folder": folder,
            "filename": fname,
            "gt_mass_g": gt_m,
            "pred_m_protocol1_g": round(pred_m_p1, 1),
            "rel_err_p1_pct": round(rel_p1, 1),
            "pred_m_protocol2_g": round(pred_m_p2, 1),
            "rel_err_p2_pct": round(rel_p2, 1),
            "upstream_detection_status": t1_status,
            "downstream_interference_factors": t2_interferences,
            "detected_classes": [d["class"] for d in dets]
        })

    # Output stats
    print("\n" + "=" * 70)
    print("COMPARISON: PROTOCOL 1 (FULL-IMAGE) vs PROTOCOL 2 (TARGET-MATCHED)")
    print("=" * 70)
    print(f"{'Category':<18} | {'P1 MAPE (%)':<12} | {'P2 MAPE (%)':<12} | {'Improvement':<12}")
    print("-" * 70)

    p1_all_rel = []
    p2_all_rel = []

    summary_comparison = {}

    for folder in sorted(proto1_by_dish.keys()):
        p1_list = proto1_by_dish[folder]
        p2_list = proto2_by_dish[folder]

        p1_mape = sum(x[1] for x in p1_list) / len(p1_list)
        p2_mape = sum(x[1] for x in p2_list) / len(p2_list)

        p1_all_rel.extend([x[1] for x in p1_list])
        p2_all_rel.extend([x[1] for x in p2_list])

        diff = p1_mape - p2_mape
        print(f"{folder:<18} | {p1_mape:>10.1f}% | {p2_mape:>10.1f}% | {diff:>+10.1f}%")
        summary_comparison[folder] = {
            "p1_mape": round(p1_mape, 1),
            "p2_mape": round(p2_mape, 1),
            "diff": round(diff, 1)
        }

    overall_p1_mape = sum(p1_all_rel) / len(p1_all_rel)
    overall_p2_mape = sum(p2_all_rel) / len(p2_all_rel)
    print("-" * 70)
    print(f"{'OVERALL (54 images)':<18} | {overall_p1_mape:>10.1f}% | {overall_p2_mape:>10.1f}% | {overall_p1_mape - overall_p2_mape:>+10.1f}%")

    print("\n" + "=" * 70)
    print("TIER 1 — UPSTREAM TARGET DETECTION STATUS (Mutually Exclusive: Sum = 54)")
    print("=" * 70)
    for mode, cnt in sorted(tier1_counts.items()):
        pct = (cnt / len(images)) * 100.0
        print(f"- {mode:<35}: {cnt:>2} images ({pct:>5.1f}%)")

    print("\n" + "=" * 70)
    print("TIER 2 — DOWNSTREAM / EVALUATION INTERFERENCE (Multi-Label Factors)")
    print("=" * 70)
    for mode, cnt in sorted(tier2_counts.items()):
        pct = (cnt / len(images)) * 100.0
        print(f"- {mode:<42}: {cnt:>2} images ({pct:>5.1f}%)")

    # Save to JSON
    out_path = os.path.join(ROOT_DIR, "data", "benchmark_error_attribution_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "title": "Phase 4 Two-Tier Error Attribution and Target-Association Audit",
                "total_images": len(images),
                "overall_p1_mape_pct": round(overall_p1_mape, 1),
                "overall_p2_mape_pct": round(overall_p2_mape, 1),
                "attribution_protocol": {
                    "type": "two_tier_causal_taxonomy",
                    "tier_1_definition": "Upstream target detection status (Mutually exclusive: each image has exactly one status)",
                    "tier_2_definition": "Downstream and evaluation interference factors (Multi-label: each image has zero or more factors)",
                    "note": "Tier 1 sums to 54 (100%). Tier 2 factors are non-exclusive and do not sum to 100%."
                }
            },
            "summary_comparison": summary_comparison,
            "tier_1_detection_status_counts": dict(tier1_counts),
            "tier_2_interference_factor_counts": dict(tier2_counts),
            "image_audit_records": audit_records
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved two-tier audit report to: {out_path}")

if __name__ == "__main__":
    analyze()
