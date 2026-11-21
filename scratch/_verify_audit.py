import json
from pathlib import Path
from collections import defaultdict
import statistics

root = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
bench_file = root / "scratch" / "full_seg_benchmark.jsonl"

records = []
with open(bench_file, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records")

models = ["sam2", "mobilesam", "foodsam"]

# 1. Audit TH4-A boxes
th4a_boxes = []
for r in records:
    fn = r["file_name"]
    for x in r["yolo_eval"]["th4a_yolo_fp"]:
        th4a_boxes.append((fn, x))

print(f"Total TH4-A YOLO FP boxes: {len(th4a_boxes)}")

# Categories:
# 1. Duplicate detection (dup_overlap >= 50%) -> 8 boxes
# 2. Valid unannotated garnish (Chanh, Oc) -> 5 boxes
# 3. Annotation omission on cropped border (com-suon1 (4).jpg Canh) -> 1 box
# 4. Unwanted extra region (ThitKho1 (6).jpg Thit kho rim) -> 1 box

for fn, x in th4a_boxes:
    is_dup = x["duplicate_overlap"]["mobilesam"]
    cls_name = x["pred_class"]
    conf = x["pred_conf"]
    m_ms = x["extra_mass_g"]["mobilesam"]
    m_s2 = x["extra_mass_g"]["sam2"]
    m_fs = x["extra_mass_g"]["foodsam"]
    if is_dup:
        cat = "Cat 1: Duplicate detection"
    elif cls_name in ("Chanh", "Oc"):
        cat = "Cat 2: Valid unannotated garnish"
    elif fn == "com-suon1 (4).jpg" and cls_name == "Canh":
        cat = "Cat 3: Annotation omission (cropped Canh)"
    else:
        cat = "Cat 4: Unwanted extra region"
    print(f"[{cat}] {fn} - {cls_name} ({conf:.2f}) -> MSAM: {m_ms:.1f}g, SAM2: {m_s2:.1f}g, FSAM: {m_fs:.1f}g")

# 2. Compute Adjusted Full-Image Meal MAE
# Standard Full-Image vs Adjusted (excluding Cat 2 & Cat 3)
for m in models:
    raw_meal_errs = []
    raw_meal_apes = []
    adj_meal_errs = []
    adj_meal_apes = []
    
    for r in records:
        gt_meal_by_cls = {}
        valid_img = True
        for inst in r["oracle_instances"]:
            if inst["mapping_status"] in ("excluded_unknown_weight", "excluded_unverified_line_mismatch"):
                valid_img = False
                break
            if inst["gt_weight_g"] is not None:
                gt_meal_by_cls[inst["short_class"]] = inst["gt_weight_g"]
        if not valid_img or not gt_meal_by_cls:
            continue
            
        gt_meal_total = sum(gt_meal_by_cls.values())
        
        # Raw predicted meal total
        pred_matched = sum(c["models"][m]["mass_pred_class_g"] for c in r["yolo_eval"]["matched_cases"])
        pred_fp = sum(x["extra_mass_g"][m] for x in r["yolo_eval"]["th4a_yolo_fp"])
        
        pred_fsam_rec = 0.0
        pred_fsam_extra = 0.0
        if m == "foodsam":
            pred_fsam_rec = sum(
                x["recovery_detail"]["recovered_mass_g"]
                for x in r["yolo_eval"]["th3_missed"]
                if x["foodsam_recovered"] and x["recovery_detail"]
            )
            pred_fsam_extra = sum(x["extra_mass_g"] for x in r["yolo_eval"]["th4a_foodsam_false_extras"])
            
        raw_pred_total = pred_matched + pred_fp + pred_fsam_rec + pred_fsam_extra
        raw_meal_errs.append(abs(raw_pred_total - gt_meal_total))
        raw_meal_apes.append(abs(raw_pred_total - gt_meal_total) / gt_meal_total * 100.0)
        
        # Adjusted predicted meal total:
        # Exclude Cat 2 (Chanh, Oc) and Cat 3 (Canh in com-suon1 (4))
        adj_fp = 0.0
        for x in r["yolo_eval"]["th4a_yolo_fp"]:
            is_dup = x["duplicate_overlap"]["mobilesam"]
            if is_dup:
                continue
            if x["pred_class"] in ("Chanh", "Oc"):
                continue  # exclude garnish
            if r["file_name"] == "com-suon1 (4).jpg" and x["pred_class"] == "Canh":
                continue  # exclude annotation omission
            adj_fp += x["extra_mass_g"][m]
            
        adj_fsam_extra = 0.0
        if m == "foodsam":
            # in com-suon1 (5).jpg, foodsam extra is dipping sauce (garnish)
            for x in r["yolo_eval"]["th4a_foodsam_false_extras"]:
                if r["file_name"] == "com-suon1 (5).jpg":
                    continue
                adj_fsam_extra += x["extra_mass_g"]
                
        adj_pred_total = pred_matched + adj_fp + pred_fsam_rec + adj_fsam_extra
        adj_meal_errs.append(abs(adj_pred_total - gt_meal_total))
        adj_meal_apes.append(abs(adj_pred_total - gt_meal_total) / gt_meal_total * 100.0)

    print(f"\nModel: {m}")
    print(f"  Raw Full-Image Meal MAE: {statistics.mean(raw_meal_errs):.2f} g, MAPE: {statistics.mean(raw_meal_apes):.2f}%")
    print(f"  Adjusted Full-Image Meal MAE: {statistics.mean(adj_meal_errs):.2f} g, MAPE: {statistics.mean(adj_meal_apes):.2f}%")

# 3. TH3 FoodSAM recovery analysis
print("\n--- TH3 FoodSAM Recovery Analysis ---")
th3_items = []
for r in records:
    for x in r["yolo_eval"]["th3_missed"]:
        th3_items.append((r["file_name"], x))

rec_errs = []
rec_apes = []
for fn, x in th3_items:
    gt_w = x["gt_weight_g"]
    if x["foodsam_recovered"] and x["recovery_detail"]:
        rec_m = x["recovery_detail"]["recovered_mass_g"]
        err = abs(rec_m - gt_w)
        ape = err / gt_w * 100.0
        rec_errs.append(err)
        rec_apes.append(ape)
        print(f"  {fn}: {x['gt_class']} GT={gt_w}g, Recov={rec_m}g -> |err|={err:.1f}g, APE={ape:.1f}%")
    else:
        print(f"  {fn}: {x['gt_class']} GT={gt_w}g -> NOT RECOVERED")

print(f"FoodSAM TH3 Recovered Count: {len(rec_errs)}/{len(th3_items)}")
print(f"Mean Recovery Abs Error (MAE): {statistics.mean(rec_errs):.2f} g")
print(f"Mean Recovery APE (MAPE): {statistics.mean(rec_apes):.2f}%")

