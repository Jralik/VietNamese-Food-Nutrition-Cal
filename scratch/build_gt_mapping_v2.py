import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import sys

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc

COCO_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
VAL_PATH = ROOT / "portion-estimation_val.txt"
OUT_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"

def norm(s: str) -> str:
    return s.split("(")[0].strip().lower().replace(" ", "")

VF_BY_NORM = {norm(k): k for k in pc.CLASS_TO_DISH_TYPE}

def parse_weight_token(tok: str):
    """Parse weight string like '315.5', '150g', '99.3 (tong)', '?' -> (float|None, is_total_note)."""
    t = tok.strip()
    if "?" in t:
        return None, False
    is_tong = "(tong)" in t.lower()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", t)
    if not m:
        return None, is_tong
    return float(m.group(1)), is_tong

def parse_val_line(line: str):
    p_str, rest = line[1:].split('",', 1)
    rest = rest.strip().rstrip(";").strip()
    items = []
    meal_total_g = None
    for seg in rest.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if ":" in seg:
            cls_raw, rhs = seg.split(":", 1)
            cls_raw = cls_raw.strip()
            rhs = rhs.strip()
            if cls_raw.lower() == "all":
                w, _ = parse_weight_token(rhs)
                meal_total_g = w
                continue
            if "-" in rhs:
                cnt_s, w_s = rhs.split("-", 1)
                cnt = int(re.search(r"\d+", cnt_s).group(0))
                w_val, is_tong = parse_weight_token(w_s)
            else:
                cnt = 1
                w_val, is_tong = parse_weight_token(rhs)
            items.append({"cls": cls_raw, "count": cnt, "weight_g": w_val, "is_tong": is_tong})
        else:
            # Handle legacy syntax like '1 - Trung 221.2'
            m = re.match(r"^(\d+)\s*-\s*([A-Za-z\s]+)\s+([0-9.]+)$", seg)
            if m:
                items.append({
                    "cls": m.group(2).strip(),
                    "count": int(m.group(1)),
                    "weight_g": float(m.group(3)),
                    "is_tong": False,
                })
            else:
                raise ValueError(f"Unparsed segment '{seg}' in line: {line}")
    return p_str, items, meal_total_g

coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
cats = {c["id"]: c["name"] for c in coco["categories"]}
imgs_by_name = {Path(i["file_name"]).name: i for i in coco["images"]}

# Special filename alias for Loai 3
def coco_img_for_val_path(p_str: str):
    p = Path(p_str)
    if p.name in imgs_by_name:
        return imgs_by_name[p.name]
    if p.parent.name == "Loai 3" and p.name.startswith("Banh Trang "):
        num = p.stem.split()[-1]
        alias = f"Banh Trang {num}_Loai_3.jpg"
        if alias in imgs_by_name:
            return imgs_by_name[alias]
    return None

anns_by_img_id = defaultdict(list)
for a in coco["annotations"]:
    anns_by_img_id[a["image_id"]].append(a)

val_lines = [l.strip() for l in VAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

instances_out = []
status_counts = Counter()
dish_type_counts = Counter()
not_in_coco = []
meal_only_images = []
valid_groups = set()
mismatch_notes = []

for line in val_lines:
    p_str, items, meal_total_g = parse_val_line(line)
    im = coco_img_for_val_path(p_str)
    if im is None:
        not_in_coco.append(p_str)
        continue
    iid = im["id"]
    anns = anns_by_img_id.get(iid, [])
    if not anns:
        not_in_coco.append(p_str)
        continue

    # Map val items by normalized class name
    val_by_norm = defaultdict(list)
    for it in items:
        n = norm(it["cls"])
        if n == "soup":
            n = "canh"
        val_by_norm[n].append(it)

    anns_by_norm = defaultdict(list)
    for a in anns:
        cname = cats[a["category_id"]]
        anns_by_norm[norm(cname)].append(a)

    # Check if meal_total_only (e.g. BunDau with All: 353.5 and item weights ?)
    if meal_total_g is not None and all(it["weight_g"] is None for it in items):
        meal_only_images.append({"image_id": iid, "file_name": im["file_name"], "meal_total_g": meal_total_g})
        for a in anns:
            cname = cats[a["category_id"]]
            vf_name = VF_BY_NORM[norm(cname)]
            dtype = pc.CLASS_TO_DISH_TYPE[vf_name]
            bx, by, bw, bh = a["bbox"]
            status_counts["meal_total_only"] += 1
            dish_type_counts[dtype] += 1
            instances_out.append({
                "annotation_id": a["id"],
                "image_id": iid,
                "file_name": im["file_name"],
                "image_path": p_str,
                "width": im["width"],
                "height": im["height"],
                "coco_category_id": a["category_id"],
                "short_class": cname,
                "vietfood67_class_name": vf_name,
                "dish_type": dtype,
                "bbox_xywh": [bx, by, bw, bh],
                "bbox_xyxy": [bx, by, bx + bw, by + bh],
                "polygon_area": float(a.get("area", 0.0)),
                "mapping_status": "meal_total_only",
                "gt_weight_g": None,
                "meal_total_g": meal_total_g,
            })
        valid_groups.add(f"meal::{iid}")
        continue

    for n, ann_list in anns_by_norm.items():
        v_list = val_by_norm.get(n, [])
        for idx_a, a in enumerate(ann_list):
            cname = cats[a["category_id"]]
            vf_name = VF_BY_NORM[norm(cname)]
            dtype = pc.CLASS_TO_DISH_TYPE[vf_name]
            bx, by, bw, bh = a["bbox"]

            if not v_list:
                status = "excluded_missing_in_val"
                gt_w = None
                mismatch_notes.append((im["file_name"], cname, "missing in val.txt"))
            elif len(ann_list) == 1 and len(v_list) == 1 and v_list[0]["count"] == 1:
                if v_list[0]["weight_g"] is not None:
                    status = "exact_1to1"
                    gt_w = v_list[0]["weight_g"]
                    valid_groups.add(f"ann::{a['id']}")
                else:
                    status = "excluded_unknown_weight"
                    gt_w = None
            elif len(v_list) == 1 and v_list[0]["count"] == len(ann_list):
                if v_list[0]["weight_g"] is not None:
                    status = "grouped_multi_instance"
                    gt_w = v_list[0]["weight_g"]
                    valid_groups.add(f"group::{iid}::{n}")
                else:
                    status = "excluded_unknown_weight"
                    gt_w = None
            elif len(v_list) == len(ann_list) and all(v["count"] == 1 for v in v_list):
                if v_list[idx_a]["weight_g"] is not None:
                    status = "exact_1to1"
                    gt_w = v_list[idx_a]["weight_g"]
                    valid_groups.add(f"ann::{a['id']}")
                else:
                    status = "excluded_unknown_weight"
                    gt_w = None
            else:
                status = "excluded_count_mismatch"
                gt_w = None
                mismatch_notes.append((im["file_name"], cname, f"ann_count={len(ann_list)} vs val={v_list}"))

            status_counts[status] += 1
            dish_type_counts[dtype] += 1
            instances_out.append({
                "annotation_id": a["id"],
                "image_id": iid,
                "file_name": im["file_name"],
                "image_path": p_str,
                "width": im["width"],
                "height": im["height"],
                "coco_category_id": a["category_id"],
                "short_class": cname,
                "vietfood67_class_name": vf_name,
                "dish_type": dtype,
                "bbox_xywh": [bx, by, bw, bh],
                "bbox_xyxy": [bx, by, bx + bw, by + bh],
                "polygon_area": float(a.get("area", 0.0)),
                "mapping_status": status,
                "gt_weight_g": gt_w,
                "meal_total_g": meal_total_g,
            })

out_obj = {
    "total_val_lines": len(val_lines),
    "matched_images_in_coco": len({x["image_id"] for x in instances_out}),
    "total_annotations": len(instances_out),
    "valid_mass_evaluation_groups": len(valid_groups),
    "status_counts": dict(status_counts),
    "dish_type_counts": dict(dish_type_counts),
    "not_in_coco_count": len(not_in_coco),
    "not_in_coco": not_in_coco,
    "instances": instances_out,
}
OUT_PATH.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Total val lines: {len(val_lines)}")
print(f"Matched COCO images: {out_obj['matched_images_in_coco']} | Annotations: {len(instances_out)}")
print(f"Valid mass groups (incl. meal_total_only): {len(valid_groups)}")
print("Status counts:", dict(status_counts))
print("Dish type counts:", dict(dish_type_counts))
print(f"Not in COCO annotated ({len(not_in_coco)}):", [Path(x).name for x in not_in_coco])
print(f"Mismatches ({len(mismatch_notes)}):", mismatch_notes[:15])
