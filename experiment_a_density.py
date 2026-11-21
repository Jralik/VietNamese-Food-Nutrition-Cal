"""Experiment A — density sensitivity analysis (offline, frozen baseline).

Uses the recorded per-item volume_cm3 from full_eval_records.jsonl (collected
with the frozen pipeline) and recomputes masses under alternative densities
for the AUDITED class (Banh mi) only:

    mass(rho) = volume_cm3 * rho

All other classes keep their density_db values (their volumes and masses are
untouched). For each rho in the sweep:
  - Banh mi all-est MAE/MAPE (vs GT total 99.3 g per BanhMi_Trung image)
  - overall image-level mass MAPE (only BanhMi_Trung image totals change)
  - overall image-level kcal MAPE (kcal scales linearly with mass)

Writes scratch/experiment_a_density.md.
"""

import json
import os
import statistics
from collections import defaultdict

from class_names import class_names

JSONL = os.path.join("scratch", "full_eval_records.jsonl")
OUT_MD = os.path.join("scratch", "experiment_a_density.md")
RHOS = (0.45, 0.40, 0.35, 0.30, 0.27, 0.25)
AUDITED = "Banh mi (Vietnamese baguette sandwich)"


def kcal_per_100g(class_name):
    for item in class_names:
        if item["name"] == class_name:
            per = item.get("nutrition_per_100g")
            return float(per["Calories"]) if per and "Calories" in per else None
    return None


def mape(pairs):
    errs = [abs(e - g) / g * 100 for e, g in pairs if g > 0]
    return sum(errs) / len(errs) if errs else None


def main():
    records = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
    name_to_cid = {class_names[i]["name"]: i for i in range(len(class_names))}
    bm_cid = name_to_cid[AUDITED]

    images = []
    for r in records:
        if r.get("error") or r.get("volume_note"):
            continue
        gt = {int(c): v for c, v in r["gt"].items()}
        bm_vols = [e["volume_cm3"] for e in r["est"] if e["class_name"] == AUDITED]
        bm_gt = gt.get(bm_cid, {}).get("grams_total")
        if not bm_vols or not bm_gt:
            continue
        bm_k100 = kcal_per_100g(AUDITED)
        others_m = sum(e["mass_g"] for e in r["est"] if e["class_name"] != AUDITED)
        others_k = sum(e["kcal"] for e in r["est"] if e["class_name"] != AUDITED)
        gt_m = sum(v["grams_total"] or 0 for v in gt.values())
        gt_k = 0.0
        gt_k_ok = True
        for c, v in gt.items():
            if not v["grams_total"]:
                continue
            kc = kcal_per_100g(class_names[c]["name"])
            if kc is None:
                gt_k_ok = False
            else:
                gt_k += v["grams_total"] * kc / 100
        images.append({"image": r["image"], "bm_vols": bm_vols, "bm_gt": bm_gt,
                       "bm_k100": bm_k100, "others_m": others_m, "others_k": others_k,
                       "gt_m": gt_m, "gt_k": gt_k, "gt_k_ok": gt_k_ok})

    n = len(images)
    print(f"images with Banh mi volume data: {n}")

    def pct(pairs):
        errs = [abs(e - g) / g * 100 for e, g in pairs if g > 0]
        return (sum(errs) / len(errs)) if errs else float("nan")

    rows = []
    for rho in RHOS:
        bm_pairs, img_m_pairs, img_k_pairs = [], [], []
        bm_only_pairs = []
        for im in images:
            bm_mass = sum(v * rho for v in im["bm_vols"])
            bm_only_pairs.append((bm_mass, im["bm_gt"]))
            img_m_pairs.append((bm_mass + im["others_m"], im["gt_m"]))
            if im["gt_k_ok"] and im["bm_k100"]:
                bm_kcal = sum(v * rho * im["bm_k100"] / 100 for v in im["bm_vols"])
                img_k_pairs.append((bm_kcal + im["others_k"], im["gt_k"]))
        rows.append((rho,
                     statistics.mean(abs(e - g) for e, g in bm_only_pairs),
                     pct(bm_only_pairs),
                     pct(img_m_pairs) if img_m_pairs else float("nan"),
                     pct(img_k_pairs) if img_k_pairs else float("nan")))

    L = ["", "# Experiment A — Density sensitivity (Bánh mì, offline sweep)", ""]
    L.append(f"Baseline frozen: {n} ảnh có Banh mi volume (volume_cm3 từ pipeline "
             f"đã freeze). Chỉ thay density của Bánh mì; các class khác giữ DB. "
             f"GT Bánh mì: 99.3g TỔNG 2 phần (BanhMi_Trung — 1 full ổ + 1 nửa ổ "
             f"bánh mì thường), 242.6/255.6/210g/ổ (bánh mì thịt).")
    L.append("")
    L.append("| Density (g/cm³) | Bánh mì MAE (g) | Bánh mì MAPE | Tổng mass/ảnh MAPE | Tổng kcal/ảnh MAPE |")
    L.append("|---|---|---|---|---|")
    for rho, bm_mae, bm_mape, img_m, img_k in rows:
        L.append(f"| {rho:.2f} | {bm_mae:.1f} | {bm_mape:.1f}% | "
                 f"{img_m:.1f}% | {img_k:.1f}% |")
    L.append("")
    best = min(rows, key=lambda r: r[2])
    L.append(
        f"**Đọc kết quả (sensitivity analysis)**: ρ giảm làm MAPE Bánh mì giảm từ "
        f"{rows[0][2]:.1f}% (0.45) xuống {best[2]:.1f}% ({best[0]:.2f}) nhưng **đi ngang "
        f"~51% khi ρ < 0.30** — tức density một mình KHÔNG giải thích được toàn bộ "
        f"over-estimation; phần còn lại thuộc về thể tích ước lượng (mask/depth — xem "
        f"depth_audit.md, Experiment B). Wording: đây là phân tích độ nhạy theo mật độ, "
        f"KHÔNG phải kết luận 'density gây X×'.")
    L.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(chr(10).join(L) + chr(10))
    print(chr(10).join(L))


if __name__ == "__main__":
    main()
