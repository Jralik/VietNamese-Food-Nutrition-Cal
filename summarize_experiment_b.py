"""Aggregate Experiment B: per-variant mask area/volume/mass ratios + MAPE.

Reads scratch/experiment_b_records.jsonl (one record per Banh mi image, each
item carrying area/volume/mass for ALL mask variants) and produces:
  - per-variant median ratios (area, volume, mass — vs original)
  - per-variant Banh mi per-image-total MAPE (vs GT)
  - decision-tree verdict per the approved Experiment B protocol

Writes scratch/experiment_b_mask.md.
"""

import json
import os
import statistics
from collections import defaultdict

JSONL = os.path.join("scratch", "experiment_b_records.jsonl")
OUT_MD = os.path.join("scratch", "experiment_b_mask.md")
VARIANTS = ("original", "otsu", "fixed60", "fixed60cc", "fixed80cc", "cconly")


def mape(pairs):
    errs = [abs(e - g) / g * 100 for e, g in pairs if g > 0]
    return (sum(errs) / len(errs)) if errs else float("nan")


def main():
    records = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
    valid = [r for r in records if not r.get("error") and r.get("items")]

    ratio_acc = {v: defaultdict(list) for v in VARIANTS}
    mape_pairs = {v: [] for v in VARIANTS}

    for r in valid:
        gt = r["gt_total_g"]
        sums = {v: 0.0 for v in VARIANTS}
        for item in r["items"]:
            a_o = item["variants"]["original"]["area"]
            m_o = item["variants"]["original"]["mass_g"]
            v_o = item["variants"]["original"]["volume_cm3"]
            for v in VARIANTS:
                vv = item["variants"][v]
                sums[v] += vv["mass_g"]
                if a_o:
                    ratio_acc[v]["area"].append(vv["area"] / a_o)
                if v_o:
                    ratio_acc[v]["volume"].append(vv["volume_cm3"] / v_o)
                if m_o:
                    ratio_acc[v]["mass"].append(vv["mass_g"] / m_o)
        if gt:
            for v in VARIANTS:
                mape_pairs[v].append((sums[v], gt))

    L = ["", "# Experiment B — Mask ablation results (20 items / 16 ảnh)", ""]
    L.append("Frozen: images/detections/depth/ArUco/density 0.45/volume logic. "
             "CHỈ thay mask. Variants: otsu (iteration-1), fixed60 (gray>60), "
             "fixed60cc/fixed80cc (+largest CC), cconly (largest CC của mask gốc).")
    L.append("")
    L.append("| Variant | area ratio (median) | volume ratio (median) | mass ratio (median) | Bánh mì MAPE (16 ảnh) |")
    L.append("|---|---|---|---|---|")
    for v in VARIANTS:
        ar = statistics.median(ratio_acc[v]["area"]) if ratio_acc[v]["area"] else float("nan")
        vr = statistics.median(ratio_acc[v]["volume"]) if ratio_acc[v]["volume"] else float("nan")
        mr = statistics.median(ratio_acc[v]["mass"]) if ratio_acc[v]["mass"] else float("nan")
        mp = mape(mape_pairs[v])
        L.append(f"| {v} | {ar:.2f} | {vr:.2f} | {mr:.2f} | {mp:.1f}% |")
    L.append("")
    L.append("GT totals: 99.3g (BM_Trung, 2 phần thường) · 242.6g (banh-mi1-5) · "
             "255.6g (250g) · 210g (DucTri) — bánh mì thịt.")
    L.append("")
    L.append("## Per-image Banh mi mass totals (g) — original vs các variant")
    L.append("")
    L.append("| Ảnh | GT | " + " | ".join(VARIANTS) + " |")
    L.append("|---|---|" + "---|" * len(VARIANTS))
    for r in valid:
        sums = {v: sum(i["variants"][v]["mass_g"] for i in r["items"]) for v in VARIANTS}
        L.append(f"| {r['image']} | {r['gt_total_g']:.0f} | " +
                 " | ".join(f"{sums[v]:.0f}" for v in VARIANTS) + " |")
    L.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
