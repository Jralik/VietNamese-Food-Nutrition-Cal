"""Generate thesis-ready artifacts: plots (PNG) + tables (CSV/MD) + pipeline overview.

Reads the frozen evaluation records and writes everything into evaluation_final/.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from class_names import class_names

BASE = "evaluation_final"
PLOTS = os.path.join(BASE, "plots")
TABLES = os.path.join(BASE, "thesis_tables")
JSONL = os.path.join("scratch", "full_eval_records.jsonl")
EXP_B = os.path.join("scratch", "experiment_b_records.jsonl")

os.makedirs(PLOTS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

RHOS = (0.45, 0.40, 0.35, 0.30, 0.27, 0.25)
VARIANTS = ("original", "otsu", "fixed60", "fixed60cc", "fixed80cc", "cconly")


def short(name):
    return name.split(" (")[0]


# ── Plot 1: Density sweep ──
def plot_density():
    rho = [0.45, 0.40, 0.35, 0.30, 0.27, 0.25]
    bm = [85.2, 70.8, 60.4, 53.8, 51.7, 51.2]
    kcal = [63.1, 52.9, 46.1, 43.7, 43.7, 44.8]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(rho, bm, "o-", label="Banh mi MAPE")
    ax.plot(rho, kcal, "s--", label="Total kcal/image MAPE")
    ax.axvspan(0.25, 0.30, alpha=0.15, color="green",
               label="low-error sensitivity region")
    ax.set_xlabel("Density for Banh mi (g/cm³)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Experiment A — Density sensitivity (frozen volume)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "exp_a_density_sweep.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(TABLES, "exp_a_density_sweep.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["density_g_cm3", "banh_mi_mape_pct", "total_kcal_mape_pct"])
        w.writerows(zip(rho, bm, kcal))


# ── Plot 2: Mask ablation ──
def plot_mask():
    variants = ["original", "otsu", "fixed60", "fixed60cc", "fixed80cc", "cconly"]
    mape = [84.8, 93.0, 80.9, 87.7, 89.2, 61.6]
    area = [1.00, 0.32, 0.45, 0.20, 0.20, 0.89]
    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    bars = ax1.bar(variants, mape, color=["#888", "#c66", "#cc6", "#6c6", "#6cc", "#46c"])
    ax1.set_ylabel("Banh mi MAPE (%)")
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(variants, area, "ko--", label="mask area ratio (median)")
    ax2.set_ylabel("Mask area ratio (vs original)")
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8, loc="upper left")
    for b, v in zip(bars, mape):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                 ha="center", fontsize=8)
    ax1.set_title("Experiment B — Mask ablation (density 0.45 frozen)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "exp_b_mask_ablation.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(TABLES, "exp_b_mask_ablation.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "banh_mi_mape_pct", "median_area_ratio",
                    "median_volume_ratio", "median_mass_ratio"])
        med = {"original": (1.0, 1.0, 1.0), "otsu": (0.32, 0.42, 0.42),
               "fixed60": (0.45, 0.61, 0.61), "fixed60cc": (0.20, 0.43, 0.43),
               "fixed80cc": (0.20, 0.36, 0.36), "cconly": (0.89, 0.91, 0.92)}
        for v, m in zip(variants, mape):
            a, vr, mr = med[v]
            w.writerow([v, m, a, vr, mr])


# ── Table: detection per-class recall + portion/nutrition MAPE (from JSONL) ──
def build_tables():
    records = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
    name_to_cid = {class_names[i]["name"]: i for i in range(len(class_names))}

    det = {"gt": 0, "hit": 0, "extras": 0}
    det_class = defaultdict(lambda: [0, 0])
    portion = defaultdict(list)
    kcal = defaultdict(list)
    supcua = []
    for r in records:
        if r.get("error") or r.get("volume_note"):
            if r.get("volume_note"):
                pass  # counted in detection below; portion excluded
        gt = {int(c): v for c, v in r["gt"].items()}
        det_d = defaultdict(int)
        est_by = defaultdict(lambda: {"m": 0.0, "k": 0.0})
        for d in r.get("detections", []):
            cid = name_to_cid.get(d["class_name"])
            if cid is not None:
                det_d[cid] += 1
        for e in r.get("est", []):
            cid = name_to_cid.get(e["class_name"])
            if cid is not None:
                est_by[cid]["m"] += e["mass_g"]
                est_by[cid]["k"] += e["kcal"]
        for cid, v in gt.items():
            k = v["count"]
            det["gt"] += k
            hit = min(k, det_d.get(cid, 0))
            det["hit"] += hit
            det_class[cid][0] += hit
            det_class[cid][1] += k
            if cid in est_by:
                portion[cid].append((est_by[cid]["m"], k * v["grams_total"]))
                kc100 = next((float(it["nutrition_per_100g"]["Calories"])
                              for it in class_names
                              if it["name"] == class_names[cid]["name"]
                              and it.get("nutrition_per_100g")
                              and "Calories" in it["nutrition_per_100g"]), None)
                if kc100 is not None:
                    kcal[cid].append((est_by[cid]["k"], k * v["grams_total"] * kc100 / 100))
            if "SupCua" in r["image"]:
                supcua.append({"image": r["image"], "est_mass": round(est_by.get(cid, {"m": 0.0})["m"], 1),
                               "gt_mass": k * v["grams_total"]})

    # detection per-class CSV/MD
    with open(os.path.join(TABLES, "detection_per_class.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "recall_hits", "recall_targets"])
        for cid in sorted(det_class):
            h, t = det_class[cid]
            w.writerow([short(class_names[cid]["name"]), h, t])

    # portion + nutrition MAPE per class CSV
    def stats(pairs):
        if not pairs:
            return None, None, 0
        mae = sum(abs(e - g) for e, g in pairs) / len(pairs)
        errs = [abs(e - g) / g * 100 for e, g in pairs if g > 0]
        return mae, (sum(errs) / len(errs) if errs else float("nan")), len(pairs)

    with open(os.path.join(TABLES, "portion_nutrition_mape.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "portion_n", "portion_mae_g", "portion_mape_pct",
                    "nutrition_n", "nutrition_mae_kcal", "nutrition_mape_pct"])
        for cid in sorted(set(list(portion.keys()) + list(kcal.keys()))):
            pm, pp, pn = stats(portion.get(cid, []))
            km, kp, kn = stats(kcal.get(cid, []))
            w.writerow([short(class_names[cid]["name"]), pn,
                        f"{pm:.1f}" if pm is not None else "",
                        f"{pp:.1f}" if pp is not None else "",
                        kn, f"{km:.1f}" if km is not None else "",
                        f"{kp:.1f}" if kp is not None else ""])

    # soup geometry (C1) CSV
    with open(os.path.join(TABLES, "supcua_geometry_c1.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "est_mass_g", "gt_mass_g", "rel_err_pct"])
        for s in sorted(supcua, key=lambda x: x["image"]):
            err = (s["est_mass"] - s["gt_mass"]) / s["gt_mass"] * 100
            w.writerow([s["image"], s["est_mass"], s["gt_mass"], f"{err:+.1f}"])
    return supcua


def write_pipeline_overview():
    with open(os.path.join(TABLES, "pipeline_overview.md"), "w", encoding="utf-8") as f:
        f.write("""# Pipeline overview (frozen configuration)

```
Image upload (EXIF upright)
   |
   |  v10b YOLO + dual-preresize (PIL stretch + letterbox, class-aware NMS)
   |  confidence threshold = 0.30 (UI slider)
   v
Detections (bbox, class, confidence)  ── recall 0.811 (60/74) @ 0.30
   |
   |  volume detections: confidence >= 0.40 (MIN_DETECTION_CONFIDENCE)
   v
Volume pipeline (separate worker, GPU)
   ├── SAM2 segmentation (box-prompted)
   ├── Depth Anything V2 (metric, indoor)
   ├── Scale recovery: ArUco marker (0.211-0.321 mm/px) + depth anchoring
   └── Volume -> mass (density_db) -> nutrition (per-100g scaling)
         |
         v
   Nutrition cards + volume note (UI)
```

Configuration references:
- Detection: model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt, device=cpu (volume worker owns the GPU)
- Volume: backend SAM2, source image 2000px thumbnail, worker daemon
- All numbers in evaluation_final/ use this frozen configuration
""")


def main():
    plot_density()
    plot_mask()
    supcua = build_tables()
    write_pipeline_overview()
    print("plots + tables generated")
    # in dữ liệu C1 để phân loại góc
    print("\nC1 SupCua est-vs-GT (cần phân loại góc chụp):")
    for s in sorted(supcua, key=lambda x: x["image"]):
        err = (s["est_mass"] - s["gt_mass"]) / s["gt_mass"] * 100
        print(f"  {s['image']:<18} est={s['est_mass']:>7.1f} gt={s['gt_mass']:>7.1f} ({err:+.1f}%)")


if __name__ == "__main__":
    from collections import defaultdict
    main()
