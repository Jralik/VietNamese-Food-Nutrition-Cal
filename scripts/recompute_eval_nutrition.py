"""
scripts/recompute_eval_nutrition.py
===================================
Recomputes the NUTRITION fields of the stored portion-estimation results using
the post-freeze, source-aligned per-100g values (2026-09-28 realignment).

Why offline recompute is exact here:
  - Portion (mass) results are UNCHANGED — no density value was modified, so
    volume_cm3 / mass_g / MAE / MAPE-g / RMSE all stay as measured.
  - kcal per estimation was computed at eval time as mass × per-100g-Calories/100
    (eval_full_pipeline.py:191). With the aligned per-100g values in
    class_names.py, the same formula reproduces what the pipeline would output
    today — no GPU re-run needed.

Updates (each with a .bak_pre_nutrition_realign_20260928 backup):
  1. scratch/full_eval_records.jsonl          — est[].kcal (Chain A, 54 ảnh)
  2. evaluation_final/records/full_eval_records.jsonl — same export copy
  3. data/benchmark_portion_results.json      — detected_items[].calories +
      image pred_nutrition (12 nutrients) (Chain B)

Then re-run scripts/summarize_full_eval.py to regenerate the report MD from
the updated records (its GT-kcal side already reads current class_names).

Usage: python scripts/recompute_eval_nutrition.py [--apply]  (default dry run)
"""

import os
import sys
import json
import shutil

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

BAK_SUFFIX = ".bak_pre_nutrition_realign_20260928"
APPLY = "--apply" in sys.argv

CHAIN_A_PATHS = [
    os.path.join(ROOT_DIR, "scratch", "full_eval_records.jsonl"),
    os.path.join(ROOT_DIR, "evaluation_final", "records", "full_eval_records.jsonl"),
]
CHAIN_B_PATH = os.path.join(ROOT_DIR, "data", "benchmark_portion_results.json")

from class_names import class_names
from density_db import estimate_nutrition

KCAL_BY_NAME = {it["name"]: float(it["nutrition_per_100g"]["Calories"])
                for it in class_names}


def chain_a():
    for path in CHAIN_A_PATHS:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

        total_items = changed = 0
        for r in records:
            for e in r.get("est") or []:
                cn = e.get("class_name")
                kc = KCAL_BY_NAME.get(cn)
                if kc is None or e.get("mass_g") is None:
                    continue
                new_kcal = round(float(e["mass_g"]) * kc / 100.0, 1)
                if abs(new_kcal - float(e.get("kcal", 0.0))) > 1e-9:
                    changed += 1
                e["kcal"] = new_kcal
                total_items += 1

        if APPLY:
            bak = path + BAK_SUFFIX
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Chain A] {os.path.relpath(path, ROOT_DIR)}: "
              f"{total_items} est items scanned, {changed} kcal changed "
              f"({'WRITTEN' if APPLY else 'dry run'})")


def chain_b():
    with open(CHAIN_B_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    results = doc.get("image_results") or []
    items = imgs = 0
    changed = 0
    for res in results:
        dets = res.get("detected_items") or []
        new_total = {}
        for d in dets:
            cn, mass = d.get("class"), d.get("mass_g")
            if cn is None or mass is None:
                continue
            nut = estimate_nutrition(float(mass), cn)
            new_kcal = nut.get("Calories", 0.0)
            if abs(new_kcal - float(d.get("calories", 0.0))) > 1e-9:
                changed += 1
            d["calories"] = new_kcal
            for k, v in nut.items():
                new_total[k] = round(new_total.get(k, 0.0) + float(v), 1)
            items += 1
        res["pred_nutrition"] = new_total
        imgs += 1

    if APPLY:
        bak = CHAIN_B_PATH + BAK_SUFFIX
        if not os.path.exists(bak):
            shutil.copy2(CHAIN_B_PATH, bak)
        doc.setdefault("metadata", {})["nutrition_recomputed_at"] = \
            "2026-09-28 — per-100g source-aligned (post-freeze realignment); mass results unchanged"
        with open(CHAIN_B_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[Chain B] benchmark_portion_results.json: {imgs} ảnh, "
          f"{items} detected items, {changed} calories changed "
          f"({'WRITTEN' if APPLY else 'dry run'})")


if __name__ == "__main__":
    print(f"=== Recompute eval nutrition ({'APPLY' if APPLY else 'DRY RUN'}) ===")
    chain_a()
    chain_b()
    if APPLY:
        print("\nNext: python scripts/summarize_full_eval.py  (regenerate the report MD — "
              "its GT-kcal side already reads the current class_names)")
