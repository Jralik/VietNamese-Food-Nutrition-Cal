"""Dev-set threshold tuning driver (FOODSAM ingredient feature).

Runs the pipeline on the DEV split (splits/dev.txt) once per threshold
variant, in-process, and emits a comparison table + dev_tuning.json.

Tuning discipline (thesis):
  - run this driver ONLY on the dev split; pick the winner; write the chosen
    values into pipeline_config.py (that IS the freeze); then run the eval
    split with the frozen config. Never tune on eval.

Metrics per variant (no ground truth — sanity + recovery, NOT accuracy):
  done / failed   images processed (fail-fast aborts a variant on fallback)
  extras          total accepted ingredient items
  zero_yolo       dev images where YOLO found no dish
  recovered       zero_yolo images with >= 1 accepted extra
  extras/img      mean accepted extras per image (exploding = over-extraction)
  egg_mass_avg    mean mass per "Trung (Egg)" item (plausibility ~20-150 g)

Usage:
    .venv/Scripts/python.exe run_dev_tuning.py            # default variants
    .venv/Scripts/python.exe run_dev_tuning.py --limit 6  # quick smoke
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pipeline_config as pc
import run_batch_estimation as rb

WORK_DIR = rb.WORK_DIR
DEV_LIST = os.path.join(WORK_DIR, "splits", "dev.txt")

# Sensitivity grid (thesis: threshold chosen experimentally on dev).
# Add/remove entries as needed; keys must be pipeline_config attribute names.
VARIANTS = [
    ("overlap_0.3", {"FOODSAM_INGREDIENT_OVERLAP_BREAKDOWN": 0.3}),
    ("overlap_0.5", {"FOODSAM_INGREDIENT_OVERLAP_BREAKDOWN": 0.5}),
    ("overlap_0.7", {"FOODSAM_INGREDIENT_OVERLAP_BREAKDOWN": 0.7}),
]


def dev_images():
    with open(DEV_LIST, encoding="utf-8") as f:
        rels = [ln.strip() for ln in f if ln.strip()]
    return [os.path.join(rb.TEST_IMAGE_DIR, rel) for rel in rels], rels


def variant_metrics(rels):
    """Aggregate per-image results currently on disk for the dev split."""
    m = {"done": 0, "failed": 0, "extras": 0, "zero_yolo": 0, "recovered": 0,
         "egg_masses": [], "duplicates": 0, "wall_s": [],
         "suppressed": 0, "promotions": 0, "coverage": []}
    for rel in rels:
        path = os.path.join(WORK_DIR, "results", "foodsam", rel + ".json")
        if not os.path.exists(path):
            m["failed"] += 1
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        m["done"] += 1
        if data.get("perf", {}).get("wall_s"):
            m["wall_s"].append(data["perf"]["wall_s"])
        items = data.get("items", [])
        ing = [i for i in items if i.get("is_ingredient")]
        dish = [i for i in items if i.get("source", "yolo") == "yolo"]
        m["extras"] += len(ing)
        sup_list = data.get("suppressed_dishes", [])
        if sup_list:
            m["suppressed"] += len(sup_list)
            m["promotions"] += sum(
                len(s.get("promoted_labels", [])) for s in sup_list)
            covs = [s["promoted_area_coverage"] for s in sup_list
                    if s.get("promoted_area_coverage") is not None]
            if covs:
                m["coverage"].append(sum(covs) / len(covs))
        if not dish:
            m["zero_yolo"] += 1
            if ing:
                m["recovered"] += 1
        for i in ing:
            if i["class_name"] == "Trung (Egg)":
                m["egg_masses"].append(i["mass_g"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Trim the dev list for a quick smoke of the driver.")
    args_cli = ap.parse_args()

    if not os.path.exists(DEV_LIST):
        sys.exit(f"missing {DEV_LIST} — run: "
                 f"python run_batch_estimation.py --stage split")

    images, rels = dev_images()
    if args_cli.limit:
        images, rels = images[:args_cli.limit], rels[:args_cli.limit]
    print(f"dev tuning on {len(images)} images | variants: "
          f"{[name for name, _ in VARIANTS]}")

    originals = {k: getattr(pc, k) for _, overrides in VARIANTS for k in overrides}
    table = {}
    try:
        for name, overrides in VARIANTS:
            for k, v in overrides.items():
                setattr(pc, k, v)
            print(f"\n===== variant {name}: {overrides} =====")
            args = SimpleNamespace(
                stage="pipeline", conf=rb.CONF_DEFAULT, seg_backend="foodsam",
                img_dir=None, images_list=None, limit=None, force=True)
            # run exactly the dev images through the real batch stage
            rb_args = args
            failed_variant = False
            for img, rel in zip(images, rels):
                res_path = rb.res_file_for(img, "foodsam")
                if os.path.exists(res_path):
                    os.remove(res_path)  # force overwrite per variant
            try:
                # stage_pipeline loops collect_images(); feed it the dev list
                # by patching the collector for this call
                orig_collect = rb.collect_images
                rb.collect_images = lambda *a, **k: images
                try:
                    rb.stage_pipeline(rb_args)
                finally:
                    rb.collect_images = orig_collect
            except RuntimeError as e:
                # fail-fast: a fallback invalidates the variant
                print(f"[variant {name}] FAIL-FAST: {e}")
                failed_variant = True

            m = variant_metrics(rels)
            m["failed_variant"] = failed_variant
            m["extras_per_img"] = (round(m["extras"] / m["done"], 2)
                                   if m["done"] else None)
            m["recovery_rate"] = (round(m["recovered"] / m["zero_yolo"], 3)
                                  if m["zero_yolo"] else None)
            m["egg_mass_avg"] = (round(sum(m["egg_masses"]) / len(m["egg_masses"]), 1)
                                 if m["egg_masses"] else None)
            m["avg_wall_s"] = (round(sum(m["wall_s"]) / len(m["wall_s"]), 1)
                               if m["wall_s"] else None)
            m["suppressed"] = m["suppressed"]
            m["promoted_area_coverage_mean"] = (
                round(sum(m["coverage"]) / len(m["coverage"]), 3)
                if m["coverage"] else None)
            table[name] = m
            print(f"[variant {name}] done={m['done']} failed={m['failed']} "
                  f"extras={m['extras']} ({m['extras_per_img']}/img) "
                  f"zero_yolo={m['zero_yolo']} recovered={m['recovered']} "
                  f"egg_mass_avg={m['egg_mass_avg']} "
                  f"suppressed={m['suppressed']} "
                  f"coverage={m['promoted_area_coverage_mean']} "
                  f"avg_wall={m['avg_wall_s']}s")
    finally:
        for k, v in originals.items():
            setattr(pc, k, v)
        print("\n[config restored to pre-tuning values]")

    out_path = os.path.join(WORK_DIR, "dev_tuning.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {out_path}")
    print("choose the winner, write its values into pipeline_config.py "
          "(= freeze), then run the eval split.")


if __name__ == "__main__":
    main()
