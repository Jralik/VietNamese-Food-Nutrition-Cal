"""Post-freeze ON/OFF impact evaluation of cross-class duplicate suppression.

Same protocol for BOTH runs — only the suppression flag differs:
    model YOLOv10b PT, conf 0.30, dual-preresize, class-aware NMS 0.55,
    val set portion-estimation_val.txt, GT parser eval_full_pipeline.parse_gt.

OFF run: predict_with_dual_preresize(..., suppress_cross_class=False)
ON run:  predict_with_dual_preresize(...)  # pipeline_config default (0.70)

Metric (instance level, mirrors the frozen greedy one-to-one protocol):
per class, matched = min(n_gt, n_det); recall = sum(matched)/sum(n_gt);
precision = sum(matched)/sum(n_det).

Output: scratch/post_freeze_on_off_records.jsonl
Run:  .venv/Scripts/python.exe scripts/eval_post_freeze_on_off.py
"""
import json
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.stdout.reconfigure(encoding="utf-8")

from eval_val_detection import VAL_FILE, parse_val_file
from utils import load_model, predict_with_dual_preresize

CONF = 0.30
OUT_JSONL = os.path.join(ROOT_DIR, "scratch", "post_freeze_on_off_records.jsonl")


def detect(model, pil, suppress):
    result = predict_with_dual_preresize(
        model, pil, CONF, suppress_cross_class=suppress)[0]
    dets = {}
    for b in result.boxes:
        cid = int(b.cls[0].item())
        dets[cid] = dets.get(cid, 0) + 1
    return dets, list(getattr(result, "suppressed_cross_class", []) or [])


def main():
    entries = parse_val_file(os.path.join(ROOT_DIR, VAL_FILE))
    model = load_model()
    out = []
    t0 = time.time()
    for mode, suppress in (("OFF", False), ("ON", None)):
        n_gt_total = n_det_total = matched_total = 0
        n_suppressed = n_images = 0
        for img_path, gt_counts in entries:
            if not os.path.exists(img_path):
                continue
            from PIL import Image, ImageOps
            pil = ImageOps.exif_transpose(Image.open(img_path))
            dets, sup = detect(model, pil, suppress)
            n_images += 1
            n_suppressed += len(sup)
            for cid, n_gt in gt_counts.items():
                n_gt_total += n_gt
                matched_total += min(n_gt, dets.get(cid, 0))
            n_det_total += sum(dets.values())
        recall = matched_total / n_gt_total if n_gt_total else 0.0
        precision = matched_total / n_det_total if n_det_total else 0.0
        rec = {
            "mode": mode, "images": n_images,
            "gt_instances": n_gt_total, "det_instances": n_det_total,
            "matched_instances": matched_total,
            "recall": round(recall, 4), "precision": round(precision, 4),
            "suppressed_cross_class": n_suppressed,
            "elapsed_s": round(time.time() - t0, 1),
        }
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
