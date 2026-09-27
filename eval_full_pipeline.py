"""Full-pipeline evaluation on portion-estimation_val.txt (54 images).

For every image, runs the EXACT app path:
  1. Detection : v10b + predict_with_dual_preresize @ conf=0.30 (app default)
  2. Portion   : volume_integration.estimate_nutrition_volume (SAM2 backend,
                 ArUco scale, 2000px source, daemon worker)
  3. Nutrition : per-item Calories from the estimated mass

Ground truth per image: {class: {box_count, grams_total, mass_semantics}} parsed from the val
file. Metrics (frozen definitions carried over from Phase D + new):
  Detection : greedy one-to-one per class (matched = min(n_gt, n_det)),
              recall, FP/extras.
  Portion   : for classes present in BOTH est and GT — est total mass vs
              GT total mass (grams_total, user-confirmed semantics): abs/rel error per class,
              MAE/MAPE aggregated; also per-image total-mass error.
  Nutrition : estimated Calories vs GT Calories (gt_mass x kcal/100g from
              class_names nutrition_per_100g); MAE/MAPE; coverage reported
              (classes lacking nutrition_per_100g are skipped).

Writes one JSON line per image to scratch/full_eval_records.jsonl as it goes,
then aggregates into scratch/full_pipeline_eval.md.
"""

import json
import os
import re
import time

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)

from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from eval_val_detection import VAL_FILE, SHORT_TO_ID, TOKEN_ALIASES
from utils import predict_with_dual_preresize
import volume_integration

ONNX = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
OUT_MD = os.path.join("scratch", "full_pipeline_eval.md")
OUT_JSONL = os.path.join("scratch", "full_eval_records.jsonl")
CONF = 0.30
BACKEND = "sam2"

TOKEN_ALIASES = dict(TOKEN_ALIASES)


def parse_gt(path):
    """Returns list of (img_path, {class_id: gt_dict}).

    gt_dict = {"count": int, "grams_total": float | None,
               "mass_semantics": "total" | "unknown" | "partial"}

    Formats (user-confirmed 2026-09-27):
      "Name: K - G (tong)" -> K boxes, G = TOTAL mass of all K boxes
      "Name: K - G"        -> K boxes, G = TOTAL mass (K=1 equivalent)
      "K - Name G"         -> K boxes, G = TOTAL mass
      "Name: G"            -> 1 box, G = total mass
      "Name: K - ?"        -> K boxes, count-only (no mass GT)
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^"(.+?)",\s*(.+?)\s*;?\s*$', line)
            if not m:
                continue
            img_path, rest = m.group(1), m.group(2)
            gt = {}
            for part in rest.split(","):
                part = part.strip()
                name = k = grams = None
                unknown = False
                m1 = re.match(
                    r"^([A-Za-z][A-Za-z ]*?)\s*:\s*(\d+)\s*-\s*([\d.]+)\s*g?"
                    r"(?:\s*\(tong\))?$", part)
                m2 = re.match(r"^(\d+)\s*-\s*([A-Za-z][A-Za-z ]*?)\s+([\d.]+)\s*g?$", part)
                m3 = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*:\s*([\d.]+)\s*g?$", part)
                mq = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*:\s*(\d+)\s*-\s*\?$", part)
                mq2 = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*:\s*\?$", part)
                if m1:
                    name, k, grams = m1.group(1).strip(), int(m1.group(2)), float(m1.group(3))
                elif m2:
                    name, k, grams = m2.group(2).strip(), int(m2.group(1)), float(m2.group(3))
                elif m3:
                    name, k, grams = m3.group(1).strip(), 1, float(m3.group(2))
                elif mq:
                    name, k, unknown = mq.group(1).strip(), int(mq.group(2)), True
                elif mq2:
                    name, k, unknown = mq2.group(1).strip(), 1, True
                if not name:
                    continue
                name = TOKEN_ALIASES.get(name, name)
                cid = SHORT_TO_ID.get(name)
                if cid is not None:
                    prev = gt.get(cid, {"count": 0, "grams_total": 0.0,
                                        "mass_semantics": "total"})
                    count = prev["count"] + k
                    if unknown:
                        grams_total = prev["grams_total"]
                        semantics = "partial" if prev["grams_total"] else "unknown"
                    else:
                        grams_total = prev["grams_total"] + grams
                        semantics = "partial" if prev["grams_total"] is not None and \
                            prev["count"] > 0 and unknown else "total"
                    gt[cid] = {"count": count, "grams_total": grams_total,
                               "mass_semantics": semantics}
            entries.append((img_path, gt))
    return entries


def kcal_per_100g(cid):
    item = class_names[cid]
    per = item.get("nutrition_per_100g") if isinstance(item, dict) else None
    if not per or "Calories" not in per:
        return None
    return float(per["Calories"])


def main():
    entries = parse_gt(VAL_FILE)
    model = YOLO(ONNX)
    model.overrides["device"] = "cpu"
    records = []
    t0 = time.time()

    # OneDrive hydration: cloud-only placeholders transiently fail
    # os.path.exists/open during collection — force-download once up front.
    for img_path, _gt in entries:
        for attempt in range(2):
            try:
                with open(img_path, "rb") as f:
                    f.read(4)
                break
            except Exception:
                if attempt == 0:
                    print(f"hydrating {os.path.basename(img_path)}...", flush=True)
                    time.sleep(5)

    for idx, (img_path, gt) in enumerate(entries):
        fname = os.path.basename(img_path)
        rec = {"image": fname, "group": img_path.split("\\")[-2], "gt": {}, "est": [],
               "detections": [], "scale_source": "", "worker_s": None, "error": None}
        for cid, v in gt.items():
            rec["gt"][str(cid)] = dict(v)
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)
            upright = ImageOps.exif_transpose(Image.open(img_path))
            w_up, h_up = upright.size
            volume_source = upright.copy()
            volume_source.thumbnail((2000, 2000))
            bbox_scale = (volume_source.width / w_up, volume_source.height / h_up)

            t_d = time.time()
            res = predict_with_dual_preresize(model, upright, CONF)[0]
            rec["detect_s"] = round(time.time() - t_d, 2)

            dets = volume_integration.extract_yolo_detections(
                [res], class_names, bbox_scale=bbox_scale)
            rec["detections"] = [
                {"class_name": d["class_name"], "conf": round(d["confidence"], 3)}
                for d in dets]

            # record the DISPLAY detections (>= 0.30 slider) — the volume
            # path applies its own 0.40 floor below
            rec["detections"] = [
                {"class_name": class_names[int(b.cls[0].item())]["name"],
                 "conf": round(float(b.conf[0].item()), 3)} for b in res.boxes]

            vol = volume_integration.estimate_nutrition_volume(
                volume_source, dets, backend=BACKEND)
            if vol is None:
                # Record and FALL THROUGH to the JSONL write + print — a bare
                # continue here silently dropped the record (SupCua1 (4) hit
                # this: its only detection is 0.338 < the 0.40 volume floor).
                rec["volume_note"] = ("volume skipped: no detections above "
                                      "MIN_DETECTION_CONFIDENCE=0.40"
                                      if not dets else
                                      "volume pipeline returned None")
            else:
                rec["scale_source"] = vol.scale_result.scale_source
                rec["worker_s"] = round(vol.worker_elapsed_s, 2)
                for e in vol.estimations:
                    rec["est"].append({
                        "class_name": e.class_name,
                        "mass_g": round(float(e.mass_g), 1),
                        "volume_cm3": round(float(e.volume_cm3), 1),
                        "kcal": round(float(e.nutrition.get("Calories", 0.0)), 1),
                        "method": e.estimation_method,
                    })
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        records.append(rec)
        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        err = f" ERROR={rec['error']}" if rec.get("error") else ""
        print(f"[{idx + 1}/{len(entries)}] {fname}: "
              f"{len(rec['detections'])} det, {len(rec['est'])} est{err}", flush=True)

    print(f"collection done in {time.time() - t0:.0f}s -> {OUT_JSONL}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    if os.path.exists(OUT_JSONL):
        os.remove(OUT_JSONL)
    main()
