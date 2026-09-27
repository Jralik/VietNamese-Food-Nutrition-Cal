"""Model comparison on the 54-image portion-estimation val set:

  v26m-dual    : yolov26m best.onnx + utils.predict_with_dual_preresize
                 (the CURRENT app configuration)
  v10b-dual    : yolov10b .pt + the SAME dual-preresize config
  v10b-stretch : yolov10b .pt + PIL stretch (the reference project pipeline)

All three run at conf=0.05 with thresholds applied locally (end2end
postprocess is a pure conf filter), so recall@0.50/@0.30 and the near-miss
confidence of every missed expected class come from the same single pass.

Writes scratch/model_compare.md.
"""

import os
import time

import av  # noqa: F401  (venv DLL quirk guard: must precede ultralytics)

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from class_names import class_names
from eval_val_detection import parse_val_file, pre_stretch, VAL_FILE
from utils import predict_with_dual_preresize

ONNX26 = "./model/yolov26/best.onnx"
PT10B = "./model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
OUT_PATH = os.path.join("scratch", "model_compare.md")

CONFIGS = ("v26m-dual (app hien tai)", "v10b-dual", "v10b-stretch (ref pipeline)")


def predict_local(model, src):
    t0 = time.time()
    r = model.predict(src, conf=0.05, imgsz=640, half=False, verbose=False)[0]
    ms = (time.time() - t0) * 1000
    data = r.boxes.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    return data, ms


def run_config(name, model, pil):
    if name.startswith("v10b-stretch"):
        return predict_local(model, pre_stretch(pil))
    # dual-preresize via the app's actual helper (works for both models)
    t0 = time.time()
    res = predict_with_dual_preresize(model, pil, conf=0.05)[0]
    ms = (time.time() - t0) * 1000
    data = res.boxes.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    return data, ms


def main():
    entries = parse_val_file(VAL_FILE)
    m26 = YOLO(ONNX26, task="detect")
    m10 = YOLO(PT10B)
    # .pt checkpoints carry a baked-in cuda device; force CPU so the eval is
    # reproducible regardless of GPU state (reference app also runs CPU)
    for _m in (m26, m10):
        try:
            _m.overrides["device"] = "cpu"
        except Exception:
            pass
    print("v10b names spot-check:",
          {i: m10.names[i] for i in (4, 56, 67)},
          "| num classes:", len(m10.names))

    results = {c: [] for c in CONFIGS}
    for idx, (img_path, expected) in enumerate(entries):
        if not os.path.exists(img_path):
            continue
        pil = ImageOps.exif_transpose(Image.open(img_path))
        for name, model in (("v26m-dual (app hien tai)", m26),
                            ("v10b-dual", m10),
                            ("v10b-stretch (ref pipeline)", m10)):
            data, ms = run_config(name, model, pil)
            cls050 = {int(b[5]) for b in data if b[4] >= 0.50}
            cls030 = {int(b[5]) for b in data if b[4] >= 0.30}
            n50 = int(sum(1 for b in data if b[4] >= 0.50))
            # near-miss: expected classes missed @0.50 -> their max conf here
            near = {}
            for c in expected - cls050:
                m = max((float(b[4]) for b in data if int(b[5]) == c), default=0.0)
                near[c] = m
            results[name].append((img_path, expected, cls050, cls030, n50, ms, near))
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(entries)}")

    lines = ["", "# Model comparison — v26m (app) vs yolov10b, 54 anh val", ""]
    lines.append("| Config | Recall@0.50 | Recall@0.30 | Anh 0 box @0.50 | ms/anh (CPU) |")
    lines.append("|---|---|---|---|---|")
    for name in CONFIGS:
        rows = [r for r in results[name] if r[1]]
        rec5 = sum(len(e & d) / len(e) for _, e, d, _, _, _, _ in rows) / len(rows)
        rec3 = sum(len(e & c) / len(e) for _, e, c, _, _, _, _ in rows) / len(rows)
        emp = sum(1 for _, _, _, _, nb, _, _ in results[name] if nb == 0)
        avg_ms = sum(r[5] for r in results[name]) / len(results[name])
        lines.append(f"| {name} | {rec5:.3f} | {rec3:.3f} | {emp}/{len(results[name])} | {avg_ms:.0f} |")

    # per-group recall
    import re
    def group_key(p):
        m = re.search(r"test-image[\\/]([^\\/]+)[\\/]", p)
        return m.group(1) if m else os.path.basename(p)

    lines.append("")
    lines.append("## Recall@0.50 theo nhom mon")
    lines.append("")
    lines.append("| Nhom | " + " | ".join(CONFIGS) + " |")
    lines.append("|---|" + "---|" * len(CONFIGS))
    groups = {}
    for name in CONFIGS:
        for img_path, expected, cls050, _, _, _, _ in results[name]:
            groups.setdefault(group_key(img_path), {}).setdefault(name, []).append(
                (expected, cls050))
    for g in sorted(groups):
        cells = []
        for name in CONFIGS:
            rows = groups[g].get(name, [])
            rec = sum(len(e & d) / len(e) for e, d in rows) / len(rows) if rows else 0
            cells.append(f"{rec:.2f}")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")

    # per-class recall
    lines.append("")
    lines.append("## Recall@0.50 theo class (tren cac anh co class do)")
    lines.append("")
    lines.append("| Class | So anh | " + " | ".join(CONFIGS) + " |")
    lines.append("|---|---|" + "---|" * len(CONFIGS))
    all_classes = sorted({c for name in CONFIGS for _, e, *_ in results[name] for c in e})
    for c in all_classes:
        cells = []
        n_img = 0
        for name in CONFIGS:
            hit = tot = 0
            for img_path, expected, cls050, _, _, _, _ in results[name]:
                if c in expected:
                    tot += 1
                    hit += c in cls050
            n_img = tot
            cells.append(f"{hit}/{tot}" if tot else "-")
        cname = class_names[c]["name"]
        lines.append(f"| {cname} | {n_img} | " + " | ".join(cells) + " |")

    # near-miss analysis
    lines.append("")
    lines.append("## Near-miss: class bi miss @0.50 nhung co box o conf thap hon")
    lines.append("")
    lines.append("| Config | Class | So anh miss | median max-conf | max max-conf | Sẽ bat duoi slider 0.30? |")
    lines.append("|---|---|---|---|---|---|")
    for name in CONFIGS:
        per_class = {}
        for img_path, expected, cls050, _, _, _, near in results[name]:
            for c, m in near.items():
                per_class.setdefault(c, []).append(m)
        for c, ms_list in sorted(per_class.items(), key=lambda x: -len(x[1])):
            med = float(np.median(ms_list))
            mx = float(np.max(ms_list))
            lines.append(f"| {name} | {class_names[c]['name']} | {len(ms_list)} "
                         f"| {med:.3f} | {mx:.3f} | {'Co' if med >= 0.30 else 'Khong'} |")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
