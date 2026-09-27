"""Aggregate scratch/full_eval_records.jsonl into the final evaluation report.

Reads the per-image records written by eval_full_pipeline.py and compiles:
  1. Detection: greedy one-to-one per class vs GT box counts (recall, extras)
  2. Portion:   estimated mass vs GT mass (grams_total), MAE/MAPE
                per class + per-image totals
  3. Nutrition: estimated Calories vs GT Calories (gt mass x kcal/100g),
                MAE/MAPE + coverage
  4. Scale source distribution + worker timing

Writes scratch/full_pipeline_eval.md.
"""

import json
import os
import statistics
from collections import defaultdict

from class_names import class_names

JSONL = os.path.join("scratch", "full_eval_records.jsonl")
OUT_MD = os.path.join("scratch", "full_pipeline_eval.md")


def cname(cid):
    return class_names[int(cid)]["name"]


def short(cid):
    return cname(cid).split(" (")[0]



def kcal_per_100g(cid):
    item = class_names[int(cid)]
    per = item.get("nutrition_per_100g") if isinstance(item, dict) else None
    if not per or "Calories" not in per:
        return None
    return float(per["Calories"])


def mape(pairs):
    """pairs: list of (est, gt). Mean absolute percentage error in %."""
    if not pairs:
        return None, 0
    errs = [abs(e - g) / g * 100 for e, g in pairs if g > 0]
    return (sum(errs) / len(errs) if errs else None), len(errs)


def main():
    records = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)
    n_err = sum(1 for r in records if r.get("error"))
    print(f"loaded {n} records ({n_err} with errors)")

    name_to_cid = {class_names[i]["name"]: i for i in range(len(class_names))}

    # ── 1. detection ──
    det = {"hit": 0, "gt": 0, "extras": 0}
    det_class = defaultdict(lambda: [0, 0])
    # ── 2/3. portion + nutrition accumulators ──
    portion_pairs = defaultdict(list)   # cid -> [(est_total_g, gt_total_g)]
    kcal_pairs = defaultdict(list)      # cid -> [(est_kcal, gt_kcal)]
    missing_nutrition = defaultdict(int)
    img_total = []                      # (est_total_g, gt_total_g)
    img_kcal = []                       # (est_kcal, gt_kcal)
    unmatched_mass = 0.0
    scale_dist = defaultdict(int)
    worker_s = []
    # matched-only: est↔det pair by index (worker preserves Results order),
    # per class keep top-k by confidence (k = GT count) → pure volume quality,
    # detection extras excluded and reported separately
    m_portion = defaultdict(list)
    m_kcal = defaultdict(list)
    m_extras_mass = defaultdict(float)
    m_img_total = []
    m_img_kcal = []
    group_stats = defaultdict(lambda: {"mass": [], "kcal": []})

    for r in records:
        if r.get("error"):
            continue
        gt = {int(k): v for k, v in r["gt"].items()}
        est_by_cid = defaultdict(lambda: [0.0, 0.0])  # cid -> [mass, kcal]
        pairs_by_class = defaultdict(list)
        for _i, e in enumerate(r["est"]):
            cid = name_to_cid.get(e["class_name"])
            if cid is None:
                continue
            est_by_cid[cid][0] += e["mass_g"]
            est_by_cid[cid][1] += e["kcal"]
            _conf = 0.0
            if _i < len(r["detections"]):
                _conf = r["detections"][_i].get("conf", 0.0)
            pairs_by_class[cid].append((_conf, e["mass_g"], e["kcal"]))

        # detection: greedy one-to-one per class
        det_d = defaultdict(int)
        for d in r["detections"]:
            cid = name_to_cid.get(d["class_name"])
            if cid is not None:
                det_d[cid] += 1
        for cid, v in gt.items():
            k = v["count"]
            det["gt"] += k
            hit = min(k, det_d.get(cid, 0))
            det["hit"] += hit
            det_class[cid][0] += hit
            det_class[cid][1] += k
        det["extras"] += sum(v for cid, v in det_d.items()
                             if cid not in gt) + sum(
            max(0, v - gt.get(cid, {"count": 0})["count"]) for cid, v in det_d.items() if cid in gt)

        # portion + nutrition per class
        img_est_m = img_gt_m = 0.0
        img_est_k = img_gt_k = 0.0
        img_has_kcal = True
        for cid, v in gt.items():
            g = v["grams_total"]
            if g is None:
                continue  # unknown-mass target: count-only
            gt_mass = g  # gram value is the TOTAL mass of all K boxes
            img_gt_m += gt_mass
            kc = kcal_per_100g(cid)
            gt_kcal = gt_mass * kc / 100 if kc is not None else None
            if gt_kcal is None:
                missing_nutrition[cid] += 1
                img_has_kcal = False
            else:
                img_gt_k += gt_kcal
            if cid in est_by_cid:
                est_m, est_k = est_by_cid[cid]
                img_est_m += est_m
                portion_pairs[cid].append((est_m, gt_mass))
                if gt_kcal is not None:
                    img_est_k += est_k
                    kcal_pairs[cid].append((est_k, gt_kcal))
            else:
                if gt_kcal is not None:
                    img_has_kcal = False
        # unmatched extras mass (detections whose class has no GT)
        for cid, (m, _k) in est_by_cid.items():
            if cid not in gt:
                unmatched_mass += m
        if r.get("volume_note"):
            pass  # volume unavailable (conf < 0.40 floor) — exclude image-level
        else:
            img_total.append((img_est_m, img_gt_m))
        if img_has_kcal and img_gt_k > 0 and not r.get("volume_note"):
            img_kcal.append((img_est_k, img_gt_k))
            g = r["group"]
            group_stats[g]["mass"].append((img_est_m, img_gt_m))
            group_stats[g]["kcal"].append((img_est_k, img_gt_k))
        # matched-only computation
        m_est_m = m_gt_m = 0.0
        m_est_k = m_gt_k = 0.0
        m_img_ok_k = True
        for cid, v in gt.items():
            k, g, sem = v["count"], v["grams_total"], v.get("mass_semantics", "total")
            if g is None:
                continue  # unknown-mass target: count-only
            gt_mass = g  # gram value is the TOTAL mass of all K boxes
            m_gt_m += gt_mass
            kc = kcal_per_100g(cid)
            if kc is None:
                m_img_ok_k = False
            else:
                m_gt_k += gt_mass * kc / 100
            pairs = sorted(pairs_by_class.get(cid, []), key=lambda p: -p[0])
            if sem == "partial":
                # GT mass assignment rule for partially-annotated targets:
                # the measured mass pairs with the LARGEST-mass estimation
                # (the main portion); remaining est belong to the unknown
                # target and are excluded from mass comparison. Documented
                # assumption — not absolute GT.
                if not pairs:
                    continue  # detection miss
                mm = max(p[1] for p in pairs)
                m_est_m += mm
                m_portion[cid].append((mm, gt_mass))
                if kc is not None:
                    mk = max(p[2] for p in pairs)
                    m_est_k += mk
                    m_kcal[cid].append((mk, gt_mass * kc / 100))
                else:
                    m_img_ok_k = False
                continue
            matched, extra = pairs[:k], pairs[k:]
            m_extras_mass[cid] += sum(p[1] for p in extra)
            if not matched:
                continue  # detection miss — counted in section 1, no portion comparison
            mm = sum(p[1] for p in matched)
            m_est_m += mm
            m_portion[cid].append((mm, gt_mass))
            if kc is not None:
                mk = sum(p[2] for p in matched)
                m_est_k += mk
                m_kcal[cid].append((mk, gt_mass * kc / 100))
            else:
                m_img_ok_k = False
        m_img_total.append((m_est_m, m_gt_m))
        if m_img_ok_k and m_gt_k > 0:
            m_img_kcal.append((m_est_k, m_gt_k))
        scale_dist[r.get("scale_source", "?")] += 1
        if r.get("worker_s"):
            worker_s.append(r["worker_s"])

    # ── report ──
    total_targets = sum(v["count"] for r in records if not r.get("error")
                        for v in r["gt"].values())
    L = ["", "# Full-pipeline evaluation — 54 ảnh portion-estimation_val", ""]
    L.append(f"Pipeline: v10b + dual-preresize @ conf=0.30 (app default) → "
             f"volume SAM2 daemon (ArUco scale, 2000px source) → nutrition. "
             f"GT: {total_targets} targets / "
             f"{n - n_err} ảnh hợp lệ.")
    L.append("")
    L.append("## 1. Detection (class-level, greedy one-to-one theo counts)")
    L.append("")
    rec = det["hit"] / det["gt"] if det["gt"] else 0
    L.append(f"- **Recall: {rec:.3f} ({det['hit']}/{det['gt']})** · "
             f"FP/extras: {det['extras']} · 0-box: "
             f"{sum(1 for r in records if not r.get('error') and not r['detections'])}/{n - n_err}")
    L.append("")
    L.append("| Class | Recall |")
    L.append("|---|---|")
    for cid in sorted(det_class):
        h, t = det_class[cid]
        L.append(f"| {short(cid)} | {h}/{t} |")
    L.append("")

    L.append("## 2. Portion (khối lượng ước lượng vs GT)")
    L.append("")
    L.append("| Class | n so sánh | GT mass (g) | MAE (g) | MAPE |")
    L.append("|---|---|---|---|---|")
    rows = []
    for cid in sorted(portion_pairs):
        pairs = portion_pairs[cid]
        mae = statistics.mean(abs(e - g) for e, g in pairs)
        mp, _ = mape(pairs)
        gt_total = sum(g for _e, g in pairs)
        rows.append((mp if mp is not None else 999, cid, len(pairs), gt_total, mae, mp))
    for _mp, cid, np_, gt_total, mae, mp in sorted(rows):
        L.append(f"| {short(cid)} | {np_} | {gt_total:.0f} | {mae:.1f} | "
                 f"{mp:.1f}%" if mp is not None else
                 f"| {short(cid)} | {np_} | {gt_total:.0f} | {mae:.1f} | n/a |")
    if img_total:
        mae_t = statistics.mean(abs(e - g) for e, g in img_total)
        mp_t, _ = mape(img_total)
        L.append("")
        L.append(f"**Tổng khối lượng/bức ảnh**: MAE = {mae_t:.1f} g · "
                 f"MAPE = {mp_t:.1f}% (trên {len(img_total)} ảnh) · "
                 f"khối lượng unmatched extras cộng dồn: {unmatched_mass:.0f} g")
    L.append("")

    L.append("## 3. Nutrition (Calories ước lượng vs GT)")
    L.append("")
    L.append("| Class | n so sánh | GT kcal | MAE (kcal) | MAPE |")
    L.append("|---|---|---|---|---|")
    rows = []
    for cid in sorted(kcal_pairs):
        pairs = kcal_pairs[cid]
        mae = statistics.mean(abs(e - g) for e, g in pairs)
        mp, _ = mape(pairs)
        gt_total = sum(g for _e, g in pairs)
        rows.append((mp if mp is not None else 999, cid, len(pairs), gt_total, mae, mp))
    for _mp, cid, np_, gt_total, mae, mp in sorted(rows):
        L.append(f"| {short(cid)} | {np_} | {gt_total:.0f} | {mae:.1f} | {mp:.1f}% |")
    if img_kcal:
        mae_t = statistics.mean(abs(e - g) for e, g in img_kcal)
        mp_t, _ = mape(img_kcal)
        L.append("")
        L.append(f"**Tổng kcal/bức ảnh**: MAE = {mae_t:.1f} kcal · "
                 f"MAPE = {mp_t:.1f}% (trên {len(img_kcal)} ảnh có đủ DB dinh dưỡng)")
    if missing_nutrition:
        L.append("")
        L.append("GT class thiếu nutrition_per_100g (bỏ qua phần kcal): " +
                 ", ".join(f"{short(c)} ({v} ảnh)" for c, v in sorted(missing_nutrition.items())))
    L.append("")

    L.append("## 2b. Portion MATCHED-ONLY (tách khỏi detection extras)")
    L.append("")
    L.append("Chỉ tính top-k box theo confidence khớp GT counts; phần thừa ghi riêng.")
    L.append("")
    L.append("| Class | n | GT mass (g) | Matched MAE (g) | Matched MAPE | Extras mass (g) |")
    L.append("|---|---|---|---|---|---|")
    for cid in sorted(m_portion):
        pairs = m_portion[cid]
        mae = statistics.mean(abs(e - g) for e, g in pairs)
        mp, _ = mape(pairs)
        gt_total = sum(g for _e, g in pairs)
        ex = m_extras_mass.get(cid, 0.0)
        L.append(f"| {short(cid)} | {len(pairs)} | {gt_total:.0f} | {mae:.1f} | "
                 f"{mp:.1f}% | {ex:.0f} |")
    if m_img_total:
        mae_t = statistics.mean(abs(e - g) for e, g in m_img_total)
        mp_t, _ = mape(m_img_total)
        L.append("")
        L.append(f"**Matched-only tổng khối lượng/ảnh**: MAE = {mae_t:.1f} g · "
                 f"MAPE = {mp_t:.1f}% (trên {len(m_img_total)} ảnh)")
    L.append("")
    L.append("## 3b. Nutrition MATCHED-ONLY")
    L.append("")
    L.append("| Class | n | GT kcal | MAE (kcal) | MAPE |")
    L.append("|---|---|---|---|---|")
    for cid in sorted(m_kcal):
        pairs = m_kcal[cid]
        mae = statistics.mean(abs(e - g) for e, g in pairs)
        mp, _ = mape(pairs)
        gt_total = sum(g for _e, g in pairs)
        L.append(f"| {short(cid)} | {len(pairs)} | {gt_total:.0f} | {mae:.1f} | {mp:.1f}% |")
    if m_img_kcal:
        mae_t = statistics.mean(abs(e - g) for e, g in m_img_kcal)
        mp_t, _ = mape(m_img_kcal)
        L.append("")
        L.append(f"**Matched-only tổng kcal/ảnh**: MAE = {mae_t:.1f} kcal · "
                 f"MAPE = {mp_t:.1f}% (trên {len(m_img_kcal)} ảnh)")
    L.append("")
    L.append("## 4. Scale source + timing")
    L.append("")
    L.append(f"- Scale source: {dict(scale_dist)}")
    if worker_s:
        L.append(f"- Volume worker: mean {statistics.mean(worker_s):.1f}s · "
                 f"median {statistics.median(worker_s):.1f}s · max {max(worker_s):.1f}s "
                 f"({len(worker_s)} requests)")
    L.append("")
    L.append("## 5. Theo nhóm món (MAPE tổng khối lượng / tổng kcal)")
    L.append("")
    L.append("| Nhóm | n | MAPE mass | MAPE kcal |")
    L.append("|---|---|---|---|")
    for g in sorted(group_stats):
        ms = group_stats[g]["mass"]
        ks = group_stats[g]["kcal"]
        mm, _ = mape(ms)
        mk, _ = mape(ks)
        L.append(f"| {g} | {len(ms)} | {mm:.1f}% | {mk:.1f}% |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
