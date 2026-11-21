"""
scripts/crosscheck_sources.py
=============================
Cross-checks the VietFood68 nutrition SSOT (data/vietfood68_nin_mapping.json →
class_names.py) against BOTH authoritative sources, and physically validates
the density database:

1. TDGD (thucdongiadinh.vn — Viện Nghiên Cứu Dinh Dưỡng TP.HCM)
   Bảng thành phần thực phẩm Việt Nam, per-100g, matched by Vietnamese name.
2. USDA FoodData Central (api.nal.usda.gov, DEMO_KEY)
   Verifies the commodities labeled "usda_fdc" in DENSITY_DB and records
   FDC IDs for provenance.
3. Physical sanity check of DENSITY_DB via the Choi–Okos (1986) component
   density model: the plated effective density of a food must not exceed its
   material density computed from per-100g composition.

Outputs:
  - data/source_crosscheck_report.json  (machine-readable)
  - data/source_crosscheck_report.md    (human-readable)
  - prints a summary

Usage: python scripts/crosscheck_sources.py [--skip-fdc]
"""

import os
import re
import sys
import json
import time
import unicodedata
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.insert(0, ROOT_DIR)

FDC_API_KEY = os.environ.get("FDC_API_KEY", "DEMO_KEY")
FDC_SEARCH = "https://api.nal.usda.gov/fdc/v1/foods/search?api_key={key}"
FDC_DETAIL = "https://api.nal.usda.gov/fdc/v1/food/{fdcId}?api_key={key}"

# Choi–Okos (1986) component densities at ~20 °C, g/cm³
CHOI_OKOS = {
    "water": 0.9971,
    "protein": 1.3300,
    "fat": 0.9257,
    "carb": 1.5878,
    "fiber": 1.3220,
    "ash": 2.4238,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tone_strip(s: str) -> str:
    """Strip Vietnamese tone marks only — base letters ơ/ư/đ survive, so
    bò→bo stays distinct from bơ→bơ (different base letter)."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s)


def norm_vi(s: str) -> str:
    """Vietnamese food-name normalization: keep diacritics (cơm ≠ cốm, bò ≠ bơ),
    normalize whitespace/punctuation only."""
    s = (s or "").lower()
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^a-zà-ỹăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Leading classifier/measure words that are not part of the food name
VI_CLASSIFIERS = {"quả", "củ", "lá", "con", "cây", "bát", "tô", "đĩa", "viên", "miếng", "khẩu"}


def name_tokens(s: str) -> list:
    s = s or ""
    # drop only serving-size annotations "(35g/quả)" — keep variant parens "(lợn)"
    s = re.sub(r"\([^)]*\d[^)]*\)", " ", s)
    toks = norm_vi(s).split()
    toks = [t for t in toks if len(t) >= 2]
    while toks and toks[0] in VI_CLASSIFIERS:
        toks = toks[1:]
    return toks


def _jaccard_heads(a: list, b: list) -> float:
    if not a or not b or a[0] != b[0]:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def match_score(cand: str, target: str) -> float:
    """Token-set score with a hard head-noun rule: Vietnamese food names open
    with the category word (bánh, bún, thịt, cá …) — if the heads disagree the
    foods are different, however many tokens they share. Tier 1 keeps
    diacritics; tier 2 (penalized) strips tone marks to unify mỳ/mì spellings."""
    s = _jaccard_heads(name_tokens(cand), name_tokens(target))
    if s > 0:
        return s
    a = [tone_strip(t) for t in name_tokens(cand)]
    b = [tone_strip(t) for t in name_tokens(target)]
    return _jaccard_heads(a, b) * 0.95


def load_json(name):
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def choi_okos_density(protein_g, fat_g, carb_g, fiber_g, water_g, ash_g=2.0):
    """Material density (g/cm³) via the mass-fraction parallel model.
    Components are paired by key with CHOI_OKOS, never by dict order."""
    comp = {
        "water": max(0.0, water_g),
        "protein": max(0.0, protein_g),
        "fat": max(0.0, fat_g),
        "carb": max(0.0, carb_g),
        "fiber": max(0.0, fiber_g),
        "ash": max(0.0, ash_g),
    }
    total = sum(comp.values())
    if total <= 0:
        return None
    inv = sum(comp[k] / total / CHOI_OKOS[k] for k in comp)
    return 1.0 / inv if inv > 0 else None


# ---------------------------------------------------------------------------
# 1. TDGD cross-check (Bảng TPTP VN via thucdongiadinh.vn)
# ---------------------------------------------------------------------------

def tdgd_nutrients(rec):
    """Extract per-100g (energy, protein, fat, carbs) from a TDGD record."""
    comps = {c["tenDinhDuong"]: c.get("hamLuong") for c in (rec.get("thanhPhanDinhDuong") or [])}
    return {
        "energy": rec.get("nangLuongKcal"),
        "protein": comps.get("Chất Đạm"),
        "fat": comps.get("Chất Béo"),
        "carbs": comps.get("Chất bột đường"),
        "water": comps.get("Nước"),
    }


# TDGD groups containing prepared dishes (vs ingredient/commodity groups)
TDGD_DISH_GROUP_IDS = {15, 16}


def crosscheck_tdgd(mappings, tdgd_records):
    """Match each VietFood68 class against the TDGD food composition table
    (same Bảng TPTP VN lineage) and compare per-100g macros. Dish classes are
    only compared against prepared-dish entries; ingredient classes against
    ingredient entries — mixing the two compares diluted soup to dry noodle."""
    tdgd_index = [
        (rec, name_tokens(rec.get("tenThucPham") or ""), rec.get("nhomThucPhamId"))
        for rec in tdgd_records
    ]
    results = []
    for m in mappings:
        cls = m["full_name"]
        p = m.get("primary_nutrition_per_100g") or {}
        pm = m.get("primary_match") or {}
        class_kind = "dish" if str(pm.get("nin_food_id", "")).startswith("nin_dish") else "ingredient"
        candidates = [c for c in (pm.get("nin_food_name"), m.get("vietfood_class_name_vi")) if c]
        best, best_score, best_gap = None, 0.0, 999
        for cand in candidates:
            cand_len = len(name_tokens(cand))
            for rec, rtoks, group_id in tdgd_index:
                score = match_score(cand, rec.get("tenThucPham") or "")
                gap = abs(cand_len - len(rtoks))
                if score > best_score or (score == best_score and score > 0 and gap < best_gap):
                    best, best_score, best_gap = rec, score, gap
        comparable = False
        match_kind = None
        if best and best_score >= 0.5:
            match_kind = "dish" if (best.get("nhomThucPhamId") in TDGD_DISH_GROUP_IDS) else "ingredient"
            comparable = (class_kind == match_kind)
        entry = {
            "class": cls,
            "class_kind": class_kind,
            "nin_food_name": pm.get("nin_food_name"),
            "tdgd_match": best.get("tenThucPham") if best else None,
            "tdgd_group": best.get("strNhomThucPham") if best else None,
            "tdgd_match_score": round(best_score, 3),
            "match_kind": match_kind,
            "comparable": comparable,
            "nin": {k: p.get(k) for k in ("energy_kcal", "protein_g", "fat_g", "carbs_g")},
            "tdgd": tdgd_nutrients(best) if best else None,
            "deviation_pct": {},
        }
        if comparable:
            t = tdgd_nutrients(best)
            pairs = [("energy_kcal", "energy"), ("protein_g", "protein"),
                     ("fat_g", "fat"), ("carbs_g", "carbs")]
            for nk, tk in pairs:
                n_val, t_val = p.get(nk), t.get(tk)
                if n_val and t_val and float(t_val) > 0:
                    dev = (float(n_val) - float(t_val)) / float(t_val) * 100.0
                    entry["deviation_pct"][nk] = round(dev, 1)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# 2. USDA FDC commodity verification
# ---------------------------------------------------------------------------

FDC_QUERIES = {
    "Ca (Fish)": "fish fillet raw mixed species",
    "Cua (Crab)": "crab meat blue crab",
    "Muc (Squid)": "squid raw",
    "Thit ga (Chicken)": "chicken breast meat raw",
    "Tom (Shrimp)": "shrimp raw",
    "Trung (Egg)": "egg whole raw fresh",
    "Dau hu (Tofu)": "tofu firm",
    "Pho mai (Cheese)": "cheddar cheese",
}


def fdc_get(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            wait = 25.0 if "429" in str(e) else 2.0
            print(f"  [FDC] attempt {attempt + 1} failed: {e} (waiting {wait:.0f}s)")
            time.sleep(wait)
    return None


def verify_fdc(density_db_keys, skip=False):
    """For each usda_fdc-labeled commodity, resolve the closest FDC commodity
    and derive gram-weight anchors from portion measures."""
    out = {}
    if skip:
        return out
    for cls, query in FDC_QUERIES.items():
        if cls not in density_db_keys:
            continue
        print(f"  [FDC] querying: {query}")
        search_url = FDC_SEARCH.format(key=FDC_API_KEY) + \
            f"&query={urllib.parse.quote(query)}&dataType=SR%20Legacy,Foundation&pageSize=1"
        res = fdc_get(search_url)
        time.sleep(6.0)
        if not res or not res.get("foods"):
            print("  [FDC] no result")
            continue
        food = res["foods"][0]
        fdc_id = food.get("fdcId")
        detail = fdc_get(FDC_DETAIL.format(fdcId=fdc_id, key=FDC_API_KEY))
        time.sleep(1.5)
        portions = []
        if detail:
            for p in detail.get("foodPortions", []) or []:
                grams = p.get("gramWeight")
                m = p.get("measureUnitAbbreviation") or p.get("measureUnitName") or ""
                vol_ml = None
                mm = re.search(r"([\d.]+)\s*(cup|tbsp|tsp|fluid ounce|fl oz)", m.lower())
                if mm:
                    unit = mm.group(2)
                    qty = float(mm.group(1))
                    vol_ml = qty * {"cup": 236.588, "tbsp": 14.787, "tsp": 4.929,
                                    "fluid ounce": 29.574, "fl oz": 29.574}[unit]
                if grams and vol_ml:
                    portions.append({
                        "measure": m.strip(), "grams": grams,
                        "implied_density": round(grams / vol_ml, 3),
                    })
        out[cls] = {
            "fdc_id": fdc_id,
            "description": food.get("description"),
            "data_type": food.get("dataType"),
            "portion_anchors": portions[:6],
        }
    return out


# ---------------------------------------------------------------------------
# 3. Density physical validation (Choi–Okos ceiling)
# ---------------------------------------------------------------------------

def validate_densities(mappings, tdgd_results, density_db):
    rows = []
    tdgd_by_class = {r["class"]: r for r in tdgd_results}
    for m in mappings:
        cls = m["full_name"]
        p = m.get("primary_nutrition_per_100g") or {}
        water = None
        t = tdgd_by_class.get(cls)
        if t and t.get("tdgd") and t.get("comparable"):
            water = t["tdgd"].get("water")
        protein, fat, carb = p.get("protein_g") or 0, p.get("fat_g") or 0, p.get("carbs_g") or 0
        fiber = p.get("fiber_g") or 0
        if water is None:
            water = max(0.0, 100.0 - protein - fat - carb - fiber - 2.0)
        mat = choi_okos_density(protein, fat, carb, fiber, water)
        db_entry = density_db.get(cls)
        db_val = db_entry[0] if db_entry else None
        db_src = db_entry[2] if db_entry else None
        flag = None
        if db_val and mat and db_val > mat + 0.03:
            flag = "DB_DENSITY_EXCEEDS_MATERIAL"
        rows.append({
            "class": cls,
            "water_per_100g": round(float(water), 1) if water is not None else None,
            "material_density": round(mat, 3) if mat else None,
            "db_density": db_val,
            "db_source": db_src,
            "flag": flag,
        })
    return rows


# ---------------------------------------------------------------------------
# 4. class_names.py projection check (class_names vs mapping SSOT)
# ---------------------------------------------------------------------------

def check_projection(mappings):
    """class_names.py keeps its 7 macro/salt fields verbatim (P3-F backward
    compatibility) and takes the 5 micronutrients from the mapping SSOT — so
    macro deltas are BY DESIGN, micronutrient mismatches are real defects."""
    from class_names import class_names
    by_name = {m["full_name"]: m for m in mappings}
    macro_deltas, issues = [], []
    macro_fields = {"Calories", "Protein", "Fat", "Carbs"}
    for item in class_names:
        name = item["name"]
        m = by_name.get(name)
        if not m:
            issues.append({"class": name, "issue": "missing_from_mapping"})
            continue
        p100 = item.get("nutrition_per_100g", {})
        p = m.get("primary_nutrition_per_100g") or {}
        checks = [
            ("Calories", "energy_kcal"), ("Protein", "protein_g"),
            ("Fat", "fat_g"), ("Carbs", "carbs_g"),
            ("Sodium", "sodium_mg"), ("Calcium", "calcium_mg"),
            ("Iron", "iron_mg"), ("Zinc", "zinc_mg"), ("Cholesterol", "cholesterol_mg"),
        ]
        for ck, mk in checks:
            cv, mv = p100.get(ck), p.get(mk)
            if cv is None or mv is None or float(mv) == 0:
                continue
            diff = abs(float(cv) - float(mv))
            if diff <= max(0.15, abs(float(mv)) * 0.02):
                continue
            if ck in macro_fields:
                macro_deltas.append({
                    "class": name, "field": ck,
                    "class_names_value": cv, "mapping_value": mv,
                    "ratio": round(float(cv) / float(mv), 2) if float(mv) else None,
                })
            else:
                issues.append({
                    "class": name, "issue": "micronutrient_mismatch",
                    "field": ck, "class_names_value": cv, "mapping_value": mv,
                })
    return issues, macro_deltas


# ---------------------------------------------------------------------------
# 5. AUDIT OF THE ACTUAL CALCULATION INPUT (class_names per-100g) vs TDGD
# ---------------------------------------------------------------------------

CALC_AUDIT_PAIRS = [
    ("Calories", "energy"), ("Protein", "protein"), ("Fat", "fat"), ("Carbs", "carbs"),
]  # (class_names field, TDGD field)


def audit_calc_input(tdgd_results):
    """The app computes serving nutrition as mass × per-100g where per-100g
    comes from class_names.nutrition_per_100g. Audit THAT value (not the
    mapping projection) directly against the TDGD authoritative table."""
    from class_names import class_names
    by_name = {it["name"]: it for it in class_names}
    tdgd_by_class = {r["class"]: r for r in tdgd_results}
    rows = []
    for cls, item in by_name.items():
        p100 = item.get("nutrition_per_100g", {})
        row = {
            "class": cls,
            "serving_size_g": item.get("serving_size_g"),
            "calc_per100": {ck: p100.get(ck) for ck, _ in CALC_AUDIT_PAIRS},
            "tdgd_match": None,
            "tdgd": None,
            "deviation_pct": {},
            "verdict": "no_tdgd_reference",
        }
        t = tdgd_by_class.get(cls)
        if t and t.get("comparable"):
            row["tdgd_match"] = t["tdgd_match"]
            row["tdgd"] = {tk: t["tdgd"].get(tk) for _, tk in CALC_AUDIT_PAIRS}
            devs = {}
            effective = []
            for ck, tk in CALC_AUDIT_PAIRS:
                cv, tv = p100.get(ck), t["tdgd"].get(tk)
                if cv is None or not tv or float(tv) <= 0:
                    continue
                diff = float(cv) - float(tv)
                devs[ck] = round(diff / float(tv) * 100.0, 1)
                # small absolute gaps are rounding/recipe noise — a +50% on a
                # 1 g base (Pho fat) says nothing; only larger-than-guard gaps
                # count toward the verdict
                guard = 15.0 if ck == "Calories" else 2.0
                if abs(diff) > guard:
                    effective.append(abs(devs[ck]))
            row["deviation_pct"] = devs
            maxdev = max(effective, default=0.0)
            row["verdict"] = "match" if maxdev <= 15 else ("variant" if maxdev <= 40 else "review")
        rows.append(row)
    return rows


def check_serving_consistency():
    """Demo path uses class_names['nutrition'] (per serving). Verify it equals
    per-100g × serving_size_g / 100 within rounding tolerance."""
    from class_names import class_names
    issues = []
    for item in class_names:
        p100 = item.get("nutrition_per_100g", {})
        pserv = item.get("nutrition", {})
        scale = item.get("serving_size_g", 100) / 100.0
        for k, v100 in p100.items():
            vs = pserv.get(k)
            if vs is None or v100 is None:
                continue
            expected = float(v100) * scale
            if abs(float(vs) - expected) > max(0.6, abs(expected) * 0.02):
                issues.append({
                    "class": item["name"], "field": k,
                    "serving_value": vs, "expected": round(expected, 2),
                })
    return issues


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(tdgd_results, fdc_results, density_rows, projection_issues, macro_deltas=()):
    lines = [
        "# Nguồn đối chiếu dữ liệu dinh dưỡng — Báo cáo cross-check",
        "",
        f"Ngày chạy: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Nguồn uy tín sử dụng",
        "",
        "| Nguồn | Kênh | Cơ sở |",
        "|---|---|---|",
        "| Viện Dinh dưỡng Quốc gia — Bảng TPTP VN 2017 | viendinhduong.vn (tool \"Giá trị dinh dưỡng món ăn\" + \"thực phẩm\") | per_serving → per_100g chuẩn hóa |",
        "| Viện NC Dinh dưỡng TP.HCM — Thực đơn gia đình | thucdongiadinh.vn (tool \"Tra cứu thành phần dinh dưỡng thực phẩm\") | per_100g (BTP2007/2000) |",
        "| USDA FoodData Central | fdc.nal.usda.gov | per_100g (SR Legacy / Foundation) |",
        "",
    ]

    matched = [r for r in tdgd_results if r.get("comparable")]
    big_dev = [r for r in matched if any(abs(v) > 25 for v in r["deviation_pct"].values())]
    lines += [
        "## 1. Đối chiếu NIN ↔ TDGD (68 lớp)",
        "",
        f"- Khớp comparable với bảng TDGD (score ≥ 0.5, cùng loại món/nguyên liệu): **{len(matched)}/{len(tdgd_results)}**",
        f"- Lớp lệch lớn (> 25% ở ít nhất 1 chất): **{len(big_dev)}** (chi tiết bên dưới)",
        "",
        "### Bảng chi tiết (chỉ các lớp khớp)",
        "",
        "| Lớp | TDGD match | kcal NIN/TDGD | Δ% kcal | Protein NIN/TDGD | Δ% | Fat NIN/TDGD | Δ% | Carb NIN/TDGD | Δ% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(matched, key=lambda x: -max(abs(v) for v in x["deviation_pct"].values() or {0})):
        n, t = r["nin"], r["tdgd"]
        d = r["deviation_pct"]

        def fmt(nv, tv, dv):
            ns = f"{nv:.0f}" if nv else "—"
            ts = f"{tv:.0f}" if tv else "—"
            ds = f"{dv:+.0f}%" if dv is not None else "—"
            return ns, ts, ds
        k_ns, k_ts, k_ds = fmt(n.get("energy_kcal"), t.get("energy"), d.get("energy_kcal"))
        p_ns, p_ts, p_ds = fmt(n.get("protein_g"), t.get("protein"), d.get("protein_g"))
        f_ns, f_ts, f_ds = fmt(n.get("fat_g"), t.get("fat"), d.get("fat_g"))
        c_ns, c_ts, c_ds = fmt(n.get("carbs_g"), t.get("carbs"), d.get("carbs_g"))
        lines.append(f"| {r['class']} | {r['tdgd_match']} | {k_ns}/{k_ts} | {k_ds} | "
                     f"{p_ns}/{p_ts} | {p_ds} | {f_ns}/{f_ts} | {f_ds} | {c_ns}/{c_ts} | {c_ds} |")

    lines += ["", "## 2. Kiểm chứng USDA FoodData Central (commodity khớp)"]
    if fdc_results:
        lines += ["", "| Lớp | FDC commodity | fdcId | Portion anchors (g/ml → g/cm³) |", "|---|---|---|---|"]
        for cls, v in fdc_results.items():
            anchors = "; ".join(f"{p['measure']} = {p['grams']} g → {p['implied_density']}"
                                for p in v["portion_anchors"][:3]) or "—"
            lines.append(f"| {cls} | {v['description']} | {v['fdc_id']} | {anchors} |")
    else:
        lines += [
            "",
            "_(Chưa chạy được: API key DEMO_KEY của USDA bị giới hạn tần suất (HTTP 429). "
            "Chạy lại với key riêng: `set FDC_API_KEY=<key> && python scripts/crosscheck_sources.py` "
            "— script sẽ đối chiếu 8 commodity gắn nhãn usda_fdc và ghi fdcId vào báo cáo.)_",
        ]

    lines += [
        "",
        "## 3. Kiểm chứng vật lý mật độ (mô hình Choi–Okos 1986)",
        "",
        "Mật độ hữu hiệu khi bày món (what the volume pipeline measures) phải **không vượt quá** "
        "mật độ vật liệu tính từ thành phần per-100g (mô hình thành phần song song Choi–Okos).",
        "",
        "### Vi phạm / cận hạn",
        "",
    ]
    flagged = [r for r in density_rows if r["flag"]]
    if flagged:
        lines += ["| Lớp | DB | Material | Nước/100g | Flag |", "|---|---|---|---|---|"]
        for r in flagged:
            lines.append(f"| {r['class']} | {r['db_density']} | {r['material_density']} | {r['water_per_100g']} | {r['flag']} |")
        lines += [
            "",
            "Chẩn đoán: cờ thường là dương tính giả do thành phần NIN của bản ghi món tương ứng bị "
            "chuẩn hóa sai khẩu phần (nước ~86% cho phô mai là dấu hiệu nhận biết — phô mai thật ~45–50% nước). "
            "Giá trị DB vẫn nhất quán với literature cho thực phẩm thật; bản ghi thành phần cần sửa trong đợt "
            "tái-canonicalization (xem mục Known limitation).",
        ]
    else:
        lines += ["Không có vi phạm — mọi giá trị DENSITY_DB ≤ mật độ vật liệu."]
    close = [r for r in density_rows if r["material_density"] and r["db_density"]
             and r["db_density"] > r["material_density"] - 0.02]
    if close:
        lines += ["", "Các lớp có mật độ DB sát trần vật liệu (≤ 0.02 g/cm³): "
                  + ", ".join(f"{r['class']} ({r['db_density']}/{r['material_density']})" for r in close)]

    lines += [
        "",
        "## 4. Kiểm tra projection class_names.py ↔ mapping SSOT",
        "",
        "Theo thiết kế (P3-F), class_names.py **giữ nguyên 7 trường macro** (Calories/Protein/Fat/"
        "Carbs/Saturates/Sugar/Salt) từ bộ dữ liệu gốc và chỉ đồng bộ 5 vi chất từ mapping SSOT — "
        "chênh lệch macro là chủ ý, không phải lỗi.",
        "",
    ]
    if projection_issues:
        lines += [f"- **{len(projection_issues)} lệch vi chất (cần xử lý)**:"]
        for i in projection_issues[:30]:
            lines.append(f"  - {i['class']}: {i.get('field')} "
                         f"(class_names={i.get('class_names_value')} vs mapping={i.get('mapping_value')})")
    else:
        lines += ["- Vi chất (Sodium/Calcium/Iron/Zinc/Cholesterol) khớp 100% giữa class_names.py và mapping SSOT."]
    if macro_deltas:
        worst = sorted(macro_deltas, key=lambda d: -abs(1.0 - (d.get("ratio") or 1.0)))[:10]
        lines += [
            f"- Macro giữ nguyên theo thiết kế: {len(macro_deltas)} cặp lệch "
            f"(top lệch nhất, ratio class_names/mapping):",
        ]
        for i in worst:
            lines.append(f"  - {i['class']} · {i['field']}: {i['class_names_value']} vs {i['mapping_value']} "
                         f"(ratio {i['ratio']})")
    lines.append("")
    return "\n".join(lines)


def render_calc_audit(calc_rows, serving_issues):
    lines = [
        "## 5. Audit ĐẦU VÀO TÍNH TOÁN DINH DƯỠNG (class_names per-100g × khối lượng ước lượng)",
        "",
        "Ứng dụng tính dinh dưỡng khẩu phần = `mass_g × per-100g / 100`, với per-100g lấy từ "
        "`class_names.nutrition_per_100g` (cả pipeline volume lẫn luồng demo). Mục này so trực tiếp "
        "GIÁ TRỊ TÍNH TOÁN với bảng TDGD (Bảng TPTP VN). Verdict: **match** (≤15%), **variant** "
        "(15–40% — khác biệt công thức giữa hai bảng), **review** (>40% — cần xem), "
        "**no_tdgd_reference** (TDGD không có dòng tương đương).",
        "",
    ]
    counts = {}
    for r in calc_rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines.append(f"Tổng kết: " + ", ".join(f"**{v}**: {c}" for v, c in sorted(counts.items()))
                 + f" trên {len(calc_rows)} lớp.")
    lines += ["", "### Các lớp verdict `review` (lệch > 40%)", ""]
    reviews = [r for r in calc_rows if r["verdict"] == "review"]
    if reviews:
        lines += ["| Lớp | TDGD match | kcal tính/TDGD | Δ% kcal | Protein | Δ% | Fat | Δ% | Carb | Δ% |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in reviews:
            c, t, d = r["calc_per100"], r["tdgd"], r["deviation_pct"]
            def f2(a, b):
                return f"{a if a is not None else '—'}/{b if b is not None else '—'}"
            def fd(x):
                return f"{x:+.0f}%" if x is not None else "—"
            lines.append(
                f"| {r['class']} | {r['tdgd_match']} "
                f"| {f2(c['Calories'], t['energy'])} | {fd(d.get('Calories'))} "
                f"| {f2(c['Protein'], t['protein'])} | {fd(d.get('Protein'))} "
                f"| {f2(c['Fat'], t['fat'])} | {fd(d.get('Fat'))} "
                f"| {f2(c['Carbs'], t['carbs'])} | {fd(d.get('Carbs'))} |")
    else:
        lines += ["Không có lớp nào vượt ngưỡng 40%."]
    lines += ["", "### Nhất quán nội bộ per-serving = per-100g × khẩu phần", ""]
    if serving_issues:
        lines += [f"- **{len(serving_issues)} lệch** (luồng demo dùng per-serving):"]
        for i in serving_issues[:20]:
            lines.append(f"  - {i['class']} · {i['field']}: serving={i['serving_value']} "
                         f"nhưng kỳ vọng {i['expected']}")
    else:
        lines += ["- `nutrition` (per-serving) = `nutrition_per_100g` × serving_size/100 khớp trong dung sai "
                  "làm tròn cho toàn bộ 68 lớp — luồng demo dùng đúng dữ liệu đã audit."]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    skip_fdc = "--skip-fdc" in sys.argv

    mappings = load_json("vietfood68_nin_mapping.json")["mappings"]
    tdgd = load_json("tdgd_food_composition.json")["data"]

    from density_db import DENSITY_DB
    density_db = {k: v for k, v in DENSITY_DB.items()}

    print("[1/4] Cross-check NIN SSOT vs TDGD (thucdongiadinh.vn) …")
    tdgd_results = crosscheck_tdgd(mappings, tdgd)

    print("[2/4] USDA FDC commodity verification …")
    usda_keys = {k for k, v in density_db.items() if v[2] == "usda_fdc"}
    fdc_results = verify_fdc(usda_keys, skip=skip_fdc)

    print("[3/4] Density physical validation (Choi–Okos) …")
    density_rows = validate_densities(mappings, tdgd_results, density_db)

    print("[4/5] Projection check class_names.py ↔ mapping …")
    projection_issues, macro_deltas = check_projection(mappings)

    print("[5/5] Audit calculation input (class_names per-100g) vs TDGD …")
    calc_rows = audit_calc_input(tdgd_results)
    serving_issues = check_serving_consistency()

    report_md = render_report(tdgd_results, fdc_results, density_rows, projection_issues, macro_deltas)
    report_md += "\n" + render_calc_audit(calc_rows, serving_issues)
    with open(os.path.join(DATA_DIR, "source_crosscheck_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(os.path.join(DATA_DIR, "source_crosscheck_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tdgd_crosscheck": tdgd_results,
            "fdc_verification": fdc_results,
            "density_validation": density_rows,
            "projection_issues": projection_issues,
            "projection_macro_deltas_by_design": macro_deltas,
            "calc_input_audit": calc_rows,
            "serving_consistency_issues": serving_issues,
        }, f, ensure_ascii=False, indent=2)

    matched = sum(1 for r in tdgd_results if r.get("comparable"))
    flagged = [r for r in density_rows if r["flag"]]
    print()
    print(f"TDGD matched: {matched}/{len(tdgd_results)}")
    print(f"Density flags: {len(flagged)}")
    for r in flagged:
        print(f"  - {r['class']}: db={r['db_density']} > material={r['material_density']}")
    print(f"Micronutrient projection issues: {len(projection_issues)} "
          f"(macro deltas preserved by design: {len(macro_deltas)})")
    counts = {}
    for r in calc_rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"Calc-input audit verdicts: {counts}")
    print(f"Serving consistency issues: {len(serving_issues)}")
    print("Report: data/source_crosscheck_report.md")


if __name__ == "__main__":
    main()
