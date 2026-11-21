"""
scripts/make_class_metrics_table.py
===================================
Derived table cho Chương 4: per-class detection coverage + target-matched MAPE.

Nguồn dữ liệu (Chain B — historical, pre-freeze pipeline v26m ONNX):
  data/benchmark_error_attribution_audit.json  (54 image_audit_records + summary_comparison)

Output:
  results/class_metrics.md

Quy tắc trình bày (theo results/frozen_numbers.md §0):
  - Bảng chính: Class | N | Detected (D1) | Exact-class | Coverage (D1) | MAPE (target-matched)
  - Hai phép đo detection được trình bày CẠNH NHAU:
      * D1 (audit): "vùng món mục tiêu được phát hiện" — KỂ CẢ khi nhãn là generic/sibling
        (Com thay cho Com tam, Canh thay cho Sup cua, Bun bo Hue thay cho Bun rieu).
        Đây là taxonomy frozen, khớp tier_1_detection_status_counts (D1=34).
      * Exact-class: chỉ đếm khi đúng chuỗi class mục tiêu xuất hiện trong detected_classes.
    Không được trình bày riêng lẻ một trong hai — sẽ gây hiểu nhầm coverage.
  - Nhóm Coverage = 0% theo CẢ HAI phép đo (BanhCuon): MAPE (target-matched) = N/A;
    số MAPE frozen chuyển sang trường "Auxiliary D2 error" kèm footnote.
  - Bảng derived, KHÔNG phải nguồn độc lập; mọi số truy về audit JSON.
"""

import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_JSON = os.path.join(ROOT_DIR, "data", "benchmark_error_attribution_audit.json")
OUT_MD = os.path.join(ROOT_DIR, "results", "class_metrics.md")

TARGET_CLASS = {
    "Pho": "Pho (Vietnamese noodle soup)",
    "BanhCuon": "Banh cuon (Rolled rice pancake)",
    "ComSuon": "Com tam (Broken rice)",
    "BunBoHue": "Bun bo Hue (Hue beef noodle soup)",
    "BanhMi": "Banh mi (Vietnamese baguette sandwich)",
    "BunRieu-CanhBun": "Bun rieu (Crab noodle soup)",
    "SupCua": "Sup cua (Crab soup)",
}

D1 = "D1_correct_target_detected"


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def main():
    with open(AUDIT_JSON, encoding="utf-8") as f:
        audit = json.load(f)

    records = audit["image_audit_records"]
    summary = audit["summary_comparison"]
    groups = list(TARGET_CLASS)  # fixed order

    rows = []
    d1_only_rows = []
    exact_rows = []
    status_notes = []
    generic_labels = []  # (group, D1 count, ví dụ nhãn generic/sibling)
    auxiliary = []

    for g in groups:
        recs = [r for r in records if r["folder"] == g]
        n = len(recs)
        d1 = [r for r in recs if r["upstream_detection_status"] == D1]
        detected = len(d1)
        coverage = 100.0 * detected / n if n else 0.0
        exact = sum(1 for r in recs if TARGET_CLASS[g] in r["detected_classes"])

        p2 = summary[g]["p2_mape"]
        if detected == 0 and exact == 0:
            main_mape = "N/A"
            cls_name = TARGET_CLASS[g].split(" (")[0]
            auxiliary.append(
                f"- **{g}** — Auxiliary D2 error: **{fmt(p2)}%** — {fmt(p2)}% được tính từ detection "
                f"nhầm lớp (không phải MAPE trên detection đúng lớp {cls_name})."
            )
        else:
            main_mape = f"{fmt(p2)}%"

        rows.append((g, n, detected, exact, fmt(coverage, 0) + "%", main_mape))

        d1_mape = mean([r["rel_err_p2_pct"] for r in d1])
        d1_only_rows.append((g, detected, fmt(d1_mape) + "%" if d1_mape is not None else "N/A"))

        ex_recs = [r for r in recs if TARGET_CLASS[g] in r["detected_classes"]]
        ex_mape = mean([r["rel_err_p2_pct"] for r in ex_recs])
        exact_rows.append((g, exact, fmt(ex_mape) + "%" if ex_mape is not None else "N/A"))

        if detected > exact:
            generic_labels.append((g, detected, exact))

        statuses = Counter(r["upstream_detection_status"] for r in recs)
        non_d1 = {k.split("_")[0]: v for k, v in statuses.items() if k != D1}
        if non_d1:
            status_notes.append(f"- {g}: " + ", ".join(f"{k}={v}" for k, v in sorted(non_d1.items())))

    lines = []
    a = lines.append
    a("# Class Metrics — coverage & target-matched MAPE (bảng derived)")
    a("")
    a("> **Chain B (historical)** — nguồn: `data/benchmark_error_attribution_audit.json` "
      "(54 records, GT = khối lượng món mục tiêu, pre-freeze pipeline v26m ONNX).")
    a("> Bảng derived theo schema khóa trong `results/frozen_numbers.md` §0. "
      "Hai cột detection trình bày cạnh nhau — xem phần 'Hai phép đo detection' bên dưới.")
    a("")
    a("## Bảng chính")
    a("")
    a("| Class | N | Detected (D1) | Exact-class | Coverage (D1) | MAPE (target-matched) |")
    a("|---|---:|---:|---:|---:|---:|")
    for g, n, det, ex, cov, mape in rows:
        a(f"| {g} | {n} | {det} | {ex} | {cov} | {mape} |")
    total_n = sum(r[1] for r in rows)
    total_d1 = sum(r[2] for r in rows)
    total_ex = sum(r[3] for r in rows)
    a(f"| **Tổng** | **{total_n}** | **{total_d1}** | **{total_ex}** | "
      f"**{fmt(100.0 * total_d1 / total_n, 1)}%** | — |")
    a("")
    a("### Hai phép đo detection (đọc kỹ trước khi trích)")
    a("")
    a("- **Detected (D1)** — taxonomy frozen của audit: *vùng món mục tiêu được phát hiện*, "
      "kể cả khi nhãn là generic/sibling. Khớp `tier_1_detection_status_counts` (D1 = 34, "
      "asserted trong `tests/test_phase4_acceptance.py`).")
    a("- **Exact-class** — chỉ đếm khi đúng chuỗi class mục tiêu xuất hiện trong `detected_classes` "
      "(phép đo nghiêm ngặt hơn, KHÔNG nằm trong taxonomy frozen).")
    a("- Chênh lệch D1 − Exact-class phát sinh từ nhãn generic/sibling: "
      + "; ".join(f"{g} ({d} ảnh D1, {e} exact)" for g, d, e in generic_labels)
      + " — ví dụ SupCua D1 chỉ có `Canh (Soup)`, ComSuon D1 đa số `Com (Rice)`, "
        "BunRieu có ảnh D1 mang nhãn `Bun bo Hue (Hue beef noodle soup)`.")
    a("")
    a("## MAPE | D1-only (chỉ ảnh có trạng thái D1 theo audit)")
    a("")
    a("| Class | n (D1) | MAPE |")
    a("|---|---:|---:|")
    for g, n, mape in d1_only_rows:
        a(f"| {g} | {n} | {mape} |")
    a("")
    a("## MAPE | Exact-class-only (chỉ ảnh có đúng class mục tiêu)")
    a("")
    a("| Class | n (exact) | MAPE |")
    a("|---|---:|---:|")
    for g, n, mape in exact_rows:
        a(f"| {g} | {n} | {mape} |")
    a("")
    a("## Auxiliary D2 error (các nhóm Coverage = 0% theo cả hai phép đo)")
    a("")
    if auxiliary:
        lines.extend(auxiliary)
    else:
        a("- (không có nhóm nào Coverage = 0%)")
    a("")
    a("## Phân bổ trạng thái không phải D1 (theo nhóm)")
    a("")
    if status_notes:
        lines.extend(status_notes)
    else:
        a("- (tất cả ảnh đều D1)")
    a("")
    a("## Ghi chú")
    a("")
    a("- MAPE (target-matched) là số frozen từ `summary_comparison.p2_mape` — tính trên TOÀN BỘ ảnh "
      "nhóm (gồm cả ảnh không phải D1). Hai bảng MAPE con (D1-only / exact-only) tính từ "
      "`rel_err_p2_pct` của các subset tương ứng — phục vụ so sánh MAPE|all vs MAPE|detect-đúng.")
    a("- ⚠️ Lưu ý nhất quán: báo cáo frozen `data/benchmark_portion_report.md` nêu ví dụ D2 "
      "\"Súp cua → Canh\", nhưng JSON audit xếp các ảnh SupCua chỉ có `Canh (Soup)` vào **D1** "
      "(D2 của SupCua là các ảnh chỉ thấy Thit heo/Dau hu/Banh mi). Số count (34/12/5/1/2) là "
      "chuẩn theo JSON + test; ví dụ văn xuôi trong báo cáo là minh họa, không đụng vào.")
    overall = audit["metadata"]
    a(f"- Overall Chain B: P1 = {overall['overall_p1_mape_pct']}% · "
      f"P2 = {overall['overall_p2_mape_pct']}% (population 54 ảnh).")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {OUT_MD}")
    print(f"Summary: D1 total={total_d1}/54, exact-class total={total_ex}/54")


if __name__ == "__main__":
    main()
