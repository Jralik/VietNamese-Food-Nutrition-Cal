"""(a) Kiểm tra parser gốc có chạy được trên txt hiện tại không (KHÔNG ghi file).
(b) Ước tính tác động lên MAPE nếu GT hiệu chỉnh đúng.
"""
import os, json, re, collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT = os.path.join(BASE, 'portion-estimation_val.txt')
info = open(TXT, encoding='utf-8').read()

print("=== (a) Thử parse lại bằng LOGIC CỦA scratch/parse_gt_portions.py ===")
for l in info.splitlines():
    l = l.strip()
    if not l:
        continue
    parts = l.split('",', 1)
    if len(parts) != 2:
        continue
    raw = parts[1].strip().rstrip(';')
    for it in [x.strip() for x in raw.split(',') if x.strip()]:
        if ':' in it:
            cname, cval = it.split(':', 1)
            try:
                float(cval.strip())          # <- đúng như parser gốc
            except ValueError:
                fname = os.path.basename(parts[0].strip('" '))
                print("  Parser gốc BUỘC phải crash trên dòng hiện tại:")
                print("    file=" + fname)
                print("    float('" + cval.strip() + "') -> ValueError")
                print("    => data/real_images_ground_truth.json là sản phẩm của "
                      "phiên bản txt CŨ HƠN, chưa được regenerate.")
                raise SystemExit(0)
print("  Parser gốc vẫn parse sạch (không có dạng count).")

# === (b) Tác động của việc thiếu count lên kết quả ComSuon / BanhMi ===
print("\n=== (b) Tác động lên kết quả đã công bố trong benchmark_portion_report.md ===")
# Sai số/target-matched được trích trực tiếp từ báo cáo Phase 4
report = [
    ("com-suon1 (4).jpg", "ComSuon", "target", 426.0, 390.5),
    ("com-suon1 (2).jpg", "ComSuon", "target", 431.2, 390.5),
    ("com-suon1 (3).jpg", "ComSuon", "target", 434.3, 390.5),
    ("com-suon1 (5).jpg", "ComSuon", "target", 475.5, 390.5),
    ("com-suon1 (1).jpg", "ComSuon", "target", 486.9, 390.5),
    ("BanhMi_Trung (4).jpg", "BanhMi", "target", None, 320.5),
]
# GT hiệu chỉnh theo portion-estimation_val.txt (count x mass + component phụ)
corrected = {
    "com-suon1 (1).jpg": 390.5 + 150.0,
    "com-suon1 (3).jpg": 390.5 * 2 + 150.0,
    "com-suon1 (5).jpg": 390.5 + 150.0,
    "BanhMi_Trung (4).jpg": 99.3 * 2 + 221.2,
}
errs_json, errs_corr = [], []
for fn, grp, kind, pred, gt_json in report:
    if pred is None:
        continue
    gt_c = corrected.get(fn, gt_json)
    e1 = abs(pred - gt_json) / gt_json * 100
    e2 = abs(pred - gt_c) / gt_c * 100
    errs_json.append(e1); errs_corr.append(e2)
    tag = "KHAC" if abs(gt_c - gt_json) > 0.5 else "giong"
    print(f"  {fn:22s} pred={pred:7.1f}  GT_json={gt_json:7.1f} -> APE={e1:5.1f}%   |   "
          f"GT_dung={gt_c:7.1f} -> APE={e2:5.1f}%   [{tag}]")

# ComSuon Protocol-2 trong báo cáo = 15.4% (trên 5 ảnh)
com = [x for x in report if x[1] == "ComSuon"]
ej = [abs(p - g) / g * 100 for _, _, _, p, g in com]
ec = [abs(p - corrected.get(f, g)) / corrected.get(f, g) * 100 for f, _, _, p, g in com]
print(f"\n  ComSuon Protocol-2 MAPE  ->  nhu bao cao: {sum(ej)/len(ej):.1f}%   |   "
      f"khi dung GT hieu chinh: {sum(ec)/len(ec):.1f}%")
print(f"  (Báo cáo ghi 15.4% và khẳng định 'toàn bộ 5 ảnh Cơm sườn đạt sai số 9.1–24.7%')")
