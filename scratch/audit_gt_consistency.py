"""Kiểm tra tính nhất quán giữa Ground Truth nguồn (.txt) và JSON đã sinh ra."""
import os, re, json, collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT = os.path.join(BASE, 'portion-estimation_val.txt')
JSONP = os.path.join(BASE, 'data', 'real_images_ground_truth.json')

# ---- Parse lại txt bản hiện tại, có xử lý dạng "Name: <count> - <mass>" ----
raw_lines = open(TXT, encoding='utf-8').read().splitlines()
nonblank = [l for l in raw_lines if l.strip()]
print(f"txt: {len(raw_lines)} dòng tổng, {len(nonblank)} dòng có nội dung")

count_pattern = re.compile(r'^(.*?):\s*(\d+)\s*-\s*([\d.]+)\s*(g|)?$')
parsed = {}
unparsed, multi = [], []
for l in nonblank:
    parts = l.split('",', 1)
    if len(parts) != 2:
        unparsed.append(l)
        continue
    img = parts[0].strip('" ')
    info = parts[1].strip().rstrip(';')
    comps, total, has_count = {}, 0.0, False
    for it in [x.strip() for x in info.split(',') if x.strip()]:
        m = count_pattern.match(it)
        if m:
            name, cnt, mass = m.group(1).strip(), int(m.group(2)), float(m.group(3))
            comps[name] = mass * cnt
            total += mass * cnt
            has_count = True
            if cnt > 1:
                multi.append((os.path.basename(img), name, cnt, mass))
        elif ':' in it:
            n, v = it.split(':', 1)
            try:
                val = float(v.strip().rstrip('g'))
            except ValueError:
                val = float('nan')
            comps[n.strip()] = val
            total += val
        else:
            t = it.rsplit(None, 1)
            if len(t) == 2:
                try:
                    val = float(t[1].strip().rstrip('g'))
                except ValueError:
                    val = float('nan')
                comps[t[0].strip()] = val
                total += val
    parsed[os.path.basename(img)] = (round(total, 2), comps, has_count)

print(f"\nDòng không parse được: {len(unparsed)}")
print("\nCác dòng có COUNT > 1 (nhiều phần ăn / nhiều món trong 1 ảnh):")
for fn, name, cnt, mass in multi:
    print(f"  {fn:26s} {name:12s} x{cnt} @ {mass}g  -> GT đúng = {cnt*mass}g")

# ---- So sánh với JSON ----
db = json.load(open(JSONP, encoding='utf-8'))['records']
print(f"\nJSON: {len(db)} records   |  txt parsed: {len(parsed)} files")

mismatch = []
missing_in_txt = []
for r in db:
    fn = r['filename']
    if fn not in parsed:
        missing_in_txt.append(fn)
        continue
    txt_total, comps, had_count = parsed[fn]
    if abs(txt_total - r['total_mass_g']) > 0.51:
        mismatch.append((fn, r['total_mass_g'], txt_total, comps, had_count))

print(f"\n>>> GT JSON khác với GT txt: {len(mismatch)} / {len(db)} ảnh")
for fn, j, t, comps, hc in mismatch:
    print(f"  {fn:26s} JSON={j:8.1f}g   txt={t:8.1f}g   diff={t-j:+8.1f}g  count_form={hc}")
if missing_in_txt:
    print(f"\nẢnh có trong JSON nhưng không còn trong txt ({len(missing_in_txt)}): {missing_in_txt[:10]}")

# ---- Tác động lên MAPE toàn benchmark nếu dùng GT bản txt ----
print("\n== GT trùng lặp theo nhóm (cùng một con số cân cho nhiều ảnh) ==")
by_folder = collections.defaultdict(list)
for r in db:
    by_folder[r['folder']].append(r['total_mass_g'])
for f, vals in sorted(by_folder.items()):
    uniq = sorted(set(vals))
    print(f"  {f:12s} n={len(vals):3d}  distinct_GT={uniq}")
