import re, collections, io, os, json

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def counts(fname, pattern):
    p = os.path.join(base, fname)
    if not os.path.exists(p):
        return {'MISSING': 1}
    s = io.open(p, encoding='utf-8').read()
    return collections.Counter(re.findall(pattern, s))

print("== density_db.py source labels ==")
for k, v in counts('density_db.py', r'"([a-z_]*(?:fdc|nin_2017|literature|estimate|measured|displacement)[a-z_0-9]*)"\s*[,}\)]').most_common():
    print(f"  {k:32s} {v}")

print("\n== density_db total entries ==")
s = io.open(os.path.join(base, 'density_db.py'), encoding='utf-8').read()
print("  DENSITY_DB triples:", len(re.findall(r'\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*"', s)))

print("\n== class_names.py provenance keywords ==")
for k, v in counts('class_names.py', r'"(vn_nin[^"]*|usda[^"]*|estimated[^"]*|derived[^"]*|assumed[^"]*)"').most_common(12):
    print(f"  {k:40s} {v}")

print("\n== water displacement references across repo ==")
hits = 0
for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d not in ('.venv', '.git', '__pycache__', 'ultralytics', 'node_modules')]
    for f in files:
        if f.endswith(('.py', '.md', '.txt')):
            p = os.path.join(root, f)
            try:
                t = io.open(p, encoding='utf-8', errors='ignore').read().lower()
            except Exception:
                continue
            if 'water displacement' in t or 'water_displacement' in t:
                print("  HIT:", p.replace(base, ''))
                hits += 1
print("  total hits:", hits)

print("\n== mAP claims in md/txt ==")
for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d not in ('.venv', '.git', '__pycache__', 'ultralytics')]
    for f in files:
        if f.endswith(('.md',)) and 'README' in f.upper():
            p = os.path.join(root, f)
            t = io.open(p, encoding='utf-8', errors='ignore').read()
            for m in re.finditer(r'.{0,60}mAP.{0,60}', t):
                print(f"  {f}: {m.group(0).strip()}")
