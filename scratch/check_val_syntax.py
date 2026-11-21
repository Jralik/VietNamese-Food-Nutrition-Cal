import os
import sys
import re

sys.path.insert(0, os.path.abspath("."))
from eval_full_pipeline import VAL_FILE
from eval_val_detection import SHORT_TO_ID, TOKEN_ALIASES
from class_names import class_names

print('Total classes in class_names:', len(class_names))
print('Checking tokens in portion-estimation_val.txt...')

with open(VAL_FILE, encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

print(f'Total non-empty lines: {len(lines)}')

all_tokens = set()
missing_files = []
syntax_issues = []

for line in lines:
    m = re.match(r'^"(.+?)",\s*(.+?)\s*;?\s*$', line)
    if not m:
        syntax_issues.append(('Line match fail', line))
        continue
    img_path, rest = m.group(1), m.group(2)
    if not os.path.exists(img_path):
        missing_files.append(img_path)
    for part in rest.split(','):
        part = part.strip()
        m1 = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*:\s*(\d+)\s*-\s*([\d.]+)\s*g?(?:\s*\(tong\))?$", part)
        m2 = re.match(r"^(\d+)\s*-\s*([A-Za-z][A-Za-z ]*?)\s+([\d.]+)\s*g?$", part)
        m3 = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*:\s*([\d.]+)\s*g?$", part)
        mq = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*:\s*(\d+)\s*-\s*\?$", part)
        mq2 = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*:\s*\?$", part)
        name = None
        if m1: name = m1.group(1).strip()
        elif m2: name = m2.group(2).strip()
        elif m3: name = m3.group(1).strip()
        elif mq: name = mq.group(1).strip()
        elif mq2: name = mq2.group(1).strip()
        else:
            syntax_issues.append(('Part match fail', part, line))
        if name:
            all_tokens.add(name)

print(f'Missing image files count: {len(missing_files)} / {len(lines)}')
if missing_files:
    print('Sample missing:', missing_files[:3])

print('Tokens found:', sorted(all_tokens))
unmapped = [t for t in all_tokens if TOKEN_ALIASES.get(t, t) not in SHORT_TO_ID]
print('Unmapped tokens in SHORT_TO_ID:', unmapped)
if syntax_issues:
    print('Syntax issues:', syntax_issues)
