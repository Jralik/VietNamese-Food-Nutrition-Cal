import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    records = json.load(f)['records']

def search(q):
    res = [r for r in records if q.lower() in r['name_vi'].lower()]
    for r in res[:4]:
        basis = r.get("canonical_basis", "")
        cal = r["nutrition_per_100g"]["energy_kcal"] if r["nutrition_per_100g"] else 0
        print(f"  {r['id']}: {r['name_vi']} ({r['origin_type']}, {basis}, {cal} kcal)")

queries = [
    'cơm', 'sườn', 'bì', 'đậu phụ', 'mắm tôm', 'bún', 'chả viên', 'chả nướng',
    'lẩu', 'gỏi cuốn', 'bánh tráng', 'nộm', 'hoa chuối', 'bò', 'nui',
    'hủ tiếu', 'quảng', 'bánh xèo', 'cháo lòng', 'nem rán', 'riêu cua', 'mắm cá'
]

for query in queries:
    print(f"=== Query: {query} ===")
    search(query)
