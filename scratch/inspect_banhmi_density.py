import json

lines = open('scratch/full_eval_records.jsonl', encoding='utf-8').readlines()
for l in lines:
    d = json.loads(l)
    img = d['image']
    if 'banh-mi' in img.lower() or 'banhmi' in img.lower():
        print(f"=== Image: {img}")
        print(f"  GT: {d.get('gt')}")
        total_vol = 0
        total_mass = 0
        for e in d.get('est', []):
            cname = e['class_name']
            m = e['mass_g']
            v = e['volume_cm3']
            print(f"  Est: {cname} | mass: {m}g | vol: {v}cm3 | method: {e.get('method')}")
            if 'banh mi' in cname.lower():
                total_vol += v
                total_mass += m
        print(f"  Total BM vol: {total_vol:.1f} cm3, Total BM mass: {total_mass:.1f} g")
        if total_vol > 0:
            print(f"  Current implied density used: {total_mass / total_vol:.4f}")
