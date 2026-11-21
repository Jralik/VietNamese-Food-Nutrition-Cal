import json
from collections import defaultdict

lines = open('scratch/full_eval_records.jsonl', encoding='utf-8').readlines()
classes = defaultdict(list)

for l in lines:
    d = json.loads(l)
    img = d['image']
    gt = d.get('gt', {})
    est = d.get('est', [])
    
    # group est by class_name
    est_by_cls = defaultdict(lambda: {'mass': 0.0, 'vol': 0.0, 'count': 0})
    for e in est:
        c = e['class_name'].split(' (')[0]
        est_by_cls[c]['mass'] += e['mass_g']
        est_by_cls[c]['vol'] += e['volume_cm3']
        est_by_cls[c]['count'] += 1

    # check gt
    # class mapping id to short
    from class_names import class_names
    for cid_str, g in gt.items():
        cid = int(cid_str)
        cname = class_names[cid]['name'].split(' (')[0]
        gt_mass = g.get('grams_total')
        if gt_mass is not None and gt_mass > 0:
            est_val = est_by_cls.get(cname)
            classes[cname].append({
                'img': img,
                'gt_mass': gt_mass,
                'gt_count': g.get('count'),
                'est_mass': est_val['mass'] if est_val else 0.0,
                'est_vol': est_val['vol'] if est_val else 0.0,
                'est_count': est_val['count'] if est_val else 0,
            })

print(f"{'Class':<15} | {'N':<3} | {'GT Mean':<8} | {'Est Mean':<8} | {'MAE':<8} | {'MAPE %':<8} | {'Avg Implied Density'}")
print("-" * 80)
for cname, items in sorted(classes.items()):
    n = len(items)
    gt_total = sum(x['gt_mass'] for x in items)
    est_total = sum(x['est_mass'] for x in items)
    mae = sum(abs(x['est_mass'] - x['gt_mass']) for x in items) / n
    mape = sum(abs(x['est_mass'] - x['gt_mass']) / x['gt_mass'] for x in items) / n * 100
    
    # implied density = gt_mass / est_vol for items where est_vol > 0
    densities = [x['gt_mass'] / x['est_vol'] for x in items if x['est_vol'] > 0]
    avg_dens = sum(densities) / len(densities) if densities else 0
    
    print(f"{cname:<15} | {n:<3} | {gt_total/n:<8.1f} | {est_total/n:<8.1f} | {mae:<8.1f} | {mape:<8.1f}% | {avg_dens:.3f}")
