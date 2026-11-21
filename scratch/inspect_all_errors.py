import json
from class_names import class_names

lines = open('scratch/full_eval_records.jsonl', encoding='utf-8').readlines()

errors = []
for l in lines:
    d = json.loads(l)
    img = d['image']
    gt = d.get('gt', {})
    est = d.get('est', [])
    for cid_str, g in gt.items():
        cid = int(cid_str)
        cname = class_names[cid]['name'].split(' (')[0]
        gt_m = g.get('grams_total')
        if not gt_m:
            continue
        
        # matched est
        est_items = [e for e in est if e['class_name'].split(' (')[0] == cname]
        est_m = sum(e['mass_g'] for e in est_items)
        est_v = sum(e['volume_cm3'] for e in est_items)
        err = abs(est_m - gt_m)
        pe = err / gt_m * 100
        if pe > 25.0:
            errors.append({
                'img': img,
                'class': cname,
                'gt_m': gt_m,
                'est_m': est_m,
                'est_v': est_v,
                'err': err,
                'pe': pe,
                'count_gt': g.get('count'),
                'count_est': len(est_items)
            })

errors.sort(key=lambda x: x['pe'], reverse=True)
print(f'Total high-error targets (>25%): {len(errors)}')
print(f"{'Image':<22} | {'Class':<12} | {'GT(g)':<7} | {'Est(g)':<7} | {'Vol(cm3)':<8} | {'MAPE %':<7} | {'Cnt:Est/GT'}")
print("-" * 80)
for e in errors:
    print(f"{e['img']:<22} | {e['class']:<12} | {e['gt_m']:<7.1f} | {e['est_m']:<7.1f} | {e['est_v']:<8.1f} | {e['pe']:<7.1f} | {e['count_est']}/{e['count_gt']}")
