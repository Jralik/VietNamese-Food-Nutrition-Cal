import sys, json

sys.stdout.reconfigure(encoding='utf-8')
with open('data/vietfood68_nin_mapping.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)['mappings']

print(f"Total mappings: {len(mappings)}")
target_nutrs = ['calcium_mg', 'iron_mg', 'zinc_mg', 'sodium_mg', 'cholesterol_mg']

missing_nutrs = []
for m in mappings:
    nutr = m.get('primary_nutrition_per_100g')
    if m['mapping_type'] == 'unmapped':
        print(f"Unmapped: {m['full_name']}")
        print(f"   rationale: {m['decision_rationale']}")
        print(f"   nutrition: {nutr}")
    else:
        if not nutr:
            missing_nutrs.append((m['vietfood_class_id'], m['full_name'], 'missing primary_nutrition_per_100g'))
        else:
            for tn in target_nutrs:
                if tn not in nutr:
                    missing_nutrs.append((m['vietfood_class_id'], m['full_name'], f"missing {tn}"))

print(f"\nMissing nutrients count across 65 mapped classes: {len(missing_nutrs)}")
if missing_nutrs:
    for mn in missing_nutrs[:10]:
        print(f"   {mn}")
else:
    print("ALL 65 MAPPED CLASSES HAVE 100% OF THE 5 TARGET MICRONUTRIENTS!")
