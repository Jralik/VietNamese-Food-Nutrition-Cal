import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open('data/nin_nutrition_canonical.json', 'r', encoding='utf-8') as f:
    canon = json.load(f)["records"]

rec_map = {r['id']: r for r in canon}

ids = [
    'nin_ing_6877a6b660d6c84e9bd5cb57',
    'nin_ing_6877a6b660d6c84e9bd5cb58',
    'nin_ing_6877a6b660d6c84e9bd5cb59',
    'nin_ing_6877a6b660d6c84e9bd5cd17'
]

for i in ids:
    r = rec_map[i]
    n = r['nutrition_per_100g']
    print(f"ID: {r['id']}")
    print(f"Name: {r['name_vi']}")
    print(f"Energy: {n['energy_kcal']} kcal | Protein: {n['protein_g']}g | Fat: {n['fat_g']}g | Carbs: {n['carbs_g']}g")
    print(f"Ca: {n.get('calcium_mg')}mg | Fe: {n.get('iron_mg')}mg | Na: {n.get('sodium_mg')}mg | VitC: {n.get('vitamin_c_mg')}mg\n")
