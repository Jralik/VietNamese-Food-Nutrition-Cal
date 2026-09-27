import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("data/nin_dishes.json", "r", encoding="utf-8") as f:
    dishes = json.load(f)["data"]

with open("data/nin_ingredients.json", "r", encoding="utf-8") as f:
    ingredients = json.load(f)["data"]

print("Sample dish 1:", dishes[0]["name_vi"])
for c in dishes[0].get("nutritional_components", []):
    print("  ", c.get("key"), ":", c.get("amount"), c.get("unit_name"))

print("\nSample ingredient 1:", ingredients[0]["name"])
for c in ingredients[0].get("nutritional_components", [])[:10]:
    print("  ", c.get("key"), ":", c.get("value"), c.get("unit"))
