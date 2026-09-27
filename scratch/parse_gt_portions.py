import os
import re
import json

txt_path = 'portion-estimation_val.txt'
with open(txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

records = []
for idx, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    # Extract quoted path and the rest
    parts = line.split('",', 1)
    if len(parts) == 2:
        img_path = parts[0].strip('" ')
        raw_info = parts[1].strip().rstrip(';')
        
        # Parse portions
        # Formats:
        # 1. "portion: 819.5"
        # 2. "Bun bo Hue: 804.5, Rau: 90"
        # 3. "banh mi: 99.3, trung 221.2"
        # 4. "Bun rieu: 838.9, Rau: 70"
        components = {}
        items = [x.strip() for x in raw_info.split(',') if x.strip()]
        for it in items:
            if ':' in it:
                cname, cval = it.split(':', 1)
                components[cname.strip()] = float(cval.strip())
            else:
                # e.g., "trung 221.2"
                tokens = it.rsplit(None, 1)
                if len(tokens) == 2:
                    components[tokens[0].strip()] = float(tokens[1].strip())
        
        total_g = sum(components.values())
        records.append({
            "line": idx + 1,
            "path": img_path,
            "filename": os.path.basename(img_path),
            "folder": os.path.basename(os.path.dirname(img_path)),
            "components": components,
            "total_mass_g": round(total_g, 2),
            "exists": os.path.exists(img_path)
        })

print(f"Total parsed records: {len(records)}")
missing_files = [r for r in records if not r["exists"]]
print(f"Files existing on disk: {len(records) - len(missing_files)} / {len(records)}")
if missing_files:
    print("Missing files:", [r["filename"] for r in missing_files])

print("\nSample records:")
for r in records[:5]:
    print(r)

output_path = 'data/real_images_ground_truth.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({"records": records, "total_images": len(records)}, f, ensure_ascii=False, indent=2)
print(f"\nSaved structured ground truth to {output_path}")
