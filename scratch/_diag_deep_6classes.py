import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
from eval_segmentation_benchmark import coco_poly_to_mask
from volume_nutrition import fit_plane_to_boundary, relief_scale_factor

COCO_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
DEPTH_DIR = ROOT / "scratch" / "bench_cache_v2" / "depth_ctx"

mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
coco = json.loads(COCO_PATH.read_text(encoding="utf-8"))
ann_by_id = {a["id"]: a for a in coco["annotations"]}
del coco

inst_by_file = defaultdict(list)
for x in mp["instances"]:
    inst_by_file[x["file_name"]].append(x)

for target_cls in ("Rau", "CaChua", "ThitNuong", "NuiXaoBo", "BanhTrang", "KhoQuaThit"):
    print(f"\n=== {target_cls} ===")
    for fn in sorted(inst_by_file.keys()):
        insts = [x for x in inst_by_file[fn] if x["short_class"] == target_cls and x["gt_weight_g"] is not None]
        if not insts:
            continue
        stem = Path(fn).stem
        h_full, w_full = int(insts[0]["height"]), int(insts[0]["width"])
        sc = np.load(DEPTH_DIR / f"scene_{stem}.npz", allow_pickle=True)
        depth_mm = sc["depth_mm"].astype(np.float32)
        K = sc["K"]
        mm_px = float(sc["mm_per_pixel"])
        scale_src = str(sc["scale_source"])
        d_anc = bool(sc["depth_anchored"])
        h_vol, w_vol = int(sc["h_vol"]), int(sc["w_vol"])
        sx, sy = float(sc["sx"]), float(sc["sy"])
        sc.close()

        for inst in insts:
            gf = coco_poly_to_mask(ann_by_id[inst["annotation_id"]]["segmentation"], h_full, w_full)
            gv = cv2.resize(gf.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if gf.shape[:2] != (h_vol, w_vol) else gf
            area_cm2 = float(gv.sum() * (mm_px ** 2) / 100.0)
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            w_mm = (bx2 - bx1) * sx * mm_px
            h_mm = (by2 - by1) * sy * mm_px
            plane_z = fit_plane_to_boundary(depth_mm, gv, alpha=0.79 if d_anc else 1.0)
            plane_d = plane_z[gv]
            z_med = float(np.median(plane_d[plane_d > 0])) if (plane_d > 0).any() else 0.0
            alpha = 0.79 if d_anc else relief_scale_factor(z_med, mm_px, K)
            raw_h = (plane_d - depth_mm[gv]) * alpha
            pos_h = np.clip(raw_h, 0.0, 55.0)
            p90 = float(np.percentile(raw_h, 90))
            med = float(np.median(raw_h))
            mean_pos = float(pos_h.mean())
            print(f"  {fn:30s} | {scale_src[:5]} | anc={int(d_anc)} | mm_px={mm_px:.3f} | area={area_cm2:5.1f} | d={max(w_mm,h_mm):5.1f} | med={med:5.1f} | mean+={mean_pos:4.1f} | p90={p90:5.1f} | gt={inst['gt_weight_g']:5.1f}")
        del depth_mm
