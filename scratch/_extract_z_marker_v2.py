"""Extract z_marker_mm (depth_mm at ArUco marker corners) for all 418 cached scenes."""
import json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc
from scale_recovery import ScaleRecovery

MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = ROOT / "scratch" / "bench_cache_v2"
DEPTH_DIR = CACHE_DIR / "depth_ctx"
OUT_PATH = CACHE_DIR / "z_marker_cache.json"

mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
inst_by_file = defaultdict(list)
for x in mp["instances"]:
    inst_by_file[x["file_name"]].append(x)

sr = ScaleRecovery()
out = {}

for idx, fn in enumerate(sorted(inst_by_file.keys())):
    stem = Path(fn).stem
    sc = np.load(DEPTH_DIR / f"scene_{stem}.npz", allow_pickle=True)
    scale_src = str(sc["scale_source"])
    K = sc["K"]
    depth_mm = sc["depth_mm"].astype(np.float32)
    h_vol, w_vol = int(sc["h_vol"]), int(sc["w_vol"])
    sc.close()

    if scale_src != "aruco_marker":
        out[fn] = {"z_marker_depth_mm": None, "z_marker_tvec_mm": None}
        continue

    img_path = inst_by_file[fn][0]["image_path"]
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil.thumbnail((2000, 2000))
    vol_bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    sres = sr.recover(vol_bgr, K=K, dist=None, detected_classes=[], yolo_detections=[], seg_results=None)
    z_d = None
    z_t = None
    if sres.marker_corners is not None:
        m_mask = np.zeros(depth_mm.shape[:2], dtype=np.uint8)
        cv2.fillPoly(m_mask, [sres.marker_corners.astype(np.int32)], 1)
        md = depth_mm[m_mask > 0]
        vmd = md[md > 0]
        if len(vmd) > 0:
            z_d = float(np.median(vmd))
    if sres.marker_tvec is not None:
        z_t = float(np.ravel(sres.marker_tvec)[2])
    out[fn] = {"z_marker_depth_mm": z_d, "z_marker_tvec_mm": z_t}

    if (idx + 1) % 100 == 0 or (idx + 1) == len(inst_by_file):
        print(f"[z_marker] {idx+1}/{len(inst_by_file)}", flush=True)

OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
valid = sum(1 for v in out.values() if v["z_marker_depth_mm"] is not None)
print(f"Saved z_marker for {valid}/{len(out)} scenes to {OUT_PATH}")
