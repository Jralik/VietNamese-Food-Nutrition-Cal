"""
Centralized configuration for the Food Volume Estimation Pipeline.

All model paths, variants, and tunable thresholds live here.
Change one line to swap SAM2 size, Depth Anything encoder, etc.
"""

import os

# ---------------------------------------------------------------------------
# Segmentation backend
# ---------------------------------------------------------------------------
# "sam2"    = SAM2Segmenter (SAM 2.1 box-prompted, proven default)
# "foodsam" = FoodSAMSegmenter (FoodSAM stack: SAM2.1 AMG + SETR-MLA
#             FoodSeg103 semantic + composite matching, runs in the isolated
#             FoodSAM env via subprocess — see foodsam_segmenter.py)
SEGMENTATION_BACKEND = "sam2"

# ---------------------------------------------------------------------------
# FoodSAM (external env — no dependencies installed into this project)
# ---------------------------------------------------------------------------
FOODSAM_REPO = r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodSAM"
FOODSAM_INFER_SCRIPT = os.path.join(FOODSAM_REPO, "foodsam_infer.py")
FOODSAM_POINTS_PER_SIDE = 16     # AMG grid density (paper default); 16 = 4x faster
FOODSAM_TIMEOUT_S = 300          # subprocess budget per image (model load + inference)

# ---------------------------------------------------------------------------
# FoodSAM ingredient nutrition (FoodSeg103 -> VietFood68)
# ---------------------------------------------------------------------------
# SETR semantic components that map onto a VietFood68 class become nutrition
# items. MODE:
#   "off"                 - feature disabled
#   "breakdown_only"      - components associated with a dish only reported as
#                           per-dish breakdown metadata, no extra items
#   "breakdown_and_extra" - components outside dish regions become extra
#                           nutrition items (one per connected component)
FOODSAM_INGREDIENT_MODE = "breakdown_and_extra"

# FoodSeg103 category name (exact strings from foodseg103_category_id.txt)
# -> VietFood68 class name. Many-to-one by design (5 mushroom classes -> Nam).
# mapping_type:
#   "direct"      - same foodstuff
#   "approximate" - close substitute (sausage ~ Cha; steak ~ Thit bo)
#   "generic"     - coarse grouping (many vegetables -> Rau)
FOODSAM_INGREDIENT_MAP = {
    "egg":               {"class_name": "Trung (Egg)",                    "mapping_type": "direct"},
    "rice":              {"class_name": "Com (Rice)",                     "mapping_type": "direct"},
    "fish":              {"class_name": "Ca (Fish)",                      "mapping_type": "direct"},
    "crab":              {"class_name": "Cua (Crab)",                     "mapping_type": "direct"},
    "shrimp":            {"class_name": "Tom (Shrimp)",                   "mapping_type": "direct"},
    "tomato":            {"class_name": "Ca chua (Tomato)",               "mapping_type": "direct"},
    "cucumber":          {"class_name": "Dua leo (Cucumber)",             "mapping_type": "direct"},
    "carrot":            {"class_name": "Ca rot (Carrot)",                "mapping_type": "direct"},
    "tofu":              {"class_name": "Dau hu (Tofu)",                  "mapping_type": "direct"},
    "cauliflower":       {"class_name": "Bong cai (Cauliflower)",         "mapping_type": "direct"},
    "french fries":      {"class_name": "Khoai tay chien (French fries)", "mapping_type": "direct"},
    "cheese butter":     {"class_name": "Pho mai (Cheese)",               "mapping_type": "direct"},
    "steak":             {"class_name": "Thit bo (Beef)",                 "mapping_type": "approximate"},
    "pork":              {"class_name": "Thit heo (Pork)",                "mapping_type": "approximate"},
    "chicken duck":      {"class_name": "Thit ga (Chicken)",              "mapping_type": "approximate"},
    "noodles":           {"class_name": "Bun (Rice vermicelli)",          "mapping_type": "approximate"},
    "bread":             {"class_name": "Banh mi (Vietnamese baguette sandwich)", "mapping_type": "approximate"},
    "sausage":           {"class_name": "Cha (Vietnamese pork roll)",     "mapping_type": "approximate"},
    "fried meat":        {"class_name": "Thit nuong (Grilled meat)",      "mapping_type": "approximate"},
    "pepper":            {"class_name": "Ot chuong (Bell pepper)",        "mapping_type": "approximate"},
    "soup":              {"class_name": "Canh (Soup)",                    "mapping_type": "approximate"},
    "hamburg":           {"class_name": "Hamburger",                      "mapping_type": "approximate"},
    "eggplant":          {"class_name": "Ca phao (Pickled eggplant)",     "mapping_type": "generic"},
    "salad":             {"class_name": "Salad (Salad)",                  "mapping_type": "generic"},
    "lettuce":           {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "rape":              {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "cabbage":           {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "cilantro mint":     {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "broccoli":          {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "snow peas":         {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "green beans":       {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "French beans":      {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "bean sprouts":      {"class_name": "Rau (Vegetables)",               "mapping_type": "generic"},
    "shiitake":          {"class_name": "Nam (Mushroom)",                 "mapping_type": "generic"},
    "enoki mushroom":    {"class_name": "Nam (Mushroom)",                 "mapping_type": "generic"},
    "oyster mushroom":   {"class_name": "Nam (Mushroom)",                 "mapping_type": "generic"},
    "king oyster mushroom": {"class_name": "Nam (Mushroom)",              "mapping_type": "generic"},
    "white button mushroom": {"class_name": "Nam (Mushroom)",             "mapping_type": "generic"},
}

# Thresholds (hyperparameters — tune on the dev split ONLY, then freeze;
# document as experimentally chosen in the thesis).
# Component classification: r = |C ∩ U| / area(C)   (U = union of matched
# dish bboxes; the denominator is the COMPONENT area, not the bbox area).
FOODSAM_INGREDIENT_OVERLAP_BREAKDOWN = 0.5   # r >= this -> dish-associated (breakdown)
FOODSAM_INGREDIENT_MIN_AREA_RATIO = 0.008    # component/image area floor (noise filter)
FOODSAM_INGREDIENT_LARGE_AREA_FLAG = 0.5     # > this -> suspicious_large flag (kept)
FOODSAM_INGREDIENT_MAX_AREA_RATIO = 0.75     # > this -> hard reject (extreme case only)
FOODSAM_INGREDIENT_MIN_PURITY = 0.6          # majority-vote label ratio floor

# Owner disambiguation: top-2 owner scores closer than this -> "ambiguous"
FOODSAM_INGREDIENT_OWNER_TIE_MARGIN = 0.05

# ---------------------------------------------------------------------------
# Mixed-plate contradiction heuristic
# ---------------------------------------------------------------------------
# When YOLO labels a whole mixed meal plate as one staple dish (e.g. "Xoi"
# for a plate of rice+beef+chicken+vegetables), the SETR components inside
# that bbox are dish-associated (breakdown) and never become nutrition items.
# If the composition CONTRADICTS the dish hypothesis, the dish is removed
# from nutrition accounting (audit record kept) and its large breakdown
# components are promoted into ingredient items.
#
# Terminology: a *contradiction heuristic* — the system concludes the dish
# prediction is UNRELIABLE, not that it is objectively wrong.
# Measured note (do NOT use YOLO confidence as a gate): in the target failure
# mode the wrong dish scored 0.88-0.95 while the correct "Thit bo (Beef)"
# scored 0.116 — confidence discriminates backwards here.
FOODSAM_MIXED_PLATE_SUSPECT_CLASSES = {
    "Xoi (Sticky rice)",
    "Com (Rice)",
    "Com tam (Broken rice)",
    "Bun (Rice vermicelli)",
    "Banh mi (Vietnamese baguette sandwich)",
    "Mi (Egg noodles)",
    "Mi Quang (Quang-style noodles)",
    "Hu tieu (Clear rice noodle soup)",
    "Banh canh (Vietnamese thick noodle soup)",
    "Cao lau (Cao lau noodles)",
}
# Suppression requires: suspect class + bbox >= MIN_BBOX_RATIO of the image
# + >= MIN_NONSTAPLE distinct POST-MAPPING ingredient classes (different from
# the dish class; carrot+tomato+cucumber count as ONE "Rau" group), each with
# total pixel area >= GROUP_MIN_RATIO of the image.
FOODSAM_MIXED_PLATE_MIN_NONSTAPLE = 3
FOODSAM_MIXED_PLATE_GROUP_MIN_RATIO = 0.002   # group counting (suppression decision)
FOODSAM_MIXED_PLATE_MIN_BBOX_RATIO = 0.10
# Independent knob: breakdown components >= this image fraction are promoted
# into items when their dish is suppressed (smaller ones stay metadata — the
# information gap is MEASURED as promoted_area_coverage, not assumed zero).
FOODSAM_INGREDIENT_PROMOTE_MIN_RATIO = 0.001

# Post-processing (kept separate from SAM2's so tuning one does not move the other)
FOODSAM_ERODE_KERNEL_SIZE = 3
FOODSAM_OVERSIZED_MASK_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# SAM2 Segmentation
# ---------------------------------------------------------------------------
# Supported: "sam2.1_hiera_tiny", "sam2.1_hiera_small",
#            "sam2.1_hiera_base_plus", "sam2.1_hiera_large"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_small.yaml"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"

# Post-processing
SAM2_ERODE_KERNEL_SIZE = 3          # boundary erosion to strip plate/bowl rim
SAM2_MIN_MASK_RATIO = 2.0 / 3.0     # mask must occupy at least > 2/3 (~66.7%) of bbox
SAM2_OVERSIZED_MASK_THRESHOLD = 0.95  # flag mask if area > 95% of bbox

# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------
# Supported encoder: "vits" (~25MB/90MB, fast GPU), "vitb" (~100MB), "vitl" (~350MB)
DEPTH_ENCODER = "vits"
DEPTH_USE_METRIC = True              # True = metric checkpoint (meters output)
DEPTH_MAX_DEPTH = 20.0               # max depth in meters (indoor setting)
DEPTH_INPUT_SIZE = 518               # model input resolution

# HuggingFace model IDs for metric depth
DEPTH_HF_MODELS = {
    "vits": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}

# ---------------------------------------------------------------------------
# ArUco / Scale Recovery
# ---------------------------------------------------------------------------
ARUCO_DICT_TYPE = "DICT_4X4_50"
# Physical size = side of the BLACK square of the marker, NOT counting the
# white quiet-zone border (this matches OpenCV's corner convention, which
# reports the 4 corners of the black area). Measured: 30mm.
ARUCO_MARKER_LENGTH_MM = 30.0

# Checkerboard calibration target (same board as "Camera calibration"):
# squares are 30mm; the runtime scale recovery detects this pattern to get a
# high-confidence mm/px scale when no ArUco marker is visible.
CHESSBOARD_PATTERN_SIZE = (8, 6)    # inner corners (cols, rows) → 9x7 squares
CHESSBOARD_SQUARE_MM = 30.0

# Plate/bowl diameter heuristics (mm) — used when no ArUco marker detected
# Mapping: dish_type → estimated diameter in mm
PLATE_DIAMETER_HEURISTICS = {
    "flat_plate":        220.0,  # đĩa cơm tấm, đĩa gỏi cuốn, bánh xèo, ...
    "soup_bowl":         180.0,  # bát phở, bát bún bò, hủ tiếu, canh, ...
    "large_bowl":        220.0,  # bát canh lớn, nồi lẩu, ...
    "rice_bowl":         120.0,  # chén cơm, chén xôi, ...
    "small_plate":       150.0,  # đĩa nhỏ chả, đĩa bún chả, khoai tây chiên, ...
    "default":           200.0,  # fallback
}

# Container dish types (valid for plate/bowl circle/ellipse scale recovery)
CONTAINER_DISH_TYPES = {"soup_bowl", "large_bowl", "rice_bowl", "flat_plate", "small_plate"}

# Complete mapping for all 68 classes in VietFood dataset
CLASS_TO_DISH_TYPE = {
    # ── Bowls (Noodle Soups, Porridge, Soups) ──
    "Banh canh (Vietnamese thick noodle soup)":          "soup_bowl",
    "Bo kho (Beef stew)":                                "soup_bowl",
    "Bun (Rice vermicelli)":                             "flat_plate",
    "Bun bo Hue (Hue beef noodle soup)":                 "soup_bowl",
    "Bun cha (Grilled pork with vermicelli)":            "small_plate",
    "Bun cha ca (Fish cake noodle soup)":                "soup_bowl",
    "Bun dau (Vermicelli with tofu)":                    "flat_plate",
    "Bun mam (Fermented fish noodle soup)":              "soup_bowl",
    "Bun rieu (Crab noodle soup)":                       "soup_bowl",
    "Canh (Soup)":                                       "soup_bowl",
    "Cao lau (Cao lau noodles)":                         "soup_bowl",
    "Chao long (Pork organ congee)":                     "soup_bowl",
    "Hu tieu (Clear rice noodle soup)":                  "soup_bowl",
    "Kho qua thit (Stuffed bitter melon soup)":          "soup_bowl",
    "Lau (Hotpot)":                                      "large_bowl",
    "Mi (Egg noodles)":                                  "soup_bowl",
    "Mi Quang (Quang-style noodles)":                    "soup_bowl",
    "Pho (Vietnamese noodle soup)":                      "soup_bowl",
    "Sup cua (Crab soup)":                               "soup_bowl",
    "Thit kho (Braised pork)":                           "rice_bowl",
    "Com (Rice)":                                        "rice_bowl",
    "Xoi (Sticky rice)":                                 "rice_bowl",

    # ── Plates (Main Flat Plates) ──
    "Banh beo (Vietnamese savory steamed rice cake)":    "flat_plate",
    "Banh chung (Square sticky rice cake)":              "flat_plate",
    "Banh cuon (Rolled rice pancake)":                   "flat_plate",
    "Banh khot (Mini savory pancakes)":                  "flat_plate",
    "Banh mi (Vietnamese baguette sandwich)":            "flat_plate",
    "Banh trang (Rice paper)":                           "flat_plate",
    "Banh trang tron (Rice paper salad)":                "flat_plate",
    "Banh xeo (Vietnamese sizzling pancake)":            "flat_plate",
    "Bo la lot (Grilled beef wrapped in betel leaves)":  "flat_plate",
    "Ca (Fish)":                                         "flat_plate",
    "Cha gio (Spring rolls)":                            "flat_plate",
    "Com chien duong chau (Yangzhou fried rice)":        "flat_plate",
    "Com chien ga (Fried rice with chicken)":            "flat_plate",
    "Com tam (Broken rice)":                             "flat_plate",
    "Cua (Crab)":                                        "flat_plate",
    "Dau hu (Tofu)":                                     "flat_plate",
    "Goi cuon (Fresh spring rolls)":                     "flat_plate",
    "Hamburger":                                         "flat_plate",
    "Heo quay (Roast pork)":                             "flat_plate",
    "Long heo (Pork offal)":                             "flat_plate",
    "Muc (Squid)":                                       "flat_plate",
    "Nom hoa chuoi (Banana blossom salad)":              "flat_plate",
    "Nui xao bo (Stir-fried macaroni with beef)":        "flat_plate",
    "Oc (Snails)":                                       "flat_plate",
    "Thit bo (Beef)":                                    "flat_plate",
    "Thit ga (Chicken)":                                 "flat_plate",
    "Thit heo (Pork)":                                   "flat_plate",
    "Thit nuong (Grilled meat)":                         "flat_plate",
    "Tom (Shrimp)":                                      "flat_plate",

    # ── Small Plates ──
    "Cha (Vietnamese pork roll)":                        "small_plate",
    "Khoai tay chien (French fries)":                    "small_plate",
    "Pho mai (Cheese)":                                  "small_plate",

    # ── Side Vegetables (Leafy / Loose Mound Structure) ──
    "Rau (Vegetables)":                                  "side_vegetables",
    "Salad (Salad)":                                     "side_vegetables",
    "Bong cai (Cauliflower)":                            "side_vegetables",
    "Nam (Mushroom)":                                    "side_vegetables",

    # ── Small Ingredients / Garnishes (Never treated as separate 200mm plates!) ──
    "Chanh (Lime)":                                      "ingredient_small",
    "Ca chua (Tomato)":                                  "ingredient_small",
    "Ca phao (Pickled eggplant)":                        "ingredient_small",
    "Ca rot (Carrot)":                                   "ingredient_small",
    "Cu kieu (Pickled scallion head)":                   "ingredient_small",
    "Dua chua (Pickled vegetables)":                     "ingredient_small",
    "Dua leo (Cucumber)":                                "ingredient_small",
    "Ot chuong (Bell pepper)":                           "ingredient_small",
    "Trung (Egg)":                                       "ingredient_small",

    # ── Non-Food ──
    "Con nguoi (Human)":                                 "non_food",
}

# Container 3D Geometry Constants
BOWL_DEPTH_RATIO = 0.40       # H_bowl = 0.40 * D_rim
BOWL_BASE_RATIO = 0.52        # R_base = 0.52 * R_rim
BOWL_DEFAULT_FILL_RATIO = 0.78  # Standard fill fraction when depth is uniform/flat
# Fill fraction when the measured soup-surface recess is within noise of the
# rim plane (delta_z <= BOWL_DELTA_Z_NOISE_MM — a sign flip means the depth
# model cannot resolve the cavity; assuming a full-to-the-brim bowl then
# grossly overestimates). Calibrated: one bun-bo-hue bowl, 5 shots, measured
# 700-800 cm3.
BOWL_NOISE_FILL_RATIO = 0.70
BOWL_DELTA_Z_NOISE_MM = 4.0
# Max headroom as a fraction of bowl depth (was 0.25 — a 750 cm3 fill in a
# ~190 mm soup bowl sits ~0.30 * H below the rim)
BOWL_HEADROOM_MAX_RATIO = 0.30
# Calibrated rim diameters are trusted only within +-10% of the dish-type
# prior: the mask-based rim measurement varies ~+-14% across shots of the
# SAME bowl (mask coverage), and bowl volume scales with D^3.
BOWL_DIAMETER_TOLERANCE = 0.10

# ---------------------------------------------------------------------------
# Camera Intrinsics (fallback when no calibration file available)
# ---------------------------------------------------------------------------
CAMERA_CALIB_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Camera calibration", "camera_calib.npz"
)
# Heuristic: fx = fy ≈ max(W, H) for typical smartphone camera (used if calib file not found)
CAMERA_FOCAL_LENGTH_HEURISTIC = True

# ---------------------------------------------------------------------------
# Volume / Nutrition
# ---------------------------------------------------------------------------
# Reference plate height offset — how far food sits above plate surface (mm)
FOOD_HEIGHT_OFFSET_MM = 5.0

# Confidence labels
CONFIDENCE_HIGH = "high"       # ArUco marker detected, metric depth
CONFIDENCE_MEDIUM = "medium"   # plate heuristic or relative depth
CONFIDENCE_LOW = "low"         # bbox fallback, no depth

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
