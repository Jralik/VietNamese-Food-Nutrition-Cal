"""
Centralized configuration for the Food Volume Estimation Pipeline.

All model paths, variants, and tunable thresholds live here.
Change one line to swap SAM2 size, Depth Anything encoder, etc.
"""

import os

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
