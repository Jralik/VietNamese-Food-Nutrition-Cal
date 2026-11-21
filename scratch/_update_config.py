import os

# 1. Update pipeline_config.py
path_cfg = r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\pipeline_config.py"
with open(path_cfg, "r", encoding="utf-8") as f:
    text = f.read()

target_cfg = """# ---------------------------------------------------------------------------
# MobileSAM Segmentation (Ultralytics SAMPredictor + TinyViT)
# ---------------------------------------------------------------------------
MOBILESAM_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "mobile_sam.pt"
)
MOBILESAM_ERODE_KERNEL_SIZE = 3
MOBILESAM_MIN_MASK_RATIO = 2.0 / 3.0
MOBILESAM_OVERSIZED_MASK_THRESHOLD = 0.95"""

repl_cfg = """# ---------------------------------------------------------------------------
# MobileSAM Segmentation (Ultralytics SAMPredictor + TinyViT)
# ---------------------------------------------------------------------------
MOBILESAM_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "mobile_sam.pt"
)
# Candidate Area Filter (Config C / Config D):
#   Filters multimask candidates occupying >= 2/3 bbox area and picks min(area)
#   to capture the food boundary while discarding the outer bowl rim (>95%).
MOBILESAM_ENABLE_CANDIDATE_FILTER = True
MOBILESAM_MIN_MASK_RATIO = 2.0 / 3.0        # threshold (66.7% bbox) for candidate filter
# Boundary Erosion (False = Config C default; True = Config D with 3x3 erosion):
MOBILESAM_ENABLE_EROSION = False
MOBILESAM_ERODE_KERNEL_SIZE = 3
# Diagnostic threshold only (used by flag_oversized_mask, NOT a hard candidate filter cutoff):
MOBILESAM_OVERSIZED_MASK_THRESHOLD = 0.95"""

text_norm = text.replace("\r\n", "\n")
target_cfg_norm = target_cfg.replace("\r\n", "\n")
repl_cfg_norm = repl_cfg.replace("\r\n", "\n")

if target_cfg_norm in text_norm:
    text_norm = text_norm.replace(target_cfg_norm, repl_cfg_norm, 1)
    with open(path_cfg, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text_norm)
    print("pipeline_config.py updated successfully!")
else:
    print("Target not found in pipeline_config.py")

# 2. Update food_segmentation.py
path_seg = r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\food_segmentation.py"
with open(path_seg, "r", encoding="utf-8") as f:
    text_seg = f.read()

target_seg_init = """    def __init__(
        self,
        checkpoint: str = None,
        device: str = None,
        erode_kernel: int = None,
        min_mask_threshold: float = None,
        oversized_threshold: float = None,
    ):
        from pipeline_config import (
            MOBILESAM_CHECKPOINT,
            DEVICE,
            MOBILESAM_ERODE_KERNEL_SIZE,
            MOBILESAM_MIN_MASK_RATIO,
            MOBILESAM_OVERSIZED_MASK_THRESHOLD,
        )

        self.checkpoint = checkpoint or MOBILESAM_CHECKPOINT
        self.device = device or DEVICE
        self.erode_kernel = erode_kernel if erode_kernel is not None else MOBILESAM_ERODE_KERNEL_SIZE
        self.min_mask_threshold = (
            min_mask_threshold if min_mask_threshold is not None else MOBILESAM_MIN_MASK_RATIO
        )
        self.oversized_threshold = (
            oversized_threshold if oversized_threshold is not None else MOBILESAM_OVERSIZED_MASK_THRESHOLD
        )
        self._predictor = None
        self._available = None"""

repl_seg_init = """    def __init__(
        self,
        checkpoint: str = None,
        device: str = None,
        erode_kernel: int = None,
        min_mask_threshold: float = None,
        oversized_threshold: float = None,
        enable_candidate_filter: bool = None,
        enable_erosion: bool = None,
    ):
        from pipeline_config import (
            MOBILESAM_CHECKPOINT,
            DEVICE,
            MOBILESAM_ENABLE_CANDIDATE_FILTER,
            MOBILESAM_ENABLE_EROSION,
            MOBILESAM_ERODE_KERNEL_SIZE,
            MOBILESAM_MIN_MASK_RATIO,
            MOBILESAM_OVERSIZED_MASK_THRESHOLD,
        )

        self.checkpoint = checkpoint or MOBILESAM_CHECKPOINT
        self.device = device or DEVICE
        self.enable_candidate_filter = (
            enable_candidate_filter
            if enable_candidate_filter is not None
            else MOBILESAM_ENABLE_CANDIDATE_FILTER
        )
        self.enable_erosion = (
            enable_erosion if enable_erosion is not None else MOBILESAM_ENABLE_EROSION
        )
        self.erode_kernel = erode_kernel if erode_kernel is not None else MOBILESAM_ERODE_KERNEL_SIZE
        self.min_mask_threshold = (
            min_mask_threshold if min_mask_threshold is not None else MOBILESAM_MIN_MASK_RATIO
        )
        self.oversized_threshold = (
            oversized_threshold if oversized_threshold is not None else MOBILESAM_OVERSIZED_MASK_THRESHOLD
        )
        self._predictor = None
        self._available = None"""

target_seg_logic = """                    if raw_only:
                        best_idx = int(np.argmax(scores_np))
                        mask = cand_masks[best_idx]
                    else:
                        candidate_list = [cand_masks[i] for i in range(cand_masks.shape[0])]
                        valid_candidates = [
                            m for m in candidate_list
                            if (m.sum() / bbox_area) >= self.min_mask_threshold
                        ]
                        if valid_candidates:
                            mask = min(valid_candidates, key=lambda m: m.sum())
                        else:
                            mask = max(candidate_list, key=lambda m: m.sum())
                        if (mask.sum() / bbox_area) < self.min_mask_threshold:
                            fallback_fg = self._bbox_fallback_mask(image_rgb, bbox)
                            mask = mask | fallback_fg
                except Exception as e:
                    logger.warning(f"MobileSAM predict failed for '{class_name}': {e}")
                    mask = (
                        np.zeros((h_orig, w_orig), dtype=bool)
                        if raw_only
                        else self._bbox_fallback_mask(image_rgb, bbox)
                    )
                    warnings.append(f"MobileSAM prediction failed: {e}")
            else:
                mask = (
                    np.zeros((h_orig, w_orig), dtype=bool)
                    if raw_only
                    else self._bbox_fallback_mask(image_rgb, bbox)
                )
                warnings.append("MobileSAM not available")

            if not raw_only and self.erode_kernel and self.erode_kernel > 0:
                mask = erode_boundary(mask, self.erode_kernel)"""

repl_seg_logic = """                    if raw_only or not self.enable_candidate_filter:
                        best_idx = int(np.argmax(scores_np))
                        mask = cand_masks[best_idx]
                    else:
                        candidate_list = [cand_masks[i] for i in range(cand_masks.shape[0])]
                        valid_candidates = [
                            m for m in candidate_list
                            if (m.sum() / bbox_area) >= self.min_mask_threshold
                        ]
                        if valid_candidates:
                            mask = min(valid_candidates, key=lambda m: m.sum())
                        else:
                            mask = max(candidate_list, key=lambda m: m.sum())
                        if (mask.sum() / bbox_area) < self.min_mask_threshold:
                            fallback_fg = self._bbox_fallback_mask(image_rgb, bbox)
                            mask = mask | fallback_fg
                except Exception as e:
                    logger.warning(f"MobileSAM predict failed for '{class_name}': {e}")
                    mask = (
                        np.zeros((h_orig, w_orig), dtype=bool)
                        if raw_only
                        else self._bbox_fallback_mask(image_rgb, bbox)
                    )
                    warnings.append(f"MobileSAM prediction failed: {e}")
            else:
                mask = (
                    np.zeros((h_orig, w_orig), dtype=bool)
                    if raw_only
                    else self._bbox_fallback_mask(image_rgb, bbox)
                )
                warnings.append("MobileSAM not available")

            if not raw_only and self.enable_erosion and self.erode_kernel and self.erode_kernel > 0:
                mask = erode_boundary(mask, self.erode_kernel)"""

text_seg_norm = text_seg.replace("\r\n", "\n")
target_seg_init_norm = target_seg_init.replace("\r\n", "\n")
repl_seg_init_norm = repl_seg_init.replace("\r\n", "\n")
target_seg_logic_norm = target_seg_logic.replace("\r\n", "\n")
repl_seg_logic_norm = repl_seg_logic.replace("\r\n", "\n")

if target_seg_init_norm in text_seg_norm and target_seg_logic_norm in text_seg_norm:
    text_seg_norm = text_seg_norm.replace(target_seg_init_norm, repl_seg_init_norm, 1)
    text_seg_norm = text_seg_norm.replace(target_seg_logic_norm, repl_seg_logic_norm, 1)
    with open(path_seg, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text_seg_norm)
    print("food_segmentation.py updated successfully!")
else:
    print("Target not found in food_segmentation.py")

