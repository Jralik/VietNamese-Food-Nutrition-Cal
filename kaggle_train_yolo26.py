# ============================================================
# 📌 YOLO26m Training on VietFood67 Dataset - Kaggle Notebook
# ============================================================
# Purpose: Train YOLO26m on VietFood67 and compare with YOLOv10m
# Dataset: https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68
# Environment: Kaggle Notebook with GPU T4 x2 or P100
# ============================================================

# %% [markdown]
# # 🕵️ YOLO26m Training on VietFood67 Dataset
# **Goal:** Train YOLO26m and compare performance with YOLOv10m (mAP50 = 0.934)

# %% --- Cell 1: Install dependencies ---
# NOTE: Do NOT import torch before this cell.
# Only install ultralytics — it will pull compatible dependencies.
# The torchvision warning is harmless and won't block training.
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics", "-q"])

# %% --- Cell 2: Imports ---
import os
import yaml
import shutil
import glob
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% --- Cell 3: Dataset Configuration ---
# ============================================================
# 📦 DATASET SETUP
# ============================================================
# Actual Kaggle mount path for VietFood68 dataset:
#   /kaggle/input/datasets/thomasnguyen6868/vietfood68/dataset/
#     ├── images/ (train, valid, test)
#     └── labels/ (train, tvalid, test)   ← note: "tvalid" typo in dataset
# ============================================================

WORK_DIR = "/kaggle/working"
DATASET_BASE = "/kaggle/input/datasets/thomasnguyen6868/vietfood68/dataset"

TRAIN_IMAGES = os.path.join(DATASET_BASE, "images", "train")
VAL_IMAGES   = os.path.join(DATASET_BASE, "images", "valid")
TEST_IMAGES  = os.path.join(DATASET_BASE, "images", "test")

# Verify paths
print("📂 Dataset paths:")
for name, p in [("Train", TRAIN_IMAGES), ("Val", VAL_IMAGES), ("Test", TEST_IMAGES)]:
    if os.path.exists(p):
        n = len(os.listdir(p))
        print(f"   ✅ {name}: {p}  ({n} files)")
    else:
        print(f"   ❌ {name}: {p}  NOT FOUND")

# --- Fix: YOLO auto-resolves labels by replacing "images" → "labels" in path.
# The dataset has "labels/tvalid" instead of "labels/valid", so we create a symlink.
labels_valid_expected = os.path.join(DATASET_BASE, "labels", "valid")
labels_tvalid_actual  = os.path.join(DATASET_BASE, "labels", "tvalid")

if not os.path.exists(labels_valid_expected) and os.path.exists(labels_tvalid_actual):
    # Kaggle input is read-only, so copy labels to working dir instead
    import subprocess
    LABELS_WORK = os.path.join(WORK_DIR, "dataset_labels_fix", "labels", "valid")
    os.makedirs(LABELS_WORK, exist_ok=True)
    subprocess.run(["cp", "-r", labels_tvalid_actual + "/.", LABELS_WORK], check=True)
    print(f"   🔧 Fixed: copied labels/tvalid → {LABELS_WORK}")
    # We'll need to copy images/valid to the same structure so paths align
    IMAGES_WORK_VALID = os.path.join(WORK_DIR, "dataset_labels_fix", "images", "valid")
    os.makedirs(IMAGES_WORK_VALID, exist_ok=True)
    subprocess.run(["cp", "-rs", VAL_IMAGES + "/.", IMAGES_WORK_VALID], check=True)
    # Override VAL path to use the fixed copy
    VAL_IMAGES = IMAGES_WORK_VALID
    print(f"   🔧 Val images symlinked to: {VAL_IMAGES}")
elif os.path.exists(labels_valid_expected):
    print(f"   ✅ labels/valid exists, no fix needed")
else:
    print(f"   ⚠️ Neither labels/valid nor labels/tvalid found")

# %% --- Cell 4: Prepare data.yaml ---
# ============================================================
# 68 classes (67 Vietnamese dishes + 1 Human class)
# Using ABSOLUTE paths to avoid YOLO path resolution issues
# ============================================================

CLASS_NAMES = [
    "Banh canh", "Banh chung", "Banh cuon", "Banh khot", "Banh mi",
    "Banh trang", "Banh trang tron", "Banh xeo", "Bo kho", "Bo la lot",
    "Bong cai", "Bun", "Bun bo Hue", "Bun cha", "Bun dau",
    "Bun mam", "Bun rieu", "Ca", "Ca chua", "Ca phao",
    "Ca rot", "Canh", "Cha", "Cha gio", "Chanh",
    "Com", "Com tam", "Con nguoi", "Cu kieu", "Cua",
    "Dau hu", "Dua chua", "Dua leo", "Goi cuon", "Hamburger",
    "Heo quay", "Hu tieu", "Kho qua thit", "Khoai tay chien", "Lau",
    "Long heo", "Mi", "Muc", "Nam", "Oc",
    "Ot chuong", "Pho", "Pho mai", "Rau", "Salad",
    "Thit bo", "Thit ga", "Thit heo", "Thit kho", "Thit nuong",
    "Tom", "Trung", "Xoi", "Banh beo", "Cao lau",
    "Mi Quang", "Com chien duong chau", "Bun cha ca", "Com chien ga",
    "Chao long", "Nom hoa chuoi", "Nui xao bo", "Sup cua"
]

# Build data.yaml with absolute paths (no "path" key)
data_config = {
    "train": TRAIN_IMAGES,
    "val": VAL_IMAGES,
    "test": TEST_IMAGES,
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES
}

data_yaml_path = os.path.join(WORK_DIR, "data.yaml")
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

print(f"\n✅ data.yaml saved to: {data_yaml_path}")
print("   Contents:")
with open(data_yaml_path, 'r') as f:
    print(f.read())

# %% --- Cell 5: Training Configuration ---
# ============================================================
# ⚙️ TRAINING HYPERPARAMETERS
# ============================================================
# Matching YOLOv10m training setup from the project for fair comparison
# Project used: YOLOv10m + SGD optimizer
# We will train: YOLO26m + SGD and YOLO26m + MuSGD (new in YOLO26)
# ============================================================

TRAINING_CONFIG = {
    # --- Model ---
    "model_variant": "yolo26m.pt",     # Medium variant (same scale as YOLOv10m)
    
    # --- Training ---
    # NOTE: Dataset on Kaggle is ALREADY augmented offline (161k images from ~30k originals)
    #       So we reduce online augmentation to avoid double-augmentation.
    #       Also adjusted epochs/batch to fit Kaggle 12h GPU session limit.
    "epochs": 50,                       # Reduced (161k images = large dataset)
    "imgsz": 640,                       # Image size (same as project)
    "batch": 16,                        # T4 16GB: batch 32 causes OOM, 16 is safe
    "patience": 15,                     # Early stopping patience
    
    # --- Optimizer (SGD to match original project) ---
    "optimizer": "SGD",                 # Use SGD for fair comparison
    "lr0": 0.01,                        # Initial learning rate
    "lrf": 0.01,                        # Final learning rate factor
    "momentum": 0.937,                  # SGD momentum
    "weight_decay": 0.0005,             # Weight decay
    
    # --- Augmentation (REDUCED — dataset already augmented offline) ---
    "hsv_h": 0.01,                      # Reduced (brightness already augmented)
    "hsv_s": 0.3,                       # Reduced
    "hsv_v": 0.2,                       # Reduced (brightness already augmented)
    "degrees": 0.0,                     # No rotation
    "translate": 0.05,                  # Minimal translation
    "scale": 0.2,                       # Reduced (cropping already applied)
    "fliplr": 0.5,                      # Keep horizontal flip (lightweight)
    "flipud": 0.0,                      # No vertical flip
    "mosaic": 0.3,                      # Reduced significantly (mosaic already applied offline)
    "mixup": 0.0,                       # No mixup
    
    # --- Device ---
    "device": 0 if torch.cuda.is_available() else "cpu",
    "workers": 4,
    
    # --- Saving ---
    "project": os.path.join(WORK_DIR, "runs"),
    "name": "yolo26m_vietfood67_sgd",
    "save_period": 10,                  # Save checkpoint every N epochs
    "exist_ok": True,
}

print("⚙️ Training Configuration:")
for k, v in TRAINING_CONFIG.items():
    print(f"   {k}: {v}")

# %% --- Cell 6: Train YOLO26m with SGD (with resume support) ---
# ============================================================
# 🚀 TRAINING - YOLO26m with SGD (fair comparison with YOLOv10m)
# ============================================================
# Resume support: Upload last.pt from previous session as a Kaggle dataset,
# the script will auto-detect and resume training.
#
# Steps to resume:
#   1. Download last.pt from previous session output
#   2. Create a new Kaggle dataset (e.g. "yolo26-checkpoint") and upload last.pt
#   3. Add that dataset to this notebook via "Add Data"
#   4. Run — script will find it automatically
# ============================================================

# --- Search for last.pt to resume from ---
RESUME_CHECKPOINT = None

# Search locations (in priority order)
search_paths = [
    # 1. Previous run output in working dir
    os.path.join(TRAINING_CONFIG["project"], TRAINING_CONFIG["name"], "weights", "last.pt"),
    # 2. Uploaded as Kaggle dataset (search all input dirs)
]

# Check fixed paths first
for p in search_paths:
    if os.path.exists(p):
        RESUME_CHECKPOINT = p
        break

# Search all Kaggle input directories for last.pt
if RESUME_CHECKPOINT is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "last.pt":
                RESUME_CHECKPOINT = os.path.join(root, f)
                break
        if RESUME_CHECKPOINT:
            break

print("=" * 60)
if RESUME_CHECKPOINT:
    print(f"🔄 RESUMING training from: {RESUME_CHECKPOINT}")
    print("=" * 60)
    
    model_sgd = YOLO(RESUME_CHECKPOINT)
    
    start_time = time.time()
    results_sgd = model_sgd.train(resume=True)
else:
    print("🚀 Starting FRESH YOLO26m training with SGD optimizer")
    print("=" * 60)
    
    model_sgd = YOLO(TRAINING_CONFIG["model_variant"])
    
    start_time = time.time()
    results_sgd = model_sgd.train(
        data=data_yaml_path,
        epochs=TRAINING_CONFIG["epochs"],
        imgsz=TRAINING_CONFIG["imgsz"],
        batch=TRAINING_CONFIG["batch"],
        patience=TRAINING_CONFIG["patience"],
        optimizer=TRAINING_CONFIG["optimizer"],
        lr0=TRAINING_CONFIG["lr0"],
        lrf=TRAINING_CONFIG["lrf"],
        momentum=TRAINING_CONFIG["momentum"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        hsv_h=TRAINING_CONFIG["hsv_h"],
        hsv_s=TRAINING_CONFIG["hsv_s"],
        hsv_v=TRAINING_CONFIG["hsv_v"],
        degrees=TRAINING_CONFIG["degrees"],
        translate=TRAINING_CONFIG["translate"],
        scale=TRAINING_CONFIG["scale"],
        fliplr=TRAINING_CONFIG["fliplr"],
        flipud=TRAINING_CONFIG["flipud"],
        mosaic=TRAINING_CONFIG["mosaic"],
        mixup=TRAINING_CONFIG["mixup"],
        device=TRAINING_CONFIG["device"],
        workers=TRAINING_CONFIG["workers"],
        project=TRAINING_CONFIG["project"],
        name=TRAINING_CONFIG["name"],
        save_period=TRAINING_CONFIG["save_period"],
        exist_ok=TRAINING_CONFIG["exist_ok"],
        verbose=True,
    )

sgd_training_time = time.time() - start_time
print(f"\n⏱️ Training completed in {sgd_training_time/3600:.2f} hours")

# %% --- Cell 7: Validate YOLO26m (SGD) ---
# ============================================================
# 📊 VALIDATION - Evaluate on test set
# ============================================================

print("=" * 60)
print("📊 Evaluating YOLO26m (SGD) on test set")
print("=" * 60)

best_sgd_path = os.path.join(
    TRAINING_CONFIG["project"], TRAINING_CONFIG["name"], "weights", "best.pt"
)
model_sgd_best = YOLO(best_sgd_path)

val_results_sgd = model_sgd_best.val(
    data=data_yaml_path,
    split="test",
    imgsz=640,
    batch=16,
    device=TRAINING_CONFIG["device"],
    verbose=True,
)

print("\n📊 YOLO26m (SGD) Test Results:")
print(f"   mAP50:    {val_results_sgd.box.map50:.4f}")
print(f"   mAP50-95: {val_results_sgd.box.map:.4f}")
print(f"   Precision: {val_results_sgd.box.mp:.4f}")
print(f"   Recall:    {val_results_sgd.box.mr:.4f}")

# %% --- Cell 8: (Optional) Train YOLO26m with MuSGD ---
# ============================================================
# 🚀 TRAINING - YOLO26m with MuSGD (new optimizer in YOLO26)
# ============================================================
# Uncomment this cell to also train with MuSGD for additional comparison

# print("=" * 60)
# print("🚀 Starting YOLO26m training with MuSGD optimizer")
# print("=" * 60)
#
# model_musgd = YOLO("yolo26m.pt")
#
# start_time_musgd = time.time()
#
# results_musgd = model_musgd.train(
#     data=data_yaml_path,
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     patience=20,
#     optimizer="MuSGD",         # <-- New hybrid optimizer in YOLO26
#     lr0=0.01,
#     lrf=0.01,
#     device=TRAINING_CONFIG["device"],
#     workers=4,
#     project=os.path.join(WORK_DIR, "runs"),
#     name="yolo26m_vietfood67_musgd",
#     save_period=10,
#     exist_ok=True,
#     verbose=True,
# )
#
# musgd_training_time = time.time() - start_time_musgd
# print(f"\n⏱️ MuSGD Training completed in {musgd_training_time/3600:.2f} hours")
#
# # Validate MuSGD model
# best_musgd_path = os.path.join(WORK_DIR, "runs", "yolo26m_vietfood67_musgd", "weights", "best.pt")
# model_musgd_best = YOLO(best_musgd_path)
# val_results_musgd = model_musgd_best.val(data=data_yaml_path, split="test", imgsz=640, batch=16)
# print(f"\n📊 YOLO26m (MuSGD) - mAP50: {val_results_musgd.box.map50:.4f}")

# %% --- Cell 9: Comparison Summary ---
# ============================================================
# 📊 COMPARISON: YOLOv10m vs YOLO26m
# ============================================================

print("=" * 60)
print("📊 MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison_data = {
    "Metric": ["mAP50", "mAP50-95", "Precision", "Recall", "Optimizer", "Training Time (h)"],
    "YOLOv10m (Project)": [
        0.934,          # mAP50 from the project README
        "N/A",          # mAP50-95 not reported
        "N/A",          # Precision not reported
        "N/A",          # Recall not reported
        "SGD",
        "N/A"
    ],
    "YOLO26m (SGD)": [
        f"{val_results_sgd.box.map50:.4f}",
        f"{val_results_sgd.box.map:.4f}",
        f"{val_results_sgd.box.mp:.4f}",
        f"{val_results_sgd.box.mr:.4f}",
        "SGD",
        f"{sgd_training_time/3600:.2f}"
    ],
}

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Save comparison to CSV
comparison_csv_path = os.path.join(WORK_DIR, "model_comparison.csv")
df_comparison.to_csv(comparison_csv_path, index=False)
print(f"\n💾 Comparison saved to: {comparison_csv_path}")

# %% --- Cell 10: Visualization ---
# ============================================================
# 📈 TRAINING CURVES VISUALIZATION
# ============================================================

def plot_training_results(results_dir, title_suffix=""):
    """Plot training metrics from results.csv."""
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"❌ results.csv not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"YOLO26m Training Results {title_suffix}", fontsize=16, fontweight="bold")
    
    metrics = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Classification Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
    ]
    
    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        if col in df.columns:
            ax.plot(df["epoch"], df[col], linewidth=2, color="#2196F3")
            
            # Add reference line for mAP50
            if "mAP50" in col:
                ax.axhline(y=0.934, color="#FF5722", linestyle="--", linewidth=1.5,
                          label="YOLOv10m baseline (0.934)")
                ax.legend(fontsize=9)
            
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"Column '{col}'\nnot found", 
                   ha="center", va="center", transform=ax.transAxes)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 Training curves saved to: {save_path}")


sgd_results_dir = os.path.join(TRAINING_CONFIG["project"], TRAINING_CONFIG["name"])
plot_training_results(sgd_results_dir, "(SGD)")

# %% --- Cell 11: Export best model to ONNX ---
# ============================================================
# 📦 EXPORT MODEL (for integration with FoodDetector project)
# ============================================================

print("📦 Exporting best YOLO26m model to ONNX format...")

model_export = YOLO(best_sgd_path)
export_path = model_export.export(format="onnx", imgsz=640, simplify=True)
print(f"✅ ONNX model exported to: {export_path}")

# Also copy the .pt weights
pt_output = os.path.join(WORK_DIR, "YOLO26m_VietFood67_SGD_best.pt")
shutil.copy2(best_sgd_path, pt_output)
print(f"✅ PyTorch weights copied to: {pt_output}")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"📁 All outputs are in: {WORK_DIR}")
print(f"   - Best weights (.pt):  {pt_output}")
print(f"   - ONNX model:          {export_path}")
print(f"   - Comparison CSV:      {comparison_csv_path}")
print(f"   - Training logs:       {sgd_results_dir}")
print("=" * 60)
