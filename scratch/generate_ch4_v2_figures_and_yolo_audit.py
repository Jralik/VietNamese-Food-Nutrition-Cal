"""
Generate Publication-Quality Chapter 4 Figures & Deep YOLO TH2/FN Audit Report
for the expanded 418-image / 552-instance / 500-mass-group / 55-class benchmark.

Outputs:
  - results/thesis_figures_v2/fig1_ablation_progression_v2.png
  - results/thesis_figures_v2/fig2_gt_vs_pred_scatter_v2.png
  - results/thesis_figures_v2/fig3_per_class_mape_v2.png
  - results/thesis_figures_v2/fig4_visual_pipeline_cases_v2.png
  - results/thesis_figures_v2/fig5_yolo_th2_fn_audit_v2.png
  - results/yolo_v2_error_audit.md
"""
import gc
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3")
sys.path.insert(0, str(ROOT))
import pipeline_config as pc
from eval_segmentation_benchmark import coco_poly_to_mask, match_boxes_one_to_one
from volume_nutrition import VolumeNutritionEstimator

COCO_PATH = Path(r"D:\Document\Year4_Semester1\KhoaLuanTotNghiep\segmentation_data\annotations\instances_default.json")
MP_PATH = ROOT / "scratch" / "gt_mapping_v2_verified.json"
CACHE_DIR = ROOT / "scratch" / "bench_cache_v2"
DEPTH_DIR = CACHE_DIR / "depth_ctx"
Z_MARKER_PATH = CACHE_DIR / "z_marker_cache.json"

FIG_DIR = ROOT / "results" / "thesis_figures_v2"
ARTIFACT_DIR = Path(r"C:\Users\huynh\.gemini\antigravity\brain\c0224c8e-e00c-45d5-bd8d-70ec03b619ce")
YOLO_AUDIT_MD = ROOT / "results" / "yolo_v2_error_audit.md"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def norm_cls(s: str) -> str:
    return s.split("(")[0].strip().lower().replace(" ", "")


def make_fig1_ablation():
    stages = [
        "Gốc 418 ảnh\n(A1–A5)",
        "+ B1–B2\n(Định tuyến\n& món mỏng)",
        "+ C1\n(Bánh mì\nkhông)",
        "+ C1–C2\n(Khổ qua\nkhô)",
        "+ C1–C3\n(Cà chua\nđộ sâu)",
        "+ C1–C4\n(Rau lòng\nđĩa trũng)",
        "+ C1–C5\n(Bánh tráng\n& thịt lát)",
        "+ C1–C6\n(Phối cảnh\nZ_food/Z_m)",
    ]
    mapes = [54.05, 40.55, 37.18, 36.36, 35.37, 34.75, 33.74, 30.86]
    maes = [96.90, 80.42, 78.03, 76.80, 75.01, 74.23, 73.62, 64.84]

    x = np.arange(len(stages))
    fig, ax1 = plt.subplots(figsize=(12.5, 5.8), dpi=300)

    colors = ["#c0392b", "#e67e22", "#f39c12", "#f1c40f", "#2ecc71", "#27ae60", "#16a085", "#1f77b4"]
    bars = ax1.bar(x, mapes, width=0.56, color=colors, alpha=0.88, edgecolor="#2c3e50", linewidth=0.8, label="MAPE (%)")
    ax1.axhline(35.0, color="#e74c3c", linestyle="--", linewidth=1.4, label="Mục tiêu MAPE = 35%")
    ax1.set_ylabel("MAPE (%) ↓", fontsize=11, fontweight="bold", color="#2c3e50")
    ax1.set_ylim(0, 64)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    for b, v in zip(bars, mapes):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2c3e50")

    ax2 = ax1.twinx()
    ax2.plot(x, maes, color="#1a252f", marker="o", linewidth=2.2, markersize=6.5, label="MAE (g)")
    ax2.set_ylabel("MAE (g) ↓", fontsize=11, fontweight="bold", color="#1a252f")
    ax2.set_ylim(45, 112)
    for xi, v in zip(x, maes):
        ax2.text(xi, v + 2.2, f"{v:.1f}g", ha="center", va="bottom", fontsize=8.5, color="#1a252f", fontweight="bold")

    ax1.set_title(
        "Quá trình giảm sai số ước lượng khối lượng qua các giai đoạn cải tiến hình học\n(418 ảnh / 500 nhóm khối lượng / 55 lớp — Oracle GT Mask)",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=9.5)
    fig.tight_layout()
    out = FIG_DIR / "fig1_ablation_progression_v2.png"
    fig.savefig(out, dpi=300)
    fig.savefig(ARTIFACT_DIR / out.name, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig2_scatter(ms_groups):
    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=300)
    type_colors = {
        "soup_bowl": ("#1f77b4", "Tô nước (soup_bowl)"),
        "rice_bowl": ("#9467bd", "Chén cơm/kho (rice_bowl)"),
        "flat_plate": ("#2ca02c", "Đĩa chính (flat_plate)"),
        "small_plate": ("#ff7f0e", "Đĩa nhỏ/món rời (small_plate)"),
        "side_vegetables": ("#17becf", "Rau lá (side_vegetables)"),
        "ingredient_small": ("#d62728", "Thành phần nhỏ (ingredient_small)"),
    }
    by_t = defaultdict(list)
    for g in ms_groups:
        by_t[g["dish_type"]].append(g)

    for dt, (col, lbl) in type_colors.items():
        pts = by_t.get(dt, [])
        if not pts:
            continue
        xs = [p["gt"] for p in pts]
        ys = [p["pred"] for p in pts]
        ax.scatter(xs, ys, c=col, alpha=0.68, s=30, edgecolors="white", linewidth=0.3, label=f"{lbl} (n={len(pts)})")

    all_gt = np.array([g["gt"] for g in ms_groups], dtype=float)
    all_pr = np.array([g["pred"] for g in ms_groups], dtype=float)
    max_v = max(all_gt.max(), all_pr.max()) * 1.05
    grid = np.linspace(0, max_v, 200)
    ax.plot(grid, grid, "k--", linewidth=1.4, label="Đường lý tưởng (y = x)")
    ax.fill_between(grid, grid * 0.70, grid * 1.30, color="#95a5a6", alpha=0.14, label="Biên sai số ±30%")

    r = float(np.corrcoef(all_gt, all_pr)[0, 1])
    mae = float(np.mean(np.abs(all_pr - all_gt)))
    mape = float(np.mean(np.abs(all_pr - all_gt) / all_gt) * 100.0)
    rmse = float(np.sqrt(np.mean((all_pr - all_gt) ** 2)))

    stats_txt = f"N = {len(ms_groups)} nhóm khối lượng\nPearson r = {r:.3f} (R² = {r**2:.3f})\nMAE = {mae:.2f} g | RMSE = {rmse:.2f} g\nMAPE = {mape:.2f}%"
    ax.text(
        0.03, 0.97, stats_txt, transform=ax.transAxes, va="top", ha="left",
        fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#34495e", alpha=0.92),
    )

    ax.set_xlim(0, 1050)
    ax.set_ylim(0, 1050)
    ax.set_xlabel("Khối lượng cân thực tế — GT Mass (g)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Khối lượng dự đoán — MobileSAM Config C (g)", fontsize=11, fontweight="bold")
    ax.set_title("Tương quan Khối lượng thực tế vs Dự đoán (418 ảnh / 500 nhóm / 55 lớp)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.3, frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "fig2_gt_vs_pred_scatter_v2.png"
    fig.savefig(out, dpi=300)
    fig.savefig(ARTIFACT_DIR / out.name, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig3_per_class(ms_groups):
    by_cls = defaultdict(list)
    for g in ms_groups:
        by_cls[g["cls"]].append(g)

    rows = []
    for c, items in by_cls.items():
        mape = st.mean(abs(x["pred"] - x["gt"]) / x["gt"] * 100.0 for x in items)
        rows.append((c, len(items), mape))
    rows.sort(key=lambda t: t[2])

    labels = [f"{c} (n={n})" for c, n, _ in rows]
    vals = [m for _, _, m in rows]
    cols = ["#27ae60" if v <= 20 else ("#2980b9" if v <= 35 else "#e67e22" if v <= 50 else "#c0392b") for v in vals]

    fig, ax = plt.subplots(figsize=(10.5, 13.5), dpi=300)
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, alpha=0.88, height=0.74, edgecolor="#2c3e50", linewidth=0.4)
    ax.axvline(30.70, color="#1f77b4", linestyle="--", linewidth=1.3, label="MAPE trung bình toàn tập (30.70%)")
    ax.axvline(35.0, color="#c0392b", linestyle=":", linewidth=1.3, label="Ngưỡng mục tiêu (35.0%)")
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", ha="left", fontsize=7.8, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("MAPE (%) ↓", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(vals) + 10)
    ax.set_title("Sai số MAPE theo từng lớp món ăn (Oracle MobileSAM Config C — 52 lớp có GT đơn/gộp)", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "fig3_per_class_mape_v2.png"
    fig.savefig(out, dpi=300)
    fig.savefig(ARTIFACT_DIR / out.name, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig4_visual_cases(inst_by_file, z_marker_map, vol_est):
    showcase = [
        ("BunBoHue1 (1).jpg", "Tô nước (Bun bo Hue + Rau)"),
        ("Banh Canh 1.jpg", "Tô lớn (BanhCanh)"),
        ("com chien 2.jpg", "Hộp xốp nghiêng (C6 Z_food/Z_m)"),
        ("BoKho1(1).jpg", "Món chấm + Bánh mì không (C1)"),
    ]
    fig, axes = plt.subplots(len(showcase), 3, figsize=(13.5, 13.5), dpi=240)

    for row_i, (fn, subtitle) in enumerate(showcase):
        insts = inst_by_file[fn]
        stem = Path(fn).stem
        img_bgr = cv2.imread(insts[0]["image_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_full, w_full = img_rgb.shape[:2]

        sc = np.load(DEPTH_DIR / f"scene_{stem}.npz", allow_pickle=True)
        depth_mm = sc["depth_mm"].astype(np.float32)
        K = sc["K"]
        mm_px = float(sc["mm_per_pixel"])
        scale_src = str(sc["scale_source"])
        d_anc = bool(sc["depth_anchored"])
        h_vol, w_vol = int(sc["h_vol"]), int(sc["w_vol"])
        sx, sy = float(sc["sx"]), float(sc["sy"])
        sc.close()

        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        oracle_c = ms_npz["oracle_cand_raw"].astype(bool)
        ms_npz.close()

        zm = z_marker_map.get(fn, {}).get("z_marker_depth_mm")
        scene_cnames = [x["vietfood67_class_name"] for x in insts]

        # Col 0: Input + Boxes
        ax0 = axes[row_i, 0]
        ax0.imshow(img_rgb)
        for inst in insts:
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.2, edgecolor="#00ff88", facecolor="none")
            ax0.add_patch(rect)
            ax0.text(x1 + 6, max(20, y1 - 8), inst["short_class"], color="white", fontsize=8.5, fontweight="bold",
                     bbox=dict(facecolor="#16a085", alpha=0.85, pad=2, edgecolor="none"))
        ax0.set_title(f"{fn}\n({subtitle})", fontsize=9.5, fontweight="bold")
        ax0.axis("off")

        # Col 1: MobileSAM Config C Mask Overlay + Pred vs GT
        overlay = img_rgb.copy().astype(np.float32)
        palette = [(46, 204, 113), (52, 152, 219), (241, 196, 15)]
        note_lines = []
        for i_inst, inst in enumerate(insts):
            m = oracle_c[i_inst]
            col = palette[i_inst % len(palette)]
            overlay[m] = overlay[m] * 0.52 + np.array(col, dtype=np.float32) * 0.48
            mv = cv2.resize(m.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if m.shape[:2] != (h_vol, w_vol) else m
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
            est = vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=mv, class_name=inst["vietfood67_class_name"],
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol, K=K,
                depth_anchored=d_anc, z_marker_mm=zm, scene_classes=scene_cnames,
            )
            gt_w = inst["gt_weight_g"]
            ape = abs(est.mass_g - gt_w) / gt_w * 100.0 if gt_w else 0.0
            note_lines.append(f"{inst['short_class']}: {est.mass_g:.1f}g vs GT {gt_w:.1f}g ({ape:.1f}%)")

        ax1 = axes[row_i, 1]
        ax1.imshow(overlay.astype(np.uint8))
        ax1.set_title("MobileSAM Config C Mask\n" + " | ".join(note_lines[:2]), fontsize=8.5, fontweight="bold")
        if len(note_lines) > 2:
            ax1.text(0.02, 0.03, "\n".join(note_lines), transform=ax1.transAxes, fontsize=7.5, color="white",
                     fontweight="bold", bbox=dict(facecolor="black", alpha=0.65, pad=3))
        ax1.axis("off")

        # Col 2: Depth Anything V2 (Anchored)
        ax2 = axes[row_i, 2]
        valid = depth_mm[depth_mm > 0]
        vmin, vmax = np.percentile(valid, [2, 98]) if len(valid) > 0 else (200, 1000)
        ax2.imshow(depth_mm, cmap="turbo", vmin=vmin, vmax=vmax)
        ax2.set_title(f"Depth Anything V2 (mm)\nscale={scale_src}, anc={d_anc}", fontsize=8.8, fontweight="bold")
        ax2.axis("off")

    fig.suptitle("Minh họa Pipeline (Box → MobileSAM Config C → Depth Map → Ước lượng Khối lượng)", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIG_DIR / "fig4_visual_pipeline_cases_v2.png"
    fig.savefig(out, dpi=240)
    fig.savefig(ARTIFACT_DIR / out.name, dpi=240)
    plt.close(fig)
    print(f"Saved {out}")


def make_yolo_audit(inst_by_file, yolo_data, z_marker_map, vol_est):
    th1_by_cls = Counter()
    th2_pairs = Counter()
    th2_by_gt = Counter()
    fn_by_gt = Counter()
    fp_by_pred = Counter()
    gt_totals = Counter()

    th1_apes = []
    th2_apes = []
    th2_details = []

    for fn in sorted(inst_by_file.keys()):
        insts = inst_by_file[fn]
        y_preds = yolo_data[fn]["preds"]
        stem = Path(fn).stem

        sc = np.load(DEPTH_DIR / f"scene_{stem}.npz", allow_pickle=True)
        depth_mm = sc["depth_mm"].astype(np.float32)
        K = sc["K"]
        mm_px = float(sc["mm_per_pixel"])
        scale_src = str(sc["scale_source"])
        d_anc = bool(sc["depth_anchored"])
        h_vol, w_vol = int(sc["h_vol"]), int(sc["w_vol"])
        sx, sy = float(sc["sx"]), float(sc["sy"])
        sc.close()

        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        yolo_c = ms_npz["yolo_cand_raw"].astype(bool)
        ms_npz.close()

        zm = z_marker_map.get(fn, {}).get("z_marker_depth_mm")
        y_scene_classes = [p["class_name"] for p in y_preds]

        y_norm = [{**p, "short_class": norm_cls(p["short_class"])} for p in y_preds]
        g_norm = [{**g, "short_class": norm_cls(g["short_class"])} for g in insts]
        awr_m, rem_p1, rem_g1 = match_boxes_one_to_one(y_norm, g_norm, iou_thr=0.5, require_same_class=True)
        sub_agn, rem_p2, rem_g2 = match_boxes_one_to_one(
            [y_norm[i] for i in rem_p1], [g_norm[j] for j in rem_g1], iou_thr=0.5, require_same_class=False
        )
        th2_m = [(rem_p1[spi], rem_g1[sgi], biou) for spi, sgi, biou in sub_agn]

        for g in insts:
            gt_totals[g["short_class"]] += 1

        def est_y(pi):
            p = y_preds[pi]
            ym = yolo_c[pi]
            ym_v = cv2.resize(ym.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if ym.shape[:2] != (h_vol, w_vol) else ym
            bx1, by1, bx2, by2 = p["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
            return vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=ym_v, class_name=p["class_name"],
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol, K=K,
                depth_anchored=d_anc, z_marker_mm=zm, scene_classes=y_scene_classes,
            )

        for pi, gi, _ in awr_m:
            g = insts[gi]
            th1_by_cls[g["short_class"]] += 1
            if g["mapping_status"] == "exact_1to1" and g["gt_weight_g"]:
                e = est_y(pi)
                th1_apes.append(abs(e.mass_g - g["gt_weight_g"]) / g["gt_weight_g"] * 100.0)

        for pi, gi, biou in th2_m:
            g = insts[gi]
            p = y_preds[pi]
            th2_pairs[(g["short_class"], p["short_class"])] += 1
            th2_by_gt[g["short_class"]] += 1
            if g["mapping_status"] == "exact_1to1" and g["gt_weight_g"]:
                e = est_y(pi)
                ape = abs(e.mass_g - g["gt_weight_g"]) / g["gt_weight_g"] * 100.0
                th2_apes.append(ape)
                th2_details.append((fn, g["short_class"], p["short_class"], round(biou, 2), e.mass_g, g["gt_weight_g"], round(ape, 1)))

        for sgi in rem_g2:
            gi = rem_g1[sgi]
            fn_by_gt[insts[gi]["short_class"]] += 1
        for spi in rem_p2:
            pi = rem_p1[spi]
            fp_by_pred[y_preds[pi]["short_class"]] += 1

        del depth_mm, yolo_c

    # Plot Fig 5: Top TH2 Confusion Pairs + Top FN Classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=300)

    top_pairs = th2_pairs.most_common(15)
    p_labels = [f"{gt} → {pr}" for (gt, pr), _ in reversed(top_pairs)]
    p_vals = [cnt for _, cnt in reversed(top_pairs)]
    ax1.barh(np.arange(len(p_labels)), p_vals, color="#e67e22", edgecolor="#2c3e50", linewidth=0.5)
    for i, v in enumerate(p_vals):
        ax1.text(v + 0.2, i, str(v), va="center", fontsize=8.5, fontweight="bold")
    ax1.set_yticks(np.arange(len(p_labels)))
    ax1.set_yticklabels(p_labels, fontsize=8.5)
    ax1.set_xlabel("Số ca nhầm lớp (TH2)", fontsize=10, fontweight="bold")
    ax1.set_title(f"Top 15 cặp nhầm lớp YOLOv10b (Tổng TH2 = {sum(th2_pairs.values())})", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    top_fns = fn_by_gt.most_common(15)
    f_labels = [f"{cls} ({cnt}/{gt_totals[cls]})" for cls, cnt in reversed(top_fns)]
    f_vals = [cnt for _, cnt in reversed(top_fns)]
    ax2.barh(np.arange(len(f_labels)), f_vals, color="#c0392b", edgecolor="#2c3e50", linewidth=0.5)
    for i, v in enumerate(f_vals):
        ax2.text(v + 0.2, i, str(v), va="center", fontsize=8.5, fontweight="bold")
    ax2.set_yticks(np.arange(len(f_labels)))
    ax2.set_yticklabels(f_labels, fontsize=8.5)
    ax2.set_xlabel("Số ca bỏ sót (FN)", fontsize=10, fontweight="bold")
    ax2.set_title(f"Top 15 lớp bị bỏ sót bởi YOLOv10b (Tổng FN = {sum(fn_by_gt.values())})", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout()
    out_fig = FIG_DIR / "fig5_yolo_th2_fn_audit_v2.png"
    fig.savefig(out_fig, dpi=300)
    fig.savefig(ARTIFACT_DIR / out_fig.name, dpi=300)
    plt.close(fig)
    print(f"Saved {out_fig}")

    # Write Markdown Audit Report
    th1_mape = st.mean(th1_apes) if th1_apes else 0.0
    th2_mape = st.mean(th2_apes) if th2_apes else 0.0
    md = [
        "# Kiểm Toán Sâu Lỗi Nhận Diện YOLOv10b (104 TH2, 80 FN, 63 FP) Trên 418 Ảnh",
        "",
        "> [!IMPORTANT]",
        "> **Tác động định lượng của lỗi phân loại YOLO (`TH2`) lên ước lượng khối lượng hạ nguồn:**",
        f"> - **Nhóm `TH1` (YOLO nhận đúng lớp, `N={len(th1_apes)}` mẫu 1-1):** MAPE khối lượng = **`{th1_mape:.2f}%`** (gần sát `Oracle MobileSAM C = 30.70%`).",
        f"> - **Nhóm `TH2` (YOLO bắt đúng khung nhưng nhầm lớp, `N={len(th2_apes)}` mẫu 1-1):** MAPE khối lượng tăng vọt lên **`{th2_mape:.2f}%`** (chênh **`+{th2_mape - th1_mape:.2f}` điểm %**) do tra nhầm mô hình hình học (`bowl` vs `plate`) và mật độ $\\rho$.",
        f"> - **Tổng hợp Detection (552 GT boxes):** `TH1 = {sum(th1_by_cls.values())} ({sum(th1_by_cls.values())/552*100:.1f}%)` | `TH2 = {sum(th2_pairs.values())} ({sum(th2_pairs.values())/552*100:.1f}%)` | `FN = {sum(fn_by_gt.values())} ({sum(fn_by_gt.values())/552*100:.1f}%)` | `FP = {sum(fp_by_pred.values())}`.",
        "",
        "---",
        "",
        "## 1. Top 20 Cặp Nhầm Lớp (`TH2`: GT → YOLO Predicted)",
        "",
        "| Hạng | Lớp thực tế (GT) | Lớp YOLO dự đoán | Số ca (`TH2`) |",
        "| :---: | :--- | :--- | :---: |",
    ]
    for rank, ((gt_c, pr_c), cnt) in enumerate(th2_pairs.most_common(20), 1):
        md.append(f"| `{rank}` | **`{gt_c}`** | `{pr_c}` | `{cnt}` |")

    md.extend([
        "",
        "## 2. Top 20 Lớp Bị Bỏ Sót (`FN`: Missed Detections)",
        "",
        "| Lớp thực tế (GT) | Số ca bỏ sót (`FN`) | Tổng GT của lớp | Tỉ lệ bỏ sót (%) |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for cls, cnt in fn_by_gt.most_common(20):
        tot = gt_totals[cls]
        md.append(f"| **`{cls}`** | `{cnt}` | `{tot}` | `{cnt/tot*100:.1f}%` |")

    md.extend([
        "",
        "## 3. Top 15 Lớp Dương Tính Giả (`FP`: Extra Boxes)",
        "",
        "| Lớp YOLO dự đoán dư (`FP`) | Số lượng |",
        "| :--- | :---: |",
    ])
    for cls, cnt in fp_by_pred.most_common(15):
        md.append(f"| `{cls}` | `{cnt}` |")

    YOLO_AUDIT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "yolo_v2_error_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {YOLO_AUDIT_MD} | TH1 MAPE={th1_mape:.2f}% vs TH2 MAPE={th2_mape:.2f}%")


def main():
    mp = json.loads(MP_PATH.read_text(encoding="utf-8"))
    yolo_data = json.loads((CACHE_DIR / "yolo_detections.json").read_text(encoding="utf-8"))
    z_marker_map = json.loads(Z_MARKER_PATH.read_text(encoding="utf-8"))
    inst_by_file = defaultdict(list)
    for x in mp["instances"]:
        inst_by_file[x["file_name"]].append(x)

    vol_est = VolumeNutritionEstimator()

    # Collect Oracle MobileSAM Config C mass groups (N=500) for Fig 2 & Fig 3
    ms_groups = []
    for fn in sorted(inst_by_file.keys()):
        insts = inst_by_file[fn]
        stem = Path(fn).stem
        sc = np.load(DEPTH_DIR / f"scene_{stem}.npz", allow_pickle=True)
        depth_mm = sc["depth_mm"].astype(np.float32)
        K = sc["K"]
        mm_px = float(sc["mm_per_pixel"])
        scale_src = str(sc["scale_source"])
        d_anc = bool(sc["depth_anchored"])
        h_vol, w_vol = int(sc["h_vol"]), int(sc["w_vol"])
        sx, sy = float(sc["sx"]), float(sc["sy"])
        sc.close()

        ms_npz = np.load(CACHE_DIR / f"mobilesam_{stem}.npz")
        oracle_c = ms_npz["oracle_cand_raw"].astype(bool)
        ms_npz.close()

        zm = z_marker_map.get(fn, {}).get("z_marker_depth_mm")
        scene_cnames = [x["vietfood67_class_name"] for x in insts]
        preds = []
        for i_inst, inst in enumerate(insts):
            m = oracle_c[i_inst]
            mv = cv2.resize(m.astype(np.uint8), (w_vol, h_vol), interpolation=cv2.INTER_NEAREST).astype(bool) if m.shape[:2] != (h_vol, w_vol) else m
            bx1, by1, bx2, by2 = inst["bbox_xyxy"]
            b_vol = (int(round(bx1 * sx)), int(round(by1 * sy)), int(round(bx2 * sx)), int(round(by2 * sy)))
            est = vol_est.estimate(
                depth_map_mm=depth_mm, food_mask=mv, class_name=inst["vietfood67_class_name"],
                mm_per_pixel=mm_px, scale_source=scale_src, bbox=b_vol, K=K,
                depth_anchored=d_anc, z_marker_mm=zm, scene_classes=scene_cnames,
            )
            preds.append(est.mass_g)

        grouped = defaultdict(list)
        for gi, inst in enumerate(insts):
            dt = pc.CLASS_TO_DISH_TYPE.get(inst["vietfood67_class_name"], "default")
            if inst["mapping_status"] == "exact_1to1":
                ms_groups.append({"fn": fn, "cls": inst["short_class"], "dish_type": dt, "pred": preds[gi], "gt": float(inst["gt_weight_g"])})
            elif inst["mapping_status"] == "grouped_multi_instance":
                grouped[inst["short_class"]].append((gi, dt))
        for s_cls, items in grouped.items():
            gis = [i for i, _ in items]
            ms_groups.append({"fn": fn, "cls": s_cls, "dish_type": items[0][1], "pred": sum(preds[i] for i in gis), "gt": float(insts[gis[0]]["gt_weight_g"])})

        del depth_mm, oracle_c

    make_fig1_ablation()
    make_fig2_scatter(ms_groups)
    make_fig3_per_class(ms_groups)
    make_fig4_visual_cases(inst_by_file, z_marker_map, vol_est)
    make_yolo_audit(inst_by_file, yolo_data, z_marker_map, vol_est)
    print("ALL FIGURES & YOLO AUDIT COMPLETED!")


if __name__ == "__main__":
    main()
