"""Phase 1-2 & 1-3: 診療科別ヒートマップ + igakuqa年度別推移"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 日本語フォント設定
for fp in fm.findSystemFonts():
    if "Hiragino" in fp or "hiragino" in fp:
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
else:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False

SUMMARY_DIR = Path(__file__).resolve().parent.parent.parent / "summary_result"
FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

MODELS = [
    ("Base 30B", "base-Qwen3-30B-A3B-Instruct-2507.json", "30B"),
    ("Pos PP SFT", "positive-pinpoint-Qwen3-30B-A3B-Instruct-2507.json", "30B"),
    ("Neg PP SFT", "negative-pinpoint-Qwen3-30B-A3B-Instruct-2507.json", "30B"),
    ("Base 80B", "bse-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
    ("GRPO", "grpo-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
    ("GSPO", "gspo-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
    ("CHORD", "chord-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
]

CATEGORIES = [
    ("naika", "内科"), ("geka", "外科"), ("sanfuqa", "産婦人科"),
    ("seikeigekaqa", "整形外科"), ("seishin", "精神科"),
    ("shinkeinaika", "神経内科"), ("shinzogeka", "心臓外科"),
    ("masui", "麻酔科"), ("kyukyu", "救急"), ("kanzo", "肝臓"),
    ("syokaki", "消化器"), ("ganka", "眼科"), ("zibika", "耳鼻科"),
]


def load_summary(filepath):
    with open(filepath) as f:
        data = json.load(f)
    cols = data["columns"]
    return [dict(zip(cols, row)) for row in data["data"]]


def get_metric(rows, dataset, category, year, metric):
    for r in rows:
        if r["dataset"] == dataset and r["category"] == category and r["year"] == year:
            return r.get(metric)
    return None


def load_all():
    all_data = {}
    for name, fname, group in MODELS:
        fp = SUMMARY_DIR / fname
        if fp.exists():
            all_data[name] = (load_summary(fp), group)
    return all_data


def plot_heatmap_absolute(all_data):
    """Phase 1-2a: 診療科別accuracy ヒートマップ（絶対値）"""
    model_names = [m[0] for m in MODELS if m[0] in all_data]
    cat_keys = [c[0] for c in CATEGORIES]
    cat_labels = [c[1] for c in CATEGORIES]

    matrix = np.full((len(cat_keys), len(model_names)), np.nan)
    for j, name in enumerate(model_names):
        rows, _ = all_data[name]
        for i, cat in enumerate(cat_keys):
            val = get_metric(rows, "specialist_exam_test_v2", cat, "all", "accuracy")
            if val is not None:
                matrix[i, j] = val * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=30, vmax=95)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(cat_labels)))
    ax.set_yticklabels(cat_labels, fontsize=11)

    # 30B/80Bの境界線
    boundary = len([m for m in MODELS if m[2] == "30B" and m[0] in all_data]) - 0.5
    ax.axvline(x=boundary, color="black", linewidth=2, linestyle="--")
    ax.text(boundary / 2, -1.2, "30B Group", ha="center", fontsize=10, fontweight="bold")
    ax.text(boundary + (len(model_names) - boundary) / 2, -1.2, "80B Group",
            ha="center", fontsize=10, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] < 45 or matrix[i, j] > 85 else "black"
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    ax.set_title("専門医試験 診療科別 Accuracy (%)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "heatmap_specialist_absolute.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: heatmap_specialist_absolute.png")


def plot_heatmap_delta(all_data):
    """Phase 1-2b: 診療科別 Baseからの差分ヒートマップ"""
    # 30Bグループと80Bグループ別に作成
    for group, base_name in [("30B", "Base 30B"), ("80B", "Base 80B")]:
        group_models = [(n, g) for n, _, g in MODELS if g == group and n in all_data and n != base_name]
        if not group_models:
            continue

        model_names = [m[0] for m in group_models]
        cat_keys = [c[0] for c in CATEGORIES]
        cat_labels = [c[1] for c in CATEGORIES]

        base_rows = all_data[base_name][0]
        matrix = np.full((len(cat_keys), len(model_names)), np.nan)

        for j, (name, _) in enumerate(group_models):
            rows, _ = all_data[name]
            for i, cat in enumerate(cat_keys):
                val = get_metric(rows, "specialist_exam_test_v2", cat, "all", "accuracy")
                base_val = get_metric(base_rows, "specialist_exam_test_v2", cat, "all", "accuracy")
                if val is not None and base_val is not None:
                    matrix[i, j] = (val - base_val) * 100

        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(cat_labels)))
        ax.set_yticklabels(cat_labels, fontsize=11)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    color = "white" if abs(matrix[i, j]) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Δ Accuracy (pp)", shrink=0.8)
        ax.set_title(f"専門医試験 Base {group}からの変化量 (pp)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"heatmap_specialist_delta_{group}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: heatmap_specialist_delta_{group}.png")


def plot_igakuqa_yearly(all_data):
    """Phase 1-3: igakuqa年度別推移"""
    years = ["2023", "2024", "2025"]
    x = np.arange(len(years))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, group, title in [(axes[0], "30B", "30B Group"), (axes[1], "80B", "80B Group")]:
        group_models = [(n, f, g) for n, f, g in MODELS if g == group and n in all_data]
        colors = {"Base 30B": "#666", "Pos PP SFT": "#e74c3c", "Neg PP SFT": "#3498db",
                  "Base 80B": "#666", "GRPO": "#2ecc71", "GSPO": "#e67e22", "CHORD": "#9b59b6"}
        markers = {"Base 30B": "o", "Pos PP SFT": "s", "Neg PP SFT": "D",
                   "Base 80B": "o", "GRPO": "^", "GSPO": "v", "CHORD": "P"}

        for name, fname, g in group_models:
            rows, _ = all_data[name]
            vals = []
            sub_vals = []
            for y in years:
                v = get_metric(rows, "igakuqa", "igakuqa", y, "accuracy")
                sv = get_metric(rows, "igakuqa", "igakuqa", y, "subset_accuracy")
                vals.append(v * 100 if v else np.nan)
                sub_vals.append(sv * 100 if sv else np.nan)

            c = colors.get(name, "#333")
            m = markers.get(name, "o")
            ax.plot(x, vals, marker=m, color=c, linewidth=2, markersize=8, label=f"{name}")
            ax.plot(x, sub_vals, marker=m, color=c, linewidth=1, markersize=5,
                    linestyle="--", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(70, 100) if group == "80B" else ax.set_ylim(70, 100)

    fig.suptitle("医師国家試験 年度別推移 (実線=全問, 破線=テキストのみ)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "igakuqa_yearly.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: igakuqa_yearly.png")


def main():
    all_data = load_all()
    plot_heatmap_absolute(all_data)
    plot_heatmap_delta(all_data)
    plot_igakuqa_yearly(all_data)
    print("\nAll Phase 1 figures generated.")


if __name__ == "__main__":
    main()
