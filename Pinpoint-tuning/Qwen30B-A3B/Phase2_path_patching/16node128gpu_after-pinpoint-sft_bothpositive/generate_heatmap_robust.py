#!/usr/bin/env python3
"""
外れ値サンプル(|impact| > 100%)を除外して heatmap_both_positive_robust.png を生成
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUT_PNG = SCRIPT_DIR / "heatmap_both_positive_robust.png"
OUT_JSON = SCRIPT_DIR / "both_positive_heads_robust.json"

CLIP = 0.05       # ±0.05% clip (参照画像と同じ)
OUTLIER_THRESH = 100.0   # per-sample の |max| > 100% を除外


def load_robust_mean(ds: str) -> tuple[np.ndarray, int]:
    """外れ値サンプルを除いた平均を返す"""
    ps_path = SCRIPT_DIR / f"results_overlap_{ds}" / "results_per_sample_combined.pt"
    ps = torch.load(ps_path, weights_only=True).numpy()   # (N, 48, 32)

    per_sample_max = np.abs(ps).max(axis=(1, 2))
    mask = per_sample_max <= OUTLIER_THRESH
    n_outlier = (~mask).sum()
    n_kept = mask.sum()

    print(f"  [{ds}] total={len(ps)}, outliers removed={n_outlier}, kept={n_kept}")
    robust_mean = ps[mask].mean(axis=0)
    return robust_mean, int(n_kept)


def main():
    print("=== Loading per-sample data (outlier filtering) ===")
    med, n_med = load_robust_mean("medical")
    rea, n_rea = load_robust_mean("reasoning")

    n_samples = min(n_med, n_rea)
    num_layers, num_heads = med.shape
    print(f"Shape: {med.shape}")
    print(f"Medical  stats: mean={med.mean():.5f}, std={med.std():.5f}, "
          f"min={med.min():.5f}, max={med.max():.5f}")
    print(f"Reasoning stats: mean={rea.mean():.5f}, std={rea.std():.5f}, "
          f"min={rea.min():.5f}, max={rea.max():.5f}")

    # Both-positive mask
    both_pos_mask = (med > 0) & (rea > 0)
    n_both_pos = int(both_pos_mask.sum())
    print(f"Both-positive heads: {n_both_pos}")

    # --- Save JSON ---
    both_pos_heads = []
    for layer in range(num_layers):
        for head in range(num_heads):
            if both_pos_mask[layer, head]:
                both_pos_heads.append({
                    "layer": layer,
                    "head": head,
                    "medical_impact": float(round(med[layer, head], 6)),
                    "reasoning_impact": float(round(rea[layer, head], 6)),
                })
    both_pos_heads.sort(
        key=lambda x: x["medical_impact"] + x["reasoning_impact"], reverse=True
    )
    with open(OUT_JSON, "w") as f:
        json.dump({"both_positive_heads": both_pos_heads,
                   "n_samples_medical": n_med,
                   "n_samples_reasoning": n_rea,
                   "outlier_threshold": OUTLIER_THRESH}, f, indent=2)
    print(f"Saved: {OUT_JSON}")

    # --- Plot (参照画像スタイルに合わせる) ---
    fig, axes = plt.subplots(1, 2, figsize=(23.0, 11.9))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        f"Path Patching: Both-Positive Heads ({n_med:,} overlap samples)",
        fontsize=14, y=0.995
    )
    fig.text(
        0.5, 0.965,
        f"Green \u25a1 = heads where BOTH medical & reasoning replacement increases confidence"
        f"  |  \u00b1{CLIP:.2f}% clip",
        ha="center", fontsize=10, color="#444"
    )

    # 参照画像に合わせた発散カラーマップ（青←→赤, 0=白）
    cmap = matplotlib.colormaps["RdBu_r"]
    vmin, vmax = -CLIP, CLIP

    titles = [
        "Medical (terms \u2192 generic)",
        "Reasoning (keywords \u2192 other keywords)"
    ]
    data_arrays = [med, rea]

    im = None
    for ax, data, title in zip(axes, data_arrays, titles):
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_facecolor("white")

        # Green squares for both-positive heads
        for layer in range(num_layers):
            for head in range(num_heads):
                if both_pos_mask[layer, head]:
                    rect = mpatches.FancyBboxPatch(
                        (head - 0.45, layer - 0.45), 0.9, 0.9,
                        boxstyle="square,pad=0",
                        linewidth=1.2,
                        edgecolor="lime",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Head", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)

        ax.set_xticks(np.arange(0, num_heads, 2))
        ax.set_yticks(np.arange(0, num_layers, 2))
        ax.set_xticklabels(np.arange(0, num_heads, 2), fontsize=8)
        ax.set_yticklabels(np.arange(0, num_layers, 2), fontsize=8)

        green_patch = mpatches.Patch(
            facecolor="none", edgecolor="lime", linewidth=1.2,
            label=f"Both Positive ({n_both_pos})"
        )
        ax.legend(handles=[green_patch], loc="lower left", fontsize=9,
                  framealpha=0.8, facecolor="white")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Impact (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0.055, right=0.895, top=0.915, bottom=0.07, wspace=0.18)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PNG}")

    # Top heads summary
    print(f"\n=== Top 10 Medical (robust mean) ===")
    flat = np.argsort(med.flatten())
    print("Negative:")
    for idx in flat[:10]:
        l, h = idx // num_heads, idx % num_heads
        print(f"  Layer {l:2d}, Head {h:2d}: {med[l,h]:+.5f}%")
    print("Positive:")
    for idx in flat[-10:][::-1]:
        l, h = idx // num_heads, idx % num_heads
        print(f"  Layer {l:2d}, Head {h:2d}: {med[l,h]:+.5f}%")


if __name__ == "__main__":
    main()
