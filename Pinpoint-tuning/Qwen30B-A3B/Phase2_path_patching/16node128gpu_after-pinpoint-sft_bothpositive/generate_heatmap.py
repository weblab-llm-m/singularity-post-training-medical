#!/usr/bin/env python3
"""
heatmap_both_positive.png を生成するスクリプト
16node128gpu_after-pinpoint-sft_bothpositive の results_combined.pt を使用
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
MED_PT = SCRIPT_DIR / "results_overlap_medical" / "results_combined.pt"
REA_PT = SCRIPT_DIR / "results_overlap_reasoning" / "results_combined.pt"
OUT_PNG = SCRIPT_DIR / "heatmap_both_positive.png"
OUT_JSON = SCRIPT_DIR / "both_positive_heads.json"

CLIP = 0.05   # ±0.05% clip for colormap
TOTAL_SAMPLES = 4393


def main():
    med = torch.load(MED_PT, weights_only=True).numpy()  # (48, 32)
    rea = torch.load(REA_PT, weights_only=True).numpy()  # (48, 32)

    num_layers, num_heads = med.shape
    print(f"Shape: {med.shape}")

    # Both-positive: positive in both medical AND reasoning
    both_pos_mask = (med > 0) & (rea > 0)
    n_both_pos = int(both_pos_mask.sum())
    print(f"Both-positive heads: {n_both_pos}")

    # Save JSON
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
    both_pos_heads.sort(key=lambda x: x["medical_impact"] + x["reasoning_impact"], reverse=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"both_positive_heads": both_pos_heads}, f, indent=2)
    print(f"Saved: {OUT_JSON}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 18))
    fig.suptitle(
        f"Path Patching: Both-Positive Heads ({TOTAL_SAMPLES:,} overlap samples)\n"
        f"Green \u25a1 = heads where BOTH medical & reasoning replacement increases confidence"
        f"  |  \u00b1{CLIP:.2f}% clip",
        fontsize=12, y=0.998, linespacing=1.6
    )

    cmap = plt.cm.RdBu_r
    vmin, vmax = -CLIP, CLIP

    titles = ["Medical (terms \u2192 generic)", "Reasoning (keywords \u2192 other keywords)"]
    data_arrays = [med, rea]

    for ax, data, title in zip(axes, data_arrays, titles):
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
            interpolation='nearest',
            origin='upper',
        )

        # Green squares for both-positive heads
        for layer in range(num_layers):
            for head in range(num_heads):
                if both_pos_mask[layer, head]:
                    rect = mpatches.FancyBboxPatch(
                        (head - 0.45, layer - 0.45), 0.9, 0.9,
                        boxstyle="square,pad=0",
                        linewidth=1.2,
                        edgecolor='lime',
                        facecolor='none',
                    )
                    ax.add_patch(rect)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Head", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)

        # Ticks every 2
        ax.set_xticks(np.arange(0, num_heads, 2))
        ax.set_yticks(np.arange(0, num_layers, 2))
        ax.set_xticklabels(np.arange(0, num_heads, 2), fontsize=8)
        ax.set_yticklabels(np.arange(0, num_layers, 2), fontsize=8)

        # Legend patch
        green_patch = mpatches.Patch(
            facecolor='none', edgecolor='lime', linewidth=1.2,
            label=f"Both Positive ({n_both_pos})"
        )
        ax.legend(handles=[green_patch], loc='lower left', fontsize=9,
                  framealpha=0.8, facecolor='white')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Impact (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
