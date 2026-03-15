#!/usr/bin/env python3
"""
Heatmap Generator
3種類のヘッドごとにヒートマップを生成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils_common.visualization_helpers import VisualizationHelpers


def generate_classification_heatmap(
    classification_results: dict,
    patching_results: np.ndarray,
    output_dir: str,
    num_layers: int = 40,
    num_heads: int = 40
):
    """
    ヘッド分類結果のヒートマップを生成

    Args:
        classification_results: 分類結果
        patching_results: Path patching結果 [num_layers, num_heads]
        output_dir: 出力ディレクトリ
        num_layers: レイヤー数
        num_heads: ヘッド数
    """
    VisualizationHelpers.setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 12))

    # ベースのヒートマップ（Path patching results）
    vmax = max(abs(patching_results.min()), abs(patching_results.max()))
    im = ax.imshow(
        patching_results,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-vmax,
        vmax=vmax
    )
    cbar = plt.colorbar(im, ax=ax, label='Impact (%)')

    # ヘッド分類を重ねる
    for head_type, heads in classification_results.items():
        if head_type == 'unclassified':
            continue

        color = VisualizationHelpers.get_head_type_color(head_type)
        marker = VisualizationHelpers.get_head_type_marker(head_type)

        x_coords = [h + 0.5 for l, h in heads]
        y_coords = [l + 0.5 for l, h in heads]

        ax.scatter(x_coords, y_coords, c=color, marker=marker, s=50,
                  edgecolors='black', linewidths=1, label=head_type)

    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Head Classification with Path Patching Impact')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    output_path = os.path.join(output_dir, 'classification_heatmap.png')
    VisualizationHelpers.save_figure(fig, output_path)
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Heatmap Generator")
    parser.add_argument("--classification_results", type=str, required=True)
    parser.add_argument("--patching_results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Phase4_visualization/heatmaps")

    args = parser.parse_args()

    # データ読み込み
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    import torch
    patching_results = torch.load(args.patching_results, map_location='cpu').numpy()

    # ヒートマップ生成
    os.makedirs(args.output_dir, exist_ok=True)
    generate_classification_heatmap(classification_results, patching_results, args.output_dir)


if __name__ == '__main__':
    main()
