#!/usr/bin/env python3
"""
Heatmap Generator for MoE Models
3種類のヘッドごとにヒートマップを生成

Qwen3-30B-A3B対応版:
- 48レイヤー × 32ヘッド = 1536ヘッドの可視化
- 縦長のヒートマップ
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# visualization_helpersをインポート
try:
    from Phase1_data_preparation.utils_common.visualization_helpers import VisualizationHelpers
except ImportError:
    # フォールバック: ローカル定義
    class VisualizationHelpers:
        @staticmethod
        def get_head_type_color(head_type):
            color_map = {
                'medical_term_heads': 'red',
                'guideline_indicator_heads': 'blue',
                'reasoning_flow_heads': 'green',
                'unclassified': 'gray'
            }
            return color_map.get(head_type, 'gray')

        @staticmethod
        def get_head_type_marker(head_type):
            marker_map = {
                'medical_term_heads': 'o',
                'guideline_indicator_heads': 's',
                'reasoning_flow_heads': '^',
                'unclassified': '.'
            }
            return marker_map.get(head_type, '.')

        @staticmethod
        def setup_plot_style():
            plt.rcParams['figure.figsize'] = (14, 16)
            plt.rcParams['font.size'] = 10

        @staticmethod
        def save_figure(fig, output_path, dpi=150):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')


def generate_classification_heatmap(
    classification_results: dict,
    patching_results: np.ndarray,
    output_dir: str,
    num_layers: int = 48,
    num_heads: int = 32
):
    """
    ヘッド分類結果のヒートマップを生成

    Args:
        classification_results: 分類結果
        patching_results: Path patching結果 [num_layers, num_heads]
        output_dir: 出力ディレクトリ
        num_layers: レイヤー数 (default: 48)
        num_heads: ヘッド数 (default: 32)
    """
    VisualizationHelpers.setup_plot_style()

    # 48層用に縦長のfigsize
    fig, ax = plt.subplots(figsize=(14, 16))

    # ベースのヒートマップ（Path patching results）
    vmax = max(abs(patching_results.min()), abs(patching_results.max()))
    im = ax.imshow(
        patching_results,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-vmax,
        vmax=vmax
    )
    cbar = plt.colorbar(im, ax=ax, label='Impact (%)', shrink=0.8)

    # ヘッド分類を重ねる
    for head_type, heads in classification_results.items():
        if head_type == 'unclassified':
            continue

        color = VisualizationHelpers.get_head_type_color(head_type)
        marker = VisualizationHelpers.get_head_type_marker(head_type)

        x_coords = [h + 0.5 for l, h in heads if l < num_layers and h < num_heads]
        y_coords = [l + 0.5 for l, h in heads if l < num_layers and h < num_heads]

        ax.scatter(x_coords, y_coords, c=color, marker=marker, s=40,
                  edgecolors='black', linewidths=0.5, label=head_type.replace('_', ' ').title())

    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title(f'Head Classification with Path Patching Impact\n(Qwen3-30B-A3B: {num_layers} layers × {num_heads} heads)')

    # X軸のティック調整
    ax.set_xticks(np.arange(0, num_heads, 4))
    ax.set_yticks(np.arange(0, num_layers, 4))

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    output_path = os.path.join(output_dir, 'classification_heatmap.png')
    VisualizationHelpers.save_figure(fig, output_path)
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def generate_layer_distribution_plot(
    classification_results: dict,
    output_dir: str,
    num_layers: int = 48
):
    """
    レイヤー別ヘッド分布の棒グラフを生成

    Args:
        classification_results: 分類結果
        output_dir: 出力ディレクトリ
        num_layers: レイヤー数
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # レイヤー別カウント
    layers = np.arange(num_layers)
    medical_counts = np.zeros(num_layers)
    guideline_counts = np.zeros(num_layers)
    reasoning_counts = np.zeros(num_layers)

    for l, h in classification_results.get('medical_term_heads', []):
        if l < num_layers:
            medical_counts[l] += 1
    for l, h in classification_results.get('guideline_indicator_heads', []):
        if l < num_layers:
            guideline_counts[l] += 1
    for l, h in classification_results.get('reasoning_flow_heads', []):
        if l < num_layers:
            reasoning_counts[l] += 1

    width = 0.25
    ax.bar(layers - width, medical_counts, width, label='Medical Term', color='red', alpha=0.7)
    ax.bar(layers, guideline_counts, width, label='Guideline Indicator', color='blue', alpha=0.7)
    ax.bar(layers + width, reasoning_counts, width, label='Reasoning Flow', color='green', alpha=0.7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Classified Heads')
    ax.set_title('Head Classification Distribution by Layer')
    ax.set_xticks(np.arange(0, num_layers, 4))
    ax.legend()

    output_path = os.path.join(output_dir, 'layer_distribution.png')
    VisualizationHelpers.save_figure(fig, output_path)
    plt.close()

    print(f"Layer distribution plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Heatmap Generator for MoE Models")
    parser.add_argument("--classification_results", type=str, required=True,
                       help="Classification results JSON file")
    parser.add_argument("--patching_results", type=str, required=True,
                       help="Path patching results PT file")
    parser.add_argument("--output_dir", type=str, default="Phase4_visualization/heatmaps",
                       help="Output directory")
    parser.add_argument("--num_layers", type=int, default=48,
                       help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=32,
                       help="Number of heads per layer")

    args = parser.parse_args()

    # データ読み込み
    print(f"Loading classification results from: {args.classification_results}")
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    print(f"Loading patching results from: {args.patching_results}")
    import torch
    patching_results = torch.load(args.patching_results, map_location='cpu').numpy()

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # ヒートマップ生成
    generate_classification_heatmap(
        classification_results,
        patching_results,
        args.output_dir,
        args.num_layers,
        args.num_heads
    )

    # レイヤー分布プロット生成
    generate_layer_distribution_plot(
        classification_results,
        args.output_dir,
        args.num_layers
    )


if __name__ == '__main__':
    main()
