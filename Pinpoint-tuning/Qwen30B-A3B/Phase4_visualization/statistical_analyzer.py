#!/usr/bin/env python3
"""
Statistical Analyzer for MoE Models
ヘッド分類の統計分析

Qwen3-30B-A3B対応版:
- 48レイヤー × 32ヘッド = 1536ヘッドの分析
"""

import json
import argparse
import numpy as np
import os


def analyze_distribution_by_layer(classification_results: dict, num_layers: int = 48) -> dict:
    """
    レイヤーごとのヘッドタイプ分布を分析

    Args:
        classification_results: 分類結果
        num_layers: レイヤー数

    Returns:
        distribution: レイヤー別分布
    """
    distribution = {}

    for layer in range(num_layers):
        distribution[layer] = {
            'medical_term_heads': 0,
            'guideline_indicator_heads': 0,
            'reasoning_flow_heads': 0,
            'unclassified': 0
        }

    for head_type, heads in classification_results.items():
        for item in heads:
            if isinstance(item, list) and len(item) >= 2:
                layer, head = item[0], item[1]
            else:
                continue
            if layer < num_layers:
                distribution[layer][head_type] += 1

    return distribution


def analyze_layer_groups(classification_results: dict, num_layers: int = 48) -> dict:
    """
    レイヤーグループ（前半/中盤/後半）ごとの分析

    Args:
        classification_results: 分類結果
        num_layers: レイヤー数

    Returns:
        group_analysis: グループ別分析
    """
    group_size = num_layers // 3
    groups = {
        'early': list(range(0, group_size)),           # 0-15
        'middle': list(range(group_size, group_size*2)), # 16-31
        'late': list(range(group_size*2, num_layers))   # 32-47
    }

    group_analysis = {}

    for group_name, layer_indices in groups.items():
        group_analysis[group_name] = {
            'layers': f"{layer_indices[0]}-{layer_indices[-1]}",
            'medical_term_heads': 0,
            'guideline_indicator_heads': 0,
            'reasoning_flow_heads': 0
        }

        for head_type in ['medical_term_heads', 'guideline_indicator_heads', 'reasoning_flow_heads']:
            for item in classification_results.get(head_type, []):
                if isinstance(item, list) and len(item) >= 2:
                    layer = item[0]
                    if layer in layer_indices:
                        group_analysis[group_name][head_type] += 1

    return group_analysis


def correlation_with_patching_impact(classification_results: dict, patching_results_path: str) -> dict:
    """
    ヘッドタイプとPath patching impactの相関

    Args:
        classification_results: 分類結果
        patching_results_path: Path patching結果ファイルパス

    Returns:
        correlations: 相関分析結果
    """
    import torch
    patching_tensor = torch.load(patching_results_path, map_location='cpu').numpy()
    num_layers, num_heads = patching_tensor.shape

    correlations = {}

    for head_type, heads in classification_results.items():
        if head_type == 'unclassified':
            continue

        impacts = []
        for item in heads:
            if isinstance(item, list) and len(item) >= 2:
                l, h = item[0], item[1]
                if l < num_layers and h < num_heads:
                    impacts.append(patching_tensor[l, h])

        if impacts:
            correlations[head_type] = {
                'count': len(impacts),
                'mean_impact': float(np.mean(impacts)),
                'std_impact': float(np.std(impacts)),
                'median_impact': float(np.median(impacts)),
                'min_impact': float(np.min(impacts)),
                'max_impact': float(np.max(impacts))
            }

    return correlations


def main():
    parser = argparse.ArgumentParser(description="Statistical Analyzer for MoE Models")
    parser.add_argument("--classification_results", type=str, required=True,
                       help="Classification results JSON file")
    parser.add_argument("--patching_results", type=str, required=True,
                       help="Path patching results PT file")
    parser.add_argument("--output_path", type=str,
                       default="Phase4_visualization/statistical_report.json",
                       help="Output report path")
    parser.add_argument("--num_layers", type=int, default=48,
                       help="Number of layers")

    args = parser.parse_args()

    # データ読み込み
    print(f"Loading classification results from: {args.classification_results}")
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    # 統計分析
    print("Analyzing distribution by layer...")
    distribution = analyze_distribution_by_layer(classification_results, args.num_layers)

    print("Analyzing layer groups...")
    group_analysis = analyze_layer_groups(classification_results, args.num_layers)

    print("Analyzing correlation with patching impact...")
    correlations = correlation_with_patching_impact(classification_results, args.patching_results)

    # 結果を保存
    report = {
        'model_info': {
            'num_layers': args.num_layers,
            'total_heads': sum(len(v) for v in classification_results.values())
        },
        'distribution_by_layer': distribution,
        'layer_group_analysis': group_analysis,
        'impact_correlations': correlations
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w') as f:
        json.dump(report, f, indent=2)

    # サマリー出力
    print("\n" + "="*60)
    print("Statistical Analysis Summary")
    print("="*60)

    print("\nLayer Group Analysis:")
    for group, stats in group_analysis.items():
        total = sum(stats[k] for k in ['medical_term_heads', 'guideline_indicator_heads', 'reasoning_flow_heads'])
        print(f"  {group.title()} (layers {stats['layers']}): {total} classified heads")

    print("\nImpact Correlations:")
    for head_type, stats in correlations.items():
        print(f"  {head_type}: mean={stats['mean_impact']:.2f}%, count={stats['count']}")

    print("="*60)
    print(f"\nStatistical report saved to: {args.output_path}")


if __name__ == '__main__':
    main()
