#!/usr/bin/env python3
"""
Statistical Analyzer
ヘッド分類の統計分析
"""

import json
import argparse
import numpy as np
from scipy import stats
import os


def analyze_distribution_by_layer(classification_results, num_layers=40):
    """レイヤーごとのヘッドタイプ分布を分析"""
    distribution = {}

    for layer in range(num_layers):
        distribution[layer] = {
            'medical_term_heads': 0,
            'guideline_indicator_heads': 0,
            'reasoning_flow_heads': 0,
            'unclassified': 0
        }

    for head_type, heads in classification_results.items():
        for layer, head in heads:
            distribution[layer][head_type] += 1

    return distribution


def correlation_with_patching_impact(classification_results, patching_results):
    """ヘッドタイプとPath patching impactの相関"""
    import torch
    patching_tensor = torch.load(patching_results, map_location='cpu').numpy()

    correlations = {}

    for head_type, heads in classification_results.items():
        if head_type == 'unclassified':
            continue

        impacts = [patching_tensor[l, h] for l, h in heads]

        correlations[head_type] = {
            'mean_impact': float(np.mean(impacts)),
            'std_impact': float(np.std(impacts)),
            'median_impact': float(np.median(impacts)),
            'count': len(impacts)
        }

    return correlations


def main():
    parser = argparse.ArgumentParser(description="Statistical Analyzer")
    parser.add_argument("--classification_results", type=str, required=True)
    parser.add_argument("--patching_results", type=str, required=True)
    parser.add_argument("--output_path", type=str,
                       default="Phase4_visualization/statistical_report.json")

    args = parser.parse_args()

    # データ読み込み
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    # 統計分析
    distribution = analyze_distribution_by_layer(classification_results)
    correlations = correlation_with_patching_impact(classification_results, args.patching_results)

    # 結果を保存
    report = {
        'distribution_by_layer': distribution,
        'impact_correlations': correlations
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Statistical report saved to: {args.output_path}")


if __name__ == '__main__':
    main()
