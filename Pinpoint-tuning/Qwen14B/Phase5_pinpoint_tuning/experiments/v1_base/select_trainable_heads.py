#!/usr/bin/env python3
"""
Trainable Head Selector
3種類のヘッド分類結果からPinpoint Tuning対象を選択
"""

import json
import argparse
import torch
import yaml
import os


def select_trainable_heads(
    classification_results: dict,
    patching_results_path: str,
    criteria_config: dict
) -> list:
    """
    トレーニング対象ヘッドを選択

    選択戦略:
    1. Medical Term Headsで impact > threshold の全て
    2. Guideline Indicator Headsで impact > threshold の全て
    3. Reasoning Flow Headsで impact > threshold の上位50%

    Args:
        classification_results: 分類結果
        patching_results_path: Path patching結果のパス
        criteria_config: 選択基準の設定

    Returns:
        selected_heads: 選択されたヘッドのリスト
    """
    # Path patching結果を読み込み
    patching_results = torch.load(patching_results_path, map_location='cpu')

    selection_criteria = criteria_config.get('selection', {})
    impact_thresholds = selection_criteria.get('impact_thresholds', {})

    selected_heads = []

    # Medical Term Headsを優先
    for layer, head in classification_results.get('medical_term_heads', []):
        impact = patching_results[layer][head].item()
        if impact > impact_thresholds.get('medical_term', 0.05):
            selected_heads.append({
                'layer': int(layer),
                'head': int(head),
                'type': 'medical_term',
                'impact': float(impact),
                'priority': 'high'
            })

    # Guideline Indicator Heads
    for layer, head in classification_results.get('guideline_indicator_heads', []):
        impact = patching_results[layer][head].item()
        if impact > impact_thresholds.get('guideline_indicator', 0.08):
            selected_heads.append({
                'layer': int(layer),
                'head': int(head),
                'type': 'guideline_indicator',
                'impact': float(impact),
                'priority': 'high'
            })

    # Reasoning Flow Heads（上位50%のみ）
    reasoning_heads = classification_results.get('reasoning_flow_heads', [])
    reasoning_with_impact = [
        (l, h, patching_results[l][h].item())
        for l, h in reasoning_heads
    ]
    reasoning_with_impact.sort(key=lambda x: x[2], reverse=True)

    top_reasoning = reasoning_with_impact[:len(reasoning_with_impact)//2]
    for layer, head, impact in top_reasoning:
        if impact > impact_thresholds.get('reasoning_flow', 0.10):
            selected_heads.append({
                'layer': int(layer),
                'head': int(head),
                'type': 'reasoning_flow',
                'impact': float(impact),
                'priority': 'medium'
            })

    # 最大ヘッド数で制限
    max_total_heads = selection_criteria.get('max_total_heads', 64)
    if len(selected_heads) > max_total_heads:
        selected_heads.sort(key=lambda x: x['impact'], reverse=True)
        selected_heads = selected_heads[:max_total_heads]

    return selected_heads


def main():
    parser = argparse.ArgumentParser(description="Trainable Head Selector")
    parser.add_argument("--classification_results", type=str, required=True)
    parser.add_argument("--patching_results", type=str, required=True)
    parser.add_argument("--criteria_config", type=str,
                       default="configs/head_classification_params.yaml")
    parser.add_argument("--output_path", type=str,
                       default="Phase5_pinpoint_tuning/trainable_heads.json")

    args = parser.parse_args()

    # データ読み込み
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    with open(args.criteria_config, 'r') as f:
        criteria_config = yaml.safe_load(f)

    # ヘッド選択
    selected_heads = select_trainable_heads(
        classification_results,
        args.patching_results,
        criteria_config
    )

    # 保存
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(selected_heads, f, indent=2)

    print(f"\nSelected {len(selected_heads)} heads for training")
    print(f"Saved to: {args.output_path}")

    # 統計出力
    type_counts = {}
    for head in selected_heads:
        type_counts[head['type']] = type_counts.get(head['type'], 0) + 1

    print("\nBreakdown by type:")
    for head_type, count in type_counts.items():
        print(f"  {head_type}: {count}")


if __name__ == '__main__':
    main()
