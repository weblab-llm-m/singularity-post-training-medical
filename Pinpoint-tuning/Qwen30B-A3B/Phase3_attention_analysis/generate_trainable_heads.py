#!/usr/bin/env python3
"""
Generate Trainable Heads for Pinpoint Tuning
Path PatchingとHead Classificationの結果からSPT用のtrainable_heads.jsonを生成

Qwen3-30B-A3B対応版:
- 48レイヤー、32ヘッド (合計1536ヘッド)
- 最大61ヘッド (約4%) を選択
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import argparse
import yaml
from typing import Dict, List, Tuple
import numpy as np


def load_path_patching_results(results_path: str) -> torch.Tensor:
    """Path Patchingの結果を読み込み"""
    results = torch.load(results_path, map_location='cpu')
    return results


def load_head_classification(classification_path: str) -> Dict:
    """ヘッド分類結果を読み込み"""
    with open(classification_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_selection_config(config_path: str) -> Dict:
    """選択設定を読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def select_trainable_heads(
    path_patching_results: torch.Tensor,
    classification_results: Dict,
    selection_config: Dict,
    num_layers: int = 48,
    num_heads: int = 32
) -> List[Dict]:
    """
    学習対象のヘッドを選択

    Args:
        path_patching_results: [num_layers, num_heads] Path Patching結果
        classification_results: ヘッド分類結果
        selection_config: 選択設定
        num_layers: レイヤー数
        num_heads: ヘッド数

    Returns:
        trainable_heads: [{"layer": int, "head": int, "type": str, "impact": float}, ...]
    """
    selection = selection_config.get('selection', {})
    impact_thresholds = selection.get('impact_thresholds', {})
    max_heads_per_type = selection.get('max_heads_per_type', {})
    max_total_heads = selection.get('max_total_heads', 61)  # 1536 * 4% ≈ 61

    print("\nSelecting trainable heads...")
    print(f"  Max total heads: {max_total_heads}")

    trainable_heads = []

    # 各タイプごとに選択
    for head_type in ['medical_term_heads', 'guideline_indicator_heads', 'reasoning_flow_heads']:
        type_key = head_type.replace('_heads', '')
        threshold = impact_thresholds.get(type_key, 0.05)
        max_heads = max_heads_per_type.get(type_key, 20)

        heads_in_type = classification_results.get(head_type, [])

        # Impactでソート（絶対値が大きい順 = 負の値が大きい順）
        heads_with_impact = []
        for layer, head in heads_in_type:
            if layer < num_layers and head < num_heads:
                impact = path_patching_results[layer, head].item()
                heads_with_impact.append({
                    'layer': layer,
                    'head': head,
                    'type': type_key,
                    'impact': impact
                })

        # 負のimpact（モデルへの影響が大きい）を優先
        heads_with_impact.sort(key=lambda x: x['impact'])

        # 閾値以下のヘッドを選択
        selected = [h for h in heads_with_impact if h['impact'] < -threshold * 100]

        # 最大数で制限
        selected = selected[:max_heads]

        trainable_heads.extend(selected)
        print(f"  {head_type}: {len(selected)} heads selected (threshold: {threshold*100}%)")

    # 全体の最大数で制限
    trainable_heads.sort(key=lambda x: x['impact'])
    trainable_heads = trainable_heads[:max_total_heads]

    print(f"\nTotal trainable heads: {len(trainable_heads)}")

    return trainable_heads


def print_trainable_heads_summary(trainable_heads: List[Dict]):
    """選択されたヘッドのサマリーを出力"""
    print("\n" + "="*60)
    print("Trainable Heads Summary")
    print("="*60)

    # タイプ別集計
    type_counts = {}
    for head in trainable_heads:
        head_type = head['type']
        type_counts[head_type] = type_counts.get(head_type, 0) + 1

    print(f"Total: {len(trainable_heads)} heads")
    for head_type, count in sorted(type_counts.items()):
        print(f"  {head_type}: {count}")

    # レイヤー別分布
    layer_counts = {}
    for head in trainable_heads:
        layer = head['layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    print(f"\nLayer distribution:")
    print(f"  Early layers (0-15): {sum(layer_counts.get(i, 0) for i in range(16))}")
    print(f"  Middle layers (16-31): {sum(layer_counts.get(i, 0) for i in range(16, 32))}")
    print(f"  Late layers (32-47): {sum(layer_counts.get(i, 0) for i in range(32, 48))}")

    # Top 10ヘッド
    print(f"\nTop 10 most impactful heads:")
    for i, head in enumerate(trainable_heads[:10]):
        print(f"  {i+1}. Layer {head['layer']:2d}, Head {head['head']:2d} "
              f"({head['type']}): {head['impact']:+.2f}%")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate Trainable Heads for SPT")
    parser.add_argument(
        "--path_patching_results",
        type=str,
        required=True,
        help="Path patching results file (.pt)"
    )
    parser.add_argument(
        "--classification_results",
        type=str,
        required=True,
        help="Head classification results file (.json)"
    )
    parser.add_argument(
        "--selection_config",
        type=str,
        default="configs/head_classification_params.yaml",
        help="Selection config file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output trainable heads file (.json)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=48,
        help="Number of layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of heads per layer"
    )

    args = parser.parse_args()

    # データ読み込み
    print(f"Loading path patching results from: {args.path_patching_results}")
    path_patching_results = load_path_patching_results(args.path_patching_results)

    print(f"Loading classification results from: {args.classification_results}")
    classification_results = load_head_classification(args.classification_results)

    print(f"Loading selection config from: {args.selection_config}")
    selection_config = load_selection_config(args.selection_config)

    # ヘッド選択
    trainable_heads = select_trainable_heads(
        path_patching_results,
        classification_results,
        selection_config,
        args.num_layers,
        args.num_heads
    )

    # サマリー出力
    print_trainable_heads_summary(trainable_heads)

    # 結果を保存
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(trainable_heads, f, indent=2, ensure_ascii=False)

    print(f"\nTrainable heads saved to: {args.output_path}")


if __name__ == '__main__':
    main()
