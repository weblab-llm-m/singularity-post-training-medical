#!/usr/bin/env python3
"""
Head Classifier for MoE Models
抽出された注意パターンから3種類のヘッドを分類
翻訳論文の手法を踏襲

Qwen3-30B-A3B対応版:
- 48レイヤー、32ヘッド (合計1536ヘッド)
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


class HeadClassifier:
    """ヘッド分類クラス（MoE対応版）"""

    def __init__(
        self,
        attention_patterns: Dict[int, torch.Tensor],
        annotation_data: List[Dict],
        criteria_config: Dict,
        num_layers: int = 48,
        num_heads: int = 32
    ):
        """
        初期化

        Args:
            attention_patterns: {layer_idx: [batch, num_heads, seq_len]}
            annotation_data: アノテーションデータのリスト
            criteria_config: 分類基準の設定
            num_layers: レイヤー数 (default: 48 for Qwen3-30B-A3B)
            num_heads: ヘッド数 (default: 32)
        """
        self.attention_patterns = attention_patterns
        self.annotation_data = annotation_data
        self.criteria = criteria_config['classification_criteria']
        self.num_layers = num_layers
        self.num_heads = num_heads

        print("HeadClassifier initialized (MoE version)")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        print(f"  Total heads: {num_layers * num_heads}")
        print(f"  Samples: {len(annotation_data)}")

    def classify_all_heads(self) -> Dict:
        """
        全ヘッドを3種類に分類

        Returns:
            results: {
                'medical_term_heads': [(layer, head), ...],
                'guideline_indicator_heads': [(layer, head), ...],
                'reasoning_flow_heads': [(layer, head), ...],
                'unclassified': [(layer, head), ...]
            }
        """
        print("\nClassifying all heads...")

        results = {
            'medical_term_heads': [],
            'guideline_indicator_heads': [],
            'reasoning_flow_heads': [],
            'unclassified': []
        }

        total_heads = self.num_layers * self.num_heads
        classified_count = 0

        # 各レイヤー・各ヘッドを分類
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                head_type = self.classify_single_head(layer, head)
                results[head_type].append((layer, head))

                classified_count += 1

                if classified_count % 200 == 0:
                    print(f"Classified {classified_count}/{total_heads} heads...")

        print(f"\nClassification complete!")
        self.print_classification_summary(results)

        return results

    def classify_single_head(self, layer: int, head: int) -> str:
        """
        単一ヘッドを分類

        翻訳論文の分類基準を適用:
        1. Medical Term Heads (Source Headsに相当)
        2. Guideline Indicator Heads (Indicator Headsに相当)
        3. Reasoning Flow Heads (Positional Headsに相当)

        Args:
            layer: レイヤーインデックス
            head: ヘッドインデックス

        Returns:
            head_type: ヘッドタイプ
        """
        # 全サンプルの注意パターンを平均化
        avg_attention_pattern = self._get_average_attention_pattern(layer, head)

        # 各タイプの判定（優先順位順）
        if self.is_medical_term_head(avg_attention_pattern):
            return 'medical_term_heads'

        elif self.is_guideline_indicator_head(avg_attention_pattern):
            return 'guideline_indicator_heads'

        elif self.is_reasoning_flow_head(avg_attention_pattern):
            return 'reasoning_flow_heads'

        else:
            return 'unclassified'

    def _get_average_attention_pattern(
        self,
        layer: int,
        head: int
    ) -> torch.Tensor:
        """
        全サンプルでの平均注意パターンを取得

        Args:
            layer: レイヤーインデックス
            head: ヘッドインデックス

        Returns:
            avg_pattern: [seq_len] 平均注意パターン
        """
        if layer not in self.attention_patterns:
            return torch.zeros(100)

        # [batch, num_heads, seq_len] → [batch, seq_len]
        layer_patterns = self.attention_patterns[layer]
        head_patterns = layer_patterns[:, head, :]  # [batch, seq_len]

        # バッチ平均
        avg_pattern = head_patterns.mean(dim=0)  # [seq_len]

        return avg_pattern

    def is_medical_term_head(self, attn_pattern: torch.Tensor) -> bool:
        """
        医療用語ヘッドかどうかを判定

        Args:
            attn_pattern: 注意パターン [seq_len]

        Returns:
            is_medical: 医療用語ヘッドかどうか
        """
        all_medical_positions = []

        for annotation in self.annotation_data:
            positions = annotation.get('medical_term_positions', [])
            all_medical_positions.extend(positions)

        if not all_medical_positions:
            return False

        seq_len = len(attn_pattern)
        valid_positions = [p for p in all_medical_positions if p < seq_len]

        if not valid_positions:
            return False

        medical_attention_score = attn_pattern[valid_positions].mean().item()

        threshold = self.criteria['medical_term']['threshold']

        return medical_attention_score > threshold

    def is_guideline_indicator_head(self, attn_pattern: torch.Tensor) -> bool:
        """
        ガイドライン指示語ヘッドかどうかを判定

        Args:
            attn_pattern: 注意パターン [seq_len]

        Returns:
            is_guideline: ガイドライン指示語ヘッドかどうか
        """
        all_guideline_positions = []

        for annotation in self.annotation_data:
            positions = annotation.get('guideline_indicator_positions', [])
            all_guideline_positions.extend(positions)

        if not all_guideline_positions:
            return False

        seq_len = len(attn_pattern)
        valid_positions = [p for p in all_guideline_positions if p < seq_len]

        if not valid_positions:
            return False

        max_attn_to_guideline = attn_pattern[valid_positions].max().item()

        other_positions = [i for i in range(seq_len) if i not in valid_positions]
        if not other_positions:
            return False

        mean_other_attn = attn_pattern[other_positions].mean().item()
        spike_ratio = max_attn_to_guideline / (mean_other_attn + 1e-10)

        spike_threshold = self.criteria['guideline_indicator']['spike_threshold']
        spike_ratio_threshold = self.criteria['guideline_indicator']['spike_ratio']

        return (max_attn_to_guideline > spike_threshold and
                spike_ratio > spike_ratio_threshold)

    def is_reasoning_flow_head(self, attn_pattern: torch.Tensor) -> bool:
        """
        推論フローヘッドかどうかを判定

        Args:
            attn_pattern: 注意パターン [seq_len]

        Returns:
            is_reasoning: 推論フローヘッドかどうか
        """
        all_reasoning_positions = []

        for annotation in self.annotation_data:
            positions = annotation.get('reasoning_keyword_positions', [])
            all_reasoning_positions.extend(positions)

        if not all_reasoning_positions:
            return False

        seq_len = len(attn_pattern)

        adjacent_window = self.criteria['reasoning_flow'].get('adjacent_window', 3)
        adjacent_positions = list(range(max(0, seq_len - adjacent_window), seq_len))

        relevant_positions = list(set(all_reasoning_positions + adjacent_positions))
        relevant_positions = [p for p in relevant_positions if p < seq_len]

        if not relevant_positions:
            return False

        relevant_attention = attn_pattern[relevant_positions]
        attention_std = relevant_attention.std().item()
        attention_mean = relevant_attention.mean().item()

        uniformity_threshold = self.criteria['reasoning_flow']['uniformity_threshold']
        mean_threshold = self.criteria['reasoning_flow']['attention_mean_threshold']

        relative_std_threshold = self.criteria['reasoning_flow'].get('relative_std_threshold', 999)
        relative_std = attention_std / (attention_mean + 1e-10)

        absolute_criteria = (attention_std < uniformity_threshold and
                           attention_mean > mean_threshold)
        relative_criteria = (relative_std < relative_std_threshold and
                           attention_mean > mean_threshold)

        return absolute_criteria or relative_criteria

    def print_classification_summary(self, results: Dict):
        """
        分類サマリーを出力

        Args:
            results: 分類結果
        """
        total_heads = self.num_layers * self.num_heads

        print("\n" + "="*60)
        print("Head Classification Summary (MoE Model)")
        print("="*60)
        print(f"Total heads: {total_heads}")
        print(f"\nClassification results:")
        print(f"  Medical Term Heads: {len(results['medical_term_heads'])} " +
              f"({len(results['medical_term_heads'])/total_heads*100:.1f}%)")
        print(f"  Guideline Indicator Heads: {len(results['guideline_indicator_heads'])} " +
              f"({len(results['guideline_indicator_heads'])/total_heads*100:.1f}%)")
        print(f"  Reasoning Flow Heads: {len(results['reasoning_flow_heads'])} " +
              f"({len(results['reasoning_flow_heads'])/total_heads*100:.1f}%)")
        print(f"  Unclassified: {len(results['unclassified'])} " +
              f"({len(results['unclassified'])/total_heads*100:.1f}%)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Head Classifier for MoE Models")
    parser.add_argument(
        "--attention_patterns",
        type=str,
        required=True,
        help="Attention patterns file (.pt)"
    )
    parser.add_argument(
        "--annotation_data",
        type=str,
        required=True,
        help="Annotation data file (.jsonl)"
    )
    parser.add_argument(
        "--criteria_config",
        type=str,
        default="configs/head_classification_params.yaml",
        help="Classification criteria config file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output classification results file (.json)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=48,
        help="Number of layers (default: 48 for Qwen3-30B-A3B)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of heads per layer (default: 32 for Qwen3-30B-A3B)"
    )

    args = parser.parse_args()

    # 注意パターンを読み込み
    print(f"Loading attention patterns from: {args.attention_patterns}")
    attention_patterns = torch.load(args.attention_patterns, map_location='cpu')

    # アノテーションデータを読み込み
    print(f"Loading annotation data from: {args.annotation_data}")
    annotations = []
    with open(args.annotation_data, 'r', encoding='utf-8') as f:
        for line in f:
            annotations.append(json.loads(line))

    # 分類基準を読み込み
    print(f"Loading classification criteria from: {args.criteria_config}")
    with open(args.criteria_config, 'r', encoding='utf-8') as f:
        criteria_config = yaml.safe_load(f)

    # 分類器を初期化
    classifier = HeadClassifier(
        attention_patterns=attention_patterns,
        annotation_data=annotations,
        criteria_config=criteria_config,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )

    # 全ヘッドを分類
    classification_results = classifier.classify_all_heads()

    # 結果を保存
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # タプルをリストに変換（JSON serialization用）
    json_results = {
        key: [list(item) for item in value]
        for key, value in classification_results.items()
    }

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\nClassification results saved to: {args.output_path}")


if __name__ == '__main__':
    main()
