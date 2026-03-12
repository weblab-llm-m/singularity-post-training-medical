#!/usr/bin/env python3
"""
Medical Pattern Detector
医療QA特有のパターンを検出する補助スクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import argparse
from typing import Dict, List
import numpy as np


class MedicalPatternDetector:
    """医療パターン検出クラス"""

    def __init__(
        self,
        attention_patterns: Dict[int, torch.Tensor],
        annotation_data: List[Dict]
    ):
        """
        初期化

        Args:
            attention_patterns: {layer_idx: [batch, num_heads, seq_len]}
            annotation_data: アノテーションデータ
        """
        self.attention_patterns = attention_patterns
        self.annotation_data = annotation_data

        print("MedicalPatternDetector initialized")

    def detect_all_patterns(self) -> Dict:
        """
        全パターンを検出

        Returns:
            patterns: 検出されたパターン
        """
        patterns = {
            'think_patterns': self.detect_think_patterns(),
            'choice_patterns': self.detect_choice_patterns(),
            'guideline_reference_patterns': self.detect_guideline_reference_patterns()
        }

        self.print_pattern_summary(patterns)

        return patterns

    def detect_think_patterns(self) -> Dict:
        """
        <think>タグ内での注意パターンを解析

        Returns:
            think_patterns: 思考パターン
        """
        print("\nDetecting <think> patterns...")

        think_patterns = {
            'samples_with_think': 0,
            'average_attention_in_think': [],
            'peak_layers_in_think': []
        }

        for idx, annotation in enumerate(self.annotation_data):
            if not annotation.get('has_think_section', False):
                continue

            think_patterns['samples_with_think'] += 1

            # <think>セクションのトークン位置を検出
            # （簡略化: reasoning_keyword_positionsを使用）
            think_positions = annotation.get('reasoning_keyword_positions', [])

            if not think_positions:
                continue

            # 各レイヤーでの<think>への注意を計算
            for layer_idx, layer_patterns in self.attention_patterns.items():
                # [batch, num_heads, seq_len]
                if idx >= layer_patterns.shape[0]:
                    continue

                sample_pattern = layer_patterns[idx]  # [num_heads, seq_len]

                # <think>位置への注意
                valid_positions = [p for p in think_positions if p < sample_pattern.shape[1]]
                if valid_positions:
                    think_attention = sample_pattern[:, valid_positions].mean().item()
                    think_patterns['average_attention_in_think'].append(think_attention)

        return think_patterns

    def detect_choice_patterns(self) -> Dict:
        """
        選択肢パターンを検出

        Returns:
            choice_patterns: 選択肢パターン
        """
        print("Detecting choice patterns...")

        choice_patterns = {
            'average_attention_to_choices': 0.0,
            'correct_vs_incorrect_attention': []
        }

        # 実装は省略（データ形式に依存）

        return choice_patterns

    def detect_guideline_reference_patterns(self) -> Dict:
        """
        ガイドライン参照パターンを検出

        Returns:
            guideline_patterns: ガイドライン参照パターン
        """
        print("Detecting guideline reference patterns...")

        guideline_patterns = {
            'samples_with_guidelines': 0,
            'average_attention_to_guidelines': []
        }

        for idx, annotation in enumerate(self.annotation_data):
            guideline_positions = annotation.get('guideline_indicator_positions', [])

            if not guideline_positions:
                continue

            guideline_patterns['samples_with_guidelines'] += 1

            # 各レイヤーでのガイドライン参照への注意を計算
            for layer_idx, layer_patterns in self.attention_patterns.items():
                if idx >= layer_patterns.shape[0]:
                    continue

                sample_pattern = layer_patterns[idx]  # [num_heads, seq_len]

                # ガイドライン位置への注意
                valid_positions = [p for p in guideline_positions if p < sample_pattern.shape[1]]
                if valid_positions:
                    guideline_attention = sample_pattern[:, valid_positions].mean().item()
                    guideline_patterns['average_attention_to_guidelines'].append(guideline_attention)

        return guideline_patterns

    def print_pattern_summary(self, patterns: Dict):
        """
        パターンサマリーを出力

        Args:
            patterns: 検出されたパターン
        """
        print("\n" + "="*60)
        print("Medical Pattern Detection Summary")
        print("="*60)

        # <think>パターン
        think = patterns['think_patterns']
        print(f"\n<think> Patterns:")
        print(f"  Samples with <think>: {think['samples_with_think']}")
        if think['average_attention_in_think']:
            print(f"  Average attention in <think>: {np.mean(think['average_attention_in_think']):.4f}")

        # ガイドラインパターン
        guideline = patterns['guideline_reference_patterns']
        print(f"\nGuideline Reference Patterns:")
        print(f"  Samples with guidelines: {guideline['samples_with_guidelines']}")
        if guideline['average_attention_to_guidelines']:
            print(f"  Average attention to guidelines: {np.mean(guideline['average_attention_to_guidelines']):.4f}")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Medical Pattern Detector")
    parser.add_argument(
        "--attention_patterns",
        type=str,
        required=True,
        help="Attention patterns file (.pt)"
    )
    parser.add_argument(
        "--medical_data",
        type=str,
        required=True,
        help="Annotated medical data file (.jsonl)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output patterns file (.json)"
    )

    args = parser.parse_args()

    # 注意パターンを読み込み
    print(f"Loading attention patterns from: {args.attention_patterns}")
    attention_patterns = torch.load(args.attention_patterns, map_location='cpu')

    # アノテーションデータを読み込み
    print(f"Loading medical data from: {args.medical_data}")
    annotations = []
    with open(args.medical_data, 'r', encoding='utf-8') as f:
        for line in f:
            annotations.append(json.loads(line))

    # パターン検出器を初期化
    detector = MedicalPatternDetector(
        attention_patterns=attention_patterns,
        annotation_data=annotations
    )

    # パターンを検出
    patterns = detector.detect_all_patterns()

    # 結果を保存
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # NumPy配列をリストに変換
    json_patterns = {}
    for key, value in patterns.items():
        if isinstance(value, dict):
            json_patterns[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            json_patterns[key] = value

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(json_patterns, f, indent=2, ensure_ascii=False)

    print(f"\nMedical patterns saved to: {args.output_path}")


if __name__ == '__main__':
    main()
