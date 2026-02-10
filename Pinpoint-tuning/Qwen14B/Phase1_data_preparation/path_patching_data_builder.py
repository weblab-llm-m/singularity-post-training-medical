#!/usr/bin/env python3
"""
Path Patching Data Builder (Enhanced)
アノテーション情報を含む拡張版Path Patchingデータ生成
"""

import json
import argparse
from typing import List, Dict


def generate_enhanced_path_patching_data(
    annotation_path: str,
    counterfactual_path: str,
    output_path: str
):
    """
    拡張版Path Patchingデータを生成

    既存のgenerate_gynecology_path_patching_data.pyに以下を追加:
    - medical_term_positions
    - guideline_indicator_positions
    - reasoning_keyword_positions
    - term_types

    Args:
        annotation_path: アノテーションデータのパス
        counterfactual_path: Counterfactualデータのパス
        output_path: 出力パス
    """
    print(f"\nBuilding enhanced path patching dataset...")
    print(f"Annotation data: {annotation_path}")
    print(f"Counterfactual data: {counterfactual_path}")

    # アノテーションデータを読み込み
    annotations = []
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            annotations.append(json.loads(line))

    print(f"Loaded {len(annotations)} annotated samples")

    # Counterfactualデータを読み込み
    counterfactuals = []
    with open(counterfactual_path, 'r', encoding='utf-8') as f:
        for line in f:
            counterfactuals.append(json.loads(line))

    print(f"Loaded {len(counterfactuals)} counterfactual samples")

    # データ数の整合性確認（Strategy 2はサンプル数が少ない可能性あり）
    # original_indexでマッチング
    cf_by_index = {cf['original_index']: cf for cf in counterfactuals}

    # Path Patchingデータを構築
    path_patching_data = []

    for i, annot in enumerate(annotations):
        annot_idx = annot['index']

        # 対応するcounterfactualがない場合はスキップ
        if annot_idx not in cf_by_index:
            continue

        cf = cf_by_index[annot_idx]

        # reference_data = question + answer（元データ）
        reference_data = f"{annot['question']}\n{annot['answer']}"

        # counterfactual_data = question + answer（置換後）
        counterfactual_data = f"{cf['question']}\n{cf['answer']}"

        # 正解トークンを抽出（回答の最後の文字、通常は"a", "b", "c"など）
        answer_text = annot['answer'].strip()
        # <think>セクションがある場合は除外して最後の文字を取得
        if '</think>' in answer_text:
            predict_token = answer_text.split('</think>')[-1].strip()
        else:
            predict_token = answer_text

        # record_tokensは選択肢（a, b, c, d, e）
        record_tokens = ["a", "b", "c", "d", "e"]

        # 医療用語のトークンと型を抽出
        medical_term_tokens = [term['term'] for term in annot['medical_terms']]
        term_types = {term['term']: term['category'] for term in annot['medical_terms']}

        # Path Patchingアイテムを構築
        path_item = {
            # 基本情報
            "id": annot_idx,

            # Path Patching用データ（Phase2で必須）
            "reference_data": reference_data,
            "counterfactual_data": counterfactual_data,
            "predict_token": predict_token,
            "record_tokens": record_tokens,

            # アノテーション情報（翻訳論文手法で使用）
            "medical_term_positions": annot['medical_term_positions'],
            "medical_term_tokens": medical_term_tokens,
            "term_types": term_types,
            "guideline_indicator_positions": annot['guideline_indicator_positions'],
            "reasoning_keyword_positions": annot['reasoning_keyword_positions'],

            # メタ情報
            "has_think_section": annot['has_think_section'],
            "num_medical_terms": annot['term_count'],
            "num_guideline_indicators": len(annot['guideline_indicator_positions']),
            "num_reasoning_keywords": len(annot['reasoning_keyword_positions']),

            # Counterfactual情報
            "replacements": cf['replacements'],
            "num_replacements": cf['num_replacements'],
            "counterfactual_strategy": cf['strategy'],

            # 元データも保持
            "original_question": cf['original_question'],
            "original_answer": cf['original_answer']
        }

        path_patching_data.append(path_item)

        if (i + 1) % 100 == 0:
            print(f"Built {len(path_patching_data)}/{len(annotations)} path patching samples...")

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in path_patching_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nPath patching data building complete!")
    print(f"Generated {len(path_patching_data)} enhanced path patching samples")
    print(f"Saved to: {output_path}")

    # 統計を出力
    print_statistics(path_patching_data)


def print_statistics(data: List[Dict]):
    """
    統計情報を出力

    Args:
        data: Path Patchingデータ
    """
    total_samples = len(data)

    if total_samples == 0:
        print("No data to print statistics for")
        return

    # 医療用語統計
    avg_medical_terms = sum(d['num_medical_terms'] for d in data) / total_samples
    avg_guideline_indicators = sum(d['num_guideline_indicators'] for d in data) / total_samples
    avg_reasoning_keywords = sum(d['num_reasoning_keywords'] for d in data) / total_samples
    avg_replacements = sum(d['num_replacements'] for d in data) / total_samples

    samples_with_think = sum(1 for d in data if d['has_think_section'])

    print("\n" + "="*60)
    print("Path Patching Dataset Statistics")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"\nAverage counts per sample:")
    print(f"  - Medical terms: {avg_medical_terms:.2f}")
    print(f"  - Guideline indicators: {avg_guideline_indicators:.2f}")
    print(f"  - Reasoning keywords: {avg_reasoning_keywords:.2f}")
    print(f"  - Replacements: {avg_replacements:.2f}")
    print(f"\nSamples with <think> section: {samples_with_think} ({samples_with_think/total_samples*100:.1f}%)")

    # カテゴリ別統計
    category_counts = {}
    for sample in data:
        for term, category in sample['term_types'].items():
            category_counts[category] = category_counts.get(category, 0) + 1

    print("\nTerm categories distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {category}: {count}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Path Patching Data Builder")
    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Annotation data path (.jsonl)"
    )
    parser.add_argument(
        "--counterfactual_path",
        type=str,
        required=True,
        help="Counterfactual data path (.jsonl)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path (.jsonl)"
    )

    args = parser.parse_args()

    generate_enhanced_path_patching_data(
        annotation_path=args.annotation_path,
        counterfactual_path=args.counterfactual_path,
        output_path=args.output_path
    )


if __name__ == '__main__':
    main()
