#!/usr/bin/env python3
"""
Counterfactual Data Generator for igakuQA
医療用語を一般語に置換してカウンターファクチュアルデータを生成
Strategy 2 (Medical → Generic) に特化
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional


def load_annotated_data(input_file: str) -> List[Dict]:
    """アノテーション済みデータをロード"""
    print(f"Loading annotated data from: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"  Loaded {len(data)} samples")
    return data


def load_replacement_mapping(mapping_path: str) -> Dict[str, str]:
    """置換マッピング辞書をロード（外部JSONファイルから）"""
    print(f"Loading replacement mapping from: {mapping_path}")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"  Loaded {len(mapping)} term mappings")
    return mapping


def generate_counterfactual_sample(
    original_sample: Dict,
    replacement_mapping: Dict[str, str],
    replacement_ratio: float = 0.5
) -> Optional[Dict]:
    """
    Strategy 2: 医療用語から一般語への置換
    医療用語を一般的な表現に置換

    Args:
        original_sample: 元のサンプル（annotated_medical_data_full.jsonlの1行）
        replacement_mapping: 医療用語→一般語のマッピング辞書
        replacement_ratio: 置換する用語の割合

    Returns:
        counterfactual_sample: カウンターファクチュアルサンプル
    """
    # 医療用語を抽出
    medical_terms = original_sample.get("medical_terms", [])

    if not medical_terms:
        return None

    # 置換可能な用語をフィルタ（マッピングに存在するもののみ）
    replaceable_terms = [t for t in medical_terms if t["term"] in replacement_mapping]

    if not replaceable_terms:
        return None

    # 置換する用語を選択
    num_to_replace = max(1, int(len(replaceable_terms) * replacement_ratio))
    terms_to_replace = random.sample(replaceable_terms, min(num_to_replace, len(replaceable_terms)))

    # テキストをコピー（igakuQA形式: problem_textのみ）
    problem_text = original_sample.get("problem_text", "")
    if not problem_text:
        return None

    modified_text = problem_text
    replacements = []

    for term_info in terms_to_replace:
        original_term = term_info["term"]
        replacement_term = replacement_mapping[original_term]

        # テキスト内で置換（最初の出現のみ）
        if original_term in modified_text:
            modified_text = modified_text.replace(original_term, replacement_term, 1)

            replacements.append({
                "original": original_term,
                "replacement": replacement_term,
                "category": term_info["category"]
            })

    # 置換が行われなかった場合はNoneを返す
    if not replacements:
        return None

    # カウンターファクチュアルサンプルを構築
    counterfactual = {
        "original_index": original_sample["index"],
        "problem_id": original_sample.get("problem_id", ""),
        "counterfactual_text": modified_text,
        "original_text": problem_text,
        "replacements": replacements,
        "num_replacements": len(replacements),
        "strategy": "strategy2"
    }

    return counterfactual


def generate_counterfactual_dataset(
    input_file: str,
    output_file: str,
    replacement_mapping_path: str,
    replacement_ratio: float = 0.5,
    seed: int = 42
):
    """
    カウンターファクチュアルデータセットを生成
    """
    print("="*60)
    print("Counterfactual Data Generator for igakuQA")
    print("Strategy: Medical → Generic (Strategy 2)")
    print("="*60 + "\n")

    # シード設定
    random.seed(seed)
    print(f"Random seed: {seed}")

    # データをロード
    annotated_data = load_annotated_data(input_file)
    replacement_mapping = load_replacement_mapping(replacement_mapping_path)

    print(f"\nGenerating counterfactual samples...")
    print(f"  Replacement ratio: {replacement_ratio}")
    print()

    counterfactual_data = []
    skipped_no_terms = 0
    skipped_no_mapping = 0

    for i, sample in enumerate(annotated_data):
        cf_sample = generate_counterfactual_sample(
            sample, replacement_mapping, replacement_ratio
        )
        if cf_sample:
            counterfactual_data.append(cf_sample)
        else:
            # スキップ理由を記録
            if not sample.get("medical_terms"):
                skipped_no_terms += 1
            else:
                skipped_no_mapping += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(annotated_data)} samples...")

    # 保存
    print(f"\nSaving counterfactual data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in counterfactual_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"  Saved {len(counterfactual_data)} counterfactual samples")

    # 統計
    total_replacements = sum(s["num_replacements"] for s in counterfactual_data)
    avg_replacements = total_replacements / len(counterfactual_data) if counterfactual_data else 0

    print(f"\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"  Input samples: {len(annotated_data)}")
    print(f"  Output samples: {len(counterfactual_data)}")
    print(f"  Skipped (no medical terms): {skipped_no_terms}")
    print(f"  Skipped (no mapping available): {skipped_no_mapping}")
    print(f"  Total replacements: {total_replacements}")
    print(f"  Average replacements per sample: {avg_replacements:.2f}")

    # 置換カテゴリの分布
    category_counts = {}
    for sample in counterfactual_data:
        for repl in sample["replacements"]:
            cat = repl["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\nReplacement category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat}: {count}")

    return counterfactual_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual Data Generator for igakuQA")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input annotated JSONL file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output counterfactual JSONL file")
    parser.add_argument("--replacement_mapping", type=str, default=None,
                        help="Replacement mapping JSON file")
    parser.add_argument("--replacement_ratio", type=float, default=0.5,
                        help="Ratio of terms to replace (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # デフォルトパス設定
    script_dir = Path(__file__).parent

    input_file = args.input_file or str(script_dir / "annotated_medical_data_full.jsonl")
    output_file = args.output_file or str(script_dir / "counterfactual_strategy2.jsonl")
    replacement_mapping = args.replacement_mapping or str(script_dir / "replacement_mapping.json")

    generate_counterfactual_dataset(
        input_file=input_file,
        output_file=output_file,
        replacement_mapping_path=replacement_mapping,
        replacement_ratio=args.replacement_ratio,
        seed=args.seed
    )

    print("\n" + "="*60)
    print("Counterfactual generation completed successfully!")
    print("="*60)
