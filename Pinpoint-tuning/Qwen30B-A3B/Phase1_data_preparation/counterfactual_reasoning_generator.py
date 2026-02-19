#!/usr/bin/env python3
"""
Counterfactual Data Generator - Reasoning Keywords
推論表現を辞書内の他の推論表現に置換してカウンターファクチュアルデータを生成

対象: 医療用語置換不可 かつ 推論表現ありのサンプル (2,098件)
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def generate_reasoning_counterfactual(
    sample: Dict,
    reasoning_keywords: List[str],
    replacement_mapping: Dict[str, str],
    replacement_ratio: float = 0.5,
) -> Optional[Dict]:
    """
    推論表現を他の推論表現に置換

    Args:
        sample: annotated_medical_data_full.jsonl の1行
        reasoning_keywords: 推論キーワードリスト
        replacement_mapping: 医療用語マッピング (置換可能判定用)
        replacement_ratio: 置換する割合
    """
    terms = sample.get("medical_terms", [])

    # 医療用語置換可能なサンプルはスキップ (医療用データセット側で処理)
    if any(t["term"] in replacement_mapping for t in terms):
        return None

    # reasoning_keywords カテゴリの用語を抽出
    reasoning_terms = [t for t in terms if t["category"] == "reasoning_keywords"]
    if not reasoning_terms:
        return None

    # 置換する用語を選択
    num_to_replace = max(1, int(len(reasoning_terms) * replacement_ratio))
    terms_to_replace = random.sample(reasoning_terms, min(num_to_replace, len(reasoning_terms)))

    problem_text = sample.get("problem_text", "")
    if not problem_text:
        return None

    modified_text = problem_text
    replacements = []

    for term_info in terms_to_replace:
        original_term = term_info["term"]

        # 同じ用語以外からランダムに選択
        candidates = [kw for kw in reasoning_keywords if kw != original_term]
        if not candidates:
            continue

        # 文の自然さを考慮: 長さが近い候補を優先
        orig_len = len(original_term)
        candidates_scored = sorted(candidates, key=lambda c: abs(len(c) - orig_len))
        # 上位20%からランダム選択
        top_n = max(3, len(candidates_scored) // 5)
        replacement_term = random.choice(candidates_scored[:top_n])

        if original_term in modified_text:
            modified_text = modified_text.replace(original_term, replacement_term, 1)
            replacements.append({
                "original": original_term,
                "replacement": replacement_term,
                "category": "reasoning_keywords"
            })

    if not replacements:
        return None

    return {
        "original_index": sample["index"],
        "problem_id": sample.get("problem_id", ""),
        "counterfactual_text": modified_text,
        "original_text": problem_text,
        "replacements": replacements,
        "num_replacements": len(replacements),
        "strategy": "reasoning_swap"
    }


def main():
    parser = argparse.ArgumentParser(description="Reasoning Counterfactual Generator")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--medical_dict", type=str, default=None)
    parser.add_argument("--replacement_mapping", type=str, default=None)
    parser.add_argument("--replacement_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_file = args.input_file or str(script_dir / "annotated_medical_data_full.jsonl")
    output_file = args.output_file or str(script_dir / "counterfactual_reasoning.jsonl")
    medical_dict_path = args.medical_dict or str(script_dir / "medical_terms_dictionary.json")
    mapping_path = args.replacement_mapping or str(script_dir / "replacement_mapping.json")

    random.seed(args.seed)

    print("=" * 60)
    print("Counterfactual Generator - Reasoning Keywords")
    print("Strategy: Reasoning keyword → Other reasoning keyword")
    print("=" * 60)

    # データ読み込み
    with open(input_file) as f:
        data = [json.loads(l) for l in f]
    print(f"Input: {len(data)} samples")

    with open(medical_dict_path) as f:
        med_dict = json.load(f)
    reasoning_keywords = med_dict.get("reasoning_keywords", [])
    print(f"Reasoning keywords: {len(reasoning_keywords)}")

    with open(mapping_path) as f:
        replacement_mapping = json.load(f)
    print(f"Medical replacement mapping: {len(replacement_mapping)} terms")
    print(f"Replacement ratio: {args.replacement_ratio}\n")

    # 生成
    results = []
    for i, sample in enumerate(data):
        cf = generate_reasoning_counterfactual(
            sample, reasoning_keywords, replacement_mapping, args.replacement_ratio
        )
        if cf:
            results.append(cf)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(data)}...")

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 統計
    total_repl = sum(r["num_replacements"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Output samples: {len(results)}")
    print(f"  Total replacements: {total_repl}")
    print(f"  Avg replacements/sample: {total_repl/len(results):.2f}")

    # 置換例を表示
    print(f"\nExamples:")
    for r in results[:3]:
        print(f"  ID: {r['problem_id']}")
        print(f"    Original:      {r['original_text'][:80]}...")
        print(f"    Counterfactual: {r['counterfactual_text'][:80]}...")
        for rep in r['replacements']:
            print(f"    Replace: '{rep['original']}' → '{rep['replacement']}'")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
