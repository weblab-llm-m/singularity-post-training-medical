#!/usr/bin/env python3
"""
Counterfactual Data Generator (Enhanced)
医療用語を置換してカウンターファクチュアルデータを生成（複数戦略対応）
"""

import json
import argparse
import random
from typing import List, Dict, Optional

# 医療用語から一般語への置換マッピング
REPLACEMENT_MAPPING = {
    # 疾患名 → 一般語
    "子宮内膜症": "健康状態",
    "子宮筋腫": "健康状態",
    "卵巣癌": "健康状態",
    "骨粗鬆症": "健康状態",
    "PCOS": "健康状態",
    "多嚢胞性卵巣症候群": "健康状態",
    "不妊症": "健康状態",
    "妊娠": "健康状態",

    # 診断方法 → 一般語
    "CA125測定": "検査",
    "超音波検査": "検査",
    "MRI検査": "検査",
    "経腟超音波検査": "検査",
    "骨密度測定": "検査",
    "ホルモン検査": "検査",
    "血液培養": "検査",

    # 治療法 → 一般語
    "低用量ピル": "治療",
    "ホルモン補充療法": "治療",
    "経口避妊薬": "治療",
    "抗生物質": "治療",
    "運動療法": "治療",
    "手術治療": "治療",

    # 解剖学的用語 → 一般語
    "子宮": "器官",
    "卵巣": "器官",
    "卵管": "器官",
    "骨盤": "部位",
    "子宮内膜": "組織",
    "子宮頸部": "部位"
}


def load_annotated_data(input_file: str) -> List[Dict]:
    """アノテーション済みデータをロード"""
    print(f"Loading annotated data from: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"✓ Loaded {len(data)} samples\n")
    return data


def load_medical_dictionary(dict_path: str) -> Dict[str, List[str]]:
    """医療用語辞書をロード"""
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_counterfactual_sample_strategy1(
    original_sample: Dict,
    medical_dict: Dict[str, List[str]],
    replacement_ratio: float = 0.3
) -> Optional[Dict]:
    """
    Strategy 1: 同一カテゴリ内での医療用語置換
    医療用語を同じカテゴリの別の用語に置換

    Args:
        original_sample: 元のサンプル
        medical_dict: 医療用語辞書
        replacement_ratio: 置換する用語の割合

    Returns:
        counterfactual_sample: カウンターファクチュアルサンプル
    """
    # 医療用語を抽出
    medical_terms = original_sample.get("medical_terms", [])

    if not medical_terms:
        return None

    # 置換する用語を選択
    num_to_replace = max(1, int(len(medical_terms) * replacement_ratio))
    terms_to_replace = random.sample(medical_terms, min(num_to_replace, len(medical_terms)))

    # テキストをコピー
    question = original_sample["question"]
    answer = original_sample["answer"]
    combined_text = f"{question}\n{answer}"

    replacements = []

    for term_info in terms_to_replace:
        original_term = term_info["term"]
        category = term_info["category"]

        # 同じカテゴリから別の用語を選択
        if category in medical_dict:
            candidate_terms = [t for t in medical_dict[category] if t != original_term]
            if candidate_terms:
                replacement_term = random.choice(candidate_terms)

                # テキスト内で置換
                combined_text = combined_text.replace(original_term, replacement_term, 1)

                replacements.append({
                    "original": original_term,
                    "replacement": replacement_term,
                    "category": category,
                    "strategy": "strategy1"
                })

    # 質問と回答を分離
    parts = combined_text.split('\n', 1)
    new_question = parts[0] if len(parts) > 0 else question
    new_answer = parts[1] if len(parts) > 1 else answer

    # カウンターファクチュアルサンプルを構築
    counterfactual = {
        "original_index": original_sample["index"],
        "question": new_question,
        "answer": new_answer,
        "replacements": replacements,
        "num_replacements": len(replacements),
        "original_question": question,
        "original_answer": answer,
        "strategy": "strategy1"
    }

    return counterfactual


def generate_counterfactual_sample_strategy2(
    original_sample: Dict,
    replacement_ratio: float = 0.5
) -> Optional[Dict]:
    """
    Strategy 2: 医療用語から一般語への置換
    医療用語を一般的な表現に置換

    Args:
        original_sample: 元のサンプル
        replacement_ratio: 置換する用語の割合

    Returns:
        counterfactual_sample: カウンターファクチュアルサンプル
    """
    # 医療用語を抽出
    medical_terms = original_sample.get("medical_terms", [])

    if not medical_terms:
        return None

    # 置換可能な用語をフィルタ
    replaceable_terms = [t for t in medical_terms if t["term"] in REPLACEMENT_MAPPING]

    if not replaceable_terms:
        return None

    # 置換する用語を選択
    num_to_replace = max(1, int(len(replaceable_terms) * replacement_ratio))
    terms_to_replace = random.sample(replaceable_terms, min(num_to_replace, len(replaceable_terms)))

    # テキストをコピー
    question = original_sample["question"]
    answer = original_sample["answer"]
    combined_text = f"{question}\n{answer}"

    replacements = []

    for term_info in terms_to_replace:
        original_term = term_info["term"]
        replacement_term = REPLACEMENT_MAPPING[original_term]

        # テキスト内で置換
        combined_text = combined_text.replace(original_term, replacement_term, 1)

        replacements.append({
            "original": original_term,
            "replacement": replacement_term,
            "category": term_info["category"],
            "strategy": "strategy2"
        })

    # 質問と回答を分離
    parts = combined_text.split('\n', 1)
    new_question = parts[0] if len(parts) > 0 else question
    new_answer = parts[1] if len(parts) > 1 else answer

    # カウンターファクチュアルサンプルを構築
    counterfactual = {
        "original_index": original_sample["index"],
        "question": new_question,
        "answer": new_answer,
        "replacements": replacements,
        "num_replacements": len(replacements),
        "original_question": question,
        "original_answer": answer,
        "strategy": "strategy2"
    }

    return counterfactual


def generate_counterfactual_sample_strategy3(
    original_sample: Dict,
    medical_dict: Dict[str, List[str]],
    replacement_ratio: float = 0.7
) -> Optional[Dict]:
    """
    Strategy 3: クロスカテゴリ置換
    医療用語を異なるカテゴリの用語に置換

    Args:
        original_sample: 元のサンプル
        medical_dict: 医療用語辞書
        replacement_ratio: 置換する用語の割合

    Returns:
        counterfactual_sample: カウンターファクチュアルサンプル
    """
    # 医療用語を抽出
    medical_terms = original_sample.get("medical_terms", [])

    if not medical_terms:
        return None

    # 置換する用語を選択
    num_to_replace = max(1, int(len(medical_terms) * replacement_ratio))
    terms_to_replace = random.sample(medical_terms, min(num_to_replace, len(medical_terms)))

    # テキストをコピー
    question = original_sample["question"]
    answer = original_sample["answer"]
    combined_text = f"{question}\n{answer}"

    replacements = []

    for term_info in terms_to_replace:
        original_term = term_info["term"]
        original_category = term_info["category"]

        # 異なるカテゴリから用語を選択
        other_categories = [cat for cat in medical_dict.keys() if cat != original_category]

        if other_categories:
            target_category = random.choice(other_categories)
            candidate_terms = medical_dict[target_category]

            if candidate_terms:
                replacement_term = random.choice(candidate_terms)

                # テキスト内で置換
                combined_text = combined_text.replace(original_term, replacement_term, 1)

                replacements.append({
                    "original": original_term,
                    "replacement": replacement_term,
                    "original_category": original_category,
                    "replacement_category": target_category,
                    "strategy": "strategy3"
                })

    # 質問と回答を分離
    parts = combined_text.split('\n', 1)
    new_question = parts[0] if len(parts) > 0 else question
    new_answer = parts[1] if len(parts) > 1 else answer

    # カウンターファクチュアルサンプルを構築
    counterfactual = {
        "original_index": original_sample["index"],
        "question": new_question,
        "answer": new_answer,
        "replacements": replacements,
        "num_replacements": len(replacements),
        "original_question": question,
        "original_answer": answer,
        "strategy": "strategy3"
    }

    return counterfactual


def generate_counterfactual_sample(
    original_sample: Dict,
    medical_dict: Dict[str, List[str]],
    replacement_ratio: float = 0.3,
    strategy: str = "strategy1"
) -> Optional[Dict]:
    """
    カウンターファクチュアルサンプルを生成（戦略選択対応）

    Args:
        original_sample: 元のサンプル
        medical_dict: 医療用語辞書
        replacement_ratio: 置換する用語の割合
        strategy: 置換戦略 ("strategy1", "strategy2", "strategy3")

    Returns:
        counterfactual_sample: カウンターファクチュアルサンプル
    """
    if strategy == "strategy1":
        return generate_counterfactual_sample_strategy1(
            original_sample, medical_dict, replacement_ratio
        )
    elif strategy == "strategy2":
        return generate_counterfactual_sample_strategy2(
            original_sample, replacement_ratio
        )
    elif strategy == "strategy3":
        return generate_counterfactual_sample_strategy3(
            original_sample, medical_dict, replacement_ratio
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_counterfactual_dataset(
    input_file: str,
    output_file: str,
    medical_dict_path: str,
    replacement_ratio: float = 0.3,
    strategy: str = "strategy1",
    seed: int = 42
):
    """
    カウンターファクチュアルデータセットを生成（複数戦略対応）
    """
    print("="*60)
    print("Counterfactual Data Generator (Enhanced)")
    print("="*60 + "\n")

    # シード設定
    random.seed(seed)

    # データをロード
    annotated_data = load_annotated_data(input_file)
    medical_dict = load_medical_dictionary(medical_dict_path)

    print(f"Generating counterfactual samples...")
    print(f"  Strategy: {strategy}")
    print(f"  Replacement ratio: {replacement_ratio}")
    print()

    counterfactual_data = []

    for sample in annotated_data:
        cf_sample = generate_counterfactual_sample(
            sample, medical_dict, replacement_ratio, strategy
        )
        if cf_sample:
            counterfactual_data.append(cf_sample)

    # 保存
    print(f"\nSaving counterfactual data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in counterfactual_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(counterfactual_data)} counterfactual samples")

    # 統計
    total_replacements = sum(s["num_replacements"] for s in counterfactual_data)
    avg_replacements = total_replacements / len(counterfactual_data) if counterfactual_data else 0

    print(f"\nStatistics:")
    print(f"  Total counterfactual samples: {len(counterfactual_data)}")
    print(f"  Total replacements: {total_replacements}")
    print(f"  Average replacements per sample: {avg_replacements:.2f}")

    return counterfactual_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual Data Generator (Enhanced)")
    parser.add_argument("--input_file", type=str, required=True, help="Input annotated JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output counterfactual JSONL file")
    parser.add_argument("--medical_dict", type=str, required=True, help="Medical dictionary JSON file")
    parser.add_argument("--replacement_ratio", type=float, default=0.3, help="Ratio of terms to replace")
    parser.add_argument("--strategy", type=str, default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"],
                        help="Replacement strategy (strategy1: same category, strategy2: to generic terms, strategy3: cross category)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_counterfactual_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        medical_dict_path=args.medical_dict,
        replacement_ratio=args.replacement_ratio,
        strategy=args.strategy,
        seed=args.seed
    )

    print("\n✓ Counterfactual generation completed successfully!")
