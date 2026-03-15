#!/usr/bin/env python3
"""
Path Patching Data Builder for igakuQA
3種類のアテンションヘッド抽出に対応したPath Patchingデータ生成

Head分類:
- Medical Term Heads: diseases, diagnostic_methods, biomarkers, treatments, anatomical_terms
- Profile Indicator Heads: profile (患者属性)
- Reasoning Flow Heads: reasoning_keywords (推論キーワード)
"""

import ast
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional


# カテゴリのグルーピング定義
MEDICAL_TERM_CATEGORIES = {
    'diseases',           # (1) 疾患名
    'diagnostic_methods', # (2) 診断方法
    'biomarkers',         # (3) バイオマーカー
    'treatments',         # (5) 治療法
    'anatomical_terms'    # (6) 解剖学用語
}

PROFILE_CATEGORIES = {
    'profile'             # (4) 患者属性
}

REASONING_CATEGORIES = {
    'reasoning_keywords'  # (7) 推論キーワード
}


def clean_choice(raw):
    """文字列形式のリスト（例: "['a', 'b']"）を実際のリストに変換"""
    if isinstance(raw, str) and raw.strip().startswith('['):
        try:
            parsed = ast.literal_eval(raw)
            return [item.replace('\u3000', '') for item in parsed]
        except Exception:
            return raw
    return raw


def format_choices_text(choices) -> str:
    """選択肢リストを 'a. xxx\nb. yyy\n...' 形式に変換"""
    processed = clean_choice(choices)
    if isinstance(processed, list):
        return '\n'.join([f"{chr(97+i)}. {c}" for i, c in enumerate(processed)])
    return str(processed)


def load_igakuqa_choices(igakuqa_path: str) -> Dict[str, Dict]:
    """
    igakuQAデータセットから problem_id → {choices, answer} のマッピングを構築

    Args:
        igakuqa_path: HuggingFaceデータセット名 or ローカルjsonlパス

    Returns:
        {problem_id: {"choices": [...], "answer": ["a", ...]}}
    """
    mapping = {}

    if os.path.isfile(igakuqa_path):
        # ローカルjsonlファイルから読み込み（tools/dataset.py形式）
        with open(igakuqa_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                pid = item.get('problem_id', '')
                if not pid:
                    continue
                # igakuqa.jsonlはmessages形式なので、choicesはない
                # HuggingFaceから直接読む場合のフォールバック
                mapping[pid] = {
                    "choices": item.get('choices', []),
                    "answer": item.get('answer', []),
                    "solution": item.get('solution', ''),
                }
    else:
        # HuggingFaceデータセットとして読み込み
        try:
            from datasets import load_dataset
            from dotenv import load_dotenv
            load_dotenv(os.path.expandvars(
                '$HOME/singularity-post-training-medical/.env'))
            token = os.environ.get('HF_TOKEN')
            ds = load_dataset(igakuqa_path, split='train', token=token)
            for ex in ds:
                pid = ex.get('problem_id', '')
                if not pid:
                    continue
                choices = clean_choice(ex.get('choices', []))
                answer = clean_choice(ex.get('answer', []))
                mapping[pid] = {
                    "choices": choices if isinstance(choices, list) else [],
                    "answer": answer if isinstance(answer, list) else [answer],
                }
            print(f"  Loaded {len(mapping)} problems from HuggingFace: {igakuqa_path}")
        except Exception as e:
            print(f"  Warning: Failed to load igakuQA dataset: {e}")

    return mapping


def extract_positions_by_group(medical_terms: List[Dict]) -> Dict[str, List[int]]:
    """
    医療用語リストから3グループ別のトークン位置を抽出

    Args:
        medical_terms: アノテーションデータのmedical_termsリスト

    Returns:
        {
            'medical_term_positions': [...],
            'profile_indicator_positions': [...],
            'reasoning_keyword_positions': [...]
        }
    """
    medical_positions: Set[int] = set()
    profile_positions: Set[int] = set()
    reasoning_positions: Set[int] = set()

    for term_info in medical_terms:
        category = term_info.get('category', '')
        token_start = term_info.get('token_start', 0)
        token_end = term_info.get('token_end', 0)
        positions = list(range(token_start, token_end))

        if category in MEDICAL_TERM_CATEGORIES:
            medical_positions.update(positions)
        elif category in PROFILE_CATEGORIES:
            profile_positions.update(positions)
        elif category in REASONING_CATEGORIES:
            reasoning_positions.update(positions)

    return {
        'medical_term_positions': sorted(list(medical_positions)),
        'profile_indicator_positions': sorted(list(profile_positions)),
        'reasoning_keyword_positions': sorted(list(reasoning_positions))
    }


def generate_path_patching_data(
    annotation_path: str,
    counterfactual_path: str,
    output_path: str,
    igakuqa_source: Optional[str] = None
):
    """
    Path Patchingデータを生成

    Args:
        annotation_path: アノテーションデータのパス
        counterfactual_path: Counterfactualデータのパス
        output_path: 出力パス
        igakuqa_source: igakuQAデータソース（HuggingFaceデータセット名 or ローカルパス）
    """
    print("="*60)
    print("Path Patching Data Builder for igakuQA")
    print("="*60 + "\n")

    print(f"Annotation data: {annotation_path}")
    print(f"Counterfactual data: {counterfactual_path}")

    # igakuQAの選択肢・正解データを読み込み
    igakuqa_map = {}
    if igakuqa_source:
        print(f"igakuQA source: {igakuqa_source}")
        igakuqa_map = load_igakuqa_choices(igakuqa_source)
        print(f"  Loaded choices/answers for {len(igakuqa_map)} problems")

    # アノテーションデータを読み込み
    annotations = []
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            annotations.append(json.loads(line))
    print(f"  Loaded {len(annotations)} annotated samples")

    # Counterfactualデータを読み込み
    counterfactuals = []
    with open(counterfactual_path, 'r', encoding='utf-8') as f:
        for line in f:
            counterfactuals.append(json.loads(line))
    print(f"  Loaded {len(counterfactuals)} counterfactual samples")

    # original_indexでマッチング用辞書を作成
    cf_by_index = {cf['original_index']: cf for cf in counterfactuals}

    # Path Patchingデータを構築
    path_patching_data = []
    skipped_no_cf = 0
    matched_choices = 0

    for i, annot in enumerate(annotations):
        annot_idx = annot['index']

        # 対応するcounterfactualがない場合はスキップ
        if annot_idx not in cf_by_index:
            skipped_no_cf += 1
            continue

        cf = cf_by_index[annot_idx]

        # 3グループ別のポジションを抽出
        positions = extract_positions_by_group(annot.get('medical_terms', []))

        # 医療用語のトークンと型を抽出（医療用語カテゴリのみ）
        medical_term_tokens = [
            term['term'] for term in annot.get('medical_terms', [])
            if term.get('category') in MEDICAL_TERM_CATEGORIES
        ]
        term_types = {
            term['term']: term['category']
            for term in annot.get('medical_terms', [])
        }

        # igakuQAから選択肢と正解を取得
        problem_id = annot.get('problem_id', '')
        qa_info = igakuqa_map.get(problem_id, {})
        choices = qa_info.get('choices', [])
        answer_list = qa_info.get('answer', [])

        # reference_data / counterfactual_data に選択肢を追加
        ref_text = annot.get('problem_text', '')
        cf_text = cf.get('counterfactual_text', '')
        if choices:
            matched_choices += 1
            choices_text = format_choices_text(choices)
            ref_text = f"{ref_text}\n{choices_text}"
            cf_text = f"{cf_text}\n{choices_text}"

        # predict_token: 正解があれば使用、なければ"a"
        if answer_list:
            predict_token = answer_list[0]  # 複数回答の場合は先頭を使用
        else:
            predict_token = "a"

        # record_tokens: 選択肢の数に合わせる
        num_choices = len(choices) if choices else 5
        record_tokens = [chr(97 + j) for j in range(num_choices)]

        # Path Patchingアイテムを構築
        path_item = {
            # 基本情報
            "id": annot_idx,
            "problem_id": problem_id,

            # Path Patching用データ（Phase2で必須）
            "reference_data": ref_text,
            "counterfactual_data": cf_text,

            # 予測トークン情報（Phase2のPath Patchingで必須）
            "predict_token": predict_token,
            "record_tokens": record_tokens,

            # 3グループ別のトークン位置（Phase3のHead分類で使用）
            "medical_term_positions": positions['medical_term_positions'],
            "profile_indicator_positions": positions['profile_indicator_positions'],
            "reasoning_keyword_positions": positions['reasoning_keyword_positions'],

            # 医療用語情報
            "medical_term_tokens": medical_term_tokens,
            "term_types": term_types,

            # メタ情報
            "num_medical_terms": len(positions['medical_term_positions']),
            "num_profile_indicators": len(positions['profile_indicator_positions']),
            "num_reasoning_keywords": len(positions['reasoning_keyword_positions']),

            # Counterfactual情報
            "replacements": cf.get('replacements', []),
            "num_replacements": cf.get('num_replacements', 0),
            "counterfactual_strategy": cf.get('strategy', 'strategy2'),

            # 元データ
            "original_text": cf.get('original_text', ''),
        }

        path_patching_data.append(path_item)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(annotations)} samples...")

    if igakuqa_source:
        print(f"\n  Choices matched: {matched_choices}/{len(path_patching_data)}")

    # 保存
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in path_patching_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  Saved {len(path_patching_data)} path patching samples")
    print(f"  Skipped (no counterfactual): {skipped_no_cf}")

    # 統計を出力
    print_statistics(path_patching_data)

    return path_patching_data


def print_statistics(data: List[Dict]):
    """統計情報を出力"""
    total_samples = len(data)

    if total_samples == 0:
        print("No data to print statistics for")
        return

    # ポジション統計
    total_medical = sum(d['num_medical_terms'] for d in data)
    total_profile = sum(d['num_profile_indicators'] for d in data)
    total_reasoning = sum(d['num_reasoning_keywords'] for d in data)
    total_replacements = sum(d['num_replacements'] for d in data)

    avg_medical = total_medical / total_samples
    avg_profile = total_profile / total_samples
    avg_reasoning = total_reasoning / total_samples
    avg_replacements = total_replacements / total_samples

    # サンプル数（各グループのポジションを持つサンプル）
    samples_with_medical = sum(1 for d in data if d['num_medical_terms'] > 0)
    samples_with_profile = sum(1 for d in data if d['num_profile_indicators'] > 0)
    samples_with_reasoning = sum(1 for d in data if d['num_reasoning_keywords'] > 0)

    print("\n" + "="*60)
    print("Path Patching Dataset Statistics")
    print("="*60)
    print(f"Total samples: {total_samples}")

    print(f"\n--- Position Statistics (for Head Classification) ---")
    print(f"\n  Medical Term Heads (diseases, diagnostic_methods, biomarkers, treatments, anatomical_terms):")
    print(f"    Total positions: {total_medical}")
    print(f"    Average per sample: {avg_medical:.2f}")
    print(f"    Samples with positions: {samples_with_medical} ({samples_with_medical/total_samples*100:.1f}%)")

    print(f"\n  Profile Indicator Heads (profile):")
    print(f"    Total positions: {total_profile}")
    print(f"    Average per sample: {avg_profile:.2f}")
    print(f"    Samples with positions: {samples_with_profile} ({samples_with_profile/total_samples*100:.1f}%)")

    print(f"\n  Reasoning Flow Heads (reasoning_keywords):")
    print(f"    Total positions: {total_reasoning}")
    print(f"    Average per sample: {avg_reasoning:.2f}")
    print(f"    Samples with positions: {samples_with_reasoning} ({samples_with_reasoning/total_samples*100:.1f}%)")

    print(f"\n--- Counterfactual Statistics ---")
    print(f"  Total replacements: {total_replacements}")
    print(f"  Average replacements per sample: {avg_replacements:.2f}")

    # カテゴリ別統計
    category_counts = {}
    for sample in data:
        for term, category in sample.get('term_types', {}).items():
            category_counts[category] = category_counts.get(category, 0) + 1

    print(f"\n--- Term Categories Distribution ---")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        group = "Medical" if category in MEDICAL_TERM_CATEGORIES else \
                "Profile" if category in PROFILE_CATEGORIES else \
                "Reasoning" if category in REASONING_CATEGORIES else "Other"
        print(f"    {category}: {count} [{group}]")

    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path Patching Data Builder for igakuQA")
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="Annotation data path (.jsonl)")
    parser.add_argument("--counterfactual_path", type=str, default=None,
                        help="Counterfactual data path (.jsonl)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path (.jsonl)")
    parser.add_argument("--igakuqa_source", type=str,
                        default="weblab-LLM-M/igakuqa-2001-2024-filtered",
                        help="igakuQA dataset (HuggingFace name or local jsonl path)")

    args = parser.parse_args()

    # デフォルトパス設定
    script_dir = Path(__file__).parent

    annotation_path = args.annotation_path or str(script_dir / "annotated_medical_data_full.jsonl")
    counterfactual_path = args.counterfactual_path or str(script_dir / "counterfactual_strategy2.jsonl")
    output_path = args.output_path or str(script_dir / "path_patching_data.jsonl")

    generate_path_patching_data(
        annotation_path=annotation_path,
        counterfactual_path=counterfactual_path,
        output_path=output_path,
        igakuqa_source=args.igakuqa_source
    )

    print("\n" + "="*60)
    print("Path patching data building completed successfully!")
    print("="*60)
