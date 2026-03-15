#!/usr/bin/env python3
"""
Medical Term Annotator for igakuQA
医療用語を自動抽出・アノテーション（igakuQAデータセット用）
"""

import json
import re
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# ユーティリティ関数のインポート
from utils_common.medical_nlp_utils import (
    load_medical_dictionary,
    split_think_sections,
    extract_guideline_indicators,
    extract_reasoning_keywords
)
from utils_common.tokenizer_utils import (
    initialize_qwen3_tokenizer,
    find_token_positions
)


def get_required_answer_count(problem_text: str):
    """
    問題文中の「2つ選べ」「２つ選べ」「二つ選べ」などから
    必要な解答数を推定する。見つからなければ None を返す。
    """
    text = problem_text
    z2h = str.maketrans("０１２３４５６７８９", "0123456789")
    text_norm = text.translate(z2h)

    m = re.search(r'([0-9]+)\s*つ\s*選', text_norm)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    kanji_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5}
    m2 = re.search(r'([一二三四五])\s*つ\s*選', text)
    if m2:
        return kanji_map.get(m2.group(1))

    return None


def annotate_igakuqa_data(
    medical_dict_path: str,
    model_path: str,
    output_file: str,
    num_samples: int = None
):
    """igakuQAデータに用語アノテーションを付加"""

    print("="*60)
    print("Medical Term Annotator for igakuQA")
    print("="*60 + "\n")

    # 医療用語辞書をロード
    print(f"Loading medical dictionary from: {medical_dict_path}")
    medical_dict = load_medical_dictionary(medical_dict_path)
    total_terms = sum(len(v) for v in medical_dict.values())
    print(f"✓ Loaded {len(medical_dict)} categories, {total_terms} total terms\n")

    # トークナイザーをロード
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = initialize_qwen3_tokenizer(model_path)
    print("✓ Tokenizer loaded\n")

    # igakuQAデータをロード
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)

    print("Loading igakuQA dataset from HuggingFace...")
    dataset = load_dataset("weblab-LLM-M/igakuqa-2001-2024-filtered", split="train", token=token)
    print(f"✓ Loaded {len(dataset)} samples\n")

    # アノテーション処理
    print(f"Processing samples...")
    annotated_data = []
    processed_count = 0

    for idx, row in enumerate(dataset):
        # 2023, 2024年を除外
        if str(row.get("year")) in ["2023", "2024", 2023, 2024]:
            continue

        # テキストのみの問題に限定
        if not row.get("text_only"):
            continue

        # 回答がある問題に限定
        if not row.get("answer") or len(row["answer"]) == 0:
            continue

        problem_text = row.get("problem_text", "")
        if not problem_text:
            continue

        # 解答数ミスマッチを除外（dataset.pyと同一フィルタ）
        required = get_required_answer_count(problem_text)
        if required is not None and len(row["answer"]) != required:
            continue

        processed_count += 1
        if processed_count % 500 == 0:
            print(f"  Processed {processed_count} samples...")

        if num_samples and processed_count > num_samples:
            break

        # トークン化
        token_ids = tokenizer.encode(problem_text, add_special_tokens=False)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # 医療用語を検出
        found_terms = []
        term_positions = {}  # {category: [token_indices]}

        for category, terms in medical_dict.items():
            term_positions[category] = []
            for term in terms:
                if term in problem_text:
                    # 用語の文字位置を取得
                    start_idx = problem_text.find(term)
                    if start_idx != -1:
                        end_idx = start_idx + len(term)

                        # トークン位置を推定
                        prefix_tokens = tokenizer.encode(problem_text[:start_idx], add_special_tokens=False)
                        token_start = len(prefix_tokens)
                        token_end = len(tokenizer.encode(problem_text[:end_idx], add_special_tokens=False))

                        found_terms.append({
                            "term": term,
                            "category": category,
                            "char_start": start_idx,
                            "char_end": end_idx,
                            "token_start": token_start,
                            "token_end": token_end
                        })

                        term_positions[category].extend(range(token_start, token_end))

        # プロフィール指標を検出 (profile category)
        profile_terms = medical_dict.get('profile', [])
        profile_indicators = extract_guideline_indicators(problem_text, profile_terms)

        # プロフィール指標のトークン位置を計算
        profile_indicator_positions = []
        for indicator in profile_indicators:
            char_start = indicator['char_start']
            char_end = indicator['char_end']

            # トークン位置を推定
            prefix_tokens = tokenizer.encode(problem_text[:char_start], add_special_tokens=False)
            token_start = len(prefix_tokens)
            token_end = len(tokenizer.encode(problem_text[:char_end], add_special_tokens=False))

            profile_indicator_positions.extend(range(token_start, token_end))

        # 推論キーワードを検出
        reasoning_keywords_list = medical_dict.get('reasoning_keywords', [])
        reasoning_keywords = extract_reasoning_keywords(problem_text, reasoning_keywords_list)

        # 推論キーワードのトークン位置を計算
        reasoning_keyword_positions = []
        for keyword in reasoning_keywords:
            char_start = keyword['char_start']
            char_end = keyword['char_end']

            # トークン位置を推定
            prefix_tokens = tokenizer.encode(problem_text[:char_start], add_special_tokens=False)
            token_start = len(prefix_tokens)
            token_end = len(tokenizer.encode(problem_text[:char_end], add_special_tokens=False))

            reasoning_keyword_positions.extend(range(token_start, token_end))

        # アノテーション結果を保存
        annotated_sample = {
            "index": processed_count - 1,
            "problem_id": row.get("problem_id", ""),
            "problem_text": problem_text,
            "year": row.get("year"),
            "tokens": tokens,
            "token_ids": token_ids,
            "medical_terms": found_terms,
            "term_count": len(found_terms),
            "categories_found": list(set(t["category"] for t in found_terms)),
            "medical_term_positions": sorted(list(set([pos for positions in term_positions.values() for pos in positions]))),
            "profile_indicator_positions": sorted(list(set(profile_indicator_positions))),
            "reasoning_keyword_positions": sorted(list(set(reasoning_keyword_positions))),
        }

        annotated_data.append(annotated_sample)

    # JSONLとして保存
    print(f"\nSaving annotated data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in annotated_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(annotated_data)} annotated samples")

    # 統計情報
    total_terms_found = sum(len(s["medical_terms"]) for s in annotated_data)
    avg_terms_per_sample = total_terms_found / len(annotated_data) if annotated_data else 0

    total_profile_positions = sum(len(s["profile_indicator_positions"]) for s in annotated_data)
    avg_profile_per_sample = total_profile_positions / len(annotated_data) if annotated_data else 0

    total_reasoning_positions = sum(len(s["reasoning_keyword_positions"]) for s in annotated_data)
    avg_reasoning_per_sample = total_reasoning_positions / len(annotated_data) if annotated_data else 0

    print(f"\nStatistics:")
    print(f"  Total medical terms found: {total_terms_found}")
    print(f"  Average terms per sample: {avg_terms_per_sample:.2f}")
    print(f"  Total profile indicator positions: {total_profile_positions}")
    print(f"  Average profile indicators per sample: {avg_profile_per_sample:.2f}")
    print(f"  Total reasoning keyword positions: {total_reasoning_positions}")
    print(f"  Average reasoning keywords per sample: {avg_reasoning_per_sample:.2f}")

    # カテゴリ別統計
    category_counts = {}
    for sample in annotated_data:
        for term in sample["medical_terms"]:
            cat = term["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    return annotated_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Term Annotator for igakuQA")
    parser.add_argument("--medical_dict", type=str, default=None, help="Medical dictionary JSON file")
    parser.add_argument("--model_path", type=str, default=None, help="Model path for tokenizer")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSONL file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # デフォルト値
    medical_dict = args.medical_dict or str(script_dir / "medical_terms_dictionary.json")
    model_path = args.model_path or "/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507"
    output_file = args.output_file or str(script_dir / "annotated_medical_data_full.jsonl")

    annotate_igakuqa_data(
        medical_dict_path=medical_dict,
        model_path=model_path,
        output_file=output_file,
        num_samples=args.num_samples
    )

    print("\n✓ Annotation completed successfully!")
