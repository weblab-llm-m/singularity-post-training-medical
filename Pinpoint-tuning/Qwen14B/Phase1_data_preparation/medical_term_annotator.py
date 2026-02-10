#!/usr/bin/env python3
"""
Medical Term Annotator
医療用語を自動抽出・アノテーション（Qwen辞書使用、強化版）
"""

import json
import argparse
import pandas as pd
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer

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




def annotate_medical_data(
    input_file: str,
    output_file: str,
    medical_dict_path: str,
    model_path: str,
    num_samples: int = None
):
    """医療データに用語アノテーションを付加"""

    print("="*60)
    print("Medical Term Annotator (Enhanced)")
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

    # データをロード
    print(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"✓ Loaded {len(df)} rows")

    if num_samples and num_samples < len(df):
        df = df.head(num_samples)
        print(f"  Using first {num_samples} samples\n")

    # アノテーション処理
    print(f"Processing {len(df)} samples...")
    annotated_data = []

    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(df)} samples...")

        # extra_infoから質問と回答を取得
        extra_info = row.get('extra_info', {})
        if not isinstance(extra_info, dict):
            continue

        question = extra_info.get('question', '')
        answer = extra_info.get('answer', '')

        # <think>セクションを分離
        think_content, answer_without_think, has_think = split_think_sections(answer)

        # 質問と回答を結合
        combined_text = f"{question}\n{answer}"

        # トークン化
        token_ids = tokenizer.encode(combined_text, add_special_tokens=False)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # 医療用語を検出
        found_terms = []
        term_positions = {}  # {category: [token_indices]}

        for category, terms in medical_dict.items():
            term_positions[category] = []
            for term in terms:
                if term in combined_text:
                    # 用語の文字位置を取得
                    start_idx = combined_text.find(term)
                    if start_idx != -1:
                        end_idx = start_idx + len(term)

                        # トークン位置を推定
                        prefix_tokens = tokenizer.encode(combined_text[:start_idx], add_special_tokens=False)
                        token_start = len(prefix_tokens)
                        token_end = len(tokenizer.encode(combined_text[:end_idx], add_special_tokens=False))

                        found_terms.append({
                            "term": term,
                            "category": category,
                            "char_start": start_idx,
                            "char_end": end_idx,
                            "token_start": token_start,
                            "token_end": token_end
                        })

                        term_positions[category].extend(range(token_start, token_end))

        # ガイドライン指標を検出
        guideline_terms = medical_dict.get('guidelines', [])
        guideline_indicators = extract_guideline_indicators(combined_text, guideline_terms)

        # ガイドライン指標のトークン位置を計算
        guideline_indicator_positions = []
        for indicator in guideline_indicators:
            char_start = indicator['char_start']
            char_end = indicator['char_end']

            # トークン位置を推定
            prefix_tokens = tokenizer.encode(combined_text[:char_start], add_special_tokens=False)
            token_start = len(prefix_tokens)
            token_end = len(tokenizer.encode(combined_text[:char_end], add_special_tokens=False))

            guideline_indicator_positions.extend(range(token_start, token_end))

        # 推論キーワードを検出
        reasoning_keywords_list = medical_dict.get('reasoning_keywords', [])
        reasoning_keywords = extract_reasoning_keywords(combined_text, reasoning_keywords_list)

        # 推論キーワードのトークン位置を計算
        reasoning_keyword_positions = []
        for keyword in reasoning_keywords:
            char_start = keyword['char_start']
            char_end = keyword['char_end']

            # トークン位置を推定
            prefix_tokens = tokenizer.encode(combined_text[:char_start], add_special_tokens=False)
            token_start = len(prefix_tokens)
            token_end = len(tokenizer.encode(combined_text[:char_end], add_special_tokens=False))

            reasoning_keyword_positions.extend(range(token_start, token_end))

        # アノテーション結果を保存
        annotated_sample = {
            "index": int(idx),
            "question": question,
            "answer": answer,
            "tokens": tokens,
            "token_ids": token_ids,
            "medical_terms": found_terms,
            "term_count": len(found_terms),
            "categories_found": list(set(t["category"] for t in found_terms)),
            "medical_term_positions": sorted(list(set([pos for positions in term_positions.values() for pos in positions]))),
            "guideline_indicator_positions": sorted(list(set(guideline_indicator_positions))),
            "reasoning_keyword_positions": sorted(list(set(reasoning_keyword_positions))),
            "has_think_section": has_think,
            "think_content": think_content if has_think else ""
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
    samples_with_think = sum(1 for s in annotated_data if s["has_think_section"])

    total_guideline_positions = sum(len(s["guideline_indicator_positions"]) for s in annotated_data)
    avg_guideline_per_sample = total_guideline_positions / len(annotated_data) if annotated_data else 0

    total_reasoning_positions = sum(len(s["reasoning_keyword_positions"]) for s in annotated_data)
    avg_reasoning_per_sample = total_reasoning_positions / len(annotated_data) if annotated_data else 0

    print(f"\nStatistics:")
    print(f"  Total medical terms found: {total_terms_found}")
    print(f"  Average terms per sample: {avg_terms_per_sample:.2f}")
    print(f"  Samples with <think> section: {samples_with_think} ({samples_with_think/len(annotated_data)*100:.1f}%)")
    print(f"  Total guideline indicator positions: {total_guideline_positions}")
    print(f"  Average guideline indicators per sample: {avg_guideline_per_sample:.2f}")
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
    parser = argparse.ArgumentParser(description="Medical Term Annotator")
    parser.add_argument("--input_file", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--medical_dict", type=str, required=True, help="Medical dictionary JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Model path for tokenizer")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")

    args = parser.parse_args()

    annotate_medical_data(
        input_file=args.input_file,
        output_file=args.output_file,
        medical_dict_path=args.medical_dict,
        model_path=args.model_path,
        num_samples=args.num_samples
    )

    print("\n✓ Annotation completed successfully!")
