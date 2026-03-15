#!/usr/bin/env python3
"""
Overlap Counterfactual Data Generator (dictionary_260212版)

拡張辞書を使い、医療用語置換可能 かつ 推論表現ありのサンプルに対して
2種類の反実データを生成:
  1. counterfactual_overlap_medical.jsonl  - 医療用語のみ置換
  2. counterfactual_overlap_reasoning.jsonl - 推論表現のみ置換

IDの重複を避けるため、annotated dataのindexをそのまま使用。
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def generate_medical_counterfactual(
    sample: Dict,
    replacement_mapping: Dict[str, str],
    replacement_ratio: float = 0.5,
) -> Optional[Dict]:
    medical_terms = sample.get("medical_terms", [])
    if not medical_terms:
        return None
    replaceable_terms = [t for t in medical_terms if t["term"] in replacement_mapping]
    if not replaceable_terms:
        return None
    num_to_replace = max(1, int(len(replaceable_terms) * replacement_ratio))
    terms_to_replace = random.sample(
        replaceable_terms, min(num_to_replace, len(replaceable_terms))
    )
    problem_text = sample.get("problem_text", "")
    if not problem_text:
        return None
    modified_text = problem_text
    replacements = []
    for term_info in terms_to_replace:
        original_term = term_info["term"]
        replacement_term = replacement_mapping[original_term]
        if original_term in modified_text:
            modified_text = modified_text.replace(original_term, replacement_term, 1)
            replacements.append({
                "original": original_term,
                "replacement": replacement_term,
                "category": term_info["category"],
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
        "strategy": "strategy2",
    }


def generate_reasoning_counterfactual(
    sample: Dict,
    reasoning_keywords: List[str],
    replacement_ratio: float = 0.5,
) -> Optional[Dict]:
    terms = sample.get("medical_terms", [])
    reasoning_terms = [t for t in terms if t["category"] == "reasoning_keywords"]
    if not reasoning_terms:
        return None
    num_to_replace = max(1, int(len(reasoning_terms) * replacement_ratio))
    terms_to_replace = random.sample(
        reasoning_terms, min(num_to_replace, len(reasoning_terms))
    )
    problem_text = sample.get("problem_text", "")
    if not problem_text:
        return None
    modified_text = problem_text
    replacements = []
    for term_info in terms_to_replace:
        original_term = term_info["term"]
        candidates = [kw for kw in reasoning_keywords if kw != original_term]
        if not candidates:
            continue
        orig_len = len(original_term)
        candidates_scored = sorted(candidates, key=lambda c: abs(len(c) - orig_len))
        top_n = max(3, len(candidates_scored) // 5)
        replacement_term = random.choice(candidates_scored[:top_n])
        if original_term in modified_text:
            modified_text = modified_text.replace(original_term, replacement_term, 1)
            replacements.append({
                "original": original_term,
                "replacement": replacement_term,
                "category": "reasoning_keywords",
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
        "strategy": "reasoning_swap",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    random.seed(args.seed)

    print("=" * 60)
    print("Overlap Counterfactual Generator (dictionary_260212)")
    print("=" * 60)

    with open(script_dir / "annotated_medical_data_full.jsonl") as f:
        annotated = [json.loads(l) for l in f]
    print(f"Annotated data: {len(annotated)} samples")

    with open(script_dir / "replacement_mapping.json") as f:
        replacement_mapping = json.load(f)
    print(f"Replacement mapping: {len(replacement_mapping)} terms")

    with open(script_dir / "medical_terms_dictionary.json") as f:
        med_dict = json.load(f)
    reasoning_keywords = med_dict.get("reasoning_keywords", [])
    print(f"Reasoning keywords: {len(reasoning_keywords)}")

    # IDの重複チェック
    indices = [s["index"] for s in annotated]
    assert len(indices) == len(set(indices)), "ERROR: Duplicate indices in annotated data!"
    print(f"ID uniqueness check: OK (all {len(indices)} unique)")

    # Overlap samples
    overlap_samples = []
    for sample in annotated:
        terms = sample.get("medical_terms", [])
        has_medical = any(t["term"] in replacement_mapping for t in terms)
        has_reasoning = any(t["category"] == "reasoning_keywords" for t in terms)
        if has_medical and has_reasoning:
            overlap_samples.append(sample)

    print(f"Overlap samples: {len(overlap_samples)}")

    # Medical-only
    print(f"\n--- Medical-only counterfactuals ---")
    medical_results = []
    for sample in overlap_samples:
        cf = generate_medical_counterfactual(sample, replacement_mapping, args.replacement_ratio)
        if cf:
            medical_results.append(cf)

    med_out = script_dir / "counterfactual_overlap_medical.jsonl"
    with open(med_out, "w", encoding="utf-8") as f:
        for r in medical_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Output: {len(medical_results)} samples -> {med_out.name}")

    # Reasoning-only
    print(f"\n--- Reasoning-only counterfactuals ---")
    reasoning_results = []
    for sample in overlap_samples:
        cf = generate_reasoning_counterfactual(sample, reasoning_keywords, args.replacement_ratio)
        if cf:
            reasoning_results.append(cf)

    rea_out = script_dir / "counterfactual_overlap_reasoning.jsonl"
    with open(rea_out, "w", encoding="utf-8") as f:
        for r in reasoning_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Output: {len(reasoning_results)} samples -> {rea_out.name}")

    # Verify no duplicate IDs
    med_ids = [r["original_index"] for r in medical_results]
    rea_ids = [r["original_index"] for r in reasoning_results]
    assert len(med_ids) == len(set(med_ids)), "Duplicate IDs in medical!"
    assert len(rea_ids) == len(set(rea_ids)), "Duplicate IDs in reasoning!"

    both = set(med_ids) & set(rea_ids)
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Medical-only:  {len(medical_results)} (unique IDs: {len(set(med_ids))})")
    print(f"  Reasoning-only: {len(reasoning_results)} (unique IDs: {len(set(rea_ids))})")
    print(f"  Both generated: {len(both)}")
    print(f"  ID duplicates:  None")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
