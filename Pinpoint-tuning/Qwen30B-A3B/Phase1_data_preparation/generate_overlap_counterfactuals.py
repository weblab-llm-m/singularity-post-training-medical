#!/usr/bin/env python3
"""
Overlap Counterfactual Data Generator

医療用語置換可能 かつ 推論表現ありの4,393件に対して、
2種類の反実データを生成:
  1. counterfactual_overlap_medical.jsonl  - 医療用語のみ置換
  2. counterfactual_overlap_reasoning.jsonl - 推論表現のみ置換

既存の counterfactual_generator.py / counterfactual_reasoning_generator.py と
同一のロジックを使用。
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# === 医療用語置換 (counterfactual_generator.py と同一ロジック) ===

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


# === 推論表現置換 (counterfactual_reasoning_generator.py と同一ロジック) ===

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
    parser = argparse.ArgumentParser(description="Overlap Counterfactual Generator")
    parser.add_argument("--replacement_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    random.seed(args.seed)

    print("=" * 60)
    print("Overlap Counterfactual Generator")
    print("Target: samples with BOTH medical terms AND reasoning keywords")
    print("=" * 60)

    # --- データ読み込み ---
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
    print(f"Replacement ratio: {args.replacement_ratio}\n")

    # --- 4,393件の重複サンプルを特定 ---
    overlap_samples = []
    for sample in annotated:
        terms = sample.get("medical_terms", [])
        has_medical = any(t["term"] in replacement_mapping for t in terms)
        has_reasoning = any(t["category"] == "reasoning_keywords" for t in terms)
        if has_medical and has_reasoning:
            overlap_samples.append(sample)

    print(f"Overlap samples (medical + reasoning): {len(overlap_samples)}")

    # --- 医療用語のみ置換 ---
    print(f"\n--- Generating medical-only counterfactuals ---")
    medical_results = []
    for sample in overlap_samples:
        cf = generate_medical_counterfactual(
            sample, replacement_mapping, args.replacement_ratio
        )
        if cf:
            medical_results.append(cf)

    med_out = script_dir / "counterfactual_overlap_medical.jsonl"
    with open(med_out, "w", encoding="utf-8") as f:
        for r in medical_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_med_repl = sum(r["num_replacements"] for r in medical_results)
    print(f"  Output: {len(medical_results)} samples")
    print(f"  Total replacements: {total_med_repl}")
    print(f"  Avg replacements/sample: {total_med_repl / len(medical_results):.2f}")
    print(f"  Saved to: {med_out}")

    # --- 推論表現のみ置換 ---
    print(f"\n--- Generating reasoning-only counterfactuals ---")
    reasoning_results = []
    for sample in overlap_samples:
        cf = generate_reasoning_counterfactual(
            sample, reasoning_keywords, args.replacement_ratio
        )
        if cf:
            reasoning_results.append(cf)

    rea_out = script_dir / "counterfactual_overlap_reasoning.jsonl"
    with open(rea_out, "w", encoding="utf-8") as f:
        for r in reasoning_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_rea_repl = sum(r["num_replacements"] for r in reasoning_results)
    print(f"  Output: {len(reasoning_results)} samples")
    print(f"  Total replacements: {total_rea_repl}")
    print(f"  Avg replacements/sample: {total_rea_repl / len(reasoning_results):.2f}")
    print(f"  Saved to: {rea_out}")

    # --- サマリ ---
    # 両方生成できたサンプル (path patching で使う共通集合)
    med_indices = {r["original_index"] for r in medical_results}
    rea_indices = {r["original_index"] for r in reasoning_results}
    both_indices = med_indices & rea_indices
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Overlap candidates:        {len(overlap_samples)}")
    print(f"  Medical-only generated:    {len(medical_results)}")
    print(f"  Reasoning-only generated:  {len(reasoning_results)}")
    print(f"  Both successfully generated: {len(both_indices)}")
    print(f"{'=' * 60}")

    # 例を表示
    print(f"\nExamples (first 2):")
    for i, sample in enumerate(overlap_samples[:2]):
        idx = sample["index"]
        med_cf = next((r for r in medical_results if r["original_index"] == idx), None)
        rea_cf = next((r for r in reasoning_results if r["original_index"] == idx), None)
        if med_cf and rea_cf:
            print(f"\n  [{i+1}] {sample.get('problem_id', '')}")
            print(f"    Original:    {sample['problem_text'][:80]}...")
            print(f"    Med-only:    {med_cf['counterfactual_text'][:80]}...")
            for rep in med_cf["replacements"]:
                print(f"      '{rep['original']}' -> '{rep['replacement']}'")
            print(f"    Reason-only: {rea_cf['counterfactual_text'][:80]}...")
            for rep in rea_cf["replacements"]:
                print(f"      '{rep['original']}' -> '{rep['replacement']}'")


if __name__ == "__main__":
    main()
