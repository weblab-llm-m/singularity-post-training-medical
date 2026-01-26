#!/usr/bin/env python3
"""
Merge 5 medical dictionaries and remove duplicates/errors
"""

import json
import re
from typing import List, Dict, Set

def is_valid_term(term: str) -> bool:
    """
    Validate if a term is valid (not error/noise)
    """
    if not term or not isinstance(term, str):
        return False

    # Remove whitespace
    term = term.strip()

    if len(term) == 0:
        return False

    # Reject if too short (likely noise)
    if len(term) < 2:
        return False

    # Reject if all non-alphanumeric/Japanese characters
    if not re.search(r'[a-zA-Z0-9ぁ-んァ-ヶー一-龯]', term):
        return False

    return True

def normalize_term(term: str) -> str:
    """
    Normalize term for deduplication
    """
    # Strip whitespace
    term = term.strip()

    # Remove extra spaces
    term = re.sub(r'\s+', ' ', term)

    return term

def merge_dictionaries(input_files: List[str], output_file: str):
    """
    Merge multiple dictionaries and remove duplicates
    """
    print("="*60)
    print("Merging Medical Dictionaries")
    print("="*60 + "\n")

    # Load all dictionaries
    all_dicts = []
    for file_path in input_files:
        print(f"Loading: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_dicts.append(data)
                print(f"  ✓ Loaded {sum(len(v) for v in data.values())} terms")
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}")

    print(f"\nTotal dictionaries loaded: {len(all_dicts)}\n")

    # Merge and deduplicate
    merged_dict = {
        "diseases": set(),
        "diagnostic_methods": set(),
        "biomarkers": set(),
        "guidelines": set(),
        "treatments": set(),
        "anatomical_terms": set(),
        "reasoning_keywords": set()
    }

    for category in merged_dict.keys():
        print(f"Processing category: {category}")

        for dict_data in all_dicts:
            if category in dict_data:
                for term in dict_data[category]:
                    if is_valid_term(term):
                        normalized = normalize_term(term)
                        merged_dict[category].add(normalized)

        print(f"  → {len(merged_dict[category])} unique valid terms")

    # Convert sets to sorted lists
    final_dict = {}
    for category, terms in merged_dict.items():
        final_dict[category] = sorted(list(terms))

    # Save merged dictionary
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Merged dictionary saved to: {output_file}")
    print(f"\nFinal statistics:")
    total_terms = 0
    for category, terms in final_dict.items():
        print(f"  - {category}: {len(terms)} terms")
        total_terms += len(terms)
    print(f"\nTotal unique terms: {total_terms}")

    return final_dict

if __name__ == "__main__":
    input_files = [
        "medical_terms_dictionary_qwen_generated_1.json",
        "medical_terms_dictionary_qwen_generated_2.json",
        "medical_terms_dictionary_qwen_generated_3.json",
        "medical_terms_dictionary_qwen_generated_4.json",
        "medical_terms_dictionary_qwen_generated_5.json"
    ]

    output_file = "medical_terms_dictionary_qwen_generated.json"

    merge_dictionaries(input_files, output_file)
