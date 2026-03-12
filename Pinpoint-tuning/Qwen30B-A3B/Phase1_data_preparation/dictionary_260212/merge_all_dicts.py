#!/usr/bin/env python3
"""
Merge old 5 dictionaries + new 5 dictionaries → expanded medical_terms_dictionary.json
"""

import json
import re
from pathlib import Path


def is_valid_term(term: str) -> bool:
    if not term or not isinstance(term, str):
        return False
    term = term.strip()
    if len(term) < 2:
        return False
    if not re.search(r'[a-zA-Z0-9ぁ-んァ-ヶー一-龯]', term):
        return False
    return True


def normalize_term(term: str) -> str:
    return re.sub(r'\s+', ' ', term.strip())


def main():
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent

    # Old dictionaries (5 seeds)
    old_files = [
        parent_dir / f"medical_terms_dictionary_qwen_generated_{s}.json"
        for s in [42, 123, 456, 789, 1024]
    ]

    # New dictionaries (5 seeds)
    new_files = [
        script_dir / f"dict_seed{s}.json"
        for s in [2000, 3500, 4000, 4500, 5000]
    ]

    all_files = old_files + new_files

    print("=" * 60)
    print("Merging All Dictionaries (old 5 + new 5)")
    print("=" * 60)

    categories = [
        "diseases", "diagnostic_methods", "biomarkers",
        "profile", "treatments", "anatomical_terms", "reasoning_keywords"
    ]
    merged = {cat: set() for cat in categories}
    loaded_count = 0

    for fp in all_files:
        tag = "OLD" if fp.parent == parent_dir else "NEW"
        if not fp.exists():
            print(f"  [{tag}] SKIP (not found): {fp.name}")
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total = sum(len(v) for v in data.values() if isinstance(v, list))
        print(f"  [{tag}] {fp.name}: {total} terms")
        loaded_count += 1

        for cat in categories:
            if cat in data and isinstance(data[cat], list):
                for term in data[cat]:
                    if is_valid_term(term):
                        merged[cat].add(normalize_term(term))

    print(f"\nLoaded: {loaded_count} dictionaries")

    # Convert to sorted lists
    final_dict = {cat: sorted(list(terms)) for cat, terms in merged.items()}

    # Save
    output_path = script_dir / "medical_terms_dictionary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"\nFinal statistics:")
    total = 0
    for cat, terms in final_dict.items():
        print(f"  {cat}: {len(terms)}")
        total += len(terms)
    print(f"  TOTAL: {total}")

    # Compare with old
    old_dict_path = parent_dir / "medical_terms_dictionary.json"
    if old_dict_path.exists():
        with open(old_dict_path) as f:
            old = json.load(f)
        old_total = sum(len(v) for v in old.values())
        print(f"\n  Old dictionary: {old_total} terms")
        print(f"  New dictionary: {total} terms")
        print(f"  Increase: +{total - old_total} ({(total/old_total - 1)*100:.1f}%)")


if __name__ == "__main__":
    main()
