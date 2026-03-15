#!/usr/bin/env python3
"""
Analyze original head classification results to identify heads with positive impact
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_head_results(json_path):
    """Analyze head classification results and show impact distribution"""

    print("="*80)
    print("Head Classification Results Analysis")
    print("="*80)
    print()

    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract heads information
    if isinstance(data, dict):
        if 'heads' in data:
            heads = data['heads']
        elif 'medical_term_heads' in data or 'guideline_heads' in data or 'reasoning_flow_heads' in data:
            # Combine different types
            heads = []
            for key in ['medical_term_heads', 'guideline_heads', 'reasoning_flow_heads']:
                if key in data:
                    heads.extend(data[key])
        else:
            print("Error: Unexpected data structure")
            return
    elif isinstance(data, list):
        heads = data
    else:
        print("Error: Unexpected data type")
        return

    print(f"Total heads in file: {len(heads)}")
    print()

    # Analyze by impact sign
    positive_heads = []
    negative_heads = []
    zero_heads = []

    for head in heads:
        impact = head.get('impact', 0)
        if impact > 0:
            positive_heads.append(head)
        elif impact < 0:
            negative_heads.append(head)
        else:
            zero_heads.append(head)

    # Summary statistics
    print("="*80)
    print("IMPACT DISTRIBUTION")
    print("="*80)
    print(f"Positive impact (>0):  {len(positive_heads):4d} heads ({len(positive_heads)/len(heads)*100:.1f}%)")
    print(f"Negative impact (<0):  {len(negative_heads):4d} heads ({len(negative_heads)/len(heads)*100:.1f}%)")
    print(f"Zero impact (=0):      {len(zero_heads):4d} heads ({len(zero_heads)/len(heads)*100:.1f}%)")
    print(f"Total:                 {len(heads):4d} heads")
    print()

    # Impact range
    if heads:
        impacts = [h['impact'] for h in heads]
        print("="*80)
        print("IMPACT STATISTICS")
        print("="*80)
        print(f"Minimum impact: {min(impacts):8.6f}")
        print(f"Maximum impact: {max(impacts):8.6f}")
        print(f"Average impact: {sum(impacts)/len(impacts):8.6f}")
        print()

    # Show negative impact heads
    if negative_heads:
        print("="*80)
        print(f"NEGATIVE IMPACT HEADS ({len(negative_heads)} heads)")
        print("="*80)
        negative_sorted = sorted(negative_heads, key=lambda x: x['impact'])
        print(f"{'Layer':<6} {'Head':<6} {'Impact':<12} {'Type':<20} {'Priority':<10}")
        print("-"*80)
        for h in negative_sorted:
            print(f"{h['layer']:<6} {h['head']:<6} {h['impact']:<12.6f} {h.get('type', 'N/A'):<20} {h.get('priority', 'N/A'):<10}")
        print()

    # Show lowest positive impact heads
    if positive_heads:
        print("="*80)
        print("LOWEST POSITIVE IMPACT HEADS (Bottom 20)")
        print("="*80)
        positive_sorted = sorted(positive_heads, key=lambda x: x['impact'])
        print(f"{'Layer':<6} {'Head':<6} {'Impact':<12} {'Type':<20} {'Priority':<10}")
        print("-"*80)
        for h in positive_sorted[:20]:
            print(f"{h['layer']:<6} {h['head']:<6} {h['impact']:<12.6f} {h.get('type', 'N/A'):<20} {h.get('priority', 'N/A'):<10}")
        print()

    # Show highest positive impact heads
    if positive_heads:
        print("="*80)
        print("HIGHEST POSITIVE IMPACT HEADS (Top 20)")
        print("="*80)
        positive_sorted = sorted(positive_heads, key=lambda x: x['impact'], reverse=True)
        print(f"{'Layer':<6} {'Head':<6} {'Impact':<12} {'Type':<20} {'Priority':<10}")
        print("-"*80)
        for h in positive_sorted[:20]:
            print(f"{h['layer']:<6} {h['head']:<6} {h['impact']:<12.6f} {h.get('type', 'N/A'):<20} {h.get('priority', 'N/A'):<10}")
        print()

    # Type breakdown
    print("="*80)
    print("BREAKDOWN BY TYPE")
    print("="*80)
    type_stats = defaultdict(lambda: {'total': 0, 'positive': 0, 'negative': 0, 'zero': 0})
    for head in heads:
        head_type = head.get('type', 'unknown')
        impact = head.get('impact', 0)
        type_stats[head_type]['total'] += 1
        if impact > 0:
            type_stats[head_type]['positive'] += 1
        elif impact < 0:
            type_stats[head_type]['negative'] += 1
        else:
            type_stats[head_type]['zero'] += 1

    print(f"{'Type':<20} {'Total':<8} {'Positive':<10} {'Negative':<10} {'Zero':<8}")
    print("-"*80)
    for head_type in sorted(type_stats.keys()):
        stats = type_stats[head_type]
        print(f"{head_type:<20} {stats['total']:<8} {stats['positive']:<10} {stats['negative']:<10} {stats['zero']:<8}")
    print()

    # Generate filtered list for positive heads only
    output_path = Path(json_path).parent / "trainable_heads_positive_only.json"
    with open(output_path, 'w') as f:
        json.dump(positive_heads, f, indent=2)

    print("="*80)
    print("OUTPUT")
    print("="*80)
    print(f"Filtered positive heads saved to:")
    print(f"  {output_path}")
    print(f"  Total heads: {len(positive_heads)}")
    print()

    return {
        'total': len(heads),
        'positive': len(positive_heads),
        'negative': len(negative_heads),
        'zero': len(zero_heads)
    }

if __name__ == "__main__":
    original_file = "/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase3_attention_analysis/head_classification_results_438samples.json"

    if not Path(original_file).exists():
        print(f"Error: File not found: {original_file}")
        sys.exit(1)

    result = analyze_head_results(original_file)

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Analyzed {result['total']} heads")
    print(f"✓ Found {result['positive']} heads with positive impact")
    print(f"✓ Found {result['negative']} heads with negative impact")
    print(f"✓ Found {result['zero']} heads with zero impact")
    print("="*80)
