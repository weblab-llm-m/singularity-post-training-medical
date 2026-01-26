#!/usr/bin/env python3
"""
Filter heads with positive impact from Phase3 results
Only include medical_term, guideline, and reasoning_flow heads with impact > 0
"""

import json
from pathlib import Path

# Input files
phase3_file = "/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase3_attention_analysis/head_classification_results_438samples.json"
output_file = "/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v5_pinpoint_tuning_mk2_8gpu/trainable_heads_positive_impact.json"

print("="*80)
print("Filtering Positive Impact Heads")
print("="*80)
print()

# Load Phase3 results
with open(phase3_file, 'r') as f:
    data = json.load(f)

medical_heads = data['medical_term_heads']
guideline_heads = data['guideline_heads']
reasoning_heads = data['reasoning_flow_heads']
all_impacts = data['all_head_impacts']

print(f"Medical term heads: {len(medical_heads)}")
print(f"Guideline heads: {len(guideline_heads)}")
print(f"Reasoning flow heads: {len(reasoning_heads)}")
print(f"Total classified heads: {len(medical_heads) + len(guideline_heads) + len(reasoning_heads)}")
print(f"All head impacts: {len(all_impacts)}")
print()

# Function to get impact for a head
def get_impact(layer, head):
    key = f"L{layer}H{head}"
    return all_impacts.get(key, 0)

# Process each category
positive_heads = []

# Medical term heads
print("Processing medical term heads...")
for layer, head in medical_heads:
    impact = get_impact(layer, head)
    if impact > 0:  # Positive impact only
        positive_heads.append({
            "layer": layer,
            "head": head,
            "type": "medical_term",
            "impact": impact,
            "priority": "high" if impact > 0.4 else "medium" if impact > 0.2 else "low"
        })

medical_positive = len([h for h in positive_heads if h['type'] == 'medical_term'])
print(f"  Positive impact: {medical_positive}/{len(medical_heads)}")

# Guideline heads
print("Processing guideline heads...")
guideline_start = len(positive_heads)
for layer, head in guideline_heads:
    impact = get_impact(layer, head)
    if impact > 0:  # Positive impact only
        positive_heads.append({
            "layer": layer,
            "head": head,
            "type": "guideline",
            "impact": impact,
            "priority": "high" if impact > 0.4 else "medium" if impact > 0.2 else "low"
        })

guideline_positive = len([h for h in positive_heads if h['type'] == 'guideline'])
print(f"  Positive impact: {guideline_positive}/{len(guideline_heads)}")

# Reasoning flow heads
print("Processing reasoning flow heads...")
reasoning_start = len(positive_heads)
for layer, head in reasoning_heads:
    impact = get_impact(layer, head)
    if impact > 0:  # Positive impact only
        positive_heads.append({
            "layer": layer,
            "head": head,
            "type": "reasoning_flow",
            "impact": impact,
            "priority": "high" if impact > 0.4 else "medium" if impact > 0.2 else "low"
        })

reasoning_positive = len([h for h in positive_heads if h['type'] == 'reasoning_flow'])
print(f"  Positive impact: {reasoning_positive}/{len(reasoning_heads)}")

print()

# Sort by impact (descending)
positive_heads.sort(key=lambda x: x['impact'], reverse=True)

# Statistics
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Total positive impact heads: {len(positive_heads)}")
print(f"  Medical term:   {medical_positive}")
print(f"  Guideline:      {guideline_positive}")
print(f"  Reasoning flow: {reasoning_positive}")
print()

if positive_heads:
    impacts = [h['impact'] for h in positive_heads]
    print(f"Impact range: {min(impacts):.6f} to {max(impacts):.6f}")
    print(f"Average impact: {sum(impacts)/len(impacts):.6f}")
    print()

# Priority breakdown
priority_counts = {"high": 0, "medium": 0, "low": 0}
for h in positive_heads:
    priority_counts[h['priority']] += 1

print("Priority breakdown:")
print(f"  High (>0.4):   {priority_counts['high']}")
print(f"  Medium (>0.2): {priority_counts['medium']}")
print(f"  Low (≤0.2):    {priority_counts['low']}")
print()

# Show top 20 heads
print("="*80)
print("TOP 20 POSITIVE IMPACT HEADS")
print("="*80)
print(f"{'Layer':<6} {'Head':<6} {'Impact':<12} {'Type':<20} {'Priority':<10}")
print("-"*80)
for h in positive_heads[:20]:
    print(f"{h['layer']:<6} {h['head']:<6} {h['impact']:<12.6f} {h['type']:<20} {h['priority']:<10}")
print()

# Show bottom 20 positive heads
if len(positive_heads) > 20:
    print("="*80)
    print("BOTTOM 20 POSITIVE IMPACT HEADS")
    print("="*80)
    print(f"{'Layer':<6} {'Head':<6} {'Impact':<12} {'Type':<20} {'Priority':<10}")
    print("-"*80)
    for h in positive_heads[-20:]:
        print(f"{h['layer']:<6} {h['head']:<6} {h['impact']:<12.6f} {h['type']:<20} {h['priority']:<10}")
    print()

# Save to file
with open(output_file, 'w') as f:
    json.dump(positive_heads, f, indent=2)

print("="*80)
print("OUTPUT")
print("="*80)
print(f"Saved {len(positive_heads)} positive impact heads to:")
print(f"  {output_file}")
print("="*80)
