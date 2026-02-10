#!/bin/bash

# Auto-proceed to Phase 3 after Phase 2 completion
set -e

VENV_PYTHON="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/venv/bin/python3"
PHASE2_DIR="results_8gpu_parallel"
PHASE3_DIR="../Phase3_attention_analysis"

echo "============================================================"
echo "Phase 2 → Phase 3 Auto-Transition"
echo "============================================================"
echo "Started: $(date)"

# Step 1: Merge Phase 2 results
echo ""
echo "Step 1: Merging Phase 2 results..."
$VENV_PYTHON << 'EOF'
import json
import glob
import torch

# Collect all chunk results
all_results = []
attention_patterns = {}

for gpu_id in range(8):
    # Merge JSONL results
    chunk_file = f'results_8gpu_parallel/gpu_{gpu_id}/chunk_{gpu_id:02d}.jsonl'
    try:
        with open(chunk_file, 'r') as f:
            chunk_data = [json.loads(line) for line in f]
            all_results.extend(chunk_data)
            print(f"  GPU {gpu_id}: {len(chunk_data)} samples")
    except FileNotFoundError:
        print(f"  GPU {gpu_id}: NOT FOUND")
        continue
    
    # Merge attention patterns
    attn_file = f'results_8gpu_parallel/gpu_{gpu_id}/attention_patterns.pt'
    try:
        gpu_patterns = torch.load(attn_file, map_location='cpu')
        print(f"  GPU {gpu_id}: Attention patterns loaded")
        # Merge patterns (implementation depends on structure)
    except:
        print(f"  GPU {gpu_id}: Attention patterns NOT FOUND")

print(f"\nTotal samples: {len(all_results)}")

# Save merged results
with open('results_8gpu_parallel/merged_results.jsonl', 'w') as f:
    for item in all_results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Merged results saved!")
EOF

# Step 2: Run Phase 3 Head Classification
echo ""
echo "Step 2: Running Phase 3 Head Classification..."
cd $PHASE3_DIR

# Check if classification config exists
if [ ! -f "classification_criteria.yaml" ]; then
    echo "Creating default classification criteria..."
    cat > classification_criteria.yaml << 'YAML'
classification_criteria:
  medical_term_heads:
    attention_threshold: 0.1
    focus_positions: "medical_term_positions"
    
  guideline_indicator_heads:
    attention_threshold: 0.08
    focus_positions: "guideline_indicator_positions"
    
  reasoning_flow_heads:
    attention_threshold: 0.05
    focus_positions: "reasoning_keyword_positions"
YAML
fi

# Run head classifier
$VENV_PYTHON head_classifier.py \
    --attention_dir ../Phase2_path_patching/results_8gpu_parallel \
    --annotation_file ../Phase2_path_patching/results_8gpu_parallel/merged_results.jsonl \
    --criteria_config classification_criteria.yaml \
    --output_file head_classification_results_new.json \
    2>&1 | tee phase3_execution.log

echo ""
echo "============================================================"
echo "Phase 3 Complete!"
echo "Completed: $(date)"
echo "Results: $PHASE3_DIR/head_classification_results_new.json"
echo "============================================================"
