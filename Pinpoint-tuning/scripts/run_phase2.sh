#!/bin/bash
# Phase 2: Path Patching実行

set -e

echo "==================================================="
echo "Phase 2: Path Patching Execution"
echo "==================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"

cd $BASE_DIR

echo ""
echo "Configuration:"
echo "---------------------------------------------------"
echo "Model: ${MODEL_PATH}"
echo "Data: Phase1_data_preparation/medical_path_patching_enhanced.jsonl"
echo "Batch size: 1"
echo "Sample num: ALL (processing full dataset)"
echo "Extract attention: true"
echo ""

echo "Running Path Patching with Attention Extraction on Full Dataset"
echo "---------------------------------------------------"

# Pythonパスを設定
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python3 Phase2_path_patching/path_patching_medical.py \
    --model_path ${MODEL_PATH} \
    --data_path Phase1_data_preparation/medical_path_patching_enhanced.jsonl \
    --batch_size 1 \
    --extract_attention true \
    --output_dir Phase2_path_patching/results/

echo ""
echo "==================================================="
echo "Phase 2 Completed!"
echo "==================================================="
echo "Output files:"
echo "  - Path Patching results: Phase2_path_patching/results/results.pt"
echo "  - Attention patterns: Phase2_path_patching/results/attention_patterns.pt"
echo "  - Heatmap visualization: Phase2_path_patching/results/head_map.html"
echo ""
echo "Next step: Run Phase 3 to classify attention heads"
echo "  bash scripts/run_phase3.sh"
echo ""
