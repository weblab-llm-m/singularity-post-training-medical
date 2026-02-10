#!/bin/bash
# Phase 3: 注意パターン解析とヘッド分類

set -e

echo "==================================================="
echo "Phase 3: Attention Analysis & Head Classification"
echo "==================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"

cd $BASE_DIR

echo ""
echo "Step 1: Head Classification"
echo "---------------------------------------------------"
python3 Phase3_attention_analysis/head_classifier.py \
    --attention_patterns Phase2_path_patching/results/attention_patterns.pt \
    --annotation_data Phase1_data_preparation/annotated_medical_data.jsonl \
    --criteria_config configs/head_classification_params.yaml \
    --output_path Phase3_attention_analysis/head_classification_results.json \
    --num_layers 40 \
    --num_heads 40

echo ""
echo "Step 2: Medical Pattern Detection"
echo "---------------------------------------------------"
python3 Phase3_attention_analysis/medical_pattern_detector.py \
    --attention_patterns Phase2_path_patching/results/attention_patterns.pt \
    --medical_data Phase1_data_preparation/annotated_medical_data.jsonl \
    --output_path Phase3_attention_analysis/medical_specific_patterns.json

echo ""
echo "==================================================="
echo "Phase 3 Completed!"
echo "==================================================="
echo "Output files:"
echo "  - Classification: Phase3_attention_analysis/head_classification_results.json"
echo "  - Medical patterns: Phase3_attention_analysis/medical_specific_patterns.json"
echo ""
