#!/bin/bash
# Phase 1-2-3 フルパイプラインテスト

set -e

echo "=========================================="
echo "Full Pipeline Test: Phase 1-2-3"
echo "=========================================="

cd /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching

echo ""
echo "Step 1: Phase 1 - Data Preparation"
echo "=========================================="
bash scripts/run_phase1.sh

echo ""
echo "Step 2: Phase 2 - Path Patching"
echo "=========================================="
bash scripts/run_phase2.sh

echo ""
echo "Step 3: Phase 3 - Head Classification"
echo "=========================================="
bash scripts/run_phase3.sh

echo ""
echo "=========================================="
echo "Full Pipeline Completed Successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  Phase 1:"
echo "    - Phase1_data_preparation/test_annotated.jsonl"
echo "    - Phase1_data_preparation/test_counterfactual.jsonl"
echo "    - Phase1_data_preparation/test_path_patching_data.jsonl"
echo ""
echo "  Phase 2:"
echo "    - Phase2_path_patching/results/results.pt"
echo "    - Phase2_path_patching/results/attention_patterns.pt"
echo "    - Phase2_path_patching/results/head_map.html"
echo ""
echo "  Phase 3:"
echo "    - Phase3_attention_analysis/head_classification_results.json"
echo "    - Phase3_attention_analysis/medical_specific_patterns.json"
echo ""
