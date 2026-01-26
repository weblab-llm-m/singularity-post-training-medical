#!/bin/bash
# Medical Path Patching Full Pipeline
# Phase 1-2-3-4を連続実行（Phase 5はオプション）

set -e

echo "==================================================="
echo "Medical Path Patching Full Pipeline"
echo "==================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"

cd $BASE_DIR

# タイムスタンプ
START_TIME=$(date +%s)

echo ""
echo "Phase 1: Data Preparation"
echo "==================================================="
bash scripts/run_phase1.sh

echo ""
echo "Phase 2: Path Patching Execution"
echo "==================================================="
bash scripts/run_phase2.sh

echo ""
echo "Phase 3: Attention Analysis & Head Classification"
echo "==================================================="
bash scripts/run_phase3.sh

echo ""
echo "Phase 4: Visualization & Reporting"
echo "==================================================="

echo "Step 1: Generate Heatmaps"
echo "---------------------------------------------------"
python3 Phase4_visualization/heatmap_generator.py \
    --classification_results Phase3_attention_analysis/head_classification_results.json \
    --patching_results Phase2_path_patching/results/results.pt \
    --output_dir Phase4_visualization/

echo ""
echo "Step 2: Statistical Analysis"
echo "---------------------------------------------------"
python3 Phase4_visualization/statistical_analyzer.py \
    --classification_results Phase3_attention_analysis/head_classification_results.json \
    --patching_results Phase2_path_patching/results/results.pt \
    --output_path Phase4_visualization/statistics.json

echo ""
echo "Step 3: Report Generation"
echo "---------------------------------------------------"
python3 Phase4_visualization/report_generator.py \
    --classification_results Phase3_attention_analysis/head_classification_results.json \
    --statistical_report Phase4_visualization/statistics.json \
    --output_path Phase4_visualization/medical_head_analysis_report.md

echo ""
echo "Phase 4 Completed!"
echo "Output files:"
echo "  - Heatmap: Phase4_visualization/classification_heatmap.png"
echo "  - Statistics: Phase4_visualization/statistics.json"
echo "  - Report: Phase4_visualization/medical_head_analysis_report.md"

# Phase 5（オプション）
echo ""
echo "Phase 5: Trainable Head Selection (Optional)"
echo "==================================================="
read -p "Do you want to run Phase 5 (Trainable Head Selection)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash Phase5_pinpoint_tuning/run_spt_medical.sh
else
    echo "Phase 5 skipped."
    echo "To run later: bash Phase5_pinpoint_tuning/run_spt_medical.sh"
fi

# 終了
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED_TIME / 60))
ELAPSED_SEC=$((ELAPSED_TIME % 60))

echo ""
echo "==================================================="
echo "Full Pipeline Completed!"
echo "==================================================="
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
echo "Generated files:"
echo "  Phase 1:"
echo "    - Phase1_data_preparation/annotated_medical_data.jsonl"
echo "    - Phase1_data_preparation/counterfactual_medical_data.jsonl"
echo "    - Phase1_data_preparation/medical_path_patching_enhanced.jsonl"
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
echo "  Phase 4:"
echo "    - Phase4_visualization/classification_heatmap.png"
echo "    - Phase4_visualization/statistics.json"
echo "    - Phase4_visualization/medical_head_analysis_report.md"
echo ""
echo "Next steps:"
echo "  1. Review the analysis report: cat Phase4_visualization/medical_head_analysis_report.md"
echo "  2. View the heatmap: open Phase4_visualization/classification_heatmap.png"
echo "  3. (Optional) Run Pinpoint Tuning: bash Phase5_pinpoint_tuning/run_spt_medical.sh"
echo ""
