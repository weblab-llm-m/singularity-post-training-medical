#!/bin/bash
# Phase 1: データ準備

set -e

echo "==================================================="
echo "Phase 1: Medical Data Preparation"
echo "==================================================="

# ベースディレクトリ
BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"

cd $BASE_DIR

echo ""
echo "Step 1: Medical Term Annotation"
echo "---------------------------------------------------"
echo "Processing full train.parquet dataset..."
python3 Phase1_data_preparation/medical_term_annotator.py \
    --input_path ${DATA_DIR}/train.parquet \
    --output_path Phase1_data_preparation/annotated_medical_data.jsonl \
    --medical_dict Phase1_data_preparation/medical_terms_dictionary.json \
    --model_path ${MODEL_PATH}

echo ""
echo "Step 2: Counterfactual Generation"
echo "---------------------------------------------------"
python3 Phase1_data_preparation/counterfactual_generator.py \
    --input_path Phase1_data_preparation/annotated_medical_data.jsonl \
    --output_path Phase1_data_preparation/counterfactual_medical_data.jsonl \
    --strategy medical_term_replacement

echo ""
echo "Step 3: Path Patching Data Building"
echo "---------------------------------------------------"
python3 Phase1_data_preparation/path_patching_data_builder.py \
    --annotation_path Phase1_data_preparation/annotated_medical_data.jsonl \
    --counterfactual_path Phase1_data_preparation/counterfactual_medical_data.jsonl \
    --output_path Phase1_data_preparation/medical_path_patching_enhanced.jsonl

echo ""
echo "==================================================="
echo "Phase 1 Completed!"
echo "==================================================="
echo "Output files:"
echo "  - Annotated data: Phase1_data_preparation/annotated_medical_data.jsonl"
echo "  - Counterfactual data: Phase1_data_preparation/counterfactual_medical_data.jsonl"
echo "  - Path patching data: Phase1_data_preparation/medical_path_patching_enhanced.jsonl"
echo ""
