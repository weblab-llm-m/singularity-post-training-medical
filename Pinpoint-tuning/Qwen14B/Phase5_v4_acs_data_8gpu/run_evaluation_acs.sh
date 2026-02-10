#!/bin/bash
# Evaluation script for Phase5_v4 ACS Data 8GPU Training
# Compare Base Model vs Trained Model (321 heads SPT)

set -e

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v4_acs_data_8gpu"
MODEL_BASE="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
MODEL_TRAINED="${BASE_DIR}/spt_output"
DATA_DIR="/home/Competition2025/P05/shareP05/data/ACS_data_v1"
TEST_DATA="${DATA_DIR}/test.parquet"
RESULTS_DIR="${BASE_DIR}/evaluation_results"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Activate venv
source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${BASE_DIR}:$PYTHONPATH"

cd ${BASE_DIR}

echo "=========================================================================="
echo "Phase5_v4 ACS Data Evaluation - 321 Heads SPT (8 GPUs)"
echo "=========================================================================="
echo ""
echo "Models:"
echo "  Base: ${MODEL_BASE}"
echo "  Trained: ${MODEL_TRAINED}"
echo ""
echo "Test Data: ${TEST_DATA}"
echo ""

# Evaluate Base Model
echo "=========================================================================="
echo "1. Evaluating Base Model (Qwen3-14B)"
echo "=========================================================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/evaluate_model_fixed.py \
    --model_path ${MODEL_BASE} \
    --data_path ${TEST_DATA} \
    --output_path ${RESULTS_DIR}/base_model_results.json \
    --max_new_tokens 512

echo ""
echo "Base model evaluation completed!"
echo "Results saved to: ${RESULTS_DIR}/base_model_results.json"
echo ""

# Evaluate Trained Model
echo "=========================================================================="
echo "2. Evaluating Trained Model (321 Heads SPT)"
echo "=========================================================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/evaluate_model_fixed.py \
    --model_path ${MODEL_TRAINED} \
    --data_path ${TEST_DATA} \
    --output_path ${RESULTS_DIR}/trained_model_321heads_acs_results.json \
    --max_new_tokens 512

echo ""
echo "Trained model evaluation completed!"
echo "Results saved to: ${RESULTS_DIR}/trained_model_321heads_acs_results.json"
echo ""

# Display Results
echo "=========================================================================="
echo "Evaluation Results Summary"
echo "=========================================================================="
echo ""

if [ -f "${RESULTS_DIR}/base_model_results.json" ]; then
    echo "Base Model:"
    python3 -c "import json; data=json.load(open('${RESULTS_DIR}/base_model_results.json')); print(f\"  Accuracy: {data['metrics']['accuracy']:.1f}%\"); print(f\"  Correct: {data['metrics']['correct']}/{data['metrics']['total_samples']}\")"
    echo ""
fi

if [ -f "${RESULTS_DIR}/trained_model_321heads_acs_results.json" ]; then
    echo "Trained Model (321 Heads SPT):"
    python3 -c "import json; data=json.load(open('${RESULTS_DIR}/trained_model_321heads_acs_results.json')); print(f\"  Accuracy: {data['metrics']['accuracy']:.1f}%\"); print(f\"  Correct: {data['metrics']['correct']}/{data['metrics']['total_samples']}\")"
    echo ""
fi

echo "=========================================================================="
echo "Evaluation Completed!"
echo "=========================================================================="
echo ""
echo "Results directory: ${RESULTS_DIR}"
echo ""
