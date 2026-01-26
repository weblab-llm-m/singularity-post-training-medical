#!/bin/bash
# Final Evaluation for Phase5_v5 with Qwen3 Recommended Settings

set -e

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v5_pinpoint_tuning_mk2_8gpu"
MODEL_TRAINED="${BASE_DIR}/spt_output"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"
TEST_DATA="${DATA_DIR}/test.parquet"
RESULTS_DIR="${BASE_DIR}/evaluation_results"

# Activate venv
source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${BASE_DIR}:$PYTHONPATH"

cd ${BASE_DIR}

echo "=========================================================================="
echo "Phase5_v5 Evaluation - Qwen3 Recommended Settings"
echo "=========================================================================="
echo ""
echo "Test Data: somemodels (gynecology_guideline_2023)"
echo "Model: ${MODEL_TRAINED}"
echo ""
echo "Settings:"
echo "  max_new_tokens: 32768"
echo "  do_sample: True"
echo "  temperature: 0.6"
echo "  top_p: 0.95"
echo "  top_k: 20"
echo ""

# Evaluate Trained Model
CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/evaluate_model_fixed.py \
    --model_path ${MODEL_TRAINED} \
    --data_path ${TEST_DATA} \
    --output_path ${RESULTS_DIR}/trained_model_results.json

echo ""
echo "Evaluation completed!"
echo "Results saved to: ${RESULTS_DIR}/trained_model_results.json"
echo ""

# Display Results
if [ -f "${RESULTS_DIR}/trained_model_results.json" ]; then
    echo "Phase5_v5 Trained Model (Qwen3 Recommended Settings):"
    python3 -c "import json; data=json.load(open('${RESULTS_DIR}/trained_model_results.json')); print(f\"  Accuracy: {data['metrics']['accuracy']:.1f}%\"); print(f\"  Correct: {data['metrics']['correct']}/{data['metrics']['total_samples']}\")"
    echo ""
fi

echo "=========================================================================="
echo "Evaluation Completed!"
echo "=========================================================================="
