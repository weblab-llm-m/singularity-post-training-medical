#!/bin/bash
# Evaluation Script: Compare Base vs SPT-tuned Model

set -e

echo "==================================================="
echo "Model Evaluation: Base vs SPT-tuned Qwen3-14B"
echo "==================================================="
echo ""

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
BASE_MODEL="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
TUNED_MODEL="${BASE_DIR}/Phase5_pinpoint_tuning/final_model"
TEST_DATA="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/test.parquet"

# Use train data if test data doesn't exist, but limit to 100 samples
if [ ! -f "$TEST_DATA" ]; then
    echo "Test data not found, using train data (first 100 samples)"
    TEST_DATA="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/train.parquet"
    MAX_SAMPLES=100
else
    echo "Using test data"
    MAX_SAMPLES=200  # Limit for faster evaluation
fi

OUTPUT_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/evaluation_results"
mkdir -p ${OUTPUT_DIR}

source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate

# Evaluate SPT-tuned model first (since it's more important)
echo ""
echo "==================================================="
echo "Evaluating SPT-tuned Model (144 Medical Heads)"
echo "==================================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Phase5_pinpoint_tuning/evaluate_model.py \
    --model_path ${TUNED_MODEL} \
    --data_path ${TEST_DATA} \
    --output_path ${OUTPUT_DIR}/tuned_model_results.json \
    --max_samples ${MAX_SAMPLES} \
    2>&1 | tee ${OUTPUT_DIR}/tuned_model_eval.log

echo ""
echo "==================================================="
echo "Evaluating Base Model (Qwen3-14B)"
echo "==================================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Phase5_pinpoint_tuning/evaluate_model.py \
    --model_path ${BASE_MODEL} \
    --data_path ${TEST_DATA} \
    --output_path ${OUTPUT_DIR}/base_model_results.json \
    --max_samples ${MAX_SAMPLES} \
    2>&1 | tee ${OUTPUT_DIR}/base_model_eval.log

echo ""
echo "==================================================="
echo "Generating Comparison Report"
echo "==================================================="
echo ""

python3 << 'EOF'
import json

tuned_file = "Phase5_pinpoint_tuning/evaluation_results/tuned_model_results.json"
base_file = "Phase5_pinpoint_tuning/evaluation_results/base_model_results.json"

try:
    with open(tuned_file, 'r') as f:
        tuned_data = json.load(f)

    with open(base_file, 'r') as f:
        base_data = json.load(f)

    tuned_acc = tuned_data['metrics']['accuracy']
    base_acc = base_data['metrics']['accuracy']
    improvement = tuned_acc - base_acc

    print("\n" + "="*60)
    print("EVALUATION COMPARISON REPORT")
    print("="*60)
    print(f"\nBase Model (Qwen3-14B):")
    print(f"  Accuracy: {base_acc:.2f}%")
    print(f"  Correct: {base_data['metrics']['correct']}/{base_data['metrics']['total_samples']}")

    print(f"\nSPT-tuned Model (144 Medical Heads):")
    print(f"  Accuracy: {tuned_acc:.2f}%")
    print(f"  Correct: {tuned_data['metrics']['correct']}/{tuned_data['metrics']['total_samples']}")

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.2f}%")
    print(f"  Relative: {(improvement/base_acc*100) if base_acc > 0 else 0:+.2f}%")
    print("="*60)
    print("")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Evaluation results not found. Check logs for errors.")
except Exception as e:
    print(f"Error generating report: {e}")

EOF

echo ""
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
