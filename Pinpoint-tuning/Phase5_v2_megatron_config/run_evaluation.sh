#!/bin/bash
# Evaluate Base Model vs 321-heads Trained Model

set -e

# Set ulimit to prevent RAM memory errors
ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

echo "=========================================================================="
echo "Model Evaluation - Base vs 321-heads Trained Model"
echo "=========================================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
BASE_MODEL="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
TRAINED_MODEL="${BASE_DIR}/Phase5_v2_megatron_config/spt_321heads_output/checkpoint-14"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"
OUTPUT_DIR="${BASE_DIR}/Phase5_v2_megatron_config/evaluation_results"
CACHE_DIR="${BASE_DIR}/Phase5_v2_megatron_config/cache"

# Create output directory
mkdir -p ${OUTPUT_DIR}

source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate
export PYTHONPATH="${BASE_DIR}/Phase5_v2_megatron_config:$PYTHONPATH"

echo ""
echo "Configuration:"
echo "  Base model: ${BASE_MODEL}"
echo "  Trained model: ${TRAINED_MODEL}"
echo "  Test data: ${DATA_DIR}/test.parquet"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# Evaluate Base Model
echo "=========================================="
echo "Step 1: Evaluating Base Model"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Phase5_v2_megatron_config/evaluate_model_fixed.py \
    --model_path ${BASE_MODEL} \
    --data_path ${DATA_DIR}/test.parquet \
    --output_path ${OUTPUT_DIR}/base_model_results.json \
    --batch_size 4

echo ""
echo "Base model evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}/base_model_results.json"
echo ""

# Evaluate Trained Model
echo "=========================================="
echo "Step 2: Evaluating 321-heads Trained Model"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Phase5_v2_megatron_config/evaluate_model_fixed.py \
    --model_path ${TRAINED_MODEL} \
    --data_path ${DATA_DIR}/test.parquet \
    --output_path ${OUTPUT_DIR}/trained_model_321heads_results.json \
    --batch_size 4

echo ""
echo "Trained model evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}/trained_model_321heads_results.json"
echo ""

# Compare results
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo ""

if [ -f "${OUTPUT_DIR}/base_model_results.json" ] && [ -f "${OUTPUT_DIR}/trained_model_321heads_results.json" ]; then
    python3 << EOF
import json

with open("${OUTPUT_DIR}/base_model_results.json", "r") as f:
    base_data = json.load(f)

with open("${OUTPUT_DIR}/trained_model_321heads_results.json", "r") as f:
    trained_data = json.load(f)

base_metrics = base_data.get("metrics", {})
trained_metrics = trained_data.get("metrics", {})

base_correct = base_metrics.get("correct", 0)
trained_correct = trained_metrics.get("correct", 0)
total = base_metrics.get("total_samples", 0)

print(f"Base Model:")
print(f"  Correct: {base_correct}/{total}")
print(f"  Accuracy: {base_metrics.get('accuracy', 0):.1f}%")
print(f"")
print(f"Trained Model (321 heads):")
print(f"  Correct: {trained_correct}/{total}")
print(f"  Accuracy: {trained_metrics.get('accuracy', 0):.1f}%")
print(f"")
if total > 0:
    improvement = trained_correct - base_correct
    improvement_pct = 100.0 * improvement / total
    print(f"Improvement: {improvement} samples (+{improvement_pct:.1f}%)")
EOF
fi

echo ""
echo "=========================================================================="
echo "Evaluation Completed!"
echo "=========================================================================="
echo "  Output directory: ${OUTPUT_DIR}"
echo ""
