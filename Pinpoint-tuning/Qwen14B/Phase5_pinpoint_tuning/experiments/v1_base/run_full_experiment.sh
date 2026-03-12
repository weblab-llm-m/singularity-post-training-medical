#!/bin/bash
# Full SPT Experiment: Train and Evaluate Medical Term Head
# Compares base model vs SPT-tuned model on train.parquet

set -e

echo "==================================================="
echo "Full SPT Experiment for Medical Term Head"
echo "==================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"
TRAIN_DATA="${DATA_DIR}/train.parquet"

cd $BASE_DIR

# Activate virtual environment
source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate

# Experiment parameters
EVAL_MAX_SAMPLES=50  # Reduced from 100 for memory efficiency
# For full evaluation, set to empty:
# EVAL_MAX_SAMPLES=""

echo ""
echo "Experiment Configuration:"
echo "  Base model: ${MODEL_PATH}"
echo "  Data: ${TRAIN_DATA}"
echo "  Evaluation samples: ${EVAL_MAX_SAMPLES:-All (1761)}"
echo "  Medical Term Heads: 1 (Layer 28, Head 32)"
echo ""

# Output paths
BASE_EVAL_RESULT="Phase5_pinpoint_tuning/results/base_model_evaluation.json"
TUNED_EVAL_RESULT="Phase5_pinpoint_tuning/results/tuned_model_evaluation.json"
COMPARISON_RESULT="Phase5_pinpoint_tuning/results/comparison_report.json"
TUNED_MODEL_DIR="Phase5_pinpoint_tuning/spt_medical_term_output"

mkdir -p Phase5_pinpoint_tuning/results

# ===================================================================
# Step 1: Evaluate Base Model
# ===================================================================
echo ""
echo "Step 1: Evaluating Base Model"
echo "---------------------------------------------------"

if [ -z "$EVAL_MAX_SAMPLES" ]; then
    EVAL_CMD="python3 Phase5_pinpoint_tuning/evaluate_model.py \
        --model_path ${MODEL_PATH} \
        --data_path ${TRAIN_DATA} \
        --output_path ${BASE_EVAL_RESULT}"
else
    EVAL_CMD="python3 Phase5_pinpoint_tuning/evaluate_model.py \
        --model_path ${MODEL_PATH} \
        --data_path ${TRAIN_DATA} \
        --output_path ${BASE_EVAL_RESULT} \
        --max_samples ${EVAL_MAX_SAMPLES}"
fi

echo "Running: ${EVAL_CMD}"
eval ${EVAL_CMD}

echo ""
echo "Base model evaluation completed!"
echo "  Results saved to: ${BASE_EVAL_RESULT}"

# ===================================================================
# Step 2: Run SPT Training
# ===================================================================
echo ""
echo "Step 2: Running SPT Training"
echo "---------------------------------------------------"

bash Phase5_pinpoint_tuning/run_spt_medical_term.sh

echo ""
echo "SPT training completed!"
echo "  Model saved to: ${TUNED_MODEL_DIR}"

# ===================================================================
# Step 3: Evaluate Tuned Model
# ===================================================================
echo ""
echo "Step 3: Evaluating Tuned Model"
echo "---------------------------------------------------"

# Find the latest checkpoint directory
CHECKPOINT_DIR=$(ls -td ${TUNED_MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "No checkpoint found in ${TUNED_MODEL_DIR}"
    echo "Using final model from ${TUNED_MODEL_DIR}"
    TUNED_MODEL_PATH="${TUNED_MODEL_DIR}"
else
    echo "Using checkpoint: ${CHECKPOINT_DIR}"
    TUNED_MODEL_PATH="${CHECKPOINT_DIR}"
fi

if [ -z "$EVAL_MAX_SAMPLES" ]; then
    EVAL_CMD="python3 Phase5_pinpoint_tuning/evaluate_model.py \
        --model_path ${TUNED_MODEL_PATH} \
        --data_path ${TRAIN_DATA} \
        --output_path ${TUNED_EVAL_RESULT}"
else
    EVAL_CMD="python3 Phase5_pinpoint_tuning/evaluate_model.py \
        --model_path ${TUNED_MODEL_PATH} \
        --data_path ${TRAIN_DATA} \
        --output_path ${TUNED_EVAL_RESULT} \
        --max_samples ${EVAL_MAX_SAMPLES}"
fi

echo "Running: ${EVAL_CMD}"
eval ${EVAL_CMD}

echo ""
echo "Tuned model evaluation completed!"
echo "  Results saved to: ${TUNED_EVAL_RESULT}"

# ===================================================================
# Step 4: Compare Results
# ===================================================================
echo ""
echo "Step 4: Comparing Results"
echo "---------------------------------------------------"

python3 << EOF
import json

# Load results
with open('${BASE_EVAL_RESULT}', 'r') as f:
    base_results = json.load(f)

with open('${TUNED_EVAL_RESULT}', 'r') as f:
    tuned_results = json.load(f)

# Extract metrics
base_metrics = base_results['metrics']
tuned_metrics = tuned_results['metrics']

# Calculate improvement
improvement = tuned_metrics['accuracy'] - base_metrics['accuracy']
improvement_pct = (improvement / base_metrics['accuracy'] * 100) if base_metrics['accuracy'] > 0 else 0

# Create comparison report
comparison = {
    'experiment': 'Medical Term Head SPT',
    'trainable_heads': 1,
    'head_info': 'Layer 28, Head 32',
    'base_model': {
        'path': base_results['model_path'],
        'total_samples': base_metrics['total_samples'],
        'correct': base_metrics['correct'],
        'accuracy': base_metrics['accuracy']
    },
    'tuned_model': {
        'path': tuned_results['model_path'],
        'total_samples': tuned_metrics['total_samples'],
        'correct': tuned_metrics['correct'],
        'accuracy': tuned_metrics['accuracy']
    },
    'improvement': {
        'absolute': improvement,
        'relative_pct': improvement_pct
    }
}

# Save comparison
with open('${COMPARISON_RESULT}', 'w') as f:
    json.dump(comparison, f, indent=2)

# Print summary
print("=" * 60)
print("EXPERIMENT RESULTS SUMMARY")
print("=" * 60)
print(f"Trainable Heads: 1 Medical Term Head (Layer 28, Head 32)")
print(f"Evaluation Samples: {base_metrics['total_samples']}")
print("")
print(f"Base Model:")
print(f"  Accuracy: {base_metrics['accuracy']:.2f}%")
print(f"  Correct: {base_metrics['correct']}/{base_metrics['total_samples']}")
print("")
print(f"Tuned Model (SPT):")
print(f"  Accuracy: {tuned_metrics['accuracy']:.2f}%")
print(f"  Correct: {tuned_metrics['correct']}/{tuned_metrics['total_samples']}")
print("")
print(f"Improvement:")
print(f"  Absolute: {improvement:+.2f} percentage points")
print(f"  Relative: {improvement_pct:+.2f}%")
print("=" * 60)
print(f"\nDetailed comparison saved to: ${COMPARISON_RESULT}")

EOF

echo ""
echo "==================================================="
echo "Full Experiment Completed!"
echo "==================================================="
echo "Results:"
echo "  Base model: ${BASE_EVAL_RESULT}"
echo "  Tuned model: ${TUNED_EVAL_RESULT}"
echo "  Comparison: ${COMPARISON_RESULT}"
echo ""
echo "Note: Only 1 Medical Term Head was found and trained."
echo "      Limited improvement is expected with such a small modification."
echo ""
