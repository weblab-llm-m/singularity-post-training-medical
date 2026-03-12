#!/bin/bash
# Phase 5: Pinpoint Tuning for Medical QA
# Supervised Pinpoint Tuningを実行

set -e

echo "==================================================="
echo "Phase 5: Pinpoint Tuning (Medical QA)"
echo "==================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"

cd $BASE_DIR

echo ""
echo "Step 1: Select Trainable Heads"
echo "---------------------------------------------------"
python3 Phase5_pinpoint_tuning/select_trainable_heads.py \
    --classification_results Phase3_attention_analysis/head_classification_results.json \
    --patching_results Phase2_path_patching/results/results.pt \
    --output_path Phase5_pinpoint_tuning/trainable_heads.json

echo ""
echo "Step 2: Run Supervised Pinpoint Tuning"
echo "---------------------------------------------------"

# SPT configuration
TRAINABLE_HEADS="${BASE_DIR}/Phase5_pinpoint_tuning/trainable_heads.json"
OUTPUT_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/spt_medical_output"
TRAIN_DATA="${DATA_DIR}/train.parquet"
CACHE_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/cache"

# Training parameters
USE_LORA=false  # Set to true to use LoRA
PRECISE_LEVEL=4  # 4: only qkv_proj, 3: qkv+o_proj, 2: qkv+o_proj+mlp, 1: all
LEARNING_RATE=1e-4
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
NUM_EPOCHS=3
MAX_SEQ_LENGTH=2048

echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Trainable heads: ${TRAINABLE_HEADS}"
echo "  Train data: ${TRAIN_DATA}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Use LoRA: ${USE_LORA}"
echo "  Precise level: ${PRECISE_LEVEL}"
echo ""

# Create output and cache directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

# Build training command
TRAIN_CMD="python3 ${BASE_DIR}/Phase5_pinpoint_tuning/run_spt_medical.py"
TRAIN_CMD="${TRAIN_CMD} --model_path ${MODEL_PATH}"
TRAIN_CMD="${TRAIN_CMD} --data_path ${TRAIN_DATA}"
TRAIN_CMD="${TRAIN_CMD} --output_dir ${OUTPUT_DIR}"
TRAIN_CMD="${TRAIN_CMD} --cache_dir ${CACHE_DIR}"
TRAIN_CMD="${TRAIN_CMD} --path_patching_path ${TRAINABLE_HEADS}"
TRAIN_CMD="${TRAIN_CMD} --precise_level ${PRECISE_LEVEL}"
TRAIN_CMD="${TRAIN_CMD} --attn_implementation eager"
TRAIN_CMD="${TRAIN_CMD} --torch_dtype bfloat16"
TRAIN_CMD="${TRAIN_CMD} --max_seq_length ${MAX_SEQ_LENGTH}"
TRAIN_CMD="${TRAIN_CMD} --per_device_train_batch_size ${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION}"
TRAIN_CMD="${TRAIN_CMD} --learning_rate ${LEARNING_RATE}"
TRAIN_CMD="${TRAIN_CMD} --num_train_epochs ${NUM_EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --save_strategy epoch"
TRAIN_CMD="${TRAIN_CMD} --save_total_limit 2"
TRAIN_CMD="${TRAIN_CMD} --logging_steps 10"
TRAIN_CMD="${TRAIN_CMD} --warmup_ratio 0.1"
TRAIN_CMD="${TRAIN_CMD} --lr_scheduler_type cosine"
TRAIN_CMD="${TRAIN_CMD} --bf16 true"
TRAIN_CMD="${TRAIN_CMD} --gradient_checkpointing false"
TRAIN_CMD="${TRAIN_CMD} --seed 42"

# Add LoRA config if enabled
if [ "$USE_LORA" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --peft_type lora"
    TRAIN_CMD="${TRAIN_CMD} --peft_config ${BASE_DIR}/Phase5_pinpoint_tuning/configs/lora_config.json"
    echo "  LoRA config: ${BASE_DIR}/Phase5_pinpoint_tuning/configs/lora_config.json"
fi

echo ""
echo "Running Supervised Pinpoint Tuning..."
echo "Command: ${TRAIN_CMD}"
echo ""

# Execute training
eval ${TRAIN_CMD}

echo ""
echo "==================================================="
echo "Phase 5 Completed!"
echo "==================================================="
echo "Output files:"
echo "  - Trainable heads: Phase5_pinpoint_tuning/trainable_heads.json"

if [ -d "${OUTPUT_DIR}" ]; then
    echo "  - Tuned model: ${OUTPUT_DIR}/"
fi

echo ""
echo "Next step: Evaluate the tuned model on medical QA tasks"
echo ""
