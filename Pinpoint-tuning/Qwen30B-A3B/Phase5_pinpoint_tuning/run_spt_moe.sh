#!/bin/bash
# SPT Training for Qwen3-30B-A3B-Instruct-2507 (MoE)
# Training on heads identified via Path Patching analysis

set -e

echo "==================================================="
echo "SPT Training for Qwen3-30B-A3B MoE"
echo "==================================================="

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
MODEL_PATH="/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"

cd $BASE_DIR

echo ""
echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Model Type: qwen3_moe (MoE with 128 experts, 8 active)"
echo "  Layers: 48, Heads: 32, Total: 1536"
echo "  Train data: ${DATA_DIR}/train.parquet"
echo ""

# SPT configuration
TRAINABLE_HEADS="${BASE_DIR}/Phase5_pinpoint_tuning/trainable_heads.json"
OUTPUT_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/spt_output"
TRAIN_DATA="${DATA_DIR}/train.parquet"
CACHE_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/cache"

# Training parameters (optimized for 30B MoE model)
# MoE models need more memory due to all experts being loaded
USE_LORA=false
PRECISE_LEVEL=4  # Only qkv_proj
LEARNING_RATE=1e-4  # Lower LR for larger model
BATCH_SIZE=1
GRADIENT_ACCUMULATION=32  # Higher accumulation for memory efficiency
NUM_EPOCHS=2
MAX_SEQ_LENGTH=512  # Reduced for memory constraints
SAVE_STEPS=100
LOGGING_STEPS=10

# MoE-specific settings
FREEZE_ROUTER=true
FREEZE_EXPERTS=true

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

echo "Starting SPT training..."
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Trainable heads file: ${TRAINABLE_HEADS}"
echo "  Precise level: ${PRECISE_LEVEL}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "  Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "  Max sequence length: ${MAX_SEQ_LENGTH}"
echo "  Freeze router: ${FREEZE_ROUTER}"
echo "  Freeze experts: ${FREEZE_EXPERTS}"
echo ""

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
TRAIN_CMD="${TRAIN_CMD} --save_strategy steps"
TRAIN_CMD="${TRAIN_CMD} --save_steps ${SAVE_STEPS}"
TRAIN_CMD="${TRAIN_CMD} --save_total_limit 2"
TRAIN_CMD="${TRAIN_CMD} --logging_steps ${LOGGING_STEPS}"
TRAIN_CMD="${TRAIN_CMD} --warmup_ratio 0.1"
TRAIN_CMD="${TRAIN_CMD} --lr_scheduler_type cosine"
TRAIN_CMD="${TRAIN_CMD} --bf16 true"
TRAIN_CMD="${TRAIN_CMD} --gradient_checkpointing true"
TRAIN_CMD="${TRAIN_CMD} --seed 42"
TRAIN_CMD="${TRAIN_CMD} --dataloader_num_workers 4"
TRAIN_CMD="${TRAIN_CMD} --freeze_router ${FREEZE_ROUTER}"
TRAIN_CMD="${TRAIN_CMD} --freeze_experts ${FREEZE_EXPERTS}"

# Execute training
echo "Command: ${TRAIN_CMD}"
echo ""

# Activate virtual environment
source ${BASE_DIR}/venv/bin/activate

eval ${TRAIN_CMD} 2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "==================================================="
echo "SPT Training Completed!"
echo "==================================================="
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${OUTPUT_DIR}/training.log"
echo ""
