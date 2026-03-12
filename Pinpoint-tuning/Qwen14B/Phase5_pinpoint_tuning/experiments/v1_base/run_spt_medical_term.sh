#!/bin/bash
# SPT for Medical Term Heads
# Training on 1 Medical Term Head: Layer 28, Head 32

set -e

echo "==================================================="
echo "SPT Training for Medical Term Head"
echo "==================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"

cd $BASE_DIR

echo ""
echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Trainable heads: 1 Medical Term Head (Layer 28, Head 32)"
echo "  Train data: ${DATA_DIR}/train.parquet"
echo "  Samples: All (1761)"
echo ""

# SPT configuration
TRAINABLE_HEADS="${BASE_DIR}/Phase5_pinpoint_tuning/trainable_heads_medical_term.json"
OUTPUT_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/spt_medical_term_output"
TRAIN_DATA="${DATA_DIR}/train.parquet"
CACHE_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/cache"

# Training parameters (optimized for memory efficiency)
USE_LORA=false
PRECISE_LEVEL=4  # Only qkv_proj
LEARNING_RATE=2e-4
BATCH_SIZE=1  # Reduced from 2 for memory efficiency
GRADIENT_ACCUMULATION=16  # Increased to maintain effective batch size
NUM_EPOCHS=2
MAX_SEQ_LENGTH=1024

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

echo "Starting SPT training..."
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Precise level: ${PRECISE_LEVEL}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "  Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "  Max sequence length: ${MAX_SEQ_LENGTH}"
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
TRAIN_CMD="${TRAIN_CMD} --save_strategy epoch"
TRAIN_CMD="${TRAIN_CMD} --save_total_limit 1"
TRAIN_CMD="${TRAIN_CMD} --logging_steps 20"
TRAIN_CMD="${TRAIN_CMD} --warmup_ratio 0.1"
TRAIN_CMD="${TRAIN_CMD} --lr_scheduler_type cosine"
TRAIN_CMD="${TRAIN_CMD} --bf16 true"
TRAIN_CMD="${TRAIN_CMD} --gradient_checkpointing false"
TRAIN_CMD="${TRAIN_CMD} --seed 42"
TRAIN_CMD="${TRAIN_CMD} --dataloader_num_workers 4"

# Execute training
echo "Command: ${TRAIN_CMD}"
echo ""

source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate
eval ${TRAIN_CMD}

echo ""
echo "==================================================="
echo "SPT Training Completed!"
echo "==================================================="
echo "  Output: ${OUTPUT_DIR}"
echo ""
