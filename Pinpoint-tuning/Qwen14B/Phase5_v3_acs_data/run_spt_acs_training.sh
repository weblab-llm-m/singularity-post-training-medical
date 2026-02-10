#!/bin/bash
# SPT Training for 321 Important Heads - ACS Data v1
# Using cosine scheduler, bfloat16, PRECISE_LEVEL=3

set -e

# Set ulimit to prevent RAM memory errors
ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

echo "=========================================================================="
echo "SPT Training - 321 Important Heads (ACS Data v1)"
echo "=========================================================================="
echo ""
echo "Resource Limits:"
echo "  Stack size: $(ulimit -s)"
echo "  Virtual memory: $(ulimit -v)"
echo "  Open files: $(ulimit -n)"
echo "  Max user processes: $(ulimit -u)"
echo ""

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
DATA_DIR="/home/Competition2025/P05/shareP05/data/ACS_data_v1"

cd $BASE_DIR

echo ""
echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Trainable heads: 321 Important Heads"
echo "    - Medical term heads: 144 (Layers 0-12)"
echo "    - Guideline heads: 141 (Layers 13-26)"
echo "    - Reasoning flow heads: 36 (Layers 27-39)"
echo "  Train data: ${DATA_DIR}/train.parquet"
echo "  Dataset: ACS_data_v1"
echo "  GPU: Single GPU (CUDA_VISIBLE_DEVICES=0)"
echo ""

echo "Training Parameters (ACS v1 Configuration):"
echo "  Learning rate: 2e-5"
echo "  Batch size: 1"
echo "  Gradient accumulation: 128"
echo "  Global batch size: 128"
echo "  Epochs: 1"
echo "  Max sequence length: 2048"
echo "  LR schedule: cosine with 0.1 warmup"
echo "  Weight decay: 0.01"
echo "  Adam beta1: 0.9, beta2: 0.95"
echo "  Gradient clipping: 1.0"
echo "  Dtype: bfloat16"
echo "  PRECISE_LEVEL: 3 (qkv_proj + o_proj)"
echo ""

# SPT configuration
TRAINABLE_HEADS="${BASE_DIR}/Phase5_v3_acs_data/trainable_heads_all_321.json"
OUTPUT_DIR="${BASE_DIR}/Phase5_v3_acs_data/spt_output"
TRAIN_DATA="${DATA_DIR}/train.parquet"
CACHE_DIR="${BASE_DIR}/Phase5_v3_acs_data/cache"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

# Training parameters
PRECISE_LEVEL=3  # qkv_proj + o_proj
LEARNING_RATE=2e-5
BATCH_SIZE=1
GRADIENT_ACCUMULATION=128  # Global batch size = 128
NUM_EPOCHS=1
MAX_SEQ_LENGTH=2048
WEIGHT_DECAY=0.01
ADAM_BETA1=0.9
ADAM_BETA2=0.95
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.1
SAVE_STEPS=100
LOGGING_STEPS=5

echo "Memory Optimization Strategy:"
echo "  1. Single GPU mode (avoid DataParallel overhead)"
echo "  2. Gradient checkpointing: enabled"
echo "  3. Gradient accumulation: 128 steps"
echo "  4. BF16 mixed precision training"
echo "  5. Max sequence length: 2048"
echo "  6. dataloader_num_workers: 2"
echo ""

echo "Training Configuration:"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Trainable heads: 321"
echo "  Precise level: ${PRECISE_LEVEL}"
echo ""

# Build training command
TRAIN_CMD="python3 ${BASE_DIR}/Phase5_v3_acs_data/run_spt_medical.py"
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
TRAIN_CMD="${TRAIN_CMD} --weight_decay ${WEIGHT_DECAY}"
TRAIN_CMD="${TRAIN_CMD} --adam_beta1 ${ADAM_BETA1}"
TRAIN_CMD="${TRAIN_CMD} --adam_beta2 ${ADAM_BETA2}"
TRAIN_CMD="${TRAIN_CMD} --max_grad_norm ${MAX_GRAD_NORM}"
TRAIN_CMD="${TRAIN_CMD} --save_strategy steps"
TRAIN_CMD="${TRAIN_CMD} --save_steps ${SAVE_STEPS}"
TRAIN_CMD="${TRAIN_CMD} --save_total_limit 2"
TRAIN_CMD="${TRAIN_CMD} --logging_steps ${LOGGING_STEPS}"
TRAIN_CMD="${TRAIN_CMD} --warmup_ratio ${WARMUP_RATIO}"
TRAIN_CMD="${TRAIN_CMD} --lr_scheduler_type cosine"
TRAIN_CMD="${TRAIN_CMD} --bf16 true"
TRAIN_CMD="${TRAIN_CMD} --gradient_checkpointing true"
TRAIN_CMD="${TRAIN_CMD} --seed 42"
TRAIN_CMD="${TRAIN_CMD} --dataloader_num_workers 2"

# Execute training on SINGLE GPU
echo "Command: CUDA_VISIBLE_DEVICES=0 ${TRAIN_CMD}"
echo ""

source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate

# Set PYTHONPATH to include Phase5_v3_acs_data directory
export PYTHONPATH="${BASE_DIR}/Phase5_v3_acs_data:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 eval ${TRAIN_CMD} 2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "=========================================================================="
echo "SPT Training Completed!"
echo "=========================================================================="
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${OUTPUT_DIR}/training.log"
echo ""

# Show training summary
if [ -f "${OUTPUT_DIR}/training.log" ]; then
    echo "Training Summary:"
    echo "  Initial loss: $(grep -m 1 "'loss':" ${OUTPUT_DIR}/training.log | grep -oP "'loss': \K[0-9.]+" || echo "N/A")"
    echo "  Final loss: $(grep "'loss':" ${OUTPUT_DIR}/training.log | tail -1 | grep -oP "'loss': \K[0-9.]+" || echo "N/A")"
fi
