#!/bin/bash
# DDP Distributed Training for ACS Data with 8 GPUs
# Training 321 Important Heads (Medical + Guideline + Reasoning)
# Based on Phase5_v3_acs_data configuration

set -e

# Set ulimit to prevent RAM memory errors
ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

echo "=========================================================================="
echo "SPT Training (8 GPUs DDP) - 321 Important Heads (ACS Data v1)"
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
echo "  GPUs: 8 (DDP with torchrun)"
echo ""

echo "Distributed Training Parameters (8 GPUs):"
echo "  Strategy: torchrun with 8 GPUs"
echo "  Learning rate: 2e-5"
echo "  Per-device batch size: 1"
echo "  Gradient accumulation: 16"
echo "  Global batch size: 8 × 1 × 16 = 128"
echo "  Epochs: 1"
echo "  Max sequence length: 2048 (ACS v1 Configuration)"
echo "  LR schedule: cosine with 0.1 warmup"
echo "  Weight decay: 0.01"
echo "  Adam beta1: 0.9, beta2: 0.95"
echo "  Gradient clipping: 1.0"
echo "  Dtype: bfloat16"
echo "  PRECISE_LEVEL: 3 (qkv_proj + o_proj)"
echo "  Gradient checkpointing: enabled"
echo ""

# SPT configuration
TRAINABLE_HEADS="${BASE_DIR}/Phase5_v4_acs_data_8gpu/trainable_heads_all_321.json"
OUTPUT_DIR="${BASE_DIR}/Phase5_v4_acs_data_8gpu/spt_output"
TRAIN_DATA="${DATA_DIR}/train.parquet"
CACHE_DIR="${BASE_DIR}/Phase5_v4_acs_data_8gpu/cache"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

# Training parameters (ACS v1 Configuration with 8 GPUs)
PRECISE_LEVEL=3  # qkv_proj + o_proj (same as v3)
LEARNING_RATE=2e-5
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION=16  # Global batch size = 8 × 1 × 16 = 128
NUM_EPOCHS=1
MAX_SEQ_LENGTH=2048  # Same as v3
WEIGHT_DECAY=0.01
ADAM_BETA1=0.9
ADAM_BETA2=0.95
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.1
SAVE_STEPS=100
LOGGING_STEPS=5

echo "Memory Optimization Strategy:"
echo "  1. DDP: Data parallelism across 8 GPUs"
echo "  2. Gradient checkpointing: enabled"
echo "  3. Per-device batch size: 1"
echo "  4. Gradient accumulation: 16 steps"
echo "  5. BF16 mixed precision training"
echo "  6. Max sequence length: 2048"
echo "  7. dataloader_num_workers: 4"
echo ""

echo "Training Configuration:"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Trainable heads: 321"
echo "  Precise level: ${PRECISE_LEVEL}"
echo "  GPUs: 8"
echo ""

# Activate venv
source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate

# Set PYTHONPATH to include Phase5_v4_acs_data_8gpu directory
export PYTHONPATH="${BASE_DIR}/Phase5_v4_acs_data_8gpu:$PYTHONPATH"

# Build training script path
TRAIN_SCRIPT="${BASE_DIR}/Phase5_v4_acs_data_8gpu/run_spt_medical.py"

echo "Starting DDP training with torchrun..."
echo "  Training script: ${TRAIN_SCRIPT}"
echo ""

# torchrun command
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    ${TRAIN_SCRIPT} \
    --model_path ${MODEL_PATH} \
    --data_path ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --path_patching_path ${TRAINABLE_HEADS} \
    --precise_level ${PRECISE_LEVEL} \
    --attn_implementation eager \
    --torch_dtype bfloat16 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --weight_decay ${WEIGHT_DECAY} \
    --adam_beta1 ${ADAM_BETA1} \
    --adam_beta2 ${ADAM_BETA2} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 2 \
    --logging_steps ${LOGGING_STEPS} \
    --logging_strategy steps \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type cosine \
    --bf16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --dataloader_num_workers 4 \
    --seed 42 2>&1 | tee ${OUTPUT_DIR}/training.log

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
