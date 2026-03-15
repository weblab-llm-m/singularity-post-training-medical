#!/bin/bash
# SPT Training with Megatron-compatible Configuration
# Training 321 Important Heads (Medical + Guideline + Reasoning)

set -e

echo "=========================================================================="
echo "SPT Training (Megatron Config) - 321 Important Heads"
echo "=========================================================================="

BASE_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"

cd $BASE_DIR

echo ""
echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Trainable heads: 321 Important Heads"
echo "    - Medical term heads: 144 (Layers 0-12)"
echo "    - Guideline heads: 141 (Layers 13-26)"
echo "    - Reasoning flow heads: 36 (Layers 27-39)"
echo "  Train data: ${DATA_DIR}/train.parquet"
echo "  Samples: 1,761"
echo ""

echo "Training Parameters (Megatron-compatible):"
echo "  Learning rate: 2e-5 (same as Megatron)"
echo "  Global batch size: 128 (same as Megatron)"
echo "  Epochs: 1 (same as Megatron)"
echo "  LR schedule: cosine with 0.1 warmup"
echo "  Weight decay: 0.01"
echo "  Adam beta1: 0.9, beta2: 0.95"
echo "  Gradient clipping: 1.0"
echo ""

# SPT configuration
TRAINABLE_HEADS="${BASE_DIR}/Phase5_v2_megatron_config/trainable_heads_all_321.json"
OUTPUT_DIR="${BASE_DIR}/Phase5_v2_megatron_config/spt_321heads_output"
TRAIN_DATA="${DATA_DIR}/train.parquet"
CACHE_DIR="${BASE_DIR}/Phase5_v2_megatron_config/cache"

# Training parameters (Megatron-compatible)
PRECISE_LEVEL=4  # Only qkv_proj
LEARNING_RATE=2e-5  # Megatron: 2e-5
BATCH_SIZE=1
GRADIENT_ACCUMULATION=128  # To achieve global_batch_size=128
NUM_EPOCHS=1  # Megatron: max_epochs=1
MAX_SEQ_LENGTH=16384  # Megatron uses 16384, reduced for memory
WEIGHT_DECAY=0.01  # Megatron: 0.01
ADAM_BETA1=0.9  # Megatron: 0.9
ADAM_BETA2=0.95  # Megatron: 0.95
MAX_GRAD_NORM=1.0  # Megatron: clip_grad=1.0
WARMUP_RATIO=0.1  # Megatron: lr_warmup_fraction=0.1

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

echo "Memory Optimization:"
echo "  Single GPU mode (CUDA_VISIBLE_DEVICES=0)"
echo "  Gradient checkpointing: enabled"
echo "  Max sequence length: ${MAX_SEQ_LENGTH}"
echo ""

echo "Starting SPT training..."
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Trainable heads: 321"
echo "  Precise level: ${PRECISE_LEVEL}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size per device: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "  Global batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "  Adam beta1: ${ADAM_BETA1}, beta2: ${ADAM_BETA2}"
echo "  Max grad norm: ${MAX_GRAD_NORM}"
echo "  Max sequence length: ${MAX_SEQ_LENGTH}"
echo ""

# Build training command
TRAIN_CMD="python3 ${BASE_DIR}/Phase5_v2_megatron_config/run_spt_medical.py"
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
TRAIN_CMD="${TRAIN_CMD} --save_steps 100"
TRAIN_CMD="${TRAIN_CMD} --save_total_limit 2"
TRAIN_CMD="${TRAIN_CMD} --logging_steps 5"
TRAIN_CMD="${TRAIN_CMD} --warmup_ratio ${WARMUP_RATIO}"
TRAIN_CMD="${TRAIN_CMD} --lr_scheduler_type cosine"
TRAIN_CMD="${TRAIN_CMD} --bf16 true"
TRAIN_CMD="${TRAIN_CMD} --gradient_checkpointing true"
TRAIN_CMD="${TRAIN_CMD} --seed 42"
TRAIN_CMD="${TRAIN_CMD} --dataloader_num_workers 4"

# Execute training
echo "Full command:"
echo "CUDA_VISIBLE_DEVICES=0 ${TRAIN_CMD}"
echo ""

source /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/activate
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
