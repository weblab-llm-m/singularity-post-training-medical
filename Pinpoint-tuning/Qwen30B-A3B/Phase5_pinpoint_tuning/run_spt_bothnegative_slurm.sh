#!/bin/bash
#SBATCH --job-name=DONT-KILL_spt_bothnegative
#SBATCH --partition=P08317
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# SPT Training for Qwen3-30B-A3B-Instruct-2507 (MoE) - both_negative_heads
# Uses DeepSpeed ZeRO-3 for memory-efficient training

set -e

echo "==================================================="
echo "SPT Training for Qwen3-30B-A3B MoE (8 GPU) - both_negative_heads"
echo "==================================================="

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
MODEL_PATH="/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507"

cd $BASE_DIR

echo ""
echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Model Type: qwen3_moe (MoE with 128 experts, 8 active)"
echo "  Layers: 48, Heads: 32, Total: 1536"
echo "  Trainable heads: both_negative_heads (457 heads)"
echo "  Train data: igakuqa.jsonl (9374件)"
echo "  GPUs: 8"
echo ""

# SPT configuration
TRAINABLE_HEADS="${BASE_DIR}/Phase5_pinpoint_tuning/trainable_heads_bothnegative.json"
OUTPUT_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/spt_output_bothnegative"
TRAIN_DATA="/home/yuuki.nakamura/downloads/datasets/igakuqa.jsonl"
CACHE_DIR="${BASE_DIR}/Phase5_pinpoint_tuning/cache"

# Training parameters (aligned with ms-swift Megatron config)
USE_LORA=false
PRECISE_LEVEL=4  # Only qkv_proj
LEARNING_RATE=1e-4
MIN_LR=1e-5
BATCH_SIZE=2           # micro-batch-size 2
GRADIENT_ACCUMULATION=2  # global-batch-size 32 / (2 * 8 GPUs) = 2
NUM_EPOCHS=3
MAX_SEQ_LENGTH=4096    # seq-length 4096
SAVE_STEPS=50
LOGGING_STEPS=1
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPSILON=1e-8

# MoE-specific settings
FREEZE_ROUTER=true
FREEZE_EXPERTS=true

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

echo "Starting SPT training with 8 GPUs..."
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Trainable heads file: ${TRAINABLE_HEADS}"
echo "  Precise level: ${PRECISE_LEVEL}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Per-device batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "  Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 8))"
echo "  Max sequence length: ${MAX_SEQ_LENGTH}"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "  Max grad norm: ${MAX_GRAD_NORM}"
echo "  Adam beta1/beta2/eps: ${ADAM_BETA1}/${ADAM_BETA2}/${ADAM_EPSILON}"
echo "  Freeze router: ${FREEZE_ROUTER}"
echo "  Freeze experts: ${FREEZE_EXPERTS}"
echo ""

module load cuda/12.8
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PATH=$CUDA_HOME/bin:$PATH

# Activate virtual environment
source ${BASE_DIR}/venv/bin/activate

# DeepSpeed config
DS_CONFIG="${BASE_DIR}/Phase5_pinpoint_tuning/configs/deepspeed_zero3.json"

# Create DeepSpeed config if not exists
if [ ! -f "${DS_CONFIG}" ]; then
    mkdir -p $(dirname ${DS_CONFIG})
    cat > ${DS_CONFIG} << 'EOF'
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOF
fi

# Build training command with DeepSpeed
TRAIN_CMD="deepspeed --num_gpus=8 ${BASE_DIR}/Phase5_pinpoint_tuning/run_spt_medical.py"
TRAIN_CMD="${TRAIN_CMD} --model_path ${MODEL_PATH}"
TRAIN_CMD="${TRAIN_CMD} --data_path ${TRAIN_DATA}"
TRAIN_CMD="${TRAIN_CMD} --output_dir ${OUTPUT_DIR}"
TRAIN_CMD="${TRAIN_CMD} --cache_dir ${CACHE_DIR}"
TRAIN_CMD="${TRAIN_CMD} --path_patching_path ${TRAINABLE_HEADS}"
TRAIN_CMD="${TRAIN_CMD} --precise_level ${PRECISE_LEVEL}"
TRAIN_CMD="${TRAIN_CMD} --attn_implementation flash_attention_2"
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
TRAIN_CMD="${TRAIN_CMD} --warmup_ratio 0.05"
TRAIN_CMD="${TRAIN_CMD} --lr_scheduler_type cosine"
TRAIN_CMD="${TRAIN_CMD} --weight_decay ${WEIGHT_DECAY}"
TRAIN_CMD="${TRAIN_CMD} --max_grad_norm ${MAX_GRAD_NORM}"
TRAIN_CMD="${TRAIN_CMD} --adam_beta1 ${ADAM_BETA1}"
TRAIN_CMD="${TRAIN_CMD} --adam_beta2 ${ADAM_BETA2}"
TRAIN_CMD="${TRAIN_CMD} --adam_epsilon ${ADAM_EPSILON}"
TRAIN_CMD="${TRAIN_CMD} --bf16 true"
TRAIN_CMD="${TRAIN_CMD} --gradient_checkpointing false"
TRAIN_CMD="${TRAIN_CMD} --seed 42"
TRAIN_CMD="${TRAIN_CMD} --dataloader_num_workers 8"
TRAIN_CMD="${TRAIN_CMD} --freeze_router ${FREEZE_ROUTER}"
TRAIN_CMD="${TRAIN_CMD} --freeze_experts ${FREEZE_EXPERTS}"
TRAIN_CMD="${TRAIN_CMD} --deepspeed ${DS_CONFIG}"

# Execute training
echo "Command: ${TRAIN_CMD}"
echo ""

eval ${TRAIN_CMD} 2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "==================================================="
echo "SPT Training Completed! (both_negative_heads)"
echo "==================================================="
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${OUTPUT_DIR}/training.log"
echo ""
