#!/bin/bash
#SBATCH --job-name=negative_sft_megatron
#SBATCH --partition=P08317
#SBATCH --nodes=4
#SBATCH --nodelist=osk-gpu[79-82]
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --mem=1200G
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -xeuo pipefail
source $HOME/singularity-post-training-medical/.env

# ============================================================
#  共通パス設定（GRPO スクリプトと同一構造）
# ============================================================
SWIFT_WORKDIR=${SWIFT_WORKDIR:-$HOME/swift-RL}

MS_SWIFT_DIR=${MS_SWIFT_DIR:-$SWIFT_WORKDIR/containers/ms-swift}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-$SWIFT_WORKDIR/containers/megatron-lm-core_r0.14.0}

SIF_FILE=${SIF_FILE:-$SWIFT_WORKDIR/containers/swift3.9.3.sif}

# モデル
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-$HOME/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507}
MODEL_PATH=${MODEL_PATH:-${LOCAL_MODEL_PATH}}

# データセット（SFT 用 JSONL, messages 形式）
DATASET_JSONL=${DATASET_JSONL:-$HOME/downloads/datasets/sft_igakuqa_v2.jsonl}

# ============================================================
#  学習パラメータ
# ============================================================
DTYPE=${DTYPE:-bfloat16}
MAX_LENGTH=${MAX_LENGTH:-4096}

PROJECT_NAME=${PROJECT_NAME:-megatron_swift_qwen_next80b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-negative_layer_head}
OUTPUT_DIR=${OUTPUT_DIR:-${SWIFT_WORKDIR}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
HF_CACHE=${HF_CACHE:-${SWIFT_WORKDIR}/.cache_home}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ============================================================
#  共通環境変数
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:32"}
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_MAX_CONNECTIONS=1

# キャッシュ永続化
export TORCH_EXTENSIONS_DIR="${HF_CACHE}/torch_extensions"; mkdir -p "${TORCH_EXTENSIONS_DIR}"
export XDG_CACHE_HOME="${HF_CACHE}/xdg";                    mkdir -p "${XDG_CACHE_HOME}"
export TRITON_CACHE_DIR="${HF_CACHE}/triton_cache";          mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${HF_CACHE}" "${HF_CACHE}/datasets" "${HF_CACHE}/hub" "${OUTPUT_DIR}"

module purge
module load cuda/12.8
module load singularity || true

CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
echo "[host] nvcc: $(which nvcc)"; echo "[host] CUDA_HOME: ${CUDA_HOME}"

# ============================================================
#  分散設定
# ============================================================
export NCCL_ASYNC_ERROR_HANDLING=1
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk '{print $1; exit}')
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$(/sbin/ip route show default | awk "/default/ {print \$5}")}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(echo "${NCCL_SOCKET_IFNAME}" | cut -d"," -f1)}
export NNODES=${SLURM_JOB_NUM_NODES}


# ============================================================
#  PinPointTuning設定（スクリプト冒頭で定義）
# ============================================================
# PINPOINT_EXPERTS="5:3,7_10:1,4"
PINPOINT_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47"
PINPOINT_HEADS="0:1,17,22_1:27_2:2,3,5,9,12,14,16,18,20_3:2,6,10,11,12,15,22,25,26,28,30_4:2,3,5,8,9,13,17,20,24,26_5:0,1,2,4,6,10,18,23,24,26,30_6:3,10,14,27,28_7:2,7,8,11,15,20,28,31_8:1,6,9,10,13,15,20,21,22,26,28_9:3,5,15,16,19,24,25,30_10:4,10,20,21,24,26,29,31_11:0,6,8,15,22,23,27_12:4,12,13,16,20,23_13:13,15,16,17,18,23,25,28,29_14:10,11,15,19,26_15:1,15,27,30,31_16:0,1,2,4,6,7,10,12,14,15,16,17,19,21,22,23,24,25,26,28_17:2,6,7,14,15,18,19,25,28,30_18:5,7,9,12,14,15,17,23,25,26,27,29,31_19:1,2,7,9,17,22_20:11,13,20_21:1,20,21_22:7,11,12,18,22,24,28_23:14,15,19,20,24_24:2,4,28_25:2,10,20_26:11,18,21,31_27:1,6,12,16,18,25,27_28:0,1,2,3,7,15,18,22,23,26,27,28,29,30_29:12,14,20,21,23,27,29_30:5,8,19,23,24,27_31:6,7,12,17,18,20,23,24,28_32:8,12,14,16,17,18,21_33:2,4,13,17,19,22,28_34:5,14,16,31_35:3,5,27_36:4,6,9,13,14,19,23,29_37:6,13,15,26,28,31_38:1,4,6,9,10,13,15,19,21,22,23,25_39:0,1,5,6,8,11,12,13,17,18,19,20,23,25,26,27,28,29,30_40:0,2,4,5,6,9,10,11,12,13,17,21,23,24,26,27,29_41:0,1,3,7,8,10,11,15,16,18,20,22,23,24,25,26,27,28,30_42:1,2,3,4,5,7,8,9,12,16,19,21,22,24,26,27,28,30_43:0,4,5,6,7,10,11,12,13,14,15,16,17,18,19,22,23,28,31_44:0,1,2,3,8,9,10,11,13,14,16,17,18,19,20,21,22,23,27,28_45:0,1,2,4,7,8,9,10,11,12,13,14,15,16,18,20,21,22,23,24,25,29,31_46:1,4,6,8,9,14,15,16,19,20,21,22,24,26,27,29,31_47:0,2,3,4,6,7,8,9,11,12,14,15,16,18,19,21,23,26,28,30,31"

# ============================================================
#  プリウォーム（flash-attn ビルドキャッシュ）
# ============================================================
srun --overlap -N1 -n1 -w "${MASTER_NODE}" singularity exec --nv --cleanenv \
  -B "${CUDA_HOME}:${CUDA_HOME}" \
  -B "${SWIFT_WORKDIR}:${SWIFT_WORKDIR}" \
  -B "/dev/shm:/dev/shm" \
  --home "${HF_CACHE}:/root" \
  --env HOME=/root \
  --env HF_HOME="${HF_CACHE}" \
  --env HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
  --env HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub" \
  --env TRANSFORMERS_CACHE="${HF_CACHE}/hub" \
  --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  --env TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
  --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
  "${SIF_FILE}" bash -lc '
    python - <<PY || true
import torch
torch.cuda.init()
a=torch.randn(1,128,64,device="cuda",dtype=torch.bfloat16)
b=torch.randn(1,128,64,device="cuda",dtype=torch.bfloat16)
c=torch.randn(1,128,64,device="cuda",dtype=torch.bfloat16)
try:
  from flash_attn.flash_attn_interface import flash_attn_func
  flash_attn_func(a,b,c,causal=True)
  print("[warmup] flash-attn ok")
except Exception as e:
  print("[warmup] skipped:", e)
PY
  '

# ============================================================
#  SFT 学習本体 (Megatron SFT)
# ============================================================
srun --export=ALL -N${SLURM_JOB_NUM_NODES} -n${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --kill-on-bad-exit=1 \
  singularity exec --nv --cleanenv \
    -B "${CUDA_HOME}:${CUDA_HOME}" \
    -B "${SWIFT_WORKDIR}:${SWIFT_WORKDIR}" \
    -B "${MS_SWIFT_DIR}:${MS_SWIFT_DIR}" \
    -B "${MEGATRON_LM_PATH}:${MEGATRON_LM_PATH}" \
    -B "${MODEL_PATH}:${MODEL_PATH}" \
    -B "/dev/shm:/dev/shm" \
    -B "${DATASET_JSONL}:${DATASET_JSONL}" \
    --home "${HF_CACHE}:/root" \
    --env NODE_RANK="${SLURM_NODEID}" \
    --env NNODES="${SLURM_JOB_NUM_NODES}" \
    --env NPROC_PER_NODE="${NPROC_PER_NODE}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HOME="${HF_CACHE}" \
    --env HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
    --env HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub" \
    --env TRANSFORMERS_CACHE="${HF_CACHE}/hub" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
    --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
    --env MS_SWIFT_DIR="${MS_SWIFT_DIR}" \
    --env MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_ENTITY='llm-m_wandb-weblab' \
    --env WANDB_PROJECT="${PROJECT_NAME}" \
    --env WANDB_DIR="${OUTPUT_DIR}" \
    --env CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    --env PINPOINT_LAYERS="${PINPOINT_LAYERS}" \
    --env PINPOINT_HEADS="${PINPOINT_HEADS}" \
    "${SIF_FILE}" bash -lc "
      set -xeuo pipefail
      ulimit -l unlimited || true
      ulimit -m unlimited || true
      ulimit -v unlimited || true

      export PATH=\"${CUDA_HOME}/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin:\$PATH\"
      export PYTHONPATH=\"\${MS_SWIFT_DIR}:\${MEGATRON_LM_PATH}:\${PYTHONPATH-}\"
      cd \"\${MS_SWIFT_DIR}\"
      export SWIFT_LOG_LEVEL=DEBUG

      torchrun \
        --nnodes ${SLURM_JOB_NUM_NODES} \
        --nproc_per_node ${NPROC_PER_NODE} \
        --node_rank \${NODE_RANK} \
        --rdzv_backend c10d \
        --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
        -m swift.cli._megatron.sft \
          --use_hf true \
          --model ${MODEL_PATH} \
          --model_type qwen3_moe \
          --dataset ${DATASET_JSONL} \
          --split_dataset_ratio 0.01 \
          --bf16 true \
          --torch_dtype ${DTYPE} \
          --train_type full \
          --load_safetensors true \
          --no_initialization false \
          --pinpoint_tuning true \
          --pinpoint_trainable_layers "${PINPOINT_LAYERS}" \
          --pinpoint_trainable_heads "${PINPOINT_HEADS}" \
          --pinpoint_freeze_mlp false \
          --pinpoint_freeze_attention false \
          --pinpoint_freeze_router true \
          --pinpoint_freeze_shared_expert true \
          --pinpoint_freeze_embed_lm_head true \
          --finetune true \
          --tensor_model_parallel_size 1 \
          --pipeline_model_parallel_size 4 \
          --expert_model_parallel_size 8 \
          --context_parallel_size 1 \
          --sequence_parallel true \
          --use_distributed_optimizer \
          --moe_grouped_gemm true \
          --moe_shared_expert_overlap true \
          --moe_aux_loss_coeff 1e-3 \
          --micro_batch_size 2 \
          --global_batch_size 32 \
          --padding_free true \
          --recompute_granularity full \
          --recompute_method uniform \
          --recompute_num_layers 1 \
          --cross_entropy_loss_fusion true \
          --lr 1e-4 \
          --lr_decay_style cosine \
          --lr_warmup_fraction 0.05 \
          --min_lr 1e-5 \
          --max_epochs 1 \
          --eval_interval 50 \
          --save_interval 50 \
          --log_interval 1 \
          --max_length ${MAX_LENGTH} \
          --num_workers 8 \
          --dataset_num_proc 8 \
          --attention_backend flash \
          --no_save_optim true \
          --no_save_rng true \
          --save '${OUTPUT_DIR}' \
          --wandb_project 'Ramen_PinPoint_Tuning' \
          --wandb_exp_name '${EXPERIMENT_NAME}'
    "