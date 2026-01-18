#!/bin/bash
#SBATCH --job-name=grpo_learn
#SBATCH --partition=P08317
#SBATCH --nodes=8
#SBATCH --nodelist=osk-gpu[61-68]
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=240
#SBATCH --time=200:00:00
#SBATCH --mem=1200G
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -xeuo pipefail
source $HOME/singularity-post-training-medical/.env

# ===== 共通 =====
# Swift の作業ディレクトリ（v2）
SWIFT_WORKDIR=${SWIFT_WORKDIR:-$HOME/swift-RL}

# 最新 ms-swift のソースツリー（Singularity 外）
MS_SWIFT_DIR=${MS_SWIFT_DIR:-$SWIFT_WORKDIR/containers/ms-swift}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-$SWIFT_WORKDIR/containers/megatron-lm-core_r0.14.0}

# 使用する Singularity イメージ
SIF_FILE=${SIF_FILE:-$SWIFT_WORKDIR/containers/swift3.9.3.sif}

# 元の（共有FS上の）モデル格納場所（Qwen3-Next-80B-A3B-Instruct はそのまま）
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-$HOME/downloads/models/Qwen_Qwen3-Next-80B-A3B-Instruct}
MODEL_PATH=${MODEL_PATH:-${LOCAL_MODEL_PATH}}

# GRPO 用 JSONL データ（過去IgakuQA）
# messages + answer を含む *.jsonl を想定(swift-RLリポジトリのswift-RL/src/swift/data/prepare_data_v2.py実行し作成
DATASET_JSONL=${DATASET_JSONL:-$HOME/downloads/datasets/igakuqa.jsonl}


# 学習パラメータ
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}          # max_length / vLLM max_model_len に利用
MAX_COMPLETION_LEN=${MAX_COMPLETION_LEN:-4096}

PROJECT_NAME=${PROJECT_NAME:-megatron_swift_qwen_next80b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-dapo_megatron_grpo_specialist_exam}
OUTPUT_DIR=${OUTPUT_DIR:-${SWIFT_WORKDIR}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
HF_CACHE=${HF_CACHE:-${SWIFT_WORKDIR}/.cache_home}

# vLLM / GRPO 関連
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
SP_SIZE=${SP_SIZE:-1}

# 共通環境変数
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:32"}
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ビルドキャッシュ永続化
export TORCH_EXTENSIONS_DIR="${HF_CACHE}/torch_extensions"; mkdir -p "${TORCH_EXTENSIONS_DIR}"
export XDG_CACHE_HOME="${HF_CACHE}/xdg"; mkdir -p "${XDG_CACHE_HOME}"
export TRITON_CACHE_DIR="${HF_CACHE}/triton_cache"; mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${HF_CACHE}" "${HF_CACHE}/datasets" "${HF_CACHE}/hub" "${OUTPUT_DIR}"

module purge
module load cuda/12.8
module load singularity || true

CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
echo "[host] nvcc: $(which nvcc)"; echo "[host] CUDA_HOME: ${CUDA_HOME}"

# ===== 分散設定 =====
export NCCL_ASYNC_ERROR_HANDLING=1
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk '{print $1; exit}')
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$(/sbin/ip route show default | awk "/default/ {print \$5}")}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(echo "${NCCL_SOCKET_IFNAME}" | cut -d"," -f1)}
export NNODES=${SLURM_JOB_NUM_NODES}

# ===== 軽いプリウォーム（flash-attn ビルド） =====
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

# ===== ③ 学習本体 (Megatron GRPO, vLLM colocate + DAPO 設定) =====
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
    --env WANDB_DIR=${OUTPUT_DIR} \
    --env CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
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
        -m swift.cli._megatron.rlhf \
          --custom_register_path ${SWIFT_WORKDIR}/src/swift/patch_promptid.py \
          --external_plugins \
            ${MS_SWIFT_DIR}/examples/train/grpo/plugin/plugin.py \
            ${MS_SWIFT_DIR}/examples/train/grpo/plugin/reward_ophtho_plugin.py \
            ${MS_SWIFT_DIR}/examples/train/grpo/plugin/reward_chinese_plugin.py \
          --rlhf_type grpo \
          --loss_type grpo \
          --beta 0.05 \
          --overlong_filter true \
          --reward_funcs ophtho chinese \
          --reward_weights 1.5 1.0 \
          --soft_cache_length 1024 \
          --max_epochs 5 \
          --eval_interval 50 \
          --save_interval 50 \
          --sleep_level 1 \
          --clip_grad 0.4 \
          --lr 1e-6 \
          --lr_decay_style cosine \
          --lr_warmup_fraction 0.03 \
          --log_interval 1 \
          --use_hf true \
          --dataset ${DATASET_JSONL} \
          --model ${MODEL_PATH} \
          --model_type qwen3_next \
          --bf16 true \
          --train_type full \
          --torch_dtype ${DTYPE} \
          --context_parallel_size 1 \
          --tensor_model_parallel_size 1 \
          --expert_model_parallel_size 8 \
          --pipeline_model_parallel_size 8 \
          --sequence_parallel true \
          --remove_unused_columns false \
          --load_safetensors true \
          --offload_model false \
          --offload_optimizer false \
          --use_distributed_optimizer \
          --optimizer_cpu_offload false \
          --recompute_granularity full \
          --recompute_method uniform \
          --recompute_num_layers 1 \
          --use_precision_aware_optimizer true \
          --moe_grouped_gemm true \
          --moe_shared_expert_overlap true \
          --moe_aux_loss_coeff 1e-3 \
          --finetune \
          --global_batch_size 512 \
          --micro_batch_size 1 \
          --steps_per_generation 5 \
          --num_generations ${NUM_GENERATIONS} \
          --max_length ${MAX_MODEL_LEN} \
          --max_completion_length ${MAX_COMPLETION_LEN} \
          --use_vllm true \
          --vllm_mode colocate \
          --vllm_gpu_memory_utilization 0.4 \
          --vllm_tensor_parallel_size 8 \
          --vllm_max_model_len ${MAX_MODEL_LEN} \
          --temperature 0.9 \
          --num_workers 8 \
          --dataset_num_proc 8 \
          --log_completions false \
          --attention_backend flash \
          --padding_free true \
          --save '${OUTPUT_DIR}' \
          --split_dataset_ratio 0.05 \
          --wandb_project 'Ramen_GRPO_GSPO_TRY' \
          --wandb_exp_name 'grpo_reward_chinese_1.0_5epochs_resume'
    "

# --save_safetensors trueでsafetensorsで保存する
# --log_completions trueでWandbで推論結果を出力させる