#!/bin/bash
#SBATCH --job-name=convert_mcore_to_hf
#SBATCH --partition=P08317
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu47
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=800G
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -xeuo pipefail
source $HOME/singularity-post-training-medical/.env

# ===== 設定 =====
SWIFT_WORKDIR=${SWIFT_WORKDIR:-$HOME/swift-RL}
MS_SWIFT_DIR=${MS_SWIFT_DIR:-$SWIFT_WORKDIR/containers/ms-swift}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-$SWIFT_WORKDIR/containers/megatron-lm-core_r0.14.0}
SIF_FILE=${SIF_FILE:-$SWIFT_WORKDIR/containers/swift3.9.3.sif}
HF_CACHE=${HF_CACHE:-${SWIFT_WORKDIR}/.cache_home}

# ===== 元モデルパス（HF形式のconfig.jsonが必要）=====
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-$HOME/downloads/models/Qwen_Qwen3-Next-80B-A3B-Instruct}

# ===== 変換対象のチェックポイント ！！！！ここを変更する！！！！=====
MCORE_CHECKPOINT_DIR=${MCORE_CHECKPOINT_DIR:-"$SWIFT_WORKDIR/outputs/megatron_swift_qwen_next80b/dapo_megatron_grpo_specialist_exam/v70-20260126-010259"}

# ===== 出力先 =====
OUTPUT_HF_DIR=${OUTPUT_HF_DIR:-"${MCORE_CHECKPOINT_DIR}-hf"}

# ===== データ型 =====
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}

# ===== 精度テスト =====
TEST_PRECISION=${TEST_PRECISION:-false}

# ===== GPU数 =====
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ===== 環境変数 =====
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:32"}
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false

# キャッシュディレクトリ
export TORCH_EXTENSIONS_DIR="${HF_CACHE}/torch_extensions"
export XDG_CACHE_HOME="${HF_CACHE}/xdg"
export TRITON_CACHE_DIR="${HF_CACHE}/triton_cache"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"

module purge
module load cuda/12.8
module load singularity || true

CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}

# ===== チェックポイント確認 =====
echo "=== Checkpoint Information ==="
echo "MCORE_CHECKPOINT_DIR: ${MCORE_CHECKPOINT_DIR}"
echo "OUTPUT_HF_DIR: ${OUTPUT_HF_DIR}"
echo "LOCAL_MODEL_PATH: ${LOCAL_MODEL_PATH}"

# 元モデルの存在確認
if [ -d "${LOCAL_MODEL_PATH}" ]; then
    echo "✓ Original model found at: ${LOCAL_MODEL_PATH}"
    if [ -f "${LOCAL_MODEL_PATH}/config.json" ]; then
        echo "✓ config.json found"
    else
        echo "✗ WARNING: config.json NOT found in original model path"
    fi
else
    echo "✗ ERROR: Original model NOT found at: ${LOCAL_MODEL_PATH}"
    echo "Please set LOCAL_MODEL_PATH to the correct path"
    exit 1
fi

if [ -f "${MCORE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    CHECKPOINT_ITER=$(cat "${MCORE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")
    echo "Latest checkpoint iteration: ${CHECKPOINT_ITER}"
    echo "Will convert: ${MCORE_CHECKPOINT_DIR}/iter_$(printf '%07d' ${CHECKPOINT_ITER})"
else
    echo "WARNING: latest_checkpointed_iteration.txt not found!"
    echo "Listing available iter_* directories:"
    ls -la "${MCORE_CHECKPOINT_DIR}"/iter_* 2>/dev/null || echo "No iter_* directories found"
fi

# ===== 変換実行 =====
echo "=== Starting Conversion ==="

singularity exec --nv --cleanenv \
  -B "${CUDA_HOME}:${CUDA_HOME}" \
  -B "${SWIFT_WORKDIR}:${SWIFT_WORKDIR}" \
  -B "${MS_SWIFT_DIR}:${MS_SWIFT_DIR}" \
  -B "${MEGATRON_LM_PATH}:${MEGATRON_LM_PATH}" \
  -B "${LOCAL_MODEL_PATH}:${LOCAL_MODEL_PATH}" \
  -B "/dev/shm:/dev/shm" \
  --home "${HF_CACHE}:/root" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HF_HOME="${HF_CACHE}" \
  --env HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
  --env HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub" \
  --env TRANSFORMERS_CACHE="${HF_CACHE}/hub" \
  --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  --env TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
  --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
  --env MS_SWIFT_DIR="${MS_SWIFT_DIR}" \
  --env MEGATRON_LM_PATH="${MEGATRON_LM_PATH}" \
  --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  "${SIF_FILE}" bash -lc "
    set -xeuo pipefail
    
    export PATH=\"${CUDA_HOME}/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin:\$PATH\"
    export PYTHONPATH=\"\${MS_SWIFT_DIR}:\${MEGATRON_LM_PATH}:\${PYTHONPATH-}\"
    
    cd \"\${MS_SWIFT_DIR}\"
    
    # megatron export を使用した変換（Mcore-Bridge）
    # ★★★ --model パラメータを追加 ★★★
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NPROC_PER_NODE=8 \
    megatron export \
      --model '${LOCAL_MODEL_PATH}' \
      --model_type qwen3_next \
      --load '${MCORE_CHECKPOINT_DIR}' \
      --save '${OUTPUT_HF_DIR}' \
      --to_hf true \
      --tensor_model_parallel_size 1 \
      --expert_model_parallel_size 8 \
      --pipeline_model_parallel_size 1 \
      --use_cpu_initialization true \
      --test_convert_precision ${TEST_PRECISION}
  "

# ===== 変換結果確認 =====
echo "=== Conversion Complete ==="
echo "Output directory: ${OUTPUT_HF_DIR}"

if [ -d "${OUTPUT_HF_DIR}" ]; then
    echo "Contents of output directory:"
    ls -la "${OUTPUT_HF_DIR}/"
    
    # config.jsonの存在確認
    if [ -f "${OUTPUT_HF_DIR}/config.json" ]; then
        echo "✓ config.json found"
    else
        echo "✗ config.json NOT found"
    fi
    
    # safetensorsファイルの確認
    SAFETENSOR_COUNT=$(find "${OUTPUT_HF_DIR}" -name "*.safetensors" | wc -l)
    echo "Number of safetensors files: ${SAFETENSOR_COUNT}"
else
    echo "ERROR: Output directory not created!"
    exit 1
fi

echo "=== Done ==="