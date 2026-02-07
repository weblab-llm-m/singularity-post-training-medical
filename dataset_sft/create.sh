#!/bin/bash
#SBATCH --job-name=sft_datagen_8node
#SBATCH --partition=P08317
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=120
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ===========================================================
# 8ノード並列SFTデータ生成
#
# 仕組み:
#   SLURM Job Array (0-7) で8ジョブを投入。
#   各ノードが独立して:
#     1. vLLMサーバーを起動
#     2. データを8分割し自分のシャードのみ処理
#     3. output/sft_igakuqa_shard{0-7}.jsonl に書き出し
#
#   全ジョブ完了後、merge_shards.sh でマージ。
# ===========================================================

set -euo pipefail

SHARD_INDEX=${SHARD_INDEX:?"SHARD_INDEX must be set via submit_all.sh"}
NUM_SHARDS=8
CONFIG_NAME="${1:-config}"
CONFIG_FILE="conf/${CONFIG_NAME}.yaml"


bash /home/matsuolab/scripts/cleanup_ram.sh
# 残留プロセスのクリーンアップ
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5
# GPU メモリ解放確認
nvidia-smi --gpu-reset 2>/dev/null || true

echo "========================================"
echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} on $(hostname)"
echo "========================================"

# --- 環境変数 ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_DISABLE_USAGE_STATS=1
export NCCL_DEBUG=WARN

module purge
module load cuda/12.8

# --- YAML値取得 ---
get_yaml() {
    grep -v '^\s*#' "$CONFIG_FILE" | grep "^\s*${1}:" | head -1 | \
        sed "s/.*${1}:\s*//" | sed 's/^"\(.*\)"$/\1/' | sed 's/\s*#.*//' | xargs
}

MODEL_NAME=$(get_yaml "model_name")
TP_SIZE=$(get_yaml "tensor_parallel_size")
PP_SIZE=$(get_yaml "pipeline_parallel_size")
GPU_MEM=$(get_yaml "gpu_memory_utilization")
MAX_LEN=$(get_yaml "max_model_len")
EXTRA=$(get_yaml "extra_args")
WAIT_MAX=$(get_yaml "waittime")

# ポート: シャード毎にずらす（同一ノード対策）
PORT=$((8001 + SHARD_INDEX))

# --- Conda ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sft_datagen

# --- HF ---
export HF_HOME="${HOME}/.cache/huggingface"
export HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || echo "")
export HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

mkdir -p logs output data

# ===========================================================
# [1/3] データ前処理（1回だけ、排他ロック付き）
# ===========================================================
INPUT_PATH=$(get_yaml "input_path")
LOCKFILE="/tmp/sft_datagen_prepare.lock"

if [ ! -f "$INPUT_PATH" ]; then
    (
        flock -n 200 || { echo "  別シャードが前処理中、待機..."; flock 200; }
        if [ ! -f "$INPUT_PATH" ]; then
            echo "[1/3] データ前処理中..."
            python prepare_data.py --output "$INPUT_PATH"
        fi
    ) 200>"$LOCKFILE"
fi
echo "[1/3] データ: ${INPUT_PATH} ($(wc -l < "$INPUT_PATH") 件)"

# ===========================================================
# [2/3] vLLMサーバー起動
# ===========================================================
echo "[2/3] vLLMサーバー起動 (port=${PORT})..."

vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --disable-custom-all-reduce \
    --trust-remote-code \
    $EXTRA \
    > "logs/vllm_shard${SHARD_INDEX}_${SLURM_JOB_ID}log" 2>&1 &

VLLM_PID=$!

# ヘルスチェック
elapsed=0
while true; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  vLLM準備完了 (${elapsed}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM異常終了 (shard=${SHARD_INDEX})"
        tail -30 "logs/vllm_shard${SHARD_INDEX}_${SLURM_JOB_ID}log"
        exit 1
    fi
    if [ $elapsed -ge $WAIT_MAX ]; then
        echo "ERROR: vLLMタイムアウト (shard=${SHARD_INDEX})"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 20
    elapsed=$((elapsed + 20))
done

# ===========================================================
# [3/3] SFTデータ生成（自分のシャードのみ）
# ===========================================================
echo "[3/3] SFTデータ生成 (shard=${SHARD_INDEX}/${NUM_SHARDS})..."

python generate_sft.py \
    --config-name="$CONFIG_NAME" \
    +shard_index=${SHARD_INDEX} \
    +num_shards=${NUM_SHARDS} \
    base_url="http://localhost:${PORT}/v1" \
    hydra.output_subdir=null \
    hydra.run.dir=. \
    hydra.job.chdir=false \
    hydra/job_logging=none \
    hydra/hydra_logging=none \
    2>&1 | tee "logs/generate_shard${SHARD_INDEX}_${SLURM_JOB_ID}log"

# --- クリーンアップ ---
kill $VLLM_PID 2>/dev/null || true

# --- サマリー ---
OUTPUT_BASE=$(get_yaml "output_path")
STEM="${OUTPUT_BASE%.*}"
EXT="${OUTPUT_BASE##*.}"
SHARD_FILE="${STEM}_shard${SHARD_INDEX}.${EXT}"

if [ -f "$SHARD_FILE" ]; then
    COUNT=$(wc -l < "$SHARD_FILE")
    echo "Shard ${SHARD_INDEX} 完了: ${COUNT} 件 → ${SHARD_FILE}"
fi