#!/bin/bash
# =============================================================
# Phase 2: Path Patching - 8ノード64GPU並列実行 (SLURM)
# medical / reasoning データセットを指定して実行
# Usage:
#   bash submit_all.sh medical            # 5543サンプル (医療用語置換)
#   bash submit_all.sh reasoning          # 2098サンプル (推論表現置換)
#   bash submit_all.sh overlap_medical    # 4393サンプル (重複: 医療のみ置換)
#   bash submit_all.sh overlap_reasoning  # 4393サンプル (重複: 推論のみ置換)
# =============================================================
set -e

DATASET=${1:?"Usage: bash submit_all.sh [medical|reasoning|overlap_medical|overlap_reasoning]"}

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
SCRIPT_DIR="${BASE_DIR}/Phase2_path_patching/8node64gpu"

if [ "${DATASET}" = "medical" ]; then
    DATA_PATH="${BASE_DIR}/Phase1_data_preparation/path_patching_data_medical.jsonl"
    OUTPUT_DIR="${SCRIPT_DIR}/results_medical"
elif [ "${DATASET}" = "reasoning" ]; then
    DATA_PATH="${BASE_DIR}/Phase1_data_preparation/path_patching_data_reasoning.jsonl"
    OUTPUT_DIR="${SCRIPT_DIR}/results_reasoning"
elif [ "${DATASET}" = "overlap_medical" ]; then
    DATA_PATH="${BASE_DIR}/Phase1_data_preparation/path_patching_data_overlap_medical.jsonl"
    OUTPUT_DIR="${SCRIPT_DIR}/results_overlap_medical"
elif [ "${DATASET}" = "overlap_reasoning" ]; then
    DATA_PATH="${BASE_DIR}/Phase1_data_preparation/path_patching_data_overlap_reasoning.jsonl"
    OUTPUT_DIR="${SCRIPT_DIR}/results_overlap_reasoning"
else
    echo "ERROR: Unknown dataset '${DATASET}'. Use 'medical', 'reasoning', 'overlap_medical', or 'overlap_reasoning'."
    exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Data file not found: ${DATA_PATH}"
    exit 1
fi

NUM_NODES=8
GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))  # 64

NODES=(osk-gpu61 osk-gpu62 osk-gpu63 osk-gpu64 osk-gpu65 osk-gpu66 osk-gpu67 osk-gpu68)

echo "==================================================="
echo "Phase 2: Path Patching - 8 Node × 8 GPU = 64 Parallel"
echo "Dataset:    ${DATASET}"
echo "==================================================="
echo "Data:       ${DATA_PATH}"
echo "Output:     ${OUTPUT_DIR}"
echo "Nodes:      ${NODES[*]}"
echo "Total GPUs: ${TOTAL_GPUS}"
echo ""

# -----------------------------------------------------------
# 1. 出力ディレクトリ準備
# -----------------------------------------------------------
mkdir -p ${OUTPUT_DIR}/data_chunks
for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    PROC_ID=$(printf "%02d" $i)
    mkdir -p ${OUTPUT_DIR}/process_${PROC_ID}
done

# -----------------------------------------------------------
# 2. データを64チャンクに分割
# -----------------------------------------------------------
echo "Splitting data into ${TOTAL_GPUS} chunks..."
TOTAL_LINES=$(wc -l < ${DATA_PATH})
echo "  Total lines in data: ${TOTAL_LINES}"

CHUNK_SIZE=$((TOTAL_LINES / TOTAL_GPUS))
REMAINDER=$((TOTAL_LINES % TOTAL_GPUS))

OFFSET=0
for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    PROC_ID=$(printf "%02d" $i)
    if [ $i -lt $REMAINDER ]; then
        THIS_CHUNK=$((CHUNK_SIZE + 1))
    else
        THIS_CHUNK=${CHUNK_SIZE}
    fi
    tail -n +$((OFFSET + 1)) ${DATA_PATH} | head -${THIS_CHUNK} > ${OUTPUT_DIR}/data_chunks/chunk_${PROC_ID}.jsonl
    ACTUAL=$(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_${PROC_ID}.jsonl)
    echo "  Chunk ${PROC_ID}: ${ACTUAL} samples (offset ${OFFSET})"
    OFFSET=$((OFFSET + THIS_CHUNK))
done
echo "  Total distributed: ${OFFSET}"
echo ""

# -----------------------------------------------------------
# 3. 各ノードにSLURMジョブを投入
# -----------------------------------------------------------
echo "Submitting SLURM jobs..."
JOB_IDS=()

for node_idx in $(seq 0 $((NUM_NODES - 1))); do
    NODE=${NODES[$node_idx]}

    JOB_ID=$(sbatch \
        --job-name="pp_${DATASET}_n${node_idx}" \
        --partition=P08317 \
        --nodelist=${NODE} \
        --gres=gpu:8 \
        --ntasks=1 \
        --cpus-per-task=64 \
        --mem=300G \
        --time=72:00:00 \
        --output="${OUTPUT_DIR}/slurm_node${node_idx}_%j.out" \
        --error="${OUTPUT_DIR}/slurm_node${node_idx}_%j.err" \
        --export=ALL,NODE_INDEX=${node_idx},OUTPUT_DIR=${OUTPUT_DIR},BASE_DIR=${BASE_DIR} \
        ${SCRIPT_DIR}/node_job.sh \
        | awk '{print $4}')

    JOB_IDS+=($JOB_ID)
    echo "  Node ${node_idx} (${NODE}): JobID ${JOB_ID} (process_$(printf '%02d' $((node_idx*8)))-$(printf '%02d' $((node_idx*8+7))))"
done

echo ""
echo "==================================================="
echo "All ${NUM_NODES} SLURM jobs submitted!"
echo "==================================================="
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor commands:"
echo "  squeue -u \$(whoami)                           # ジョブ状態確認"
echo "  bash ${SCRIPT_DIR}/monitor_all_nodes.sh        # 進捗確認"
echo ""
echo "完了後のマージ:"
echo "  ${BASE_DIR}/venv/bin/python3 ${SCRIPT_DIR}/merge_64gpu_results.py ${OUTPUT_DIR}"
echo ""
echo "全データセット実行例:"
echo "  bash ${SCRIPT_DIR}/submit_all.sh overlap_medical    # 4393件 (重複: 医療のみ)"
echo "  bash ${SCRIPT_DIR}/submit_all.sh overlap_reasoning  # 4393件 (重複: 推論のみ)"
echo "  bash ${SCRIPT_DIR}/submit_all.sh medical             # 5543件 (全医療)"
echo "  bash ${SCRIPT_DIR}/submit_all.sh reasoning            # 2098件 (推論のみ群)"
echo "==================================================="

# ジョブIDを保存
echo "${JOB_IDS[*]}" > ${OUTPUT_DIR}/slurm_job_ids.txt
