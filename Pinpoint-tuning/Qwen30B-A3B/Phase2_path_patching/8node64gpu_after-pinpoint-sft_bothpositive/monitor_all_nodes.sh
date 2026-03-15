#!/bin/bash
# =============================================================
# 8ノード64GPU並列 Path Patching モニタリング (positive_pinpoint)
# Usage: bash monitor_all_nodes.sh [overlap_medical|overlap_reasoning]
# =============================================================

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
SCRIPT_DIR="${BASE_DIR}/Phase2_path_patching/8node64gpu_after-pinpoint-sft_bothpositive"

DATASET=${1:-"overlap_medical"}
OUTPUT_DIR="${SCRIPT_DIR}/results_${DATASET}"

NUM_NODES=8
GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
NODES=(osk-gpu54 osk-gpu55 osk-gpu56 osk-gpu57 osk-gpu58 osk-gpu59 osk-gpu60 osk-gpu62)

echo "==================================================="
echo "Path Patching Monitor - 8 Nodes × 8 GPUs (positive_pinpoint)"
echo "Dataset: ${DATASET}"
echo "Time: $(date)"
echo "==================================================="
echo ""

# SLURM jobs status
echo "--- SLURM Jobs ---"
squeue -u $(whoami) --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null | grep -E "JOBID|DONT-KILL_MI"
echo ""

# Per-process status
COMPLETED=0
RUNNING=0
PENDING=0

echo "--- Process Status ---"
echo "Process | Node       | Samples | Status     | Progress"
echo "--------|------------|---------|------------|--------"

for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    PROC_ID=$(printf "%02d" $i)
    NODE_IDX=$((i / GPUS_PER_NODE))
    NODE=${NODES[$NODE_IDX]}

    # サンプル数
    CHUNK_FILE="${OUTPUT_DIR}/data_chunks/chunk_${PROC_ID}.jsonl"
    if [ -f "${CHUNK_FILE}" ]; then
        CHUNK_SIZE=$(wc -l < ${CHUNK_FILE})
    else
        CHUNK_SIZE="?"
    fi

    # 完了チェック
    PT_FILE="${OUTPUT_DIR}/process_${PROC_ID}/results.pt"
    LOG_FILE="${OUTPUT_DIR}/process_${PROC_ID}.log"

    if [ -f "${PT_FILE}" ]; then
        STATUS="COMPLETED"
        COMPLETED=$((COMPLETED + 1))
        PROGRESS="done"
    elif [ -f "${LOG_FILE}" ] && [ -s "${LOG_FILE}" ]; then
        STATUS="RUNNING"
        RUNNING=$((RUNNING + 1))
        PROGRESS=$(grep -oP 'Batches:.*' "${LOG_FILE}" 2>/dev/null | tail -1 | head -c 40)
        if [ -z "$PROGRESS" ]; then
            PROGRESS=$(tail -1 "${LOG_FILE}" 2>/dev/null | head -c 40)
        fi
    else
        STATUS="PENDING"
        PENDING=$((PENDING + 1))
        PROGRESS="-"
    fi

    printf "  %s    | %-10s | %7s | %-10s | %s\n" "${PROC_ID}" "${NODE}" "${CHUNK_SIZE}" "${STATUS}" "${PROGRESS}"
done

echo ""
echo "--- Summary ---"
echo "  Completed: ${COMPLETED}/${TOTAL_GPUS}"
echo "  Running:   ${RUNNING}/${TOTAL_GPUS}"
echo "  Pending:   ${PENDING}/${TOTAL_GPUS}"
echo ""

if [ ${COMPLETED} -eq ${TOTAL_GPUS} ]; then
    echo "*** ALL PROCESSES COMPLETED! ***"
    echo "Run merge:"
    echo "  ${BASE_DIR}/venv/bin/python3 ${SCRIPT_DIR}/merge_64gpu_results.py ${OUTPUT_DIR}"
fi

echo "==================================================="
