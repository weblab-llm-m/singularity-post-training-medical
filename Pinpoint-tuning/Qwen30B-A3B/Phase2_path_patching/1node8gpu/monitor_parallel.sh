#!/bin/bash
# Monitor for 8-GPU parallel path patching

OUTPUT_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B/Phase2_path_patching/results_parallel"
NUM_GPUS=8

echo "==================================================="
echo "Path Patching Monitor ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "==================================================="
echo ""

TOTAL_COMPLETED=0
TOTAL_TO_PROCESS=0
NUM_RUNNING=0
NUM_FINISHED=0

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PID=$(cat ${OUTPUT_DIR}/process_${i}.pid 2>/dev/null || echo "")
    CHUNK_SIZE=$(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_${i}.jsonl 2>/dev/null || echo 0)
    TOTAL_TO_PROCESS=$((TOTAL_TO_PROCESS + CHUNK_SIZE))

    # Check process status
    if [ -n "$PID" ] && ps -p $PID > /dev/null 2>&1; then
        STATUS="Running"
        NUM_RUNNING=$((NUM_RUNNING + 1))
    else
        STATUS="Stopped"
        NUM_FINISHED=$((NUM_FINISHED + 1))
    fi

    # Extract progress from log (count "Batches:" completions)
    COMPLETED=0
    if [ -f "${OUTPUT_DIR}/process_${i}.log" ]; then
        # Count completed batches from tqdm output
        LAST_BATCH=$(grep -oP 'Batches:\s+\d+%\|[^|]+\|\s+\K\d+(?=/)' ${OUTPUT_DIR}/process_${i}.log 2>/dev/null | tail -1)
        if [ -n "$LAST_BATCH" ]; then
            COMPLETED=$LAST_BATCH
        fi
        # Check if results.pt was saved (process complete)
        if [ -f "${OUTPUT_DIR}/process_${i}/results.pt" ]; then
            COMPLETED=$CHUNK_SIZE
        fi
    fi
    TOTAL_COMPLETED=$((TOTAL_COMPLETED + COMPLETED))

    printf "  Process %d (GPU %d): %-7s  %4d/%4d samples\n" $i $i "$STATUS" $COMPLETED $CHUNK_SIZE
done

echo ""
echo "--- Summary ---"
echo "  Running: ${NUM_RUNNING} / ${NUM_GPUS}"
echo "  Total:   ${TOTAL_COMPLETED} / ${TOTAL_TO_PROCESS} samples"
if [ "$TOTAL_TO_PROCESS" -gt 0 ]; then
    PERCENT=$((TOTAL_COMPLETED * 100 / TOTAL_TO_PROCESS))
    echo "  Progress: ${PERCENT}%"
fi

# Completed processes with results.pt
RESULTS_COUNT=$(ls ${OUTPUT_DIR}/process_*/results.pt 2>/dev/null | wc -l)
echo "  Results files: ${RESULTS_COUNT} / ${NUM_GPUS}"

echo ""
echo "--- GPU Memory ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null

echo ""
echo "==================================================="
if [ "$NUM_RUNNING" -eq 0 ]; then
    if [ "$RESULTS_COUNT" -eq "$NUM_GPUS" ]; then
        echo "STATUS: ALL COMPLETED"
    else
        echo "STATUS: Check logs for errors"
    fi
else
    echo "STATUS: Running (${NUM_RUNNING}/${NUM_GPUS} active)"
fi
echo "==================================================="
