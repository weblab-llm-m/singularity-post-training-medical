#!/bin/bash
# Monitor for parallel path patching (6478 samples total)

OUTPUT_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B/Phase2_path_patching/results_parallel"

# Check if PIDs exist
PID_0=$(cat ${OUTPUT_DIR}/process_0.pid 2>/dev/null)
PID_1=$(cat ${OUTPUT_DIR}/process_1.pid 2>/dev/null)

if [ -z "$PID_0" ] && [ -z "$PID_1" ]; then
    echo "Error: No PID files found. Has the job been started?"
    exit 1
fi

# Check process status
RUNNING_0=$(ps -p $PID_0 > /dev/null 2>&1 && echo "Running" || echo "Stopped")
RUNNING_1=$(ps -p $PID_1 > /dev/null 2>&1 && echo "Running" || echo "Stopped")

# Count completed samples
SAMPLES_0=$(ls ${OUTPUT_DIR}/process_0/*.pt 2>/dev/null | wc -l)
SAMPLES_1=$(ls ${OUTPUT_DIR}/process_1/*.pt 2>/dev/null | wc -l)
TOTAL_SAMPLES=$((SAMPLES_0 + SAMPLES_1))
CHUNK_0_SIZE=$(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_0.jsonl 2>/dev/null || echo 0)
CHUNK_1_SIZE=$(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_1.jsonl 2>/dev/null || echo 0)
TOTAL_TO_PROCESS=$((CHUNK_0_SIZE + CHUNK_1_SIZE))

echo "==================================================="
echo "Path Patching Parallel Monitor ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "==================================================="
echo ""
echo "Process 0 (GPUs 0-3): $RUNNING_0 (PID: $PID_0)"
echo "  Progress: ${SAMPLES_0}/${CHUNK_0_SIZE} samples"
echo ""
echo "Process 1 (GPUs 4-7): $RUNNING_1 (PID: $PID_1)"
echo "  Progress: ${SAMPLES_1}/${CHUNK_1_SIZE} samples"
echo ""
echo "Total: ${TOTAL_SAMPLES}/${TOTAL_TO_PROCESS} samples completed"
if [ "$TOTAL_TO_PROCESS" -gt 0 ]; then
    PERCENT=$((TOTAL_SAMPLES * 100 / TOTAL_TO_PROCESS))
    echo "Overall progress: ${PERCENT}%"
fi
echo ""

# Show latest progress from logs
echo "--- Process 0 Latest ---"
tail -5 ${OUTPUT_DIR}/process_0.log 2>/dev/null | grep -E "Sample|Progress|Error" | tail -2 || echo "(waiting for output...)"

echo ""
echo "--- Process 1 Latest ---"
tail -5 ${OUTPUT_DIR}/process_1.log 2>/dev/null | grep -E "Sample|Progress|Error" | tail -2 || echo "(waiting for output...)"

echo ""
echo "--- GPU Memory Usage ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -8

echo ""
echo "==================================================="
if [ "$RUNNING_0" == "Stopped" ] && [ "$RUNNING_1" == "Stopped" ]; then
    echo "Both processes have stopped."
    if [ "$TOTAL_SAMPLES" -eq "$TOTAL_TO_PROCESS" ]; then
        echo "STATUS: COMPLETED"
    else
        echo "STATUS: Check logs for errors"
    fi
else
    echo "STATUS: Running..."
fi
echo "==================================================="
