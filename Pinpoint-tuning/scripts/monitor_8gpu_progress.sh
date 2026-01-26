#!/bin/bash

# Monitor 8-GPU parallel Phase 2 execution progress

OUTPUT_BASE="Phase2_path_patching/results_8gpu_parallel"

echo "============================================================"
echo "Phase 2: 8-GPU Parallel Execution - Progress Monitor"
echo "============================================================"
echo ""

# Check if processes are running
echo "Process Status:"
echo "---------------"
for i in $(seq 0 7); do
    if [ -f "${OUTPUT_BASE}/gpu_${i}.pid" ]; then
        pid=$(cat "${OUTPUT_BASE}/gpu_${i}.pid")
        if ps -p $pid > /dev/null 2>&1; then
            echo "✓ GPU $i (PID $pid): Running"
        else
            echo "✗ GPU $i (PID $pid): Completed or stopped"
        fi
    else
        echo "- GPU $i: Not started"
    fi
done

echo ""
echo "Latest Progress from Logs:"
echo "--------------------------"

for i in $(seq 0 7); do
    if [ -f "${OUTPUT_BASE}/gpu_${i}.log" ]; then
        echo ""
        echo "GPU $i:"
        tail -10 "${OUTPUT_BASE}/gpu_${i}.log" | grep -E "(Batches|Path patching layers|Processed)" | tail -1
    fi
done

echo ""
echo "GPU Memory Usage:"
echo "-----------------"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "============================================================"
echo "Commands:"
for i in $(seq 0 7); do
    echo "  GPU $i log: tail -f ${OUTPUT_BASE}/gpu_${i}.log"
done
echo "  Kill all: kill \$(cat ${OUTPUT_BASE}/gpu_*.pid 2>/dev/null)"
echo "============================================================"
