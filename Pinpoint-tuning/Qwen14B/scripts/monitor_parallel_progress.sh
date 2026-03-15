#!/bin/bash

# Monitor parallel Phase 2 execution progress

OUTPUT_BASE="Phase2_path_patching/results_parallel"

echo "============================================================"
echo "Phase 2 Parallel Execution - Progress Monitor"
echo "============================================================"
echo ""

# Check if processes are running
echo "Process Status:"
echo "---------------"
if [ -f "${OUTPUT_BASE}/chunk_0.pid" ]; then
    pid0=$(cat "${OUTPUT_BASE}/chunk_0.pid")
    if ps -p $pid0 > /dev/null 2>&1; then
        echo "✓ Chunk 0 (PID $pid0): Running"
    else
        echo "✗ Chunk 0 (PID $pid0): Completed or stopped"
    fi
fi

if [ -f "${OUTPUT_BASE}/chunk_1.pid" ]; then
    pid1=$(cat "${OUTPUT_BASE}/chunk_1.pid")
    if ps -p $pid1 > /dev/null 2>&1; then
        echo "✓ Chunk 1 (PID $pid1): Running"
    else
        echo "✗ Chunk 1 (PID $pid1): Completed or stopped"
    fi
fi

echo ""
echo "Latest Progress from Logs:"
echo "--------------------------"

echo ""
echo "Chunk 0:"
if [ -f "${OUTPUT_BASE}/chunk_0.log" ]; then
    tail -5 "${OUTPUT_BASE}/chunk_0.log" | grep -E "(Batches|Path patching layers)" | tail -1
else
    echo "  Log file not found"
fi

echo ""
echo "Chunk 1:"
if [ -f "${OUTPUT_BASE}/chunk_1.log" ]; then
    tail -5 "${OUTPUT_BASE}/chunk_1.log" | grep -E "(Batches|Path patching layers)" | tail -1
else
    echo "  Log file not found"
fi

echo ""
echo "GPU Usage:"
echo "----------"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | grep -E "(pid|path_patching)"

echo ""
echo "============================================================"
echo "Commands:"
echo "  View Chunk 0 log: tail -f ${OUTPUT_BASE}/chunk_0.log"
echo "  View Chunk 1 log: tail -f ${OUTPUT_BASE}/chunk_1.log"
echo "  Kill all processes: kill \$(cat ${OUTPUT_BASE}/chunk_*.pid)"
echo "============================================================"
