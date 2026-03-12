#!/bin/bash
# Wait for parallel path patching processes and merge results

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
OUTPUT_DIR="${BASE_DIR}/Phase2_path_patching/results_parallel"

echo "Monitoring parallel path patching processes..."
echo ""

# Read PIDs
PID_0=$(cat ${OUTPUT_DIR}/process_0.pid 2>/dev/null)
PID_1=$(cat ${OUTPUT_DIR}/process_1.pid 2>/dev/null)

if [ -z "$PID_0" ] || [ -z "$PID_1" ]; then
    echo "Error: Could not find PID files"
    exit 1
fi

echo "Process 0 (GPUs 0-3): PID $PID_0"
echo "Process 1 (GPUs 4-7): PID $PID_1"
echo ""

# Monitor until both complete
while true; do
    RUNNING_0=$(ps -p $PID_0 > /dev/null 2>&1 && echo "1" || echo "0")
    RUNNING_1=$(ps -p $PID_1 > /dev/null 2>&1 && echo "1" || echo "0")

    if [ "$RUNNING_0" == "0" ] && [ "$RUNNING_1" == "0" ]; then
        echo ""
        echo "Both processes completed!"
        break
    fi

    # Show status
    STATUS=""
    [ "$RUNNING_0" == "1" ] && STATUS="${STATUS}Process 0: Running  " || STATUS="${STATUS}Process 0: Completed  "
    [ "$RUNNING_1" == "1" ] && STATUS="${STATUS}Process 1: Running" || STATUS="${STATUS}Process 1: Completed"

    echo -ne "\r${STATUS}  [$(date +%T)]"
    sleep 10
done

echo ""
echo ""
echo "==================================================="
echo "Merging results..."
echo "==================================================="

# Activate venv and merge results
source ${BASE_DIR}/venv/bin/activate

python - <<'PYTHON_SCRIPT'
import torch
import json
import os
import numpy as np

base_dir = "/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
output_dir = f"{base_dir}/Phase2_path_patching/results_parallel"

# Load results from both processes
results_0 = None
results_1 = None
total_samples = 0

for i in [0, 1]:
    result_file = f"{output_dir}/process_{i}/path_patching_results.pt"
    if os.path.exists(result_file):
        print(f"Loading process_{i} results...")
        data = torch.load(result_file, map_location='cpu')
        if results_0 is None:
            results_0 = data
            total_samples = data.shape[0] if len(data.shape) > 2 else 1
        else:
            results_1 = data
            total_samples += data.shape[0] if len(data.shape) > 2 else 1
    else:
        print(f"Warning: {result_file} not found")

# Merge results
if results_0 is not None and results_1 is not None:
    # Average the results (assuming same shape for layers/heads)
    if len(results_0.shape) == 3:  # [samples, layers, heads]
        merged = torch.cat([results_0, results_1], dim=0)
    else:  # [layers, heads]
        merged = (results_0 + results_1) / 2

    # Save merged results
    merged_path = f"{output_dir}/path_patching_results_merged.pt"
    torch.save(merged, merged_path)
    print(f"Merged results saved to: {merged_path}")
    print(f"Shape: {merged.shape}")
    print(f"Total samples processed: {total_samples}")

    # Also save a summary
    summary = {
        "total_samples": total_samples,
        "num_layers": merged.shape[-2],
        "num_heads": merged.shape[-1],
        "mean_impact": float(merged.mean()),
        "max_impact": float(merged.max()),
        "min_impact": float(merged.min())
    }

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"  Layers: {summary['num_layers']}")
    print(f"  Heads: {summary['num_heads']}")
    print(f"  Mean impact: {summary['mean_impact']:.2f}%")
    print(f"  Max impact: {summary['max_impact']:.2f}%")
else:
    print("Error: Could not load results from both processes")
    exit(1)

PYTHON_SCRIPT

echo ""
echo "==================================================="
echo "Parallel path patching completed!"
echo "==================================================="
echo "Results: ${OUTPUT_DIR}/path_patching_results_merged.pt"
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo ""
