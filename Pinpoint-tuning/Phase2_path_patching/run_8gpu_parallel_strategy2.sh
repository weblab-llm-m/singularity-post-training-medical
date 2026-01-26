#!/bin/bash

# Phase 2: Path Patching - 8 GPU Parallel Execution
# Strategy 2 (Medical -> Generic) - 200 samples

cd /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase2_path_patching

# Clean previous results
rm -rf results_8gpu_parallel/gpu_*/
mkdir -p results_8gpu_parallel/gpu_{0..7}

# Launch 8 parallel processes
for gpu_id in {0..7}; do
    echo "Starting GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id nohup /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/venv/bin/python3 path_patching_medical.py         --model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B         --data_file results_8gpu_parallel/data_chunks/chunk_0${gpu_id}.jsonl         --output_dir results_8gpu_parallel/gpu_${gpu_id}         --batch_size 1         --extract_attention         > results_8gpu_parallel/gpu_${gpu_id}.log 2>&1 &
    
    echo $! > results_8gpu_parallel/gpu_${gpu_id}.pid
    echo "GPU $gpu_id started (PID: $(cat results_8gpu_parallel/gpu_${gpu_id}.pid))"
done

echo "All 8 GPUs launched. Monitor with: tail -f results_8gpu_parallel/gpu_*.log"
