#!/bin/bash

# 8-GPU parallel Phase 2 execution
# Each process runs on a dedicated GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Phase 2: Medical Path Patching (8-GPU Parallel)"
echo "============================================================"

# Configuration
DATA_PATH="Phase1_data_preparation/medical_path_patching_enhanced.jsonl"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
OUTPUT_BASE="Phase2_path_patching/results_8gpu_parallel"
BATCH_SIZE=1
NUM_GPUS=8

# Create directories
mkdir -p "$OUTPUT_BASE"
TEMP_DATA_DIR="${OUTPUT_BASE}/data_chunks"
mkdir -p "$TEMP_DATA_DIR"

# Count total lines
TOTAL_LINES=$(wc -l < "$DATA_PATH")
echo "Total samples: $TOTAL_LINES"

# Calculate lines per chunk
LINES_PER_CHUNK=$(( (TOTAL_LINES + NUM_GPUS - 1) / NUM_GPUS ))
echo "Lines per chunk: ~$LINES_PER_CHUNK"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check if data already split
if [ ! -d "$TEMP_DATA_DIR" ] || [ -z "$(ls -A $TEMP_DATA_DIR 2>/dev/null)" ]; then
    echo "Splitting data into $NUM_GPUS chunks..."

    # Split the JSONL file
    split -l $LINES_PER_CHUNK -d -a 2 "$DATA_PATH" "${TEMP_DATA_DIR}/chunk_"

    # Rename chunks to .jsonl
    for file in "${TEMP_DATA_DIR}"/chunk_*; do
        [ -f "$file" ] && mv "$file" "${file}.jsonl"
    done

    echo "Data split complete."
fi

echo ""
echo "Chunks created:"
ls -lh "${TEMP_DATA_DIR}"/chunk_*.jsonl
echo ""

# Function to run a chunk on specific GPU
run_chunk_on_gpu() {
    local gpu_id=$1
    local chunk_file=$2
    local output_dir="${OUTPUT_BASE}/gpu_${gpu_id}"

    echo "Starting GPU $gpu_id: $(basename $chunk_file)"

    mkdir -p "$output_dir"

    # Set CUDA_VISIBLE_DEVICES to use only one GPU
    CUDA_VISIBLE_DEVICES=$gpu_id \
    /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/python3 \
        Phase2_path_patching/path_patching_medical.py \
        --data_path "$chunk_file" \
        --model_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --batch_size $BATCH_SIZE \
        --sample_num -1 \
        > "${OUTPUT_BASE}/gpu_${gpu_id}.log" 2>&1 &

    local pid=$!
    echo "GPU $gpu_id started with PID $pid"
    echo "$pid" > "${OUTPUT_BASE}/gpu_${gpu_id}.pid"
}

# Launch parallel chunks on each GPU
echo "============================================================"
echo "Launching $NUM_GPUS parallel processes (1 per GPU)..."
echo "============================================================"

gpu_id=0
for chunk_file in "${TEMP_DATA_DIR}"/chunk_*.jsonl; do
    if [ $gpu_id -lt $NUM_GPUS ]; then
        run_chunk_on_gpu $gpu_id "$chunk_file"
        gpu_id=$((gpu_id + 1))
        sleep 15  # Stagger starts to avoid resource contention
    fi
done

echo ""
echo "============================================================"
echo "All $NUM_GPUS GPU processes launched!"
echo "============================================================"
echo ""
echo "Monitor progress:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU $i: tail -f ${OUTPUT_BASE}/gpu_${i}.log"
done
echo ""
echo "Check running processes:"
echo "  ps aux | grep path_patching_medical"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Monitor script:"
echo "  bash scripts/monitor_8gpu_progress.sh"
echo ""
