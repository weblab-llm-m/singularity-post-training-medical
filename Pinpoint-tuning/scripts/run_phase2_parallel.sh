#!/bin/bash

# Parallel Phase 2 execution script
# Splits data into chunks and processes them in parallel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Phase 2: Medical Path Patching (Parallel Execution)"
echo "============================================================"

# Configuration
DATA_PATH="Phase1_data_preparation/medical_path_patching_enhanced.jsonl"
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
OUTPUT_BASE="Phase2_path_patching/results_parallel"
BATCH_SIZE=1
TOTAL_SAMPLES=1761
NUM_CHUNKS=4  # Number of parallel processes

# Calculate samples per chunk
SAMPLES_PER_CHUNK=$((TOTAL_SAMPLES / NUM_CHUNKS))
echo "Total samples: $TOTAL_SAMPLES"
echo "Number of chunks: $NUM_CHUNKS"
echo "Samples per chunk: ~$SAMPLES_PER_CHUNK"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to run a chunk
run_chunk() {
    local chunk_id=$1
    local start_idx=$2
    local num_samples=$3
    local output_dir="${OUTPUT_BASE}/chunk_${chunk_id}"

    echo "Starting chunk $chunk_id (samples $start_idx to $((start_idx + num_samples - 1)))"

    mkdir -p "$output_dir"

    /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/python3 \
        Phase2_path_patching/path_patching_medical.py \
        --data_path "$DATA_PATH" \
        --model_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --batch_size $BATCH_SIZE \
        --sample_num $num_samples \
        --start_idx $start_idx \
        > "${OUTPUT_BASE}/chunk_${chunk_id}.log" 2>&1 &

    echo "Chunk $chunk_id started with PID $!"
}

# Launch parallel chunks
echo "Launching parallel processes..."
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    start_idx=$((i * SAMPLES_PER_CHUNK))

    # Last chunk gets remaining samples
    if [ $i -eq $((NUM_CHUNKS - 1)) ]; then
        num_samples=$((TOTAL_SAMPLES - start_idx))
    else
        num_samples=$SAMPLES_PER_CHUNK
    fi

    run_chunk $i $start_idx $num_samples
    sleep 5  # Stagger starts to avoid resource contention
done

echo ""
echo "All chunks launched. Monitor progress with:"
echo "  tail -f ${OUTPUT_BASE}/chunk_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep path_patching_medical"
echo ""
