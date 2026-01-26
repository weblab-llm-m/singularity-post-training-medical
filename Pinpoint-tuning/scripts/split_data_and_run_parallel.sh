#!/bin/bash

# Split data and run Phase 2 in parallel

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
NUM_CHUNKS=2  # Number of parallel processes (adjusted for GPU memory)

# Create directories
mkdir -p "$OUTPUT_BASE"
TEMP_DATA_DIR="${OUTPUT_BASE}/data_chunks"
mkdir -p "$TEMP_DATA_DIR"

echo "Splitting data into $NUM_CHUNKS chunks..."

# Count total lines
TOTAL_LINES=$(wc -l < "$DATA_PATH")
echo "Total samples: $TOTAL_LINES"

# Calculate lines per chunk
LINES_PER_CHUNK=$(( (TOTAL_LINES + NUM_CHUNKS - 1) / NUM_CHUNKS ))
echo "Lines per chunk: ~$LINES_PER_CHUNK"
echo ""

# Split the JSONL file
split -l $LINES_PER_CHUNK -d -a 2 "$DATA_PATH" "${TEMP_DATA_DIR}/chunk_"

# Rename chunks to .jsonl
for file in "${TEMP_DATA_DIR}"/chunk_*; do
    mv "$file" "${file}.jsonl"
done

echo "Data split complete. Chunks created:"
ls -lh "${TEMP_DATA_DIR}"/chunk_*.jsonl
echo ""

# Function to run a chunk
run_chunk() {
    local chunk_file=$1
    local chunk_id=$2
    local output_dir="${OUTPUT_BASE}/chunk_${chunk_id}"

    echo "Starting chunk $chunk_id: $(basename $chunk_file)"

    mkdir -p "$output_dir"

    /home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/venv/bin/python3 \
        Phase2_path_patching/path_patching_medical.py \
        --data_path "$chunk_file" \
        --model_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --batch_size $BATCH_SIZE \
        --sample_num -1 \
        > "${OUTPUT_BASE}/chunk_${chunk_id}.log" 2>&1 &

    local pid=$!
    echo "Chunk $chunk_id started with PID $pid"
    echo "$pid" > "${OUTPUT_BASE}/chunk_${chunk_id}.pid"
}

# Launch parallel chunks
echo "============================================================"
echo "Launching parallel processes..."
echo "============================================================"

chunk_id=0
for chunk_file in "${TEMP_DATA_DIR}"/chunk_*.jsonl; do
    run_chunk "$chunk_file" "$chunk_id"
    chunk_id=$((chunk_id + 1))
    sleep 10  # Stagger starts to avoid resource contention
done

echo ""
echo "============================================================"
echo "All $NUM_CHUNKS chunks launched!"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f ${OUTPUT_BASE}/chunk_0.log"
echo "  tail -f ${OUTPUT_BASE}/chunk_1.log"
echo "  tail -f ${OUTPUT_BASE}/chunk_2.log"
echo "  tail -f ${OUTPUT_BASE}/chunk_3.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep path_patching_medical"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Results will be in:"
echo "  ${OUTPUT_BASE}/chunk_*/results.pt"
echo ""
