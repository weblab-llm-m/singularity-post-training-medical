#!/bin/bash
# Phase 2: Path Patching - 8-GPU Parallel Execution (1 GPU per process)
# Qwen3-30B-A3B fits on a single H100 (~75GB VRAM)

set -e

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
cd $BASE_DIR

MODEL_PATH="/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507"
DATA_PATH="${2:-Phase1_data_preparation/path_patching_data.jsonl}"
OUTPUT_DIR="${3:-Phase2_path_patching/results_parallel}"
NUM_GPUS=8
NUM_SAMPLES=${1:-6478}

echo "==================================================="
echo "Path Patching - 8-GPU Parallel Execution"
echo "==================================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Total samples: ${NUM_SAMPLES}"
echo "Strategy: ${NUM_GPUS} parallel processes (1 GPU each)"
echo ""

# Clean previous results
rm -rf ${OUTPUT_DIR}/process_*/
rm -rf ${OUTPUT_DIR}/data_chunks/
mkdir -p ${OUTPUT_DIR}/data_chunks

for i in $(seq 0 $((NUM_GPUS - 1))); do
    mkdir -p ${OUTPUT_DIR}/process_${i}
done

# Split data into N chunks
echo "Splitting data into ${NUM_GPUS} chunks..."
TOTAL_LINES=$(wc -l < ${DATA_PATH})
SAMPLES_TO_USE=$((NUM_SAMPLES < TOTAL_LINES ? NUM_SAMPLES : TOTAL_LINES))
CHUNK_SIZE=$((SAMPLES_TO_USE / NUM_GPUS))
REMAINDER=$((SAMPLES_TO_USE % NUM_GPUS))

OFFSET=0
for i in $(seq 0 $((NUM_GPUS - 1))); do
    # Last chunk gets the remainder
    if [ $i -lt $REMAINDER ]; then
        THIS_CHUNK=$((CHUNK_SIZE + 1))
    else
        THIS_CHUNK=${CHUNK_SIZE}
    fi
    tail -n +$((OFFSET + 1)) ${DATA_PATH} | head -${THIS_CHUNK} > ${OUTPUT_DIR}/data_chunks/chunk_${i}.jsonl
    ACTUAL=$(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_${i}.jsonl)
    echo "  Chunk ${i}: ${ACTUAL} samples (offset ${OFFSET})"
    OFFSET=$((OFFSET + THIS_CHUNK))
done
echo ""

# Activate venv
source ${BASE_DIR}/venv/bin/activate

# Launch 8 parallel processes (1 GPU each)
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Starting Process ${i} (GPU ${i})..."
    CUDA_VISIBLE_DEVICES=${i} nohup python Phase2_path_patching/path_patching_medical.py \
        --model_path ${MODEL_PATH} \
        --data_path ${OUTPUT_DIR}/data_chunks/chunk_${i}.jsonl \
        --output_dir ${OUTPUT_DIR}/process_${i} \
        --batch_size 1 \
        --sample_num -1 \
        > ${OUTPUT_DIR}/process_${i}.log 2>&1 &

    PID=$!
    echo $PID > ${OUTPUT_DIR}/process_${i}.pid
    echo "  PID: $PID"

    # Stagger launches to avoid simultaneous model loading
    sleep 10
done

echo ""
echo "==================================================="
echo "All ${NUM_GPUS} processes launched!"
echo "==================================================="
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PID=$(cat ${OUTPUT_DIR}/process_${i}.pid)
    CHUNK_SIZE=$(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_${i}.jsonl)
    echo "  Process ${i} (GPU ${i}): PID ${PID}, ${CHUNK_SIZE} samples"
done
echo ""
echo "Monitor: bash Phase2_path_patching/monitor_parallel.sh"
echo "GPU:     watch -n 5 nvidia-smi"
echo "==================================================="
