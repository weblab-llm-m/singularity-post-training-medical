#!/bin/bash
# Phase 2: Path Patching - Multi-GPU Parallel Execution for MoE
# 30B MoE model needs 4 GPUs per instance, so we run 2 parallel instances

set -e

BASE_DIR="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B"
cd $BASE_DIR

MODEL_PATH="/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507"
DATA_PATH="Phase1_data_preparation/path_patching_data.jsonl"
OUTPUT_DIR="Phase2_path_patching/results_parallel"
NUM_SAMPLES=${1:-6478}  # Default to all samples (6478)

echo "==================================================="
echo "Path Patching - Parallel Execution (MoE 30B)"
echo "==================================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Total samples: ${NUM_SAMPLES}"
echo "Strategy: 2 parallel processes (4 GPUs each)"
echo ""

# Clean previous results
rm -rf ${OUTPUT_DIR}/process_*/
mkdir -p ${OUTPUT_DIR}/data_chunks
mkdir -p ${OUTPUT_DIR}/process_{0,1}

# Split data into 2 chunks
echo "Splitting data into 2 chunks..."
TOTAL_LINES=$(wc -l < ${DATA_PATH})
SAMPLES_TO_USE=$((NUM_SAMPLES < TOTAL_LINES ? NUM_SAMPLES : TOTAL_LINES))
CHUNK_SIZE=$((SAMPLES_TO_USE / 2))

head -${SAMPLES_TO_USE} ${DATA_PATH} | head -${CHUNK_SIZE} > ${OUTPUT_DIR}/data_chunks/chunk_0.jsonl
head -${SAMPLES_TO_USE} ${DATA_PATH} | tail -$((SAMPLES_TO_USE - CHUNK_SIZE)) > ${OUTPUT_DIR}/data_chunks/chunk_1.jsonl

echo "Chunk 0: $(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_0.jsonl) samples"
echo "Chunk 1: $(wc -l < ${OUTPUT_DIR}/data_chunks/chunk_1.jsonl) samples"
echo ""

# Activate venv
source ${BASE_DIR}/venv/bin/activate

# Launch 2 parallel processes
# Process 0: GPUs 0,1,2,3
echo "Starting Process 0 (GPUs 0-3)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python Phase2_path_patching/path_patching_medical.py \
    --model_path ${MODEL_PATH} \
    --data_path ${OUTPUT_DIR}/data_chunks/chunk_0.jsonl \
    --output_dir ${OUTPUT_DIR}/process_0 \
    --batch_size 1 \
    --sample_num -1 \
    > ${OUTPUT_DIR}/process_0.log 2>&1 &

PID_0=$!
echo $PID_0 > ${OUTPUT_DIR}/process_0.pid
echo "Process 0 started (PID: $PID_0)"
sleep 5

# Process 1: GPUs 4,5,6,7
echo "Starting Process 1 (GPUs 4-7)..."
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python Phase2_path_patching/path_patching_medical.py \
    --model_path ${MODEL_PATH} \
    --data_path ${OUTPUT_DIR}/data_chunks/chunk_1.jsonl \
    --output_dir ${OUTPUT_DIR}/process_1 \
    --batch_size 1 \
    --sample_num -1 \
    > ${OUTPUT_DIR}/process_1.log 2>&1 &

PID_1=$!
echo $PID_1 > ${OUTPUT_DIR}/process_1.pid
echo "Process 1 started (PID: $PID_1)"

echo ""
echo "==================================================="
echo "Both processes launched!"
echo "==================================================="
echo "Process 0 (GPUs 0-3): PID $PID_0"
echo "Process 1 (GPUs 4-7): PID $PID_1"
echo ""
echo "Monitor logs:"
echo "  tail -f ${OUTPUT_DIR}/process_0.log"
echo "  tail -f ${OUTPUT_DIR}/process_1.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Wait for completion:"
echo "  ${BASE_DIR}/Phase2_path_patching/wait_parallel.sh"
echo ""
