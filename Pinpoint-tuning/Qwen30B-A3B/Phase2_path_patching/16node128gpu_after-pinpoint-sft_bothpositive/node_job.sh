#!/bin/bash
#SBATCH --job-name=path_patching_pps
#SBATCH --partition=P08317
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=72:00:00
# =============================================================
# Per-node job script: 8 GPU processes on this node
# モデル: positive_pinpoint (Pinpoint-SFT後)
# NODE_INDEX, OUTPUT_DIR, BASE_DIR are passed via --export
# =============================================================

echo "==================================================="
echo "Node Job Start: $(hostname)"
echo "Node Index: ${NODE_INDEX}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "==================================================="

cd ${BASE_DIR}

MODEL_PATH="/home/matsuolab/ramen_models/v2_positive_pinpoint/v2-20260227-074545-hf"
GPUS_PER_NODE=8
SCRIPT_PATH="${BASE_DIR}/Phase2_path_patching/1node8gpu/path_patching_medical.py"

# Activate venv and set PYTHONPATH so Phase2_path_patching package is found
source ${BASE_DIR}/venv/bin/activate
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# nvidia-smi for logging
nvidia-smi

# Launch 8 processes (1 per GPU)
PIDS=()
for gpu_idx in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GLOBAL_PROC_ID=$((NODE_INDEX * GPUS_PER_NODE + gpu_idx))
    PROC_ID=$(printf "%03d" ${GLOBAL_PROC_ID})

    CHUNK_FILE="${OUTPUT_DIR}/data_chunks/chunk_${PROC_ID}.jsonl"
    PROC_DIR="${OUTPUT_DIR}/process_${PROC_ID}"
    LOG_FILE="${OUTPUT_DIR}/process_${PROC_ID}.log"

    CHUNK_LINES=$(wc -l < ${CHUNK_FILE})
    echo "Starting Process ${PROC_ID} on GPU ${gpu_idx}: ${CHUNK_LINES} samples"

    CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SCRIPT_PATH} \
        --model_path ${MODEL_PATH} \
        --data_path ${CHUNK_FILE} \
        --output_dir ${PROC_DIR} \
        --batch_size 1 \
        --sample_num -1 \
        > ${LOG_FILE} 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"

    # Stagger to avoid simultaneous model loading
    sleep 15
done

echo ""
echo "All ${GPUS_PER_NODE} processes launched on $(hostname)"
echo "PIDs: ${PIDS[*]}"
echo "Waiting for all processes to complete..."

# Wait for all processes and track exit codes
FAILED=0
for i in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GLOBAL_PROC_ID=$((NODE_INDEX * GPUS_PER_NODE + i))
    PROC_ID=$(printf "%03d" ${GLOBAL_PROC_ID})
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: Process ${PROC_ID} (PID ${PIDS[$i]}) exited with code ${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    else
        echo "Process ${PROC_ID} (PID ${PIDS[$i]}) completed successfully"
    fi
done

echo ""
echo "==================================================="
echo "Node ${NODE_INDEX} ($(hostname)) finished: $(date)"
echo "  Succeeded: $((GPUS_PER_NODE - FAILED))/${GPUS_PER_NODE}"
if [ ${FAILED} -gt 0 ]; then
    echo "  FAILED: ${FAILED}/${GPUS_PER_NODE}"
fi
echo "==================================================="

# Check results files
echo ""
echo "Results check:"
for gpu_idx in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GLOBAL_PROC_ID=$((NODE_INDEX * GPUS_PER_NODE + gpu_idx))
    PROC_ID=$(printf "%03d" ${GLOBAL_PROC_ID})
    PT_FILE="${OUTPUT_DIR}/process_${PROC_ID}/results.pt"
    if [ -f "${PT_FILE}" ]; then
        SIZE=$(du -h "${PT_FILE}" | cut -f1)
        echo "  process_${PROC_ID}/results.pt: ${SIZE} [OK]"
    else
        echo "  process_${PROC_ID}/results.pt: MISSING [FAIL]"
    fi
done

exit ${FAILED}
