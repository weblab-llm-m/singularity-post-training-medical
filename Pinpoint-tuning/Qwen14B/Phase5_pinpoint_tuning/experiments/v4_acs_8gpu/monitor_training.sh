#!/bin/bash
# Monitor SPT Training Progress - ACS Data v1

OUTPUT_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v4_acs_data_8gpu/spt_output"
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "=========================================================================="
echo "SPT Training Monitor - ACS Data v1"
echo "=========================================================================="
echo ""

if [ ! -f "${LOG_FILE}" ]; then
    echo "Training log not found: ${LOG_FILE}"
    echo "Training may not have started yet."
    exit 1
fi

echo "Log file: ${LOG_FILE}"
echo ""

echo "=== Latest Training Progress ==="
tail -50 "${LOG_FILE}"

echo ""
echo "=== Training Statistics ==="
echo "Total log lines: $(wc -l < ${LOG_FILE})"
echo ""

if grep -q "'loss':" "${LOG_FILE}"; then
    echo "Loss progression:"
    grep "'loss':" "${LOG_FILE}" | tail -10
    echo ""
    echo "Initial loss: $(grep -m 1 "'loss':" ${LOG_FILE} | grep -oP "'loss': \K[0-9.]+" || echo "N/A")"
    echo "Latest loss: $(grep "'loss':" ${LOG_FILE} | tail -1 | grep -oP "'loss': \K[0-9.]+" || echo "N/A")"
fi

echo ""
echo "=== GPU Usage ==="
nvidia-smi

echo ""
echo "=========================================================================="
