#!/bin/bash
# Monitor training progress

OUTPUT_DIR="/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v2_megatron_config/spt_321heads_output"
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "Monitoring SPT Training (321 heads, 2048 tokens)..."
echo "=================================================="
echo ""

while true; do
    if [ -f "${LOG_FILE}" ]; then
        # Get latest progress
        PROGRESS=$(grep -oP '\d+%\|' "${LOG_FILE}" | tail -1 | grep -oP '\d+')
        STEP=$(grep -oP '\s+\d+/14\s' "${LOG_FILE}" | tail -1 | tr -d ' ')

        if [ ! -z "$PROGRESS" ]; then
            echo "[$(date +%H:%M:%S)] Progress: ${STEP} (${PROGRESS}%)"
        fi

        # Check if training completed
        if grep -q "Training completed" "${LOG_FILE}" 2>/dev/null || grep -q "SPT Training Completed" "../training_2048tokens_nohup.log" 2>/dev/null; then
            echo ""
            echo "=========================================="
            echo "Training Completed!"
            echo "=========================================="

            # Show final loss
            INITIAL_LOSS=$(grep -m 1 "'loss':" "${LOG_FILE}" | grep -oP "'loss': \K[0-9.]+" || echo "N/A")
            FINAL_LOSS=$(grep "'loss':" "${LOG_FILE}" | tail -1 | grep -oP "'loss': \K[0-9.]+" || echo "N/A")

            echo "Initial loss: ${INITIAL_LOSS}"
            echo "Final loss: ${FINAL_LOSS}"

            if [ "$INITIAL_LOSS" != "N/A" ] && [ "$FINAL_LOSS" != "N/A" ]; then
                REDUCTION=$(python3 -c "print(f'{(1 - $FINAL_LOSS/$INITIAL_LOSS)*100:.1f}%')" 2>/dev/null || echo "N/A")
                echo "Loss reduction: ${REDUCTION}"
            fi

            echo ""
            echo "Output directory: ${OUTPUT_DIR}"
            break
        fi
    else
        echo "[$(date +%H:%M:%S)] Waiting for training to start..."
    fi

    sleep 30
done
