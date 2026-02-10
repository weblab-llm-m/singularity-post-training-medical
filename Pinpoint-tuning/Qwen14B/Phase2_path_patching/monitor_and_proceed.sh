#!/bin/bash

# Monitor Phase 2 and auto-proceed to Phase 3
LOG_FILE="phase2_monitor.log"
RESULTS_DIR="results_8gpu_parallel"

echo "=== Phase 2 Monitor Started: $(date) ===" | tee -a $LOG_FILE

while true; do
    # Check running processes
    RUNNING=$(ps aux | grep "path_patching_medical.py" | grep -v grep | wc -l)
    echo "[$(date '+%H:%M:%S')] Running processes: $RUNNING" | tee -a $LOG_FILE
    
    if [ $RUNNING -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] All processes completed!" | tee -a $LOG_FILE
        
        # Check output files
        OUTPUT_COUNT=$(ls $RESULTS_DIR/gpu_*/chunk_*.jsonl 2>/dev/null | wc -l)
        echo "[$(date '+%H:%M:%S')] Output files: $OUTPUT_COUNT/8" | tee -a $LOG_FILE
        
        if [ $OUTPUT_COUNT -ge 7 ]; then
            echo "[$(date '+%H:%M:%S')] Phase 2 COMPLETE! Proceeding to Phase 3..." | tee -a $LOG_FILE
            exit 0
        else
            echo "[$(date '+%H:%M:%S')] ERROR: Missing output files" | tee -a $LOG_FILE
            exit 1
        fi
    fi
    
    # Show progress every 30 minutes
    for i in 0 1 2 3 4 5 6 7; do
        BATCH_INFO=$(grep "Batches:" $RESULTS_DIR/gpu_$i.log 2>/dev/null | tail -1)
        if [ -n "$BATCH_INFO" ]; then
            echo "[$(date '+%H:%M:%S')] GPU $i: $BATCH_INFO" | tee -a $LOG_FILE
        fi
    done
    
    # Wait 30 minutes
    sleep 1800
done
