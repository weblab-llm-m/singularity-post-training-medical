#!/bin/bash

# Watch Phase 2 and auto-execute Phase 3
CHECK_INTERVAL=1800  # 30 minutes

while true; do
    RUNNING=$(ps aux | grep "path_patching_medical.py" | grep -v grep | wc -l)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$TIMESTAMP] Checking... Running processes: $RUNNING"
    
    if [ $RUNNING -eq 0 ]; then
        # Check if all output files exist
        OUTPUT_COUNT=$(ls results_8gpu_parallel/gpu_*/chunk_*.jsonl 2>/dev/null | wc -l)
        echo "[$TIMESTAMP] Output files: $OUTPUT_COUNT"
        
        if [ $OUTPUT_COUNT -ge 6 ]; then
            echo "[$TIMESTAMP] Phase 2 COMPLETE! Starting Phase 3..."
            bash auto_proceed_phase3.sh
            exit 0
        else
            echo "[$TIMESTAMP] Waiting for more output files..."
        fi
    else
        # Show current progress
        for i in 0 1 2 3 4 5 6 7; do
            BATCH=$(grep "Batches:" results_8gpu_parallel/gpu_$i.log 2>/dev/null | tail -1 | grep -oP '\d+/\d+' || echo "?")
            echo "  GPU $i: $BATCH"
        done
    fi
    
    echo "[$TIMESTAMP] Sleeping for 30 minutes..."
    sleep $CHECK_INTERVAL
done
