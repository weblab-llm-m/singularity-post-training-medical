#!/bin/bash
# Monitor SPT training and generate final report

LOG_FILE="spt_144heads_optimized_nohup.log"
OUTPUT_DIR="spt_medical_144heads_output"

echo "==================================================="
echo "Monitoring SPT Training for 144 Medical Term Heads"
echo "==================================================="
echo ""

# Wait for training to complete
while ps aux | grep -q "[r]un_spt_medical.py"; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Training in progress..."
    tail -1 $LOG_FILE | grep -oP '\d+%|\d+/\d+' || echo "  (checking...)"
    sleep 60
done

echo ""
echo "==================================================="
echo "Training Completed! Generating final report..."
echo "==================================================="
echo ""

# Check if training succeeded
if [ -f "$OUTPUT_DIR/pytorch_model.bin" ] || [ -d "$OUTPUT_DIR/checkpoint-*" ]; then
    echo "✓ Model weights saved successfully"
    ls -lh $OUTPUT_DIR/checkpoint-* 2>/dev/null || ls -lh $OUTPUT_DIR/*.bin 2>/dev/null
else
    echo "✗ Warning: No model weights found"
fi

echo ""
echo "Training Log Summary:"
echo "---------------------"
tail -50 $LOG_FILE | tail -20

echo ""
echo "Output Directory Contents:"
echo "--------------------------"
ls -lh $OUTPUT_DIR/

echo ""
echo "==================================================="
echo "Next Steps:"
echo "  1. Check model outputs in: $OUTPUT_DIR"
echo "  2. Run evaluation: Phase5_pinpoint_tuning/evaluate_model.py"
echo "  3. Compare with base model performance"
echo "==================================================="
