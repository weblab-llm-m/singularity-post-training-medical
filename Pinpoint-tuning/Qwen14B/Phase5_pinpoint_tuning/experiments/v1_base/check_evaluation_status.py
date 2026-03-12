#!/usr/bin/env python3
"""Check evaluation status and display results when available"""

import json
import os
import time
import subprocess

results_dir = "/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_pinpoint_tuning/evaluation_results"
tuned_file = os.path.join(results_dir, "tuned_model_results.json")
base_file = os.path.join(results_dir, "base_model_results.json")

print("Checking evaluation status...")
print("")

# Check if processes are running
result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
eval_running = "evaluate_model.py" in result.stdout

if eval_running:
    print("✓ Evaluation is currently running")
    print("  Waiting for completion...")

    # Count down while waiting
    max_wait = 600  # 10 minutes
    for i in range(max_wait):
        time.sleep(1)
        if i % 30 == 0:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            if "evaluate_model.py" not in result.stdout:
                print("\n✓ Evaluation completed!")
                break
            else:
                elapsed = i // 60
                print(f"  Still running... ({elapsed} min elapsed)")
else:
    print("✓ No evaluation processes running")

print("")

# Check for results
if os.path.exists(tuned_file) and os.path.exists(base_file):
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    with open(tuned_file, 'r') as f:
        tuned_data = json.load(f)

    with open(base_file, 'r') as f:
        base_data = json.load(f)

    tuned_acc = tuned_data['metrics']['accuracy']
    base_acc = base_data['metrics']['accuracy']
    improvement = tuned_acc - base_acc

    print(f"\n📊 Base Model (Qwen3-14B):")
    print(f"   Accuracy: {base_acc:.2f}%")
    print(f"   Correct: {base_data['metrics']['correct']}/{base_data['metrics']['total_samples']}")

    print(f"\n📊 SPT-tuned Model (144 Medical Heads):")
    print(f"   Accuracy: {tuned_acc:.2f}%")
    print(f"   Correct: {tuned_data['metrics']['correct']}/{tuned_data['metrics']['total_samples']}")

    print(f"\n📈 Improvement:")
    print(f"   Absolute: {improvement:+.2f}%")
    if base_acc > 0:
        print(f"   Relative: {(improvement/base_acc*100):+.2f}%")

    # Show some example results
    print(f"\n📝 Sample Predictions (first 5):")
    for i, result in enumerate(tuned_data['results'][:5]):
        status = "✓" if result['correct'] else "✗"
        print(f"   {i+1}. {status} GT: {result['ground_truth']} | Pred: {result['predicted']}")

    print("="*70)
    print("")

elif os.path.exists(tuned_file):
    print("⚠️  Tuned model results found, but base model results missing")
    print(f"   Tuned model file: {tuned_file}")

elif os.path.exists(base_file):
    print("⚠️  Base model results found, but tuned model results missing")
    print(f"   Base model file: {base_file}")

else:
    print("⚠️  No result files found yet")
    if os.path.exists(results_dir):
        print(f"\n   Directory contents:")
        for f in os.listdir(results_dir):
            filepath = os.path.join(results_dir, f)
            size = os.path.getsize(filepath)
            print(f"   - {f} ({size} bytes)")
    else:
        print(f"   Results directory does not exist: {results_dir}")
