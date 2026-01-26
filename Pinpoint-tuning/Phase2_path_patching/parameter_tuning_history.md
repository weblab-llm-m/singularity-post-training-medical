# Parameter Tuning History - Medical Path Patching

This document tracks all parameter adjustments and experimental results for the medical path patching pipeline.

## Experiment Record

### Run 1: Initial Test (Completed)

**Date**: 2025-10-23
**Status**: ✅ Completed

**Dataset Configuration**:
- Input: `test.parquet`
- Samples: 10 (max_samples=10)
- Total available in test.parquet: ~N/A

**Phase 1 Parameters**:
```bash
--input_path ${DATA_DIR}/test.parquet
--max_samples 10
--model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B
```

**Phase 2 Parameters**:
```bash
--model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B
--data_path Phase1_data_preparation/medical_path_patching_enhanced.jsonl
--output_dir Phase2_path_patching/results
--batch_size 1
--attn_implementation eager
```

**Phase 3 Parameters**:
```yaml
# Head classification thresholds
medical_term:
  threshold: 0.002  # 0.2%

guideline_indicator:
  spike_threshold: 0.005  # 0.5%
  spike_ratio: 2.0

reasoning_flow:
  uniformity_threshold: 0.002
  attention_mean_threshold: 0.0005
  relative_std_threshold: 1.2
```

**Results**:
- Total Heads: 1,600 (40 layers × 40 heads)
- **Medical Term Heads: 1** (0.06%)
  - Layer 28, Head 32 (impact: 1.3245)
- Guideline Indicator Heads: 794 (49.6%)
- Reasoning Flow Heads: 5 (0.3%)
- Unclassified: 800 (50.0%)

**Analysis**:
- Very limited data (10 samples) resulted in poor Medical Term Head detection
- Only 1 Medical Term Head detected, insufficient for effective SPT
- Guideline Indicator Heads dominated (49.6%)

**Output Files**:
- `Phase1_data_preparation/annotated_medical_data.jsonl` (10 samples)
- `Phase2_path_patching/results/results.pt`
- `Phase3_attention_analysis/head_classification_results.json`

---

### Run 2: Full Training Data (In Progress - After Memory Fixes)

**Date**: 2025-10-23
**Status**: 🔄 **In Progress** (8-GPU Parallel execution)
**Phase 1 Completion**: ✅ 1,761 samples processed
**Phase 2 Execution**:
  - **Parallel Strategy**: 8-GPU parallel (8 chunks × ~221 samples each)
  - GPU 0: PID 2379002 (~4.72s/layer)
  - GPU 1: PID 2379704 (~2.64s/layer) - fastest
  - GPU 2: PID 2380117 (~3.85s/layer)
  - GPU 3: PID 2380540 (~6.77s/layer) - slowest
  - GPU 4: PID 2380954 (~4.58s/layer)
  - GPU 5: PID 2381393 (~4.72s/layer)
  - GPU 6: PID 2382873 (~1.42s/layer)
  - GPU 7: PID 2384892 (~3.30s/layer)
  - Each GPU memory: 29-34GB (fits in single H100 80GB)
  - Estimated completion: ~17 hours (vs 86 hours sequential)
  - **Speed improvement: ~5x faster** 🚀

**Objective**: Detect more Medical Term Heads using full training dataset

**Dataset Configuration**:
- Input: `train.parquet`
- Samples: **All (1,761 samples)** - no max_samples limit
- Total available in train.parquet: 1,761

**Phase 1 Parameters** (Adjusted):
```bash
--input_path ${DATA_DIR}/train.parquet
# Removed: --max_samples (use all data)
--model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B
--medical_dict Phase1_data_preparation/medical_terms_dictionary.json
```

**Changes from Run 1**:
- ✅ Changed from `test.parquet` (10 samples) → `train.parquet` (1,761 samples)
- ✅ Removed `--max_samples` limit

**Phase 2 Parameters** (Same):
```bash
--model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B
--data_path Phase1_data_preparation/medical_path_patching_enhanced.jsonl
--output_dir Phase2_path_patching/results
--batch_size 1
--attn_implementation eager
```

**Phase 3 Parameters** (Same):
```yaml
medical_term:
  threshold: 0.002  # 0.2%

guideline_indicator:
  spike_threshold: 0.005  # 0.5%
  spike_ratio: 2.0

reasoning_flow:
  uniformity_threshold: 0.002
  attention_mean_threshold: 0.0005
  relative_std_threshold: 1.2
```

**Expected Results**:
- More Medical Term Heads detected (target: 10-50 heads)
- Better representation of medical terminology patterns
- Improved SPT training effectiveness

**Execution Plan**:
```bash
# Step 1: Run Phase 1 on full train.parquet
bash scripts/run_phase1.sh

# Step 2: Run Phase 2 (path patching)
bash scripts/run_phase2.sh

# Step 3: Run Phase 3 (head classification)
bash scripts/run_phase3.sh

# Step 4: Verify Medical Term Heads
python3 -c "
import json
with open('Phase3_attention_analysis/head_classification_results.json', 'r') as f:
    results = json.load(f)
medical_term_heads = results.get('medical_term_heads', [])
print(f'Medical Term Heads: {len(medical_term_heads)}')
for h in medical_term_heads:
    print(f'  Layer {h[0]}, Head {h[1]}')
"
```

**Output Files** (Expected):
- `Phase1_data_preparation/annotated_medical_data.jsonl` (~1,761 samples)
- `Phase2_path_patching/results/results.pt` (updated)
- `Phase3_attention_analysis/head_classification_results.json` (updated)
- `Phase5_pinpoint_tuning/trainable_heads_medical_term.json` (updated)

---

### Run 3: Adjusted Thresholds (Future)

**Date**: TBD
**Status**: 📋 Future Plan

**Objective**: If Run 2 still detects too few Medical Term Heads, adjust classification thresholds

**Potential Adjustments**:

**Option A: Lower Medical Term Threshold**:
```yaml
medical_term:
  threshold: 0.001  # 0.1% (from 0.2%)
```

**Option B: Add Multi-criteria Detection**:
```yaml
medical_term:
  threshold: 0.002
  min_attention_variance: 0.001  # Detect heads with varied attention to medical terms
  medical_token_focus: true      # Require focus on medical terminology tokens
```

**Option C: Relative Threshold**:
```yaml
medical_term:
  threshold: 0.002
  relative_threshold: 1.5  # 1.5x average attention
```

---

## Summary Statistics

| Run | Dataset | Samples | Medical Term Heads | Guideline Heads | Reasoning Heads | Total Classified |
|-----|---------|---------|-------------------|-----------------|-----------------|------------------|
| 1   | test    | 10      | 1 (0.06%)        | 794 (49.6%)    | 5 (0.3%)       | 800 (50.0%)     |
| 2   | train   | 1,761   | TBD              | TBD            | TBD            | TBD             |
| 3   | TBD     | TBD     | TBD              | TBD            | TBD            | TBD             |

---

## Notes

- **Model**: Qwen3-14B (40 layers, 40 heads/layer, 1,600 total heads)
- **Attention Implementation**: eager (required for output_attentions)
- **Path Patching Strategy**: Counterfactual medical term replacement
- **Classification Method**: Attention pattern analysis with threshold-based categorization

---

## Lessons Learned

### From Run 1:
1. **Data Size Matters**: 10 samples insufficient for comprehensive head detection
2. **Medical Term Detection**: Requires larger dataset to capture diverse medical terminology patterns
3. **Guideline Dominance**: Model heavily relies on guideline indicator patterns (49.6% of heads)
4. **Threshold Sensitivity**: Current thresholds may be too strict for medical term detection

### Best Practices:
- Use full training dataset for initial analysis
- Monitor attention distribution statistics before finalizing thresholds
- Consider multi-pass analysis: broad first, then refine
- Document all parameter choices and rationale

---

## File Locations

- This file: `Phase2_path_patching/parameter_tuning_history.md`
- Phase 1 config: `scripts/run_phase1.sh`
- Phase 2 config: `scripts/run_phase2.sh`
- Phase 3 config: `configs/head_classification_params.yaml`
- Results: `Phase3_attention_analysis/head_classification_results.json`
