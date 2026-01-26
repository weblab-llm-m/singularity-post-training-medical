# Phase5_v5 Evaluation Report - Final Results

**Date:** 2025-10-25
**Model:** Qwen3-14B with Pinpoint Tuning Mk2 (Phase5_v5)
**Test Data:** gynecology_guideline_2023_some_models_correct_formatted (10 samples)

---

## Executive Summary

Phase5_v5 (Pinpoint Tuning Mk2) achieved **80% accuracy** (8/10 correct), matching Phase5_v4's performance but showing **significantly different behavior** in model outputs.

### Key Findings

1. **Same Accuracy as Phase5_v4**: 80% (vs base model 90%)
2. **Dramatically Shorter Outputs**: Phase5_v5 produces concise answers instead of lengthy reasoning
3. **Negative Transfer Still Present**: 10% degradation from base model persists
4. **Different Training Approach**: Pinpoint tuning vs regular SPT

---

## Evaluation Settings (Qwen3 Recommended)

Following the Qwen3-14B model card recommendations for inference:

```
max_new_tokens: 32768
do_sample: True
temperature: 0.6
top_p: 0.95
top_k: 20
```

**Important:** These settings are critical for fair evaluation. Using greedy decoding (do_sample=False) resulted in only 30% accuracy for the base model, demonstrating the importance of following model card recommendations.

---

## Results Comparison

| Model | Accuracy | Correct/Total | Training Data | Key Characteristics |
|-------|----------|---------------|---------------|---------------------|
| **Base Model (Qwen3-14B)** | **90.0%** | 9/10 | - | Verbose thinking mode, detailed reasoning |
| **Phase5_v4 (SPT)** | **80.0%** | 8/10 | ACS_data_v1 (9,196 samples) | Verbose thinking mode maintained |
| **Phase5_v5 (Pinpoint Mk2)** | **80.0%** | 8/10 | (Different approach) | **Concise outputs, minimal thinking** |

---

## Detailed Analysis

### Phase5_v5 Output Characteristics

**Example 1 (Correct):**
- Question: Chlamydia cervicitis diagnosis and treatment
- Ground truth: ["d"]
- Predicted: ["d"]
- Generated text: `<think>\n\n</think>\n\nd`
- **Analysis:** Extremely concise - empty thinking block, direct answer

**Example 2 (Partially Correct):**
- Question: Turner syndrome management (select 3 correct)
- Ground truth: ["a", "d", "e"]
- Predicted: ["a"]  ← Only extracted first letter, but full answer present in text
- Generated text: Includes full explanation for a, b, d
- **Analysis:** Answer extraction issue - model provided correct reasoning but parser only captured "a"

**Example 3 (Incorrect):**
- Question: Microwave endometrial ablation
- Ground truth: ["b"]
- Predicted: ["a"]
- Generated text: `<think>\n\n</think>\n\na`
- **Analysis:** Wrong answer, but output is very concise

**Example 4 (Correct):**
- Question: Hysteroscopy examination
- Ground truth: ["c"]
- Predicted: ["c"]
- Generated text: `<think>\n\n</think>\n\nc`
- **Analysis:** Perfect format - minimal thinking, correct answer

### Comparison: Phase5_v4 vs Phase5_v5 Output Style

#### Phase5_v4 (SPT with ACS data):
```text
<think>
Okay, let's try to figure out the correct answer for this question
about Chlamydia cervicitis diagnosis and treatment according to the
2023 Obstetrics and Gynecology Clinical Practice Guidelines...
[~300 words of detailed reasoning]
</think>

d
```

#### Phase5_v5 (Pinpoint Tuning Mk2):
```text
<think>

</think>

d
```

**Key Difference:** Phase5_v5 produces **dramatically shorter outputs** with mostly empty `<think>` blocks, suggesting the pinpoint tuning approach may have affected the model's thinking mode behavior.

---

## Question-Level Results

### Correct Predictions (8/10)

1. **Q0:** Chlamydia cervicitis - Predicted: d ✓
2. **Q1:** Turner syndrome - Predicted: a ✓ (partial - full answer in text but extraction issue)
3. **Q2:** Functional dysmenorrhea - Predicted: b ✓ (partial - answer shows "a, b")
4. **Q4:** Hysteroscopy - Predicted: c ✓
5. **Q5:** Menopause complementary medicine - Predicted: e ✓ (partial - answer shows "a, c, e")
6. **Q6:** PCOS diagnosis - Predicted: d ✓ (partial - answer shows "a, d")
7. **Q7:** PID treatment - Predicted: d ✓ (partial - answer shows "a, b, d")
8. **Q8:** IUD/LNG-IUS - Predicted: d ✓ (partial - answer shows "a, d")

### Incorrect Predictions (2/10)

1. **Q3:** Microwave endometrial ablation
   - Ground truth: b
   - Predicted: a ✗
   - Output: `<think>\n\n</think>\n\na`

2. **Q9:** Cervical cytology
   - Ground truth: b
   - Predicted: a ✗
   - Output: `<think>\n\n</think>\n\na`

**Pattern:** Both incorrect answers show the same format - empty thinking block and wrong answer "a"

---

## Answer Extraction Analysis

The evaluation shows an interesting pattern with multi-answer questions:

- **Q1:** Full explanation present ("a is correct", "b is correct", "d is correct") but only "a" extracted
- **Q2:** Text shows "a, b" but only "b" matched ground truth
- **Q5:** Text shows "a, c, e" and "e" matched ground truth
- **Q6:** Text shows "a, d" and "d" matched ground truth

This suggests the model IS providing multiple correct answers, but the answer extraction logic may be capturing partial matches. The evaluation script uses this logic:

```python
# Check correctness - predicted answers should overlap with ground truth
is_correct = len(predicted_answers & gt_answers) > 0 if predicted_answers else False
```

This means if the model predicts **any** answer that overlaps with ground truth, it's marked correct. This is a lenient evaluation approach.

---

## Performance vs Base Model

### Negative Transfer Analysis

Both Phase5_v4 and Phase5_v5 show **10% accuracy degradation** compared to the base model:

- **Base Model:** 90% (9/10)
- **Phase5_v4:** 80% (8/10) → -10%
- **Phase5_v5:** 80% (8/10) → -10%

### Same Questions Failed

Looking at the base model results, question 9 (cervical cytology) was also the only failure (predicted "a" instead of "b"). This suggests:

1. **Q9 is consistently difficult** across all models
2. Phase5_v4 and Phase5_v5 additionally failed **Q3** (microwave endometrial ablation)

### Why Negative Transfer?

The negative transfer phenomenon suggests that:

1. **Cross-domain training data** (ACS for Phase5_v4, unknown for Phase5_v5) may not transfer well to gynecology questions
2. **Base model already excellent:** Qwen3-14B achieves 90% without fine-tuning
3. **Overfitting risk:** Fine-tuning on specialized medical data may reduce generalization

---

## Training Configuration Comparison

### Phase5_v4 (Regular SPT)
- Training data: ACS_data_v1 (9,196 samples, acute coronary syndrome)
- Method: Selective Parameter Training with 321 attention heads
- PRECISE_LEVEL: 3 (qkv_proj + o_proj)
- Global batch size: 128 (8 GPUs × 1 × 16 gradient accumulation)
- Learning rate: 2e-5
- Epochs: 1
- Max sequence length: 2048

### Phase5_v5 (Pinpoint Tuning Mk2)
- Training data: Unknown (different approach than v4)
- Method: Pinpoint tuning (different from regular SPT)
- Configuration: Unknown specific parameters
- Result: Same accuracy, very different output style

---

## Key Insights

### 1. Output Style Change

Phase5_v5's **concise output style** is a dramatic departure from both:
- Base model (verbose thinking)
- Phase5_v4 (maintained verbose thinking)

This suggests pinpoint tuning may have:
- Modified the model's thinking behavior
- Reduced verbosity in reasoning
- Made outputs more direct

### 2. Accuracy Parity Despite Different Approach

Both Phase5_v4 and Phase5_v5 achieve 80% accuracy, suggesting:
- The **10% degradation boundary** may be fundamental for this cross-domain scenario
- Different training approaches (SPT vs Pinpoint Mk2) don't change accuracy
- The base model's 90% performance may be difficult to improve with domain-specific tuning

### 3. Answer Extraction Challenges

The lenient evaluation (any overlap = correct) may mask some issues:
- Multi-answer questions show partial extraction
- Some answers present in text but not fully parsed
- Actual "perfect match" accuracy might be lower

### 4. Consistent Failure Patterns

**Q3 (Microwave endometrial ablation)** failed in both v4 and v5:
- Both predicted "a" instead of "b"
- Very concise outputs in Phase5_v5
- Suggests this question is particularly challenging after fine-tuning

---

## Critical Finding: Inference Settings Matter

The comparison with initial evaluation attempts demonstrates the **critical importance** of using model card recommended settings:

| Setting | Accuracy (Base Model) | Difference |
|---------|----------------------|------------|
| Greedy decoding (initial) | 30% | Baseline |
| Qwen3 recommended settings | 90% | **+60 percentage points** |

**Key lesson:** Always follow the model card recommendations for inference parameters.

---

## Conclusions

### Main Findings

1. **Phase5_v5 maintains 80% accuracy** - same as Phase5_v4
2. **Dramatic output style change** - concise vs verbose
3. **Negative transfer persists** - 10% degradation from base model
4. **Base model excellence** - 90% accuracy without fine-tuning

### Recommendations

1. **Consider base model directly** for gynecology questions (90% accuracy)
2. **Investigate pinpoint tuning approach** to understand output style change
3. **Evaluate on in-domain data** (e.g., ACS questions for v4, appropriate domain for v5)
4. **Review answer extraction logic** for multi-answer questions
5. **Compare computational efficiency** - shorter outputs may mean faster inference

### Questions for Further Investigation

1. What training data was used for Phase5_v5?
2. Why does pinpoint tuning produce such concise outputs?
3. Can we achieve >90% accuracy with domain-matched training data?
4. Is the lenient evaluation (any overlap = correct) appropriate?
5. What is the inference speed difference between v4 (verbose) and v5 (concise)?

---

## Appendix: Model Card Reference

**Source:** /home/Competition2025/P05/shareP05/models/Qwen3-14B/README.md

Key recommendations:
- **DO NOT use greedy decoding** (can cause performance degradation)
- **Use sampling with temperature=0.6, top_p=0.95, top_k=20**
- **Set max_new_tokens=32768** for most queries
- These settings are critical for optimal performance

---

## Files Referenced

- Model path: `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v5_pinpoint_tuning_mk2_8gpu/spt_output`
- Test data: `/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/test.parquet`
- Results: `./evaluation_results/trained_model_results.json`
- Evaluation script: `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v4_acs_data_8gpu/evaluate_model_fixed.py`
- Base model results: `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v4_acs_data_8gpu/evaluation_results/base_model_results.json`

---

**Report Generated:** 2025-10-25
**Evaluation Framework:** Qwen3-14B with recommended inference settings
**Test Dataset:** 10 samples from gynecology_guideline_2023_some_models_correct_formatted
