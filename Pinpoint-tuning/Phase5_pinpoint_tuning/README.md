# Phase 5: Pinpoint Tuning for Medical QA

## Overview

Phase 5 implements Supervised Pinpoint Tuning (SPT) for medical QA tasks based on the path patching analysis from Phase 1-4.

## Architecture

```
Phase5_pinpoint_tuning/
├── run_spt_medical.py          # Main training script
├── run_spt_medical.sh           # Shell script to execute training
├── select_trainable_heads.py   # Head selection script (from Phase 3)
├── trainable_heads.json        # Selected heads for tuning (64 heads)
├── configs/
│   └── lora_config.json        # LoRA configuration
├── dataset/
│   ├── __init__.py
│   └── dataset_medical.py      # Medical QA dataset loader for parquet
├── model/
│   ├── __init__.py
│   ├── model_hf.py             # Hugging Face model builder
│   └── model_peft.py           # PEFT model builder
├── trainer/
│   ├── __init__.py
│   ├── trainer_hf.py           # Hugging Face trainer
│   └── callbacks.py            # Training callbacks
└── utils/
    ├── __init__.py
    ├── arguments.py            # Argument parser
    └── utils_spt.py            # SPT utilities (adapted for trainable_heads.json)
```

## Key Features

### 1. Trainable Heads Selection

64 high-priority guideline indicator heads selected based on:
- Path patching impact scores
- Head classification results
- Top head: Layer 34, Head 37 (impact: 14.55)

### 2. Dataset Adaptation

**Medical QA Dataset Loader** (`dataset/dataset_medical.py`):
- Reads parquet files from gynecology guideline dataset
- Converts to instruction tuning format
- Supports Qwen chat template
- Caches tokenized data for efficiency

### 3. SPT Implementation

**Modified SPT Utilities** (`utils/utils_spt.py`):
- `load_trainable_heads_from_json()`: Load heads from Phase 3 results
- `freeze_modules()`: Freeze all modules except selected heads
- `freeze_lora_modules()`: Combine LoRA with SPT

**Supported Models**: Qwen2, Qwen3, LLaMA, Mistral

### 4. Training Options

**Precise Levels**:
- Level 1: qkv_proj + o_proj + mlp + wte/lm_head
- Level 2: qkv_proj + o_proj + mlp
- Level 3: qkv_proj + o_proj
- Level 4: qkv_proj only (recommended)

**Training Modes**:
- Full parameter tuning on selected heads
- LoRA + SPT hybrid (more memory efficient)

## Requirements

```bash
pip install peft==0.13.2 addict==2.4.0 tensorboard==2.18.0
```

Optional (for distributed training):
```bash
pip install deepspeed==0.15.2
```

## Usage

### Quick Start

```bash
# Run full pipeline (head selection + training)
bash Phase5_pinpoint_tuning/run_spt_medical.sh
```

### Manual Execution

#### Step 1: Select Trainable Heads (if not already done)

```bash
python3 Phase5_pinpoint_tuning/select_trainable_heads.py \
    --classification_results Phase3_attention_analysis/head_classification_results.json \
    --patching_results Phase2_path_patching/results/results.pt \
    --output_path Phase5_pinpoint_tuning/trainable_heads.json
```

#### Step 2: Run Training

```bash
python3 Phase5_pinpoint_tuning/run_spt_medical.py \
    --model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B \
    --data_path /home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/train.parquet \
    --output_dir Phase5_pinpoint_tuning/spt_medical_output \
    --cache_dir Phase5_pinpoint_tuning/cache \
    --path_patching_path Phase5_pinpoint_tuning/trainable_heads.json \
    --precise_level 4 \
    --attn_implementation eager \
    --torch_dtype bfloat16 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --bf16 true \
    --seed 42
```

#### Step 3: Training with LoRA

```bash
python3 Phase5_pinpoint_tuning/run_spt_medical.py \
    --model_path /home/Competition2025/P05/shareP05/models/Qwen3-14B \
    --data_path /home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/train.parquet \
    --output_dir Phase5_pinpoint_tuning/spt_medical_output_lora \
    --cache_dir Phase5_pinpoint_tuning/cache \
    --path_patching_path Phase5_pinpoint_tuning/trainable_heads.json \
    --precise_level 3 \
    --peft_type lora \
    --peft_config Phase5_pinpoint_tuning/configs/lora_config.json \
    --attn_implementation eager \
    --torch_dtype bfloat16 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --bf16 true \
    --seed 42
```

## Configuration

### Training Parameters (in run_spt_medical.sh)

```bash
# Model and data
MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-14B"
DATA_DIR="/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted"

# SPT configuration
USE_LORA=false           # Set to true to use LoRA
PRECISE_LEVEL=4          # 4: only qkv_proj
LEARNING_RATE=1e-4
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
NUM_EPOCHS=3
MAX_SEQ_LENGTH=2048
```

### LoRA Configuration (configs/lora_config.json)

```json
{
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

## Training Results

The trained model will be saved to:
```
Phase5_pinpoint_tuning/spt_medical_output/
├── pytorch_model.bin          # Model weights
├── config.json                # Model config
├── training_args.bin          # Training arguments
└── trainer_state.json         # Training state
```

## Key Modifications from Original Pinpoint Tuning

1. **Trainable Heads Format**: Uses `trainable_heads.json` instead of `results.pt`
   - JSON format with layer, head, type, impact, priority
   - Easier to inspect and modify

2. **Dataset Loader**: Parquet support for medical QA data
   - Converts parquet to instruction tuning format
   - Handles gynecology guideline dataset structure

3. **Model Support**: Qwen3 support
   - Treated as Qwen2 architecture
   - Uses `attn_implementation="eager"` for attention extraction

4. **Flexibility**: Both full tuning and LoRA+SPT hybrid modes

## Troubleshooting

### Memory Issues

If you encounter OOM errors:
1. Reduce `BATCH_SIZE` in run_spt_medical.sh
2. Increase `GRADIENT_ACCUMULATION` to maintain effective batch size
3. Use LoRA mode (set `USE_LORA=true`)
4. Reduce `MAX_SEQ_LENGTH`

### Import Errors

Ensure all dependencies are installed:
```bash
pip install peft addict tensorboard
```

### Model Loading Issues

Qwen3 requires `attn_implementation="eager"`:
- This is already set in run_spt_medical.sh
- Required for output_attentions compatibility

## Next Steps

After training:
1. Evaluate the tuned model on validation set
2. Compare performance with baseline Qwen3-14B
3. Analyze which heads improved most
4. Fine-tune training hyperparameters if needed

## References

- Original Pinpoint Tuning: `/home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/pinpoint_tuning/`
- Phase 1-4 Results: `../Phase{1,2,3,4}_*/`
- Trainable Heads: `trainable_heads.json`
