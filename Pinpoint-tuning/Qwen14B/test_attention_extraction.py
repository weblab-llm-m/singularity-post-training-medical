#!/usr/bin/env python3
"""注意抽出のテスト"""

import sys
sys.path.append('.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Phase3_attention_analysis.attention_extractor import AttentionExtractor

# モデルとトークナイザーをロード
model_path = "/home/Competition2025/P05/shareP05/models/Qwen3-14B"
print(f"Loading model from: {model_path}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    attn_implementation="eager"  # Required for output_attentions
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

print(f"Model loaded: {model.config.model_type}")
print(f"Layers: {model.config.num_hidden_layers}")
print(f"Heads: {model.config.num_attention_heads}")

# AttentionExtractorを初期化
extractor = AttentionExtractor(
    model,
    num_layers=model.config.num_hidden_layers,
    num_heads=model.config.num_attention_heads
)

# テストテキスト
text = "産婦人科診療ガイドラインについて、正しいものを選べ"
inputs = tokenizer(text, return_tensors="pt")

# 注意パターンを抽出
print("\nExtracting attention patterns...")
patterns = extractor.extract_and_save(
    inputs['input_ids'],
    inputs['attention_mask'],
    save_path="test_attention.pt"
)

print(f"\nExtracted patterns:")
print(f"  Number of layers: {len(patterns)}")
if len(patterns) > 0:
    for layer_idx in list(patterns.keys())[:3]:
        print(f"  Layer {layer_idx}: shape={patterns[layer_idx].shape}")
    print("✅ Attention extraction successful!")
else:
    print("❌ No attention patterns extracted!")
