#!/usr/bin/env python3
"""
Phase 3: Attention Analysis & Head Classification
256サンプルの注意パターンを抽出し、3種類のヘッドに分類する

Step 1: モデルをロードし、referenceデータで注意パターンを抽出
Step 2: 注意パターン + 位置情報でヘッドを分類
Step 3: 結果を保存
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from Phase3_attention_analysis.attention_extractor import AttentionExtractor
from Phase3_attention_analysis.head_classifier import HeadClassifier


def extract_attention_patterns(
    model,
    tokenizer,
    data: list,
    num_layers: int,
    num_heads: int,
    batch_size: int = 1,
    max_samples: int = -1
) -> dict:
    """
    全サンプルの注意パターンを抽出

    Returns:
        {layer_idx: tensor[num_samples, num_heads, seq_len]}
    """
    extractor = AttentionExtractor(model, num_layers, num_heads)

    n = len(data) if max_samples < 0 else min(max_samples, len(data))
    print(f"\nExtracting attention patterns for {n} samples...")

    # 注意パターンを蓄積
    all_patterns = {}  # {layer: list of [1, num_heads, seq_len]}

    model.config.output_attentions = True

    for i in tqdm(range(n), desc="Extracting attention"):
        item = data[i]
        ref_text = item['reference_data']

        encoded = tokenizer(
            ref_text,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=512
        )
        input_ids = encoded['input_ids'].to(model.device)
        attention_mask = encoded['attention_mask'].to(model.device)

        # フックを設定
        extractor.attention_patterns = {}
        hooks = extractor.add_attention_hooks()

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # フックを削除
        for hook in hooks:
            hook.remove()
        extractor.hooks = []

        # 各レイヤーの注意パターンを蓄積
        for layer_idx in range(num_layers):
            if layer_idx in extractor.attention_patterns:
                # [1, num_heads, seq_len] → パディングして蓄積
                pattern = extractor.attention_patterns[layer_idx][0]  # [1, num_heads, seq_len]
                if layer_idx not in all_patterns:
                    all_patterns[layer_idx] = []
                all_patterns[layer_idx].append(pattern)

        # メモリ解放
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    model.config.output_attentions = False

    # seq_lenをパディングして揃える
    print("Padding and concatenating attention patterns...")
    max_seq_len = 0
    for layer_idx in all_patterns:
        for p in all_patterns[layer_idx]:
            max_seq_len = max(max_seq_len, p.shape[-1])

    padded_patterns = {}
    for layer_idx in tqdm(all_patterns, desc="Padding"):
        padded = []
        for p in all_patterns[layer_idx]:
            # p: [1, num_heads, seq_len]
            pad_len = max_seq_len - p.shape[-1]
            if pad_len > 0:
                p = torch.nn.functional.pad(p, (0, pad_len), value=0.0)
            padded.append(p)
        padded_patterns[layer_idx] = torch.cat(padded, dim=0)  # [n, num_heads, max_seq_len]

    print(f"  Max seq_len: {max_seq_len}")
    print(f"  Pattern shape per layer: [{n}, {num_heads}, {max_seq_len}]")

    return padded_patterns


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Attention Analysis & Head Classification")
    parser.add_argument("--model_path", type=str,
                        default="/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path patching data (.jsonl)")
    parser.add_argument("--criteria_config", type=str,
                        default="Phase3_attention_analysis/configs/head_classification_params.yaml")
    parser.add_argument("--output_dir", type=str,
                        default="Phase3_attention_analysis/results")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_attention", action="store_true",
                        help="Save raw attention patterns (.pt)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # === Step 1: モデルロード ===
    print("=" * 60)
    print("Phase 3: Attention Analysis & Head Classification")
    print("=" * 60)

    print(f"\nLoading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"  Layers: {num_layers}, Heads: {num_heads}")

    # === Step 2: データロード ===
    print(f"\nLoading data: {args.data_path}")
    data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    n = len(data) if args.max_samples < 0 else min(args.max_samples, len(data))
    print(f"  Total: {len(data)}, Using: {n}")

    # === Step 3: 注意パターン抽出 ===
    attention_patterns = extract_attention_patterns(
        model, tokenizer, data, num_layers, num_heads, max_samples=n
    )

    if args.save_attention:
        attn_path = os.path.join(args.output_dir, "attention_patterns.pt")
        torch.save(attention_patterns, attn_path)
        print(f"Attention patterns saved to: {attn_path}")

    # モデルをアンロード（メモリ解放）
    del model
    torch.cuda.empty_cache()

    # === Step 4: ヘッド分類 ===
    print(f"\nLoading criteria: {args.criteria_config}")
    with open(args.criteria_config, 'r', encoding='utf-8') as f:
        criteria_config = yaml.safe_load(f)

    classifier = HeadClassifier(
        attention_patterns=attention_patterns,
        annotation_data=data[:n],
        criteria_config=criteria_config,
        num_layers=num_layers,
        num_heads=num_heads
    )

    results = classifier.classify_all_heads()

    # === Step 5: 結果保存 ===
    json_results = {
        key: [list(item) for item in value]
        for key, value in results.items()
    }
    json_results['metadata'] = {
        'num_samples': n,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'data_path': args.data_path,
        'criteria_config': args.criteria_config
    }

    output_path = os.path.join(args.output_dir, f"head_classification_results_{n}samples.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
