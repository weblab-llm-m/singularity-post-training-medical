#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path Patching for Medical QA (Enhanced Version)
注意パターン抽出機能を追加した医療QA用Path Patching

Based on: sycophancy-interpretability/path_patching/path_patching_hf.py
Modified to extract attention patterns during path patching
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil

import torch
import plotly
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from Phase2_path_patching.dataset import PathPatchingDataset
from Phase2_path_patching.hook_functions import add_pre_module_hook, add_pre_module_hook_single_head
from Phase2_path_patching.utils import compute_metric, create_batch, show_path_patching_results

# Import attention extractor
from Phase3_attention_analysis.attention_extractor import AttentionExtractor


@torch.no_grad()
def path_patching_batch(
    model,
    tokenizer,
    batch_data,
    module_input_name,
    module_output_name,
    num_layers,
    num_attention_heads,
    head_dim,
    extract_attention=False,  # 【追加】注意パターン抽出フラグ
    attention_save_path=None  # 【追加】保存先パス
):
    """
    Path patching with optional attention extraction

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        batch_data: Batch of data
        module_input_name: Input module name pattern
        module_output_name: Output module name pattern
        num_layers: Number of layers
        num_attention_heads: Number of attention heads
        head_dim: Head dimension
        extract_attention: Whether to extract attention patterns
        attention_save_path: Path to save attention patterns

    Returns:
        results: Path patching results
        attention_patterns: Attention patterns (if extract_attention=True)
    """
    results = torch.zeros(size=(num_layers, num_attention_heads), device=model.device)

    # 【追加】注意パターン抽出器を初期化
    attention_extractor = None
    if extract_attention:
        attention_extractor = AttentionExtractor(model, num_layers, num_attention_heads)
        print("Attention extraction enabled")

    # Create path patching data
    xr_toks, xr_mask = create_batch(batch_data, split="xr_toks", pad_token_id=tokenizer.pad_token_id)
    xc_toks, xc_mask = create_batch(batch_data, split="xc_toks", pad_token_id=tokenizer.pad_token_id)

    xr_toks = xr_toks.to(model.device)
    xc_toks = xc_toks.to(model.device)
    xr_mask = xr_mask.to(model.device)
    xc_mask = xc_mask.to(model.device)

    default_logit_diff = compute_metric(
        model,
        xr_toks,
        xr_mask,
        [item["predict_token_id"] for item in batch_data],
        [item["record_token_ids"] for item in batch_data],
    )

    print(f"Default logit diff: {default_logit_diff}")

    Hr = {}  # Hr stores the hidden states of Xr
    Hc = {}  # Hc stores the hidden states of Xc

    # Please refer to 'Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small'
    # for the details of forward A / B / C.

    # Forward A: Record the activation of Xr
    hooks = []
    for i in range(num_layers):
        name = module_input_name.format(i=i)
        module = model.get_submodule(name)  # Get the module

        # Add hook to record the hidden states of Xr
        hook = add_pre_module_hook(
            module,
            name,
            Hr,
            -1,
            True,
        )
        hooks.append(hook)

    # Run forward pass to record
    _ = model(xr_toks.to(model.device), attention_mask=xr_mask.to(model.device))
    for hook in hooks:
        hook.remove()   # Remove the hook

    # Forward B: Record the activation of Xc
    hooks = []
    for i in range(num_layers):
        name = module_output_name.format(i=i)
        module = model.get_submodule(name)

        hook = add_pre_module_hook_single_head(
            module,
            name,
            Hc,
            -1,
            head_dim,
            True,
        )
        hooks.append(hook)

    logits = model(xc_toks.to(model.device), attention_mask=xc_mask.to(model.device)).logits
    print("Xc logits (top 10):")
    print(torch.topk(logits[0, -1], k=10))
    for hook in hooks:
        hook.remove()

    # Loop through all layers and heads
    for source_layer in tqdm(range(num_layers), desc="Path patching layers"):
        for source_head_idx in [None] + list(range(num_attention_heads)):
            if source_head_idx is None:
                continue

            # Forward C: In the forward pass of Xr, replace the hidden states of Xr with the hidden states of Xc
            #   to observe the effect of each attention head on the logits.
            hooks = []
            for j in range(source_layer+1, num_layers):
                name = module_input_name.format(i=j)
                module = model.get_submodule(name)
                hook = add_pre_module_hook(
                    module,
                    name,
                    Hr,
                    -1,
                    False,
                )
                hooks.append(hook)

            name = module_output_name.format(i=source_layer)
            module = model.get_submodule(name)
            hook = add_pre_module_hook_single_head(
                module,
                name,
                Hc,
                -1,
                head_dim,
                False,
                source_head_idx
            )
            hooks.append(hook)

            cur_logit_diff = compute_metric(
                model,
                xr_toks,
                xr_mask,
                [item["predict_token_id"] for item in batch_data],
                [item["record_token_ids"] for item in batch_data],
            )
            for hook in hooks:
                hook.remove()

            results[source_layer][source_head_idx] += (
                (cur_logit_diff - default_logit_diff) / default_logit_diff
            ).mean(dim=0)

    results *= 100

    # 【追加】注意パターンを保存
    attention_patterns = None
    if extract_attention and attention_save_path:
        # 再度フォワードパスを実行して注意パターンを抽出
        attention_patterns = attention_extractor.extract_and_save(
            xr_toks,
            xr_mask,
            attention_save_path
        )
        print(f"Attention patterns saved to: {attention_save_path}")

    return results * len(batch_data), attention_patterns


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser("Path Patching Arguments for Medical QA")
    parser.add_argument("--model_path", type=str, default="/home/Competition2025/P05/shareP05/models/Qwen3-14B")
    parser.add_argument("--data_path", type=str, required=True, help="Path to medical path patching data")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples to use for path patching (-1 for all)")

    # 【追加】注意抽出パラメータ
    parser.add_argument("--extract_attention", type=str, default="true",
                       help="Extract attention patterns (true/false)")
    parser.add_argument("--output_dir", type=str, default="Phase2_path_patching/results",
                       help="Output directory")

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size
    extract_attention = args.extract_attention.lower() == "true"

    if model_path[-1] == '/':
        model_path = model_path[:-1]

    print(f"\n{'='*60}")
    print("Medical Path Patching Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Batch size: {batch_size}")
    print(f"Extract attention: {extract_attention}")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation="eager"  # 【修正】注意重み出力のためeagerを使用
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {model.config.model_type}")

    # Get the name of module used in path patching
    # For Qwen3, use appropriate module names
    if model.config.model_type == "qwen2":
        module_input_name = "model.layers.{i}.input_layernorm"
        module_output_name = "model.layers.{i}.self_attn.o_proj"
    else:
        # Try to load from config file
        config_path = os.path.join(
            os.path.dirname(__file__),
            'configs',
            f"{model.config.model_type}.json"
        )
        if os.path.exists(config_path):
            path_patching_config = json.load(open(config_path, 'r', encoding='utf-8'))
            module_input_name = path_patching_config["module_input_name"]
            module_output_name = path_patching_config["module_output_name"]
        else:
            raise ValueError(f"Config file not found for model type: {model.config.model_type}")

    print(f"Module input name: {module_input_name}")
    print(f"Module output name: {module_output_name}")

    # Construct path patching dataset
    print(f"\nLoading dataset from: {data_path}")
    full_data = PathPatchingDataset(
        data_path,
        tokenizer,
    )

    sample_num = args.sample_num if args.sample_num > 0 else len(full_data)
    sample_num = min(sample_num, len(full_data))

    print(f"Total samples: {len(full_data)}")
    print(f"Using samples: {sample_num}")

    # Print some examples
    print(f"\nXr from dataset:\n" + "-" * 30 + f"\n{full_data.xr[0]}")
    print()
    print(f"Xc from dataset:\n" + "-" * 30 + f"\n{full_data.xc[0]}")

    # Get the number of layers, attention heads and head dimension from config
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    if hasattr(model.config, "head_dim"):
        # In some cases, head_dim is not hidden_size // num_attention_heads
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // model.config.num_attention_heads

    print(f"\nModel architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_attention_heads}")
    print(f"  Head dim: {head_dim}")

    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Copy data_path to output_dir
    shutil.copy(data_path, output_dir)

    # Initialize results
    results = torch.zeros(size=(num_layers, num_attention_heads), device=model.device)

    # Path for attention patterns
    attention_save_path = os.path.join(output_dir, "attention_patterns.pt") if extract_attention else None

    print(f"\nStarting path patching...")
    print(f"Processing {sample_num} samples in batches of {batch_size}")
    print(f"Estimated batches: {(sample_num + batch_size - 1) // batch_size}")

    cnt = 0
    for i in tqdm(range(0, sample_num, batch_size), desc="Batches"):
        batch_data = [full_data[j] for j in range(i, min(i+batch_size, sample_num))]
        batch_results, attention_patterns = path_patching_batch(
            model,
            tokenizer,
            batch_data,
            module_input_name,
            module_output_name,
            num_layers,
            num_attention_heads,
            head_dim,
            extract_attention=extract_attention,
            attention_save_path=attention_save_path if i == 0 else None  # Only save on first batch
        )
        results += batch_results
        cnt += len(batch_data)

        # Clear GPU cache after each batch to prevent OOM
        torch.cuda.empty_cache()

        # Print progress every 10 batches
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {cnt}/{sample_num} samples")

    results /= cnt

    # Find top heads
    indices = torch.topk(results.flatten(), k=min(16, results.numel()), largest=False).indices.detach().cpu().numpy()
    all_results = []
    for i in indices:
        all_results.append((i // num_attention_heads, i % num_attention_heads))

    print(f"\n{'='*60}")
    print(f"Top {len(all_results)} Attention Heads:")
    print(f"{'='*60}")
    for idx, (layer, head) in enumerate(all_results):
        impact = results[layer, head].item()
        print(f"{idx+1:2d}. Layer {layer:2d}, Head {head:2d}: {impact:+.4f}%")
    print(f"{'='*60}\n")

    # Save results
    results_path = os.path.join(output_dir, "results.pt")
    torch.save(results.detach().cpu(), results_path)
    print(f"Results saved to: {results_path}")

    # Show attention head results
    fig = show_path_patching_results(
        results.detach().cpu().numpy(),
        title=f"Medical Path Patching: Effect of patching (Heads->Final Residual Stream)",
        return_fig=True,
        show_fig=False,
        bartitle="% change in logit difference",
    )

    # Save the plot
    html_path = os.path.join(output_dir, 'head_map.html')
    plotly.offline.plot(fig, filename=html_path)
    print(f"Heatmap saved to: {html_path}")

    print(f"\n{'='*60}")
    print("Path Patching Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - results.pt")
    print(f"  - head_map.html")
    if extract_attention:
        print(f"  - attention_patterns.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
