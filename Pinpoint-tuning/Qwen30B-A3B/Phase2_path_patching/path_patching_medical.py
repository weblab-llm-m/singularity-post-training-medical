#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path Patching for Medical QA (MoE Enhanced Version)

Based on: Qwen14B/Phase2_path_patching/path_patching_medical.py
Enhanced with MoE model support for Qwen3-30B-A3B-Instruct-2507

Key additions:
1. Model type detection (qwen2, qwen3_moe, etc.)
2. MoE-specific module name patterns
3. Optional expert analysis (future extension)
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
from Phase2_path_patching.hook_functions_moe import (
    add_pre_module_hook,
    add_pre_module_hook_single_head,
    MoEPathPatchingHooks,
    analyze_router_selections
)
from Phase2_path_patching.utils import compute_metric, create_batch, show_path_patching_results


# Model type to module name mapping
MODEL_TYPE_CONFIGS = {
    "qwen2": {
        "module_input_name": "model.layers.{i}.input_layernorm",
        "module_output_name": "model.layers.{i}.self_attn.o_proj",
        "is_moe": False
    },
    "qwen3_moe": {
        "module_input_name": "model.layers.{i}.input_layernorm",
        "module_output_name": "model.layers.{i}.self_attn.o_proj",
        "router_module_name": "model.layers.{i}.mlp.gate",
        "expert_module_pattern": "model.layers.{i}.mlp.experts.{j}",
        "is_moe": True
    },
    "llama": {
        "module_input_name": "model.layers.{i}.input_layernorm",
        "module_output_name": "model.layers.{i}.self_attn.o_proj",
        "is_moe": False
    }
}


def get_model_config(model, config_dir=None):
    """Get module names based on model type.

    Parameters
    ----------
    model : PreTrainedModel
        The loaded model
    config_dir : str, optional
        Directory containing config JSON files

    Returns
    -------
    dict
        Configuration with module names and MoE flag
    """
    model_type = model.config.model_type

    # First check built-in configs
    if model_type in MODEL_TYPE_CONFIGS:
        config = MODEL_TYPE_CONFIGS[model_type].copy()
        print(f"Using built-in config for model type: {model_type}")

        # Add model-specific parameters
        if model_type == "qwen3_moe":
            config["num_experts"] = getattr(model.config, "num_experts", 128)
            config["num_experts_per_tok"] = getattr(model.config, "num_experts_per_tok", 8)

        return config

    # Try to load from config file
    if config_dir:
        config_path = os.path.join(config_dir, f"{model_type}.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Loaded config from: {config_path}")
            return config

    raise ValueError(f"No config found for model type: {model_type}")


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
    is_moe=False,
    analyze_experts=False,
    router_module_name=None,
    num_experts=None
):
    """
    Path patching with optional MoE analysis.

    Parameters
    ----------
    model : PreTrainedModel
        Transformer model
    tokenizer : AutoTokenizer
        Tokenizer
    batch_data : List[Dict]
        Batch of data
    module_input_name : str
        Input module name pattern
    module_output_name : str
        Output module name pattern
    num_layers : int
        Number of layers
    num_attention_heads : int
        Number of attention heads
    head_dim : int
        Head dimension
    is_moe : bool, optional
        Whether the model is MoE
    analyze_experts : bool, optional
        Whether to analyze expert routing (future extension)
    router_module_name : str, optional
        Router module name pattern (for MoE)
    num_experts : int, optional
        Number of experts (for MoE)

    Returns
    -------
    torch.Tensor
        Path patching results
    dict, optional
        Expert analysis results (if analyze_experts=True)
    """
    results = torch.zeros(size=(num_layers, num_attention_heads), device=model.device)
    expert_analysis = None

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

    # Forward A: Record the activation of Xr
    hooks = []
    for i in range(num_layers):
        name = module_input_name.format(i=i)
        module = model.get_submodule(name)

        hook = add_pre_module_hook(
            module,
            name,
            Hr,
            -1,
            True,
        )
        hooks.append(hook)

    _ = model(xr_toks.to(model.device), attention_mask=xr_mask.to(model.device))
    for hook in hooks:
        hook.remove()

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

            # Forward C: Replace hidden states
            hooks = []
            for j in range(source_layer + 1, num_layers):
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

    return results * len(batch_data), expert_analysis


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Path Patching Arguments for Medical QA (MoE Support)")
    parser.add_argument("--model_path", type=str,
                       default="/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--data_path", type=str, required=True, help="Path to medical path patching data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (smaller for MoE due to memory)")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (-1 for all)")
    parser.add_argument("--output_dir", type=str, default="Phase2_path_patching/results",
                       help="Output directory")
    parser.add_argument("--analyze_experts", action="store_true",
                       help="Analyze expert routing patterns (experimental)")

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size

    if model_path[-1] == '/':
        model_path = model_path[:-1]

    print(f"\n{'='*60}")
    print("Medical Path Patching Configuration (MoE Enhanced)")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Batch size: {batch_size}")
    print(f"Analyze experts: {args.analyze_experts}")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation="eager"  # Required for attention weight output
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {model.config.model_type}")

    # Get model configuration
    config_dir = os.path.join(os.path.dirname(__file__), 'configs')
    model_config = get_model_config(model, config_dir)

    module_input_name = model_config["module_input_name"]
    module_output_name = model_config["module_output_name"]
    is_moe = model_config.get("is_moe", False)

    print(f"Module input name: {module_input_name}")
    print(f"Module output name: {module_output_name}")
    print(f"Is MoE model: {is_moe}")

    if is_moe:
        num_experts = model_config.get("num_experts", getattr(model.config, "num_experts", 128))
        num_experts_per_tok = model_config.get("num_experts_per_tok",
                                                getattr(model.config, "num_experts_per_tok", 8))
        print(f"MoE Config: {num_experts} experts, {num_experts_per_tok} active per token")

    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    full_data = PathPatchingDataset(data_path, tokenizer)

    sample_num = args.sample_num if args.sample_num > 0 else len(full_data)
    sample_num = min(sample_num, len(full_data))

    print(f"Total samples: {len(full_data)}")
    print(f"Using samples: {sample_num}")

    # Print examples
    print(f"\nXr from dataset:\n" + "-" * 30 + f"\n{full_data.xr[0]}")
    print()
    print(f"Xc from dataset:\n" + "-" * 30 + f"\n{full_data.xc[0]}")

    # Get model architecture info
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    if hasattr(model.config, "head_dim"):
        head_dim = model.config.head_dim
    else:
        head_dim = model.config.hidden_size // model.config.num_attention_heads

    # GQA info
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_attention_heads)
    num_kv_groups = num_attention_heads // num_kv_heads

    print(f"\nModel architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  Query Heads: {num_attention_heads}")
    print(f"  KV Heads: {num_kv_heads}")
    print(f"  KV Groups: {num_kv_groups}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total attention heads: {num_layers * num_attention_heads}")

    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Copy data_path to output_dir
    shutil.copy(data_path, output_dir)

    # Save model info
    model_info = {
        "model_path": model_path,
        "model_type": model.config.model_type,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "is_moe": is_moe,
    }
    if is_moe:
        model_info["num_experts"] = num_experts
        model_info["num_experts_per_tok"] = num_experts_per_tok

    with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)

    # Initialize results
    results = torch.zeros(size=(num_layers, num_attention_heads), device=model.device)

    print(f"\nStarting path patching...")
    print(f"Processing {sample_num} samples in batches of {batch_size}")
    print(f"Estimated batches: {(sample_num + batch_size - 1) // batch_size}")

    cnt = 0
    for i in tqdm(range(0, sample_num, batch_size), desc="Batches"):
        batch_data = [full_data[j] for j in range(i, min(i + batch_size, sample_num))]
        batch_results, expert_analysis = path_patching_batch(
            model,
            tokenizer,
            batch_data,
            module_input_name,
            module_output_name,
            num_layers,
            num_attention_heads,
            head_dim,
            is_moe=is_moe,
            analyze_experts=args.analyze_experts,
            router_module_name=model_config.get("router_module_name"),
            num_experts=model_config.get("num_experts") if is_moe else None
        )
        results += batch_results
        cnt += len(batch_data)

        # Clear GPU cache
        torch.cuda.empty_cache()

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {cnt}/{sample_num} samples")

    results /= cnt

    # Find top heads
    indices = torch.topk(results.flatten(), k=min(16, results.numel()), largest=False).indices.detach().cpu().numpy()
    all_results = []
    for idx in indices:
        all_results.append((idx // num_attention_heads, idx % num_attention_heads))

    print(f"\n{'='*60}")
    print(f"Top {len(all_results)} Attention Heads (Most Impactful):")
    print(f"{'='*60}")
    for idx, (layer, head) in enumerate(all_results):
        impact = results[layer, head].item()
        print(f"{idx+1:2d}. Layer {layer:2d}, Head {head:2d}: {impact:+.4f}%")
    print(f"{'='*60}\n")

    # Save results
    results_path = os.path.join(output_dir, "results.pt")
    torch.save(results.detach().cpu(), results_path)
    print(f"Results saved to: {results_path}")

    # Generate heatmap
    fig = show_path_patching_results(
        results.detach().cpu().numpy(),
        title=f"Medical Path Patching ({model.config.model_type}): Effect of patching (Heads->Final Residual Stream)",
        return_fig=True,
        show_fig=False,
        bartitle="% change in logit difference",
    )

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
    print(f"  - model_info.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
