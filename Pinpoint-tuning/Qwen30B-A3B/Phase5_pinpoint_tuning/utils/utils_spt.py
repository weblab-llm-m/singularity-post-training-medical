#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised Pinpoint Tuning utilities for Qwen3-30B-A3B MoE model.

Based on the original utils_spt.py by 陈蔚 (weichen.cw@zju.edu.cn)
Modified for MoE architecture support.

Key differences from Dense model version:
1. MoE layer structure: mlp.gate (router) + mlp.experts.{j}
2. GQA ratio: 32 query heads / 4 KV heads = 8 groups
3. Additional q_norm/k_norm layers in attention
4. Expert-level freeze/unfreeze support
"""

import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Set

import torch
import torch.nn as nn

from transformers import PreTrainedModel

try:
    from utils import logging_rank
except ImportError:
    def logging_rank(msg):
        print(msg)


def load_path_patching_results(
    path_patching_path: str,
    num_key_value_groups: int,
    train_topk: int = 32
) -> Tuple[Dict[int, Set[int]], Dict[int, List[int]]]:
    """Load path patching results from results.pt file.

    Parameters
    ----------
    path_patching_path : str
        Path to path patching results directory
    num_key_value_groups : int
        Number of key-value groups (for GQA)
    train_topk : int, optional
        How many heads to train, by default 32

    Returns
    -------
    Tuple[Dict[int, Set[int]], Dict[int, List[int]]]
        all_heads: Dict mapping layer_id to set of head_ids
        all_groups: Dict mapping layer_id to list of group_ids for key-value heads
    """
    assert os.path.exists(path_patching_path), f"{path_patching_path} not exists"

    path_patching_results = torch.load(
        os.path.join(path_patching_path, 'results.pt'),
        weights_only=True
    ).numpy()
    logging_rank(f"Load path patching results from {path_patching_path}")

    all_heads = {}
    _, n_heads = path_patching_results.shape
    path_patching_results = path_patching_results.reshape(-1)

    # Heads with lower path patching results are selected, since they have
    # stronger negative influence on the model after being patched.
    min_idxs = np.argsort(path_patching_results)

    for _, idx in enumerate(min_idxs[:train_topk]):
        layer_id = idx // n_heads
        head_id = idx % n_heads
        cur_set = all_heads.get(layer_id, set())
        cur_set.add(head_id)
        all_heads[layer_id] = cur_set

    # Compute groups_to_update for key_value_heads
    all_groups = {}
    for layer, heads in all_heads.items():
        groups = set()
        for head in heads:
            if head is None:
                continue
            group = head // num_key_value_groups
            groups.add(group)
        all_groups[layer] = list(groups)

    return all_heads, all_groups


def load_trainable_heads_from_json(
    trainable_heads_path: str,
    num_key_value_groups: int
) -> Tuple[Dict[int, Set[int]], Dict[int, List[int]]]:
    """Load trainable heads from trainable_heads.json file.

    Parameters
    ----------
    trainable_heads_path : str
        Path to trainable_heads.json file
    num_key_value_groups : int
        Number of key-value groups (for GQA)

    Returns
    -------
    Tuple[Dict[int, Set[int]], Dict[int, List[int]]]
        all_heads: Dict mapping layer_id to set of head_ids
        all_groups: Dict mapping layer_id to list of group_ids for key-value heads
    """
    assert os.path.exists(trainable_heads_path), f"{trainable_heads_path} not exists"

    with open(trainable_heads_path, 'r') as f:
        trainable_heads = json.load(f)

    logging_rank(f"Load {len(trainable_heads)} trainable heads from {trainable_heads_path}")

    all_heads = {}
    for head_info in trainable_heads:
        layer_id = head_info['layer']
        head_id = head_info['head']

        cur_set = all_heads.get(layer_id, set())
        cur_set.add(head_id)
        all_heads[layer_id] = cur_set

    # Compute groups_to_update for key_value_heads
    all_groups = {}
    for layer, heads in all_heads.items():
        groups = set()
        for head in heads:
            if head is None:
                continue
            group = head // num_key_value_groups
            groups.add(group)
        all_groups[layer] = list(groups)

    return all_heads, all_groups


def freeze_modules_moe(
    model: PreTrainedModel,
    path_patching_path: str,
    precise_level: int = 0,
    train_topk: int = 32,
    train_kv: bool = False,
    freeze_router: bool = True,
    freeze_experts: bool = True,
    train_selected_experts: Optional[Dict[int, List[int]]] = None
) -> None:
    """Freeze some unimportant modules in Qwen3-MoE model.

    This function supports MoE-specific freezing strategies:
    - Router (mlp.gate): Controls expert selection, usually frozen for stability
    - Experts (mlp.experts.{j}): Individual expert MLPs, can be selectively unfrozen

    Parameters
    ----------
    model : PreTrainedModel
        Qwen3-MoE model.
    path_patching_path : str
        Path to path patching results directory or trainable_heads.json file.
    precise_level : int, optional
        Precise level used in pinpoint tuning:
        - 0: Train everything (no freezing)
        - 1: Freeze embeddings
        - 2: Freeze MLP layers
        - 3: Freeze o_proj (keep q_proj trainable)
        - 4: Freeze q_proj (use gradient hooks)
        - 5: functional_classification mode (recommended for MoE)
        by default 0
    train_topk : int, optional
        How many heads to train, by default 32 (only used if path_patching_path is a directory)
    train_kv : bool, optional
        Whether to train key, value of heads in attention, by default False
    freeze_router : bool, optional
        Whether to freeze the MoE router (mlp.gate), by default True
    freeze_experts : bool, optional
        Whether to freeze all expert layers, by default True
    train_selected_experts : Optional[Dict[int, List[int]]], optional
        Dict mapping layer_id to list of expert_ids to train.
        Only used if freeze_experts=False. by default None
    """
    assert os.path.exists(path_patching_path), f"{path_patching_path} not exists"

    # Get model configuration
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads

    if hasattr(model.config, 'head_dim'):
        head_dim = model.config.head_dim
    else:
        head_dim = hidden_size // num_attention_heads

    num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads

    # MoE specific config
    num_experts = getattr(model.config, 'num_experts', 128)
    num_experts_per_tok = getattr(model.config, 'num_experts_per_tok', 8)

    logging_rank(f"MoE Config: {num_experts} experts, {num_experts_per_tok} active per token")
    logging_rank(f"Attention Config: {num_attention_heads} heads, {num_key_value_groups} KV groups")

    # Load path patching results or trainable heads
    if path_patching_path.endswith('.json'):
        all_heads, all_groups = load_trainable_heads_from_json(path_patching_path, num_key_value_groups)
    else:
        all_heads, all_groups = load_path_patching_results(path_patching_path, num_key_value_groups, train_topk)

    # Define gradient hooks for selective head training
    def freeze_q_proj_hook(layer_module: nn.Module, heads_to_update: Set[int]) -> None:
        """Freeze attention layer q_proj except for selected heads"""
        def hook(grad):
            new_grad = torch.zeros_like(grad, device=grad.device)
            for head in heads_to_update:
                new_grad[head * head_dim: (head + 1) * head_dim] = grad[head * head_dim: (head + 1) * head_dim]
            return new_grad

        layer_module.self_attn.q_proj.weight.register_hook(hook)

    def freeze_kv_proj_hook(layer_module: nn.Module, groups_to_update: List[int]) -> None:
        """Freeze attention layer k_proj/v_proj except for selected groups"""
        def hook(grad):
            new_grad = torch.zeros_like(grad, device=grad.device)
            for head in groups_to_update:
                new_grad[head * head_dim: (head + 1) * head_dim] = grad[head * head_dim: (head + 1) * head_dim]
            return new_grad

        layer_module.self_attn.k_proj.weight.register_hook(hook)
        layer_module.self_attn.v_proj.weight.register_hook(hook)

    def freeze_o_proj_hook(layer_module: nn.Module, heads_to_update: Set[int]) -> None:
        """Freeze attention layer o_proj except for selected heads"""
        def hook(grad):
            new_grad = torch.zeros_like(grad, device=grad.device)
            for head in heads_to_update:
                new_grad[:, head * head_dim: (head + 1) * head_dim] = grad[:, head * head_dim: (head + 1) * head_dim]
            return new_grad

        layer_module.self_attn.o_proj.weight.register_hook(hook)

    # Keep the gradient of embedding layer outputs
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    logging_rank("@@@ Start freezing modules (MoE version)...")

    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    assert precise_level <= 5, "precise_level should be <= 5"

    # Level 0-1: Unfreeze embeddings
    if precise_level <= 1:
        logging_rank("--- Unfreezing embeddings and lm_head...")
        model.lm_head.weight.requires_grad = True
        model.model.embed_tokens.weight.requires_grad = True

    # Iterate through layers
    for index, layer in enumerate(model.model.layers):

        # Check if this layer has heads to train
        if index not in all_heads:
            continue

        logging_rank(f">>> Processing Layer {index}. Heads to update: {all_heads[index]}")

        # Level 0-2: Unfreeze MLP (experts + router)
        if precise_level <= 2:
            logging_rank(f"--- Layer {index}: Unfreezing MLP...")

            for name, parameter in layer.named_parameters():
                if 'mlp' in name:
                    # Check router freeze setting
                    if 'gate' in name and freeze_router:
                        logging_rank(f"    Keeping router frozen: {name}")
                        continue

                    # Check expert freeze setting
                    if 'experts' in name:
                        if freeze_experts:
                            # Check if this expert should be trained
                            if train_selected_experts and index in train_selected_experts:
                                # Extract expert id from name (e.g., "mlp.experts.42.down_proj")
                                for expert_id in train_selected_experts[index]:
                                    if f'experts.{expert_id}.' in name:
                                        logging_rank(f"    Unfreezing selected expert: {name}")
                                        parameter.requires_grad = True
                                        break
                            continue  # Keep other experts frozen
                        else:
                            logging_rank(f"    Unfreezing expert: {name}")

                    parameter.requires_grad = True

        # Skip if no heads to update
        if len(all_heads[index]) == 0:
            continue

        # Level 0-3: Unfreeze o_proj with selective gradient masking
        if precise_level <= 3:
            logging_rank(f"--- Layer {index}: Unfreezing self_attn.o_proj...")

            for name, parameter in layer.named_parameters():
                if 'self_attn.o_proj' in name:
                    parameter.requires_grad = True

            freeze_o_proj_hook(layer, all_heads[index])

        # Level 0-4: Unfreeze q_proj with selective gradient masking
        if precise_level <= 4:
            logging_rank(f"--- Layer {index}: Unfreezing self_attn.q_proj...")

            for name, parameter in layer.named_parameters():
                if 'self_attn.q' in name:
                    parameter.requires_grad = True

            freeze_q_proj_hook(layer, all_heads[index])

            # Optionally unfreeze k_proj/v_proj
            if train_kv:
                logging_rank(f">>> Layer {index}: Groups to update: {all_groups[index]}")
                logging_rank(f"--- Layer {index}: Unfreezing self_attn.k_proj/v_proj...")

                for name, parameter in layer.named_parameters():
                    if 'self_attn.k' in name or 'self_attn.v' in name:
                        parameter.requires_grad = True

                freeze_kv_proj_hook(layer, all_groups[index])

    # Log trainable parameter summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging_rank(f"@@@ Freezing complete. Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return


def freeze_lora_modules_moe(
    model: PreTrainedModel,
    path_patching_path: str,
    precise_level: int,
    train_topk: int,
    train_kv: bool,
    freeze_router: bool = True
) -> None:
    """Combine SPT and LoRA freezing for MoE model.

    Parameters
    ----------
    model : PreTrainedModel
        Qwen3-MoE model with LoRA adapters.
    path_patching_path : str
        Path to path patching results or trainable_heads.json
    precise_level : int
        Precise level (must be >= 3 for LoRA)
    train_topk : int
        Number of heads to train
    train_kv : bool
        Whether to train KV projections
    freeze_router : bool, optional
        Whether to freeze router layers, by default True
    """
    assert os.path.exists(path_patching_path), f"{path_patching_path} not exists"
    assert precise_level >= 3, "Currently only support precise_level >= 3 when using LoRA."

    # Get model configuration
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads

    if hasattr(model.config, 'head_dim'):
        head_dim = model.config.head_dim
    else:
        head_dim = hidden_size // num_attention_heads

    num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads

    # Load path patching results or trainable heads
    if path_patching_path.endswith('.json'):
        all_heads, all_groups = load_trainable_heads_from_json(path_patching_path, num_key_value_groups)
    else:
        all_heads, all_groups = load_path_patching_results(path_patching_path, num_key_value_groups, train_topk)

    def freeze_lora_A_hook(parameter: nn.Parameter, heads_to_update: Set[int]):
        """Freeze the parameter for lora_A, used if freeze along input dimension"""
        def hook(grad):
            new_grad = torch.zeros_like(grad, device=grad.device)
            for head in heads_to_update:
                new_grad[:, head * head_dim: (head + 1) * head_dim] = grad[:, head * head_dim: (head + 1) * head_dim]
            return new_grad

        parameter.register_hook(hook)

    def freeze_lora_B_hook(parameter: nn.Parameter, heads_to_update: Set[int]):
        """Freeze the parameter for lora_B, used if freeze along output dimension"""
        def hook(grad):
            new_grad = torch.zeros_like(grad, device=grad.device)
            for head in heads_to_update:
                new_grad[head * head_dim: (head + 1) * head_dim, :] = grad[head * head_dim: (head + 1) * head_dim, :]
            return new_grad

        parameter.register_hook(hook)

    def get_update_heads(parameter_name: str, all_update_heads: Dict[int, Set[int]]) -> Optional[Set[int]]:
        """Get the heads to update for the current layer"""
        for index in all_update_heads:
            if f"layers.{index}." in parameter_name:
                return all_update_heads[index]
        return None

    logging_rank("@@@ Start freezing LoRA parameters (MoE version)...")

    for name, parameter in model.named_parameters():
        # Skip non-trainable parameters
        if not parameter.requires_grad:
            logging_rank(f">>> Skipping {name}...")
            continue

        # Skip router layers if freeze_router is True
        if freeze_router and 'gate' in name and 'lora' in name:
            logging_rank(f"--- Disabling router LoRA gradients for {name}...")
            parameter.requires_grad = False
            continue

        # If the layer is not in all_heads, disable gradients
        if get_update_heads(name, all_heads) is None:
            logging_rank(f"--- Disabling gradients for {name}...")
            parameter.requires_grad = False
            continue

        # Enable o_proj if precise_level <= 3
        if precise_level <= 3:
            if 'lora_A' in name and 'o_proj' in name:
                logging_rank(f"||| Masking for Layer: {name}...")
                freeze_lora_A_hook(parameter, get_update_heads(name, all_heads))

        # Enable q_proj / k_proj / v_proj if precise_level <= 4
        if precise_level <= 4:
            if 'lora_B' in name and 'q_proj' in name:
                logging_rank(f"||| Masking for Layer: {name}...")
                freeze_lora_B_hook(parameter, get_update_heads(name, all_heads))

            if train_kv:
                if 'lora_B' in name and ('k_proj' in name or 'v_proj' in name):
                    logging_rank(f"||| Masking for Layer: {name}...")
                    freeze_lora_B_hook(parameter, get_update_heads(name, all_heads))

    return


# Alias for backward compatibility
freeze_modules = freeze_modules_moe
freeze_lora_modules = freeze_lora_modules_moe
