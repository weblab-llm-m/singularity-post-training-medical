#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hook functions for Path Patching with MoE model support.

Based on the original hook_functions.py by 陈蔚 (cy424151@alibaba-inc.com)
Extended with MoE-specific hooks for router and expert layers.

Key additions for MoE:
1. Router output hooks - capture expert selection logits
2. Expert-level hooks - capture individual expert activations
3. Combined attention + expert analysis support
"""

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Standard Attention Hooks (from original hook_functions.py)
# ============================================================

@torch.no_grad()
def add_pre_module_hook(
    module: nn.Module,
    name: str,
    cache: Dict[str, torch.Tensor],
    pos: int = None,
    read: bool = False
):
    """Add a pre-hook to a module.

    Parameters
    ----------
    module : nn.Module
        Module to add the hook to.
    name : str
        Name of the cache entry.
    cache : Dict[str, torch.Tensor]
        Cache to save the input to.
    pos : int, optional
        Position in the sequence, by default None
    read : bool, optional
        Whether to add a read hook or a write hook, by default False

    Returns
    -------
    RemovableHandle
        Hook handle for removal
    """

    def read_hook(module, args, kwargs):
        if pos is None:
            cache[name] = kwargs
        else:
            # Handle both positional and keyword arguments
            if 'hidden_states' in kwargs:
                cache[name] = kwargs['hidden_states'][:, pos]
            elif len(args) > 0:
                cache[name] = args[0][:, pos]
            else:
                raise ValueError("Hooked module must have hidden_states")

    def write_hook(module, args, kwargs):
        if pos is None:
            return cache[name]

        # Handle both positional and keyword arguments
        if 'hidden_states' in kwargs:
            kwargs['hidden_states'][:, pos] = cache[name]
            return args, kwargs
        elif len(args) > 0:
            # Create new args tuple with modified hidden_states
            new_hidden_states = args[0].clone()
            new_hidden_states[:, pos] = cache[name]
            return (new_hidden_states,) + args[1:], kwargs
        else:
            raise ValueError("Hooked module must have hidden_states")

    if read:
        return module.register_forward_pre_hook(read_hook, with_kwargs=True)
    else:
        return module.register_forward_pre_hook(write_hook, with_kwargs=True)


@torch.no_grad()
def add_pre_module_hook_single_head(
    module: nn.Module,
    name: str,
    cache: Dict[str, torch.Tensor],
    pos: int,
    head_dim: int,
    read: bool = False,
    head_idx: int = None
):
    """Add a pre-hook to a module to modify a single head.

    Parameters
    ----------
    module : nn.Module
        Module to add the hook to.
    name : str
        Name of the cache entry.
    cache : Dict[str, torch.Tensor]
        Cache to save the input to.
    pos : int
        Position in the sequence.
    head_dim : int
        Head dimension.
    read : bool, optional
        Whether to add a read hook or a write hook, by default False
    head_idx : int, optional
        The index of the head to be modified, by default None

    Returns
    -------
    RemovableHandle
        Hook handle for removal
    """

    def read_hook(module, args, kwargs):
        # [batch, seq_len, dim]
        cache[name] = args[0][:, pos]

    def write_hook(module, args, kwargs):
        args[0][:, pos, head_idx * head_dim: (head_idx + 1) * head_dim] = \
            cache[name][:, head_idx * head_dim: (head_idx + 1) * head_dim]
        return args, kwargs

    if read:
        return module.register_forward_pre_hook(read_hook, with_kwargs=True)
    else:
        return module.register_forward_pre_hook(write_hook, with_kwargs=True)


# ============================================================
# MoE-Specific Hooks
# ============================================================

@torch.no_grad()
def add_router_hook(
    module: nn.Module,
    name: str,
    cache: Dict[str, torch.Tensor],
    read: bool = True
):
    """Add a hook to capture router (gate) outputs.

    The router determines which experts are selected for each token.
    This hook captures the router logits before expert selection.

    Parameters
    ----------
    module : nn.Module
        Router module (mlp.gate)
    name : str
        Name of the cache entry
    cache : Dict[str, torch.Tensor]
        Cache to save router outputs
    read : bool, optional
        Whether to read or write, by default True

    Returns
    -------
    RemovableHandle
        Hook handle for removal
    """

    def read_hook(module, args, kwargs, output):
        # Router output shape: [batch, seq_len, num_experts]
        cache[name] = output.clone()

    def write_hook(module, args, kwargs, output):
        # Replace router output with cached values
        return cache[name]

    if read:
        return module.register_forward_hook(read_hook, with_kwargs=True)
    else:
        return module.register_forward_hook(write_hook, with_kwargs=True)


@torch.no_grad()
def add_expert_activation_hook(
    module: nn.Module,
    name: str,
    cache: Dict[str, torch.Tensor],
    expert_idx: int,
    read: bool = True
):
    """Add a hook to capture a specific expert's activation.

    Parameters
    ----------
    module : nn.Module
        Expert module (mlp.experts.{idx})
    name : str
        Name of the cache entry
    cache : Dict[str, torch.Tensor]
        Cache to save expert activations
    expert_idx : int
        Index of the expert
    read : bool, optional
        Whether to read or write, by default True

    Returns
    -------
    RemovableHandle
        Hook handle for removal
    """

    def read_hook(module, args, kwargs, output):
        # Expert output shape depends on implementation
        cache[f"{name}_expert_{expert_idx}"] = output.clone()

    def write_hook(module, args, kwargs, output):
        # Replace expert output with cached values
        return cache[f"{name}_expert_{expert_idx}"]

    if read:
        return module.register_forward_hook(read_hook, with_kwargs=True)
    else:
        return module.register_forward_hook(write_hook, with_kwargs=True)


@torch.no_grad()
def add_moe_output_hook(
    module: nn.Module,
    name: str,
    cache: Dict[str, torch.Tensor],
    pos: int = None,
    read: bool = True
):
    """Add a hook to capture the combined MoE layer output.

    This captures the weighted sum of all expert outputs after routing.

    Parameters
    ----------
    module : nn.Module
        MoE layer module (mlp)
    name : str
        Name of the cache entry
    cache : Dict[str, torch.Tensor]
        Cache to save MoE outputs
    pos : int, optional
        Position in sequence to capture, by default None (all positions)
    read : bool, optional
        Whether to read or write, by default True

    Returns
    -------
    RemovableHandle
        Hook handle for removal
    """

    def read_hook(module, args, kwargs, output):
        if pos is None:
            cache[name] = output.clone()
        else:
            cache[name] = output[:, pos].clone()

    def write_hook(module, args, kwargs, output):
        if pos is None:
            return cache[name]
        else:
            output[:, pos] = cache[name]
            return output

    if read:
        return module.register_forward_hook(read_hook, with_kwargs=True)
    else:
        return module.register_forward_hook(write_hook, with_kwargs=True)


# ============================================================
# Utility Functions for MoE Analysis
# ============================================================

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a module by its full name.

    Parameters
    ----------
    model : nn.Module
        The model
    name : str
        Full module name (e.g., "model.layers.0.mlp.gate")

    Returns
    -------
    nn.Module
        The requested module
    """
    return model.get_submodule(name)


def get_all_expert_modules(
    model: nn.Module,
    layer_idx: int,
    num_experts: int
) -> List[Tuple[int, nn.Module]]:
    """Get all expert modules for a given layer.

    Parameters
    ----------
    model : nn.Module
        The model
    layer_idx : int
        Layer index
    num_experts : int
        Number of experts

    Returns
    -------
    List[Tuple[int, nn.Module]]
        List of (expert_idx, module) tuples
    """
    experts = []
    for i in range(num_experts):
        try:
            module = model.get_submodule(f"model.layers.{layer_idx}.mlp.experts.{i}")
            experts.append((i, module))
        except AttributeError:
            break
    return experts


def analyze_router_selections(
    router_logits: torch.Tensor,
    top_k: int = 8
) -> Dict[str, torch.Tensor]:
    """Analyze router selection patterns.

    Parameters
    ----------
    router_logits : torch.Tensor
        Router logits [batch, seq_len, num_experts]
    top_k : int, optional
        Number of experts selected per token, by default 8

    Returns
    -------
    Dict[str, torch.Tensor]
        Analysis results including:
        - selected_experts: indices of selected experts
        - selection_probs: probabilities of selected experts
        - expert_load: how often each expert is selected
    """
    # Softmax to get probabilities
    probs = torch.softmax(router_logits, dim=-1)

    # Get top-k experts
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

    # Calculate expert load (how often each expert is selected)
    batch_size, seq_len, num_experts = router_logits.shape
    expert_load = torch.zeros(num_experts, device=router_logits.device)

    for expert_idx in range(num_experts):
        expert_load[expert_idx] = (topk_indices == expert_idx).float().sum()

    expert_load /= (batch_size * seq_len * top_k)  # Normalize

    return {
        'selected_experts': topk_indices,
        'selection_probs': topk_probs,
        'expert_load': expert_load,
        'router_entropy': -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    }


# ============================================================
# Combined Attention + MoE Analysis
# ============================================================

class MoEPathPatchingHooks:
    """Manager class for MoE path patching hooks.

    This class manages the lifecycle of hooks for combined
    attention head and expert analysis.
    """

    def __init__(self, model: nn.Module, num_layers: int, num_heads: int, num_experts: int):
        """Initialize the hook manager.

        Parameters
        ----------
        model : nn.Module
            The MoE model
        num_layers : int
            Number of layers
        num_heads : int
            Number of attention heads
        num_experts : int
            Number of experts per layer
        """
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts

        self.hooks = []
        self.attention_cache = {}
        self.router_cache = {}
        self.expert_cache = {}

    def register_attention_hooks(
        self,
        layer_indices: List[int] = None,
        pos: int = -1,
        read: bool = True
    ):
        """Register hooks for attention layers.

        Parameters
        ----------
        layer_indices : List[int], optional
            Layers to hook, by default all layers
        pos : int, optional
            Sequence position, by default -1 (last)
        read : bool, optional
            Read or write mode, by default True
        """
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        for i in layer_indices:
            # Input layernorm hook
            input_module = self.model.get_submodule(f"model.layers.{i}.input_layernorm")
            hook = add_pre_module_hook(
                input_module,
                f"layer_{i}_input",
                self.attention_cache,
                pos,
                read
            )
            self.hooks.append(hook)

            # Output projection hook
            output_module = self.model.get_submodule(f"model.layers.{i}.self_attn.o_proj")
            hook = add_pre_module_hook(
                output_module,
                f"layer_{i}_output",
                self.attention_cache,
                pos,
                read
            )
            self.hooks.append(hook)

    def register_router_hooks(
        self,
        layer_indices: List[int] = None,
        read: bool = True
    ):
        """Register hooks for router layers.

        Parameters
        ----------
        layer_indices : List[int], optional
            Layers to hook, by default all layers
        read : bool, optional
            Read or write mode, by default True
        """
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        for i in layer_indices:
            try:
                router_module = self.model.get_submodule(f"model.layers.{i}.mlp.gate")
                hook = add_router_hook(
                    router_module,
                    f"layer_{i}_router",
                    self.router_cache,
                    read
                )
                self.hooks.append(hook)
            except AttributeError:
                # Layer might not have MoE
                pass

    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_cache(self):
        """Clear all caches."""
        self.attention_cache.clear()
        self.router_cache.clear()
        self.expert_cache.clear()

    def get_attention_cache(self) -> Dict[str, torch.Tensor]:
        """Get the attention cache."""
        return self.attention_cache

    def get_router_cache(self) -> Dict[str, torch.Tensor]:
        """Get the router cache."""
        return self.router_cache

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all_hooks()
        self.clear_cache()
