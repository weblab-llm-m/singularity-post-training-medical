#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinpoint Tuning utilities for Qwen3-30B-A3B MoE model.
"""

import json
import logging
import os
import shutil
from typing import List

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.utils import is_peft_available
from transformers.utils.logging import set_verbosity_info, enable_default_handler, enable_explicit_format

from .utils_spt import (
    freeze_modules,
    freeze_modules_moe,
    freeze_lora_modules,
    freeze_lora_modules_moe,
    load_path_patching_results,
    load_trainable_heads_from_json,
)

# Constants
IGNORE_INDEX = -100

SUPPORTED_MODEL_CLASSES = (PreTrainedModel, ) if not is_peft_available() else (
    PreTrainedModel, PeftModel)

logger = logging.getLogger('DeepSpeed')


def transformers_logging_setup() -> None:
    """Setup logging basics for transformers."""
    set_verbosity_info()
    enable_default_handler()
    enable_explicit_format()


def logging_rank(message: str, rank: int = 0) -> None:
    """Log message on the rank provided.

    Parameters
    ----------
    message : str
        Message to log.
    rank : int, optional
        Rank used to log, by default 0
    """
    if not dist.is_initialized():
        logger.info(message)
        return

    if dist.get_rank() == rank:
        logger.info(message)


def delete_zero_hf(output_dir: str) -> None:
    """Delete zero states in previous output checkpoints.

    Parameters
    ----------
    output_dir : str
        Output directory whose zero states will be deleted.
    """
    if torch.distributed.get_rank() == 0:
        if os.path.exists(output_dir):
            for file_name in os.listdir(output_dir):
                if "global_step" in file_name:
                    shutil.rmtree(os.path.join(output_dir, file_name))


def read_jsonl(path: str) -> List[dict]:
    """Read jsonl file."""
    return [json.loads(line) for line in open(path, 'r', encoding='utf-8')]


def write_jsonl(datapoints: List[dict], path: str, overwrite: bool = True) -> None:
    """Write datapoints to jsonl file."""
    base_dir = os.path.dirname(path)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(path, 'w' if overwrite else 'a', encoding='utf-8') as file:
        for datapoint in datapoints:
            file.write(json.dumps(datapoint, ensure_ascii=False) + "\n")


__all__ = [
    'freeze_modules',
    'freeze_modules_moe',
    'freeze_lora_modules',
    'freeze_lora_modules_moe',
    'load_path_patching_results',
    'load_trainable_heads_from_json',
    'IGNORE_INDEX',
    'SUPPORTED_MODEL_CLASSES',
    'transformers_logging_setup',
    'logging_rank',
    'delete_zero_hf',
    'read_jsonl',
    'write_jsonl',
]
