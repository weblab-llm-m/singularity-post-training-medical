#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model module for Pinpoint Tuning (MoE version)
'''

from argparse import Namespace

from transformers import PreTrainedModel, PreTrainedTokenizer

from model.model_hf import build_model_hf, build_tokenizer_hf


def build_model(args: Namespace) -> PreTrainedModel:
    """Build model from args

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.

    Returns
    -------
    PreTrainedModel
        Transformer model.
    """
    model = build_model_hf(args)

    # Build PEFT models
    if args.peft_type is not None:
        from model.model_peft import build_model_peft
        model = build_model_peft(args, model)

    # Enable gradient checkpointing
    if hasattr(args, "gradient_checkpointing") and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def build_tokenizer(args: Namespace) -> PreTrainedTokenizer:
    """Build tokenizer from args

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.

    Returns
    -------
    PreTrainedTokenizer
        Tokenizer.
    """
    tokenizer = build_tokenizer_hf(args)
    return tokenizer
