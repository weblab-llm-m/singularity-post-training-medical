#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Medical Pinpoint Tuning Training Script for MoE Models
Adapted for Qwen3-30B-A3B-Instruct-2507 (MoE)
'''

import logging

from transformers import set_seed

from dataset import build_dataset
from model import build_model, build_tokenizer
from trainer import build_trainer
from utils import logging_rank, transformers_logging_setup
from utils.arguments import parse_args

from utils.utils_spt import freeze_modules, freeze_modules_moe, freeze_lora_modules, freeze_lora_modules_moe

logger = logging.getLogger('DeepSpeed')


def main():

    transformers_logging_setup()
    args = parse_args()

    # Set seed before initializing model.
    set_seed(args.seed)

    model = build_model(args)
    tokenizer = build_tokenizer(args)

    logging_rank("tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id))
    logging_rank("tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id))
    logging_rank("tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id))

    if args.precise_level > 0 and args.path_patching_path is not None:
        # MoE-specific freeze handling
        if args.model_type == 'qwen3_moe':
            logging_rank("Using MoE-specific freeze_modules")

            # Parse train_selected_experts if provided
            train_selected_experts = None
            if hasattr(args, 'train_selected_experts') and args.train_selected_experts:
                train_selected_experts = [int(x) for x in args.train_selected_experts.split(',')]
                logging_rank(f"Training selected experts: {train_selected_experts}")

            if args.peft_type is not None and args.peft_type == 'lora':
                freeze_lora_modules_moe(
                    model,
                    args.path_patching_path,
                    args.precise_level,
                    args.train_topk,
                    args.train_kv,
                    freeze_router=getattr(args, 'freeze_router', True),
                    freeze_experts=getattr(args, 'freeze_experts', True),
                    train_selected_experts=train_selected_experts
                )
            else:
                freeze_modules_moe(
                    model,
                    args.path_patching_path,
                    args.precise_level,
                    args.train_topk,
                    args.train_kv,
                    freeze_router=getattr(args, 'freeze_router', True),
                    freeze_experts=getattr(args, 'freeze_experts', True),
                    train_selected_experts=train_selected_experts
                )

        # Dense model handling (qwen2, qwen3, llama, mistral)
        elif args.model_type in ['qwen2', 'qwen3', 'llama', 'mistral']:
            logging_rank("Using standard freeze_modules for dense model")
            if args.peft_type is not None and args.peft_type == 'lora':
                freeze_lora_modules(model, args.path_patching_path, args.precise_level, args.train_topk, args.train_kv)
            else:
                freeze_modules(model, args.path_patching_path, args.precise_level, args.train_topk, args.train_kv)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}. You need to implement freeze_modules for your model type.")

    train_data = build_dataset(args, tokenizer)
    logging_rank(f"length of train dataset: {len(train_data)}")

    trainer = build_trainer(args, model, tokenizer, train_data)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir, keep_zero=True)

    logger.info("-*-" * 25 + "\nTraining completed! Congratulations!")


if __name__ == '__main__':
    main()
