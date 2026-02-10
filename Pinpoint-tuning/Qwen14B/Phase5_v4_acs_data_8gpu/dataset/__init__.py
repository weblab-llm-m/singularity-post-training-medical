#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Dataset module for medical pinpoint tuning
'''

from dataset.dataset_medical import build_dataset_medical


def build_dataset(args, tokenizer):
    """Build dataset based on data type.

    For medical QA (parquet files), use build_dataset_medical.
    """
    # Check if data_path is a parquet file
    if args.data_path.endswith('.parquet'):
        return build_dataset_medical(args, tokenizer)
    else:
        raise ValueError(f"Unsupported data format: {args.data_path}")


__all__ = ['build_dataset', 'build_dataset_medical']
