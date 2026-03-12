#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
JSONL Dataset Loader for Pinpoint Tuning
Supports igakuqa.jsonl format: {"messages": [...], "solution": "..."}
'''

import json
import logging
import os
import concurrent.futures
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import torch
import torch.distributed
from addict import Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import IGNORE_INDEX, logging_rank

logger = logging.getLogger("DeepSpeed")


def build_dataset_jsonl(args: Namespace, tokenizer: PreTrainedTokenizer) -> Dataset:
    return JsonlDataset(args, tokenizer)


class JsonlDataset(Dataset):
    """
    JSONL dataset for igakuqa format:
      {"messages": [{"role": "user", "content": "..."}], "solution": "c"}
    """

    def __init__(self, args: Namespace, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.rank, self.world_size = self._get_rank_world()
        self.load_data()

    def _get_rank_world(self) -> Tuple[int, int]:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def load_data(self) -> None:
        cache_path = self._get_cache_path()

        if not os.path.exists(cache_path) and self.rank == 0:
            raw_data = []
            with open(self.args.data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    # igakuqa format: messages (user only) + solution
                    user_msgs = item.get('messages', [])
                    solution = item.get('solution', '')
                    if not user_msgs or not solution:
                        continue
                    messages = list(user_msgs) + [{"role": "assistant", "content": str(solution)}]
                    raw_data.append({"messages": messages})

            logging_rank(f"Loaded {len(raw_data)} samples from {self.args.data_path}")

            self.tokenized_data = self._tokenize_data(
                raw_data,
                self.tokenizer,
                self.args.max_seq_length,
                self.args.padding,
                train_on_prompt=self.args.train_on_prompt,
            )
            torch.save(self.tokenized_data, cache_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        logger.info(f"Loading tokenized data from {cache_path}.")
        self.tokenized_data = torch.load(cache_path, weights_only=True)

    def _get_cache_path(self) -> str:
        assert self.args.cache_dir is not None
        dataset_name = os.path.basename(self.args.data_path)
        tokenizer_name = type(self.tokenizer).__name__
        model_type = self.args.model_type
        max_seq_length = self.args.max_seq_length
        padding = self.args.padding
        train_on_prompt = self.args.train_on_prompt
        return os.path.join(
            self.args.cache_dir,
            f"{dataset_name}.cache_{model_type}_{tokenizer_name}_{max_seq_length}_{padding}_{train_on_prompt}.pt"
        )

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i) -> Dict:
        return self.tokenized_data[i]

    @staticmethod
    def _tokenize_data(data, tokenizer, max_seq_length, padding, num_workers=16, **kwargs):
        futures = []
        tokenized_data = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for index, datapoint in enumerate(data):
                future = executor.submit(
                    JsonlDataset._tokenize_datapoint,
                    datapoint, index, tokenizer, max_seq_length, padding, **kwargs
                )
                futures.append(future)
                if index % 1000 == 0:
                    logging_rank(f"Prepare Tokenizing {index} / {len(data)} datapoints.")

            logging_rank(f"Prepare tokenizing {len(data)} datapoints.")
            index = 0
            for future in concurrent.futures.as_completed(futures):
                if index % 1000 == 0:
                    logging_rank(f"Complete Tokenizing {index} / {len(data)} datapoints.")
                result = future.result()
                if result is not None:
                    tokenized_data.append(result)
                index += 1
            logging_rank(f"Finish tokenizing {len(data)} datapoints.")

        tokenized_data = sorted(tokenized_data, key=lambda x: x['index'])
        return tokenized_data

    @staticmethod
    def _tokenize_datapoint(datapoint, index, tokenizer, max_seq_length, padding, **kwargs):
        messages = datapoint['messages']
        while messages[-1]["role"] != "assistant":
            messages.pop()

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        labels = [IGNORE_INDEX] * len(input_ids)

        train_on_prompt = kwargs.get('train_on_prompt', False)
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                ids_without = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=True)
                ids_with = tokenizer.apply_chat_template(messages[:i + 1], add_generation_prompt=False, tokenize=True)
                label_start = len(ids_without)
                label_end = len(ids_with)
                if train_on_prompt or i == len(messages) - 1:
                    labels[label_start:label_end] = input_ids[label_start:label_end]

        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

        if len(input_ids) == 0 or all(x == IGNORE_INDEX for x in labels):
            return None

        attention_mask = [1] * len(input_ids)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]

        if padding:
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {'index': index, 'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
