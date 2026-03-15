#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Medical QA Dataset Loader for Pinpoint Tuning (MoE version)
Adapted for gynecology guideline parquet data
'''

import json
import logging
import os
import concurrent.futures
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Tuple

import pandas as pd
import torch
import torch.distributed
from addict import Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import IGNORE_INDEX, logging_rank

logger = logging.getLogger("DeepSpeed")


def build_dataset_medical(args: Namespace,
                          tokenizer: PreTrainedTokenizer) -> Dataset:
    """Build dataset from parquet file.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to tokenize data.

    Returns
    -------
    Dataset
        Tokenized dataset.
    """
    train_dataset = MedicalDataset(args, tokenizer)
    return train_dataset


class MedicalDataset(Dataset):
    """
    Medical QA dataset from parquet files.
    """

    def __init__(self, args: Namespace,
                 tokenizer: PreTrainedTokenizer) -> None:
        """Build dataset from parquet file.

        Parameters
        ----------
        args : Namespace
            Arguments from argparse.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        """

        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.rank, self.world_size = self.get_rank_and_word_size()
        self.load_data()

    def get_rank_and_word_size(self) -> Tuple[int, int]:
        """Get rank and world size in distributed training.

        Returns
        -------
        Tuple[int, int]
            Rank and world size.
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        return rank, world_size

    def logging_message(self, msg) -> None:
        """Logging message on rank 0.

        Parameters
        ----------
        msg : _type_
            Message to log.
        """
        if self.rank == 0:
            logger.info(msg)
        return

    def load_data(self) -> None:
        """Load data from parquet file. If not exists, tokenize data and save to cache.
        """
        cache_path = self.get_cache_path()

        if not os.path.exists(cache_path) and self.rank == 0:
            # Load parquet file
            df = pd.read_parquet(self.args.data_path)

            # Convert to messages format
            data = []
            for idx, row in df.iterrows():
                # Extract prompt from first message
                prompt_msgs = row['prompt']
                if len(prompt_msgs) > 0:
                    question = prompt_msgs[0]['content']
                else:
                    continue

                # Extract ground truth answer
                if 'reward_model' in row and isinstance(row['reward_model'], dict):
                    ground_truth = row['reward_model'].get('ground_truth', [])
                    if isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
                        answer = str(ground_truth[0])
                    else:
                        answer = str(ground_truth)
                else:
                    continue

                # Create messages format
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]

                data.append({"messages": messages})

            self.logging_message(f"Loaded {len(data)} samples from parquet")
            self.logging_message(f"Tokenizing data and saving to {cache_path}")

            self.tokenized_data = self._tokenize_data(
                data,
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

    def get_cache_path(self) -> str:
        """Get the name of the cache file.

        Returns
        -------
        str
            The path of the cache file.
        """
        assert self.args.cache_dir is not None, "Cache directory must be specified."

        dataset_name = os.path.basename(self.args.data_path)
        tokenizer_name = type(self.tokenizer).__name__
        model_type = self.args.model_type
        max_seq_length = self.args.max_seq_length
        padding = self.args.padding
        train_on_prompt = self.args.train_on_prompt

        cache_path = os.path.join(
            self.args.cache_dir,
            f"{dataset_name}.cache_{model_type}_{tokenizer_name}_{max_seq_length}_{padding}_{train_on_prompt}.pt"
        )
        return cache_path

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.tokenized_data[i]

    @staticmethod
    def _tokenize_data(data: List[dict],
                       tokenizer: PreTrainedTokenizer,
                       max_seq_length: int,
                       padding: bool,
                       num_workers: int = 16,
                       **kwargs) -> List[Dict[str, List[int]]]:
        """Tokenize a list of data in parallel.

        Parameters
        ----------
        data : List[dict]
            Data to tokenize.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        max_seq_length : int
            Max sequence length.
        padding : bool
            Whether to pad the sequence.
        num_workers : int, optional
            Number of workers in parallel, by default 16

        Returns
        -------
        List[Dict[str, List[int]]]
            Tokenized data.
        """

        futures = []
        tokenized_data = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for index, datapoint in enumerate(data):
                future = executor.submit(MedicalDataset._tokenize_datapoint,
                                         datapoint, index, tokenizer,
                                         max_seq_length, padding,
                                         **kwargs)
                futures.append(future)
                if index % 1000 == 0:
                    logging_rank(
                        f"Prepare Tokenizing {index} / {len(data)} datapoints."
                    )

            logging_rank(f"Prepare tokenizing {len(data)} datapoints.")

            index = 0
            for future in concurrent.futures.as_completed(futures):
                if index % 1000 == 0:
                    logging_rank(
                        f"Complete Tokenizing {index} / {len(data)} datapoints."
                    )
                result = future.result()
                if result is not None:
                    tokenized_data.append(result)
                index += 1

            logging_rank(f"Finish tokenizing {len(data)} datapoints.")

        tokenized_data = sorted(tokenized_data, key=lambda x: x['index'])
        return tokenized_data

    @staticmethod
    def _tokenize_datapoint(datapoint: dict, index: int,
                            tokenizer: PreTrainedTokenizer,
                            max_seq_length: int, padding: bool,
                            **kwargs) -> Dict[str, List[int]]:
        """Tokenize a single datapoint.

        Parameters
        ----------
        datapoint : dict
            Datapoint to tokenize.
        index : int
            Index of the datapoint.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        max_seq_length : int
            Max sequence length.
        padding : bool
            Whether to pad the sequence.

        Returns
        -------
        Dict[str, List[int]]
            Tokenized result.
        """

        input_ids, labels = MedicalDataset._tokenize_messages(
            datapoint, tokenizer, **kwargs)

        # Skip empty datapoints.
        if len(input_ids) == 0 or len(labels) == 0 or all(x == IGNORE_INDEX
                                                          for x in labels):
            return None

        attention_mask = [1] * len(input_ids)
        max_seq_length = max_seq_length

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]

        if padding:
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [IGNORE_INDEX] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        tokenized_datapoint = {
            'index': index,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        return tokenized_datapoint

    @staticmethod
    def _tokenize_messages(datapoint: dict,
                           tokenizer: PreTrainedTokenizer,
                           train_on_prompt: bool = True,
                           **kwargs) -> Tuple[List[int], List[int]]:
        """Tokenize messages for instruction tuning.

        Parameters
        ----------
        datapoint : dict
            Datapoint containing 'messages' key.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        train_on_prompt : bool, optional
            Whether to train on prompt, by default True

        Returns
        -------
        Tuple[List[int], List[int]]
            Input ids and labels.
        """

        assert 'messages' in datapoint, "Datapoint must contain 'messages' key."
        messages = datapoint['messages']

        # Make sure the last message is from assistant
        while messages[-1]["role"] != "assistant":
            _ = messages.pop()

        assert hasattr(tokenizer, 'apply_chat_template'
                       ), "Tokenizer must have .apply_chat_template() method."

        input_ids = tokenizer.apply_chat_template(messages,
                                                  add_generation_prompt=False,
                                                  tokenize=True)
        labels = [IGNORE_INDEX] * len(input_ids)

        # Use the whole message as the label
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                input_ids_without_target = tokenizer.apply_chat_template(
                    messages[:i], add_generation_prompt=True, tokenize=True)
                input_ids_with_target = tokenizer.apply_chat_template(
                    messages[:i + 1],
                    add_generation_prompt=False,
                    tokenize=True)

                label_start = len(input_ids_without_target)
                label_end = len(input_ids_with_target)

                # Check if the input_ids match
                assert input_ids_without_target[:
                                                -1] == input_ids[:label_start -
                                                                 1], "input_ids without target mismatch"
                assert input_ids_with_target == input_ids[:
                                                          label_end], "input_ids mismatch"

                if train_on_prompt or i == len(messages) - 1:
                    labels[label_start:label_end] = input_ids[
                        label_start:label_end]

        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

        return input_ids, labels
