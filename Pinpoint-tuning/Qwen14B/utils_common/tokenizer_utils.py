"""
Tokenizer Utilities for Qwen3-14B Medical Path Patching
トークナイザー関連の共通処理
"""

from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import torch


class TokenizerUtils:
    """Qwen3トークナイザー関連のユーティリティクラス"""

    @staticmethod
    def initialize_qwen3_tokenizer(model_path: str):
        """
        Qwen3トークナイザーの初期化

        Args:
            model_path: モデルのパス

        Returns:
            tokenizer: 初期化されたトークナイザー
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )

        # pad_tokenが設定されていない場合はeos_tokenを使用
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    @staticmethod
    def find_token_positions(
        text: str,
        tokenizer,
        target_words: List[str]
    ) -> Dict[str, List[int]]:
        """
        特定の単語のトークン位置を検出

        Args:
            text: 入力テキスト
            tokenizer: トークナイザー
            target_words: 検出したい単語のリスト

        Returns:
            positions: {単語: [トークン位置のリスト]}
        """
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        positions = {}

        for word in target_words:
            word_tokens = tokenizer.tokenize(word)

            # トークン列のマッチング
            for i in range(len(tokens) - len(word_tokens) + 1):
                if tokens[i:i+len(word_tokens)] == word_tokens:
                    if word not in positions:
                        positions[word] = []
                    positions[word].extend(list(range(i, i+len(word_tokens))))

        return positions

    @staticmethod
    def get_token_spans(
        text: str,
        tokenizer,
        target_phrase: str
    ) -> List[Tuple[int, int]]:
        """
        フレーズの開始・終了トークン位置を取得

        Args:
            text: 入力テキスト
            tokenizer: トークナイザー
            target_phrase: 検出したいフレーズ

        Returns:
            spans: [(start_pos, end_pos), ...]
        """
        tokens = tokenizer.tokenize(text)
        phrase_tokens = tokenizer.tokenize(target_phrase)

        spans = []
        for i in range(len(tokens) - len(phrase_tokens) + 1):
            if tokens[i:i+len(phrase_tokens)] == phrase_tokens:
                spans.append((i, i + len(phrase_tokens)))

        return spans

    @staticmethod
    def batch_encode_with_special_tokens(
        texts: List[str],
        tokenizer,
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        バッチエンコーディング

        Args:
            texts: テキストのリスト
            tokenizer: トークナイザー
            max_length: 最大長
            padding: パディング方法
            truncation: 切り詰めを行うか

        Returns:
            encoded: {input_ids, attention_mask}
        """
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

        return encoded

    @staticmethod
    def decode_token_ids(
        token_ids: torch.Tensor,
        tokenizer,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        トークンIDをテキストにデコード

        Args:
            token_ids: トークンIDのテンソル [batch_size, seq_len]
            tokenizer: トークナイザー
            skip_special_tokens: 特殊トークンをスキップするか

        Returns:
            texts: デコードされたテキストのリスト
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        texts = tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

        return texts

    @staticmethod
    def get_special_token_positions(
        text: str,
        tokenizer
    ) -> Dict[str, List[int]]:
        """
        特殊トークンの位置を取得

        Args:
            text: 入力テキスト
            tokenizer: トークナイザー

        Returns:
            positions: {特殊トークン名: [位置のリスト]}
        """
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)

        special_positions = {}

        # BOS, EOS, PADなどの特殊トークンを検出
        if tokenizer.bos_token_id is not None:
            bos_positions = [i for i, tid in enumerate(token_ids)
                           if tid == tokenizer.bos_token_id]
            if bos_positions:
                special_positions['bos'] = bos_positions

        if tokenizer.eos_token_id is not None:
            eos_positions = [i for i, tid in enumerate(token_ids)
                           if tid == tokenizer.eos_token_id]
            if eos_positions:
                special_positions['eos'] = eos_positions

        if tokenizer.pad_token_id is not None:
            pad_positions = [i for i, tid in enumerate(token_ids)
                           if tid == tokenizer.pad_token_id]
            if pad_positions:
                special_positions['pad'] = pad_positions

        return special_positions
