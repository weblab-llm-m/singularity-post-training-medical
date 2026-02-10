#!/usr/bin/env python3
"""
Tokenizer Utilities
トークナイザー用の共通ユーティリティ関数
"""

from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer


def initialize_qwen3_tokenizer(model_path: str):
    """
    Qwen3トークナイザーを初期化

    Args:
        model_path: モデルのパス

    Returns:
        tokenizer: 初期化されたトークナイザー
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return tokenizer


def find_token_positions(
    text: str,
    term: str,
    tokenizer,
    add_special_tokens: bool = False
) -> List[Dict]:
    """
    テキスト内の特定の用語のトークン位置を検索

    Args:
        text: 検索対象のテキスト
        term: 検索する用語
        tokenizer: トークナイザー
        add_special_tokens: 特殊トークンを追加するかどうか

    Returns:
        positions: 見つかった用語のトークン位置情報のリスト
            [{"term": str, "char_start": int, "char_end": int,
              "token_start": int, "token_end": int}, ...]
    """
    positions = []

    # 用語の出現位置をすべて検索
    start_idx = 0
    while True:
        start_idx = text.find(term, start_idx)
        if start_idx == -1:
            break

        end_idx = start_idx + len(term)

        # トークン位置を計算
        prefix_tokens = tokenizer.encode(
            text[:start_idx],
            add_special_tokens=add_special_tokens
        )
        full_tokens = tokenizer.encode(
            text[:end_idx],
            add_special_tokens=add_special_tokens
        )

        token_start = len(prefix_tokens)
        token_end = len(full_tokens)

        positions.append({
            "term": term,
            "char_start": start_idx,
            "char_end": end_idx,
            "token_start": token_start,
            "token_end": token_end
        })

        start_idx = end_idx

    return positions


def tokenize_with_offsets(
    text: str,
    tokenizer,
    add_special_tokens: bool = False
) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    """
    テキストをトークン化し、各トークンの文字位置オフセットを返す

    Args:
        text: トークン化するテキスト
        tokenizer: トークナイザー
        add_special_tokens: 特殊トークンを追加するかどうか

    Returns:
        token_ids: トークンIDのリスト
        tokens: トークン文字列のリスト
        char_offsets: 各トークンの文字位置オフセット [(start, end), ...]
    """
    # トークン化
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # 各トークンの文字位置を推定
    char_offsets = []
    current_pos = 0

    for token in tokens:
        # トークンの開始位置を検索
        start_pos = text.find(token, current_pos)

        if start_pos != -1:
            end_pos = start_pos + len(token)
            char_offsets.append((start_pos, end_pos))
            current_pos = end_pos
        else:
            # 見つからない場合は前のトークンの終了位置を使用
            char_offsets.append((current_pos, current_pos))

    return token_ids, tokens, char_offsets


def get_token_positions_for_terms(
    text: str,
    terms: List[str],
    tokenizer,
    add_special_tokens: bool = False
) -> Dict[str, List[int]]:
    """
    複数の用語のトークン位置を一度に取得

    Args:
        text: 検索対象のテキスト
        terms: 検索する用語のリスト
        tokenizer: トークナイザー
        add_special_tokens: 特殊トークンを追加するかどうか

    Returns:
        term_positions: 用語ごとのトークン位置のマッピング
            {term: [token_idx1, token_idx2, ...], ...}
    """
    term_positions = {}

    for term in terms:
        positions = find_token_positions(text, term, tokenizer, add_special_tokens)

        # トークンインデックスのリストを作成
        token_indices = []
        for pos in positions:
            token_indices.extend(range(pos["token_start"], pos["token_end"]))

        term_positions[term] = token_indices

    return term_positions


def align_char_to_token_positions(
    char_positions: List[Tuple[int, int]],
    token_char_offsets: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    文字位置からトークン位置への変換

    Args:
        char_positions: 文字位置のリスト [(char_start, char_end), ...]
        token_char_offsets: 各トークンの文字オフセット [(start, end), ...]

    Returns:
        token_positions: トークン位置のリスト [(token_start, token_end), ...]
    """
    token_positions = []

    for char_start, char_end in char_positions:
        token_start = None
        token_end = None

        # 開始トークンを検索
        for i, (tok_start, tok_end) in enumerate(token_char_offsets):
            if tok_start <= char_start < tok_end:
                token_start = i
                break

        # 終了トークンを検索
        for i, (tok_start, tok_end) in enumerate(token_char_offsets):
            if tok_start < char_end <= tok_end:
                token_end = i + 1
                break

        if token_start is not None and token_end is not None:
            token_positions.append((token_start, token_end))

    return token_positions


def decode_tokens(token_ids: List[int], tokenizer) -> List[str]:
    """
    トークンIDのリストを文字列のリストにデコード

    Args:
        token_ids: トークンIDのリスト
        tokenizer: トークナイザー

    Returns:
        tokens: デコードされた文字列のリスト
    """
    return [tokenizer.decode([tid]) for tid in token_ids]


def get_token_count(text: str, tokenizer, add_special_tokens: bool = False) -> int:
    """
    テキストのトークン数を取得

    Args:
        text: カウントするテキスト
        tokenizer: トークナイザー
        add_special_tokens: 特殊トークンを追加するかどうか

    Returns:
        count: トークン数
    """
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return len(token_ids)


if __name__ == "__main__":
    # テスト用コード
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "/home/Competition2025/P05/shareP05/models/Qwen3-14B"

    print("Initializing tokenizer...")
    tokenizer = initialize_qwen3_tokenizer(model_path)
    print("✓ Tokenizer initialized\n")

    test_text = "産婦人科診療ガイドラインでは、子宮内膜症の診断にCA125測定が推奨されています。"

    print(f"Test text: {test_text}\n")

    print("Testing tokenize_with_offsets...")
    token_ids, tokens, char_offsets = tokenize_with_offsets(test_text, tokenizer)
    print(f"Token count: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"First 10 offsets: {char_offsets[:10]}\n")

    print("Testing find_token_positions...")
    term = "CA125"
    positions = find_token_positions(test_text, term, tokenizer)
    print(f"Positions for '{term}': {positions}\n")

    print("Testing get_token_count...")
    count = get_token_count(test_text, tokenizer)
    print(f"Token count: {count}")
