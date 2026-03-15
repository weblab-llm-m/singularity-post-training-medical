#!/usr/bin/env python3
"""
Medical NLP Utilities
医療NLP用の共通ユーティリティ関数
"""

import json
import re
from typing import Dict, List, Tuple, Optional


def load_medical_dictionary(dict_path: str) -> Dict[str, List[str]]:
    """
    医療用語辞書をロード

    Args:
        dict_path: 医療用語辞書のJSONファイルパス

    Returns:
        medical_dict: カテゴリ別の医療用語辞書
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        medical_dict = json.load(f)
    return medical_dict


def split_think_sections(text: str) -> Tuple[str, str, bool]:
    """
    テキストから<think>セクションを分離

    Args:
        text: 入力テキスト

    Returns:
        think_content: <think>タグ内のコンテンツ（存在しない場合は空文字列）
        main_content: <think>タグ外のメインコンテンツ
        has_think: <think>セクションが存在するかどうか
    """
    # <think>...</think> パターンを検索
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, text, re.DOTALL)

    if match:
        think_content = match.group(1).strip()
        # <think>タグを除いた部分を取得
        main_content = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        return think_content, main_content, True
    else:
        return '', text, False


def normalize_medical_term(term: str) -> str:
    """
    医療用語を正規化

    Args:
        term: 医療用語

    Returns:
        normalized_term: 正規化された医療用語
    """
    # 前後の空白を削除
    normalized = term.strip()

    # 全角・半角の統一
    # 数字とアルファベットを半角に統一
    normalized = normalized.translate(
        str.maketrans(
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        )
    )

    # 特殊文字の正規化
    normalized = normalized.replace('　', ' ')  # 全角スペースを半角に
    normalized = normalized.replace('－', '-')  # 全角ハイフンを半角に

    return normalized


def classify_term_type(term: str, medical_dict: Dict[str, List[str]]) -> Optional[str]:
    """
    医療用語のカテゴリを分類

    Args:
        term: 医療用語
        medical_dict: 医療用語辞書

    Returns:
        category: 該当するカテゴリ名（見つからない場合はNone）
    """
    normalized_term = normalize_medical_term(term)

    for category, terms in medical_dict.items():
        for dict_term in terms:
            if normalize_medical_term(dict_term) == normalized_term:
                return category

    return None


def find_terms_in_text(text: str, terms: List[str]) -> List[Dict]:
    """
    テキスト内から用語を検索し、位置情報を返す

    Args:
        text: 検索対象のテキスト
        terms: 検索する用語のリスト

    Returns:
        found_terms: 見つかった用語の情報リスト
            [{"term": str, "char_start": int, "char_end": int}, ...]
    """
    found_terms = []

    for term in terms:
        # 用語の出現位置をすべて検索
        start_idx = 0
        while True:
            start_idx = text.find(term, start_idx)
            if start_idx == -1:
                break

            end_idx = start_idx + len(term)
            found_terms.append({
                "term": term,
                "char_start": start_idx,
                "char_end": end_idx
            })

            start_idx = end_idx

    # 位置順にソート
    found_terms.sort(key=lambda x: x["char_start"])

    return found_terms


def extract_guideline_indicators(text: str, guideline_terms: List[str]) -> List[Dict]:
    """
    テキストからガイドライン指標を抽出

    Args:
        text: 検索対象のテキスト
        guideline_terms: ガイドライン用語のリスト

    Returns:
        indicators: ガイドライン指標の情報リスト
    """
    return find_terms_in_text(text, guideline_terms)


def extract_reasoning_keywords(text: str, reasoning_keywords: List[str]) -> List[Dict]:
    """
    テキストから推論キーワードを抽出

    Args:
        text: 検索対象のテキスト
        reasoning_keywords: 推論キーワードのリスト

    Returns:
        keywords: 推論キーワードの情報リスト
    """
    return find_terms_in_text(text, reasoning_keywords)


def validate_medical_term(term: str, min_length: int = 2, max_length: int = 50) -> bool:
    """
    医療用語の妥当性を検証

    Args:
        term: 検証する医療用語
        min_length: 最小文字数
        max_length: 最大文字数

    Returns:
        is_valid: 妥当性（True/False）
    """
    if not term or not isinstance(term, str):
        return False

    term_len = len(term.strip())
    if term_len < min_length or term_len > max_length:
        return False

    # 空白のみの場合は無効
    if not term.strip():
        return False

    return True


def get_term_statistics(medical_dict: Dict[str, List[str]]) -> Dict[str, int]:
    """
    医療用語辞書の統計情報を取得

    Args:
        medical_dict: 医療用語辞書

    Returns:
        statistics: カテゴリ別の用語数
    """
    statistics = {}
    for category, terms in medical_dict.items():
        statistics[category] = len(terms)

    statistics['total'] = sum(statistics.values())

    return statistics


if __name__ == "__main__":
    # テスト用コード
    test_text = """
    <think>
    この症例では子宮内膜症が疑われます。
    ガイドラインに従って診断を進めます。
    </think>
    したがって、経腟超音波検査とCA125測定を実施します。
    """

    print("Testing split_think_sections...")
    think_content, main_content, has_think = split_think_sections(test_text)
    print(f"Has think section: {has_think}")
    print(f"Think content: {think_content}")
    print(f"Main content: {main_content}")

    print("\nTesting normalize_medical_term...")
    print(normalize_medical_term("ＣＡ１２５測定　"))

    print("\nTesting validate_medical_term...")
    print(validate_medical_term("CA125"))
    print(validate_medical_term(""))
    print(validate_medical_term("A"))
