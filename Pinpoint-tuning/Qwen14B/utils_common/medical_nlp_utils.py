"""
Medical NLP Utilities
医療NLP処理の共通ユーティリティ
"""

import json
from typing import List, Dict, Optional
import re


class MedicalNLPUtils:
    """医療用語処理のユーティリティクラス"""

    # 同義語マッピング
    SYNONYM_DICT = {
        "PCOS": "多嚢胞性卵巣症候群",
        "IgA抗体": "IgA",
        "IgG抗体": "IgG",
        "PCR": "核酸増幅法",
        "婦人科外来編2023": "産婦人科診療ガイドライン婦人科外来編2023",
    }

    # 医療用語パターン（正規表現）
    MEDICAL_PATTERNS = {
        'disease': r'(症候群|炎|癌|腫瘍|疾患)',
        'biomarker': r'(IgA|IgG|IgM|抗体)',
        'diagnostic': r'(検査|診断|測定|培養)',
        'treatment': r'(投与|治療|手術|療法)',
    }

    @staticmethod
    def normalize_medical_term(term: str) -> str:
        """
        医療用語の正規化

        Args:
            term: 医療用語

        Returns:
            normalized_term: 正規化された用語
        """
        # 同義語があれば置き換え
        normalized = MedicalNLPUtils.SYNONYM_DICT.get(term, term)

        # 空白の正規化
        normalized = re.sub(r'\s+', '', normalized)

        return normalized

    @staticmethod
    def classify_term_type(term: str, medical_dict: Dict[str, List[str]]) -> str:
        """
        用語タイプの判定

        Args:
            term: 医療用語
            medical_dict: 医療用語辞書

        Returns:
            category: カテゴリ名 (diseases, biomarkers, etc.)
        """
        # 辞書内を検索
        for category, terms in medical_dict.items():
            if term in terms:
                return category

        # パターンマッチングで推定
        for category, pattern in MedicalNLPUtils.MEDICAL_PATTERNS.items():
            if re.search(pattern, term):
                return category

        return "unknown"

    @staticmethod
    def extract_medical_terms_by_pattern(text: str) -> List[str]:
        """
        パターンマッチングで医療用語を抽出

        Args:
            text: 入力テキスト

        Returns:
            terms: 抽出された医療用語のリスト
        """
        terms = []

        for category, pattern in MedicalNLPUtils.MEDICAL_PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                terms.append(match.group(0))

        return list(set(terms))  # 重複除去

    @staticmethod
    def load_medical_dictionary(dict_path: str) -> Dict[str, List[str]]:
        """
        医療用語辞書をロード

        Args:
            dict_path: 辞書ファイルのパス

        Returns:
            medical_dict: 医療用語辞書
        """
        with open(dict_path, 'r', encoding='utf-8') as f:
            medical_dict = json.load(f)

        return medical_dict

    @staticmethod
    def expand_synonyms(term: str, medical_dict: Dict[str, List[str]]) -> List[str]:
        """
        同義語を展開

        Args:
            term: 医療用語
            medical_dict: 医療用語辞書

        Returns:
            synonyms: 同義語のリスト
        """
        synonyms = [term]

        # 辞書内の同義語を検索
        if term in MedicalNLPUtils.SYNONYM_DICT:
            synonyms.append(MedicalNLPUtils.SYNONYM_DICT[term])

        # 逆引き
        for key, value in MedicalNLPUtils.SYNONYM_DICT.items():
            if value == term:
                synonyms.append(key)

        return list(set(synonyms))

    @staticmethod
    def is_guideline_indicator(text: str) -> bool:
        """
        ガイドライン指示語かどうかを判定

        Args:
            text: 入力テキスト

        Returns:
            is_indicator: ガイドライン指示語かどうか
        """
        indicators = [
            "産婦人科診療ガイドライン",
            "婦人科外来編",
            "2023",
            "CQ",
            "推奨度",
            "エビデンスレベル",
        ]

        for indicator in indicators:
            if indicator in text:
                return True

        return False

    @staticmethod
    def is_reasoning_keyword(text: str) -> bool:
        """
        推論キーワードかどうかを判定

        Args:
            text: 入力テキスト

        Returns:
            is_reasoning: 推論キーワードかどうか
        """
        reasoning_keywords = [
            "<think>",
            "</think>",
            "選択肢",
            "正解は",
            "検討します",
            "考えられます",
            "から",
            "ため",
            "よって",
        ]

        for keyword in reasoning_keywords:
            if keyword in text:
                return True

        return False

    @staticmethod
    def extract_guideline_references(text: str) -> List[str]:
        """
        ガイドライン参照を抽出

        Args:
            text: 入力テキスト

        Returns:
            references: ガイドライン参照のリスト
        """
        references = []

        # CQパターン
        cq_pattern = r'CQ\s*\d+'
        cq_matches = re.findall(cq_pattern, text)
        references.extend(cq_matches)

        # ガイドラインパターン
        if "産婦人科診療ガイドライン" in text:
            references.append("産婦人科診療ガイドライン")

        return references

    @staticmethod
    def split_think_sections(text: str) -> Dict[str, str]:
        """
        <think>タグでテキストを分割

        Args:
            text: 入力テキスト

        Returns:
            sections: {before_think, think_content, after_think}
        """
        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, text, re.DOTALL)

        if match:
            think_start = match.start()
            think_end = match.end()

            return {
                'before_think': text[:think_start],
                'think_content': match.group(1),
                'after_think': text[think_end:],
                'has_think': True
            }
        else:
            return {
                'before_think': '',
                'think_content': '',
                'after_think': text,
                'has_think': False
            }
