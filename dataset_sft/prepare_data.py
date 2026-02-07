"""
igakuQAデータセットをHugging Faceからダウンロードし、
SFTデータ生成用にフィルタリング・整形する。

フィルタ条件（元スクリプト準拠）:
  - 2023/2024年度を除外
  - text_only == True のみ
  - answer が空でないもの
  - 「◯つ選べ」の数と answer 数が一致しないものを除外
"""

import ast
import json
import re
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# ユーティリティ（元スクリプトから移植）
# ============================================================

def clean_choice(raw):
    """文字列形式のリスト（例: "['a','b']"）を実際のリストに変換"""
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            return [item.replace("\u3000", "") for item in parsed]
        except Exception:
            return raw
    return raw


def get_required_answer_count(problem_text: str):
    """問題文中の「◯つ選べ」から必要な解答数を推定"""
    z2h = str.maketrans("０１２３４５６７８９", "0123456789")
    text_norm = problem_text.translate(z2h)

    m = re.search(r"([0-9]+)\s*つ\s*選", text_norm)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    kanji_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    m2 = re.search(r"([一二三四五])\s*つ\s*選", problem_text)
    if m2:
        return kanji_map.get(m2.group(1))

    return None


def format_choices(choices) -> str:
    """選択肢をフォーマット"""
    processed = clean_choice(choices)
    if isinstance(processed, list):
        return "\n".join([f"{chr(97+i)}. {c}" for i, c in enumerate(processed)])
    return str(processed)


def format_answer(answer) -> str:
    """正解をカンマ区切り文字列に変換"""
    cleaned = clean_choice(answer)
    if isinstance(cleaned, list):
        return ",".join(cleaned)
    return str(cleaned)


# ============================================================
# メイン処理
# ============================================================

def prepare(
    dataset_name: str = "weblab-LLM-M/igakuqa-2001-2024-filtered",
    split: str = "train",
    output_path: str = "data/igakuqa_filtered.jsonl",
    exclude_years: list = None,
):
    if exclude_years is None:
        exclude_years = [2023, 2024]

    exclude_set = {str(y) for y in exclude_years}

    # HFログイン
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)

    # データセット読み込み
    print(f"データセット読み込み中: {dataset_name} ({split})")
    ds = load_dataset(dataset_name, split=split, token=token)
    print(f"  全件数: {len(ds)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            # --- フィルタ（元スクリプト準拠） ---
            if str(ex.get("year")) in exclude_set:
                skipped += 1
                continue
            if not ex.get("text_only"):
                skipped += 1
                continue
            if not ex.get("answer") or len(ex["answer"]) == 0:
                skipped += 1
                continue

            required = get_required_answer_count(ex["problem_text"])
            if required is not None and len(ex["answer"]) != required:
                skipped += 1
                continue

            # --- 整形 ---
            item = {
                "problem_id": ex.get("problem_id"),
                "year": ex.get("year"),
                "problem_text": ex["problem_text"],
                "choices_text": format_choices(ex["choices"]),
                "answer": format_answer(ex["answer"]),
                "category": ex.get("category", ""),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"  フィルタ後: {kept} 件 (除外: {skipped} 件)")
    print(f"  出力: {output_path}")
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weblab-LLM-M/igakuqa-2001-2024-filtered")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/igakuqa_filtered.jsonl")
    parser.add_argument("--exclude-years", nargs="+", type=int, default=[2023, 2024])
    args = parser.parse_args()
    prepare(args.dataset, args.split, args.output, args.exclude_years)