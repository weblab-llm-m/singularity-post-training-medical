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
# 重複排除・連問ID付与
# ============================================================

def deduplicate_and_fix_ids(items: list[dict]) -> list[dict]:
    """純粋な重複を排除し、連問（同一IDで内容が異なる問題）には連番IDを付与する。

    - 同一 problem_id かつ同一 problem_text → 純粋な重複として1件だけ残す
    - 同一 problem_id だが problem_text が異なる → 連問として _part1, _part2 を付与
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for item in items:
        groups[item["problem_id"]].append(item)

    result = []
    stats = {"unique": 0, "dup_removed": 0, "sequential_fixed": 0}

    for pid, recs in groups.items():
        if len(recs) == 1:
            result.append(recs[0])
            stats["unique"] += 1
            continue

        # 同一IDが複数存在: 内容で判別
        seen_texts = {}
        unique_recs = []
        for rec in recs:
            text_key = rec["problem_text"].strip()
            if text_key not in seen_texts:
                seen_texts[text_key] = rec
                unique_recs.append(rec)
            else:
                stats["dup_removed"] += 1

        if len(unique_recs) == 1:
            # 純粋な重複のみだった
            result.append(unique_recs[0])
            stats["unique"] += 1
        else:
            # 連問: 連番IDを付与して全件保持
            for i, rec in enumerate(unique_recs, 1):
                rec["problem_id"] = f"{pid}_part{i}"
                result.append(rec)
                stats["sequential_fixed"] += 1

    print(f"  重複排除: 純粋重複 {stats['dup_removed']} 件除外, "
          f"連問ID修正 {stats['sequential_fixed']} 件")
    return result


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

    # --- Pass 1: フィルタリング ---
    filtered = []
    skipped = 0
    for ex in ds:
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

        item = {
            "problem_id": ex.get("problem_id"),
            "year": ex.get("year"),
            "problem_text": ex["problem_text"],
            "choices_text": format_choices(ex["choices"]),
            "answer": format_answer(ex["answer"]),
            "category": ex.get("category", ""),
        }
        filtered.append(item)

    print(f"  フィルタ後: {len(filtered)} 件 (除外: {skipped} 件)")

    # --- Pass 2: 重複排除・連問ID付与 ---
    deduped = deduplicate_and_fix_ids(filtered)

    # --- 書き出し ---
    with out.open("w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  最終件数: {len(deduped)} 件")
    print(f"  出力: {output_path}")
    return len(deduped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weblab-LLM-M/igakuqa-2001-2024-filtered")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/igakuqa_filtered.jsonl")
    parser.add_argument("--exclude-years", nargs="+", type=int, default=[2023, 2024])
    args = parser.parse_args()
    prepare(args.dataset, args.split, args.output, args.exclude_years)