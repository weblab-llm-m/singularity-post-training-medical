#!/usr/bin/env python3
"""
Extract problem_text from igakuQA dataset for medical dictionary generation.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def extract_problem_texts(
    dataset_name: str = "weblab-LLM-M/igakuqa-2001-2024-filtered",
    split: str = "train",
    out_path: str = None,
    num_samples: int = None
) -> None:
    """
    HuggingFaceからigakuQAデータセットを読み込み、problem_textのみを抽出してjsonlに保存

    Args:
        dataset_name: HuggingFaceのデータセット名
        split: データセットのsplit名
        out_path: 出力ファイルパス
        num_samples: 抽出するサンプル数（Noneの場合は全件）
    """

    # デフォルト出力パス
    if out_path is None:
        out_path = Path(__file__).parent / "problem_texts.jsonl"
    else:
        out_path = Path(out_path)

    # アクセストークンの読み込み
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)
        print("Hugging Faceにログインしました")

    # データセットの読み込み
    print(f"データセット '{dataset_name}' の '{split}' splitを読み込み中...")
    dataset = load_dataset(dataset_name, split=split, token=token)
    print(f"読み込み完了: {len(dataset)} サンプル")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for ex in dataset:
            # 2023, 2024年を除外（テストデータとの重複回避）
            if str(ex.get("year")) in ["2023", "2024", 2023, 2024]:
                continue

            # テキストのみの問題に限定
            if not ex.get("text_only"):
                continue

            # 回答がある問題に限定
            if not ex.get("answer") or len(ex["answer"]) == 0:
                continue

            problem_text = ex.get("problem_text", "")
            if not problem_text:
                continue

            item = {
                "problem_id": ex.get("problem_id"),
                "problem_text": problem_text,
                "year": ex.get("year"),
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

            if num_samples and count >= num_samples:
                break

    print(f"✓ {count} 件のproblem_textを保存しました: {out_path}")


if __name__ == "__main__":
    extract_problem_texts()
