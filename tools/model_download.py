#!/usr/bin/env python3
"""
Hugging Face Model & Dataset Downloader
モデルやデータセットを簡単にダウンロードするスクリプト
"""

from huggingface_hub import snapshot_download
import argparse
import os
from dotenv import load_dotenv

# envファイル読み込み
load_dotenv()

def download_model(model_id: str, save_dir: str = "./models"):
    """
    Hugging Faceからモデルをダウンロード
    
    Args:
        model_id: モデルID (例: "bert-base-uncased")
        save_dir: 保存先ディレクトリ
    """
    print(f"モデルをダウンロード中: {model_id}")
    local_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=f"{save_dir}/{model_id.replace('/', '_')}"
    )
    print(f"✓ ダウンロード完了: {local_path}")
    return local_path

def download_dataset(dataset_id: str, save_dir: str = "./datasets"):
    """
    Hugging Faceからデータセットをダウンロード
    
    Args:
        dataset_id: データセットID (例: "squad")
        save_dir: 保存先ディレクトリ
    """
    print(f"データセットをダウンロード中: {dataset_id}")
    local_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=f"{save_dir}/{dataset_id.replace('/', '_')}"
    )
    print(f"✓ ダウンロード完了: {local_path}")
    return local_path

def main():
    parser = argparse.ArgumentParser(
        description="Hugging Faceからモデル/データセットをダウンロード"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="モデルID (例: bert-base-uncased)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="データセットID (例: squad)"
    )
    parser.add_argument(
        "--save-dir", "-s",
        type=str,
        default=os.path.expandvars("$HOME/downloads"),
        help="保存先ディレクトリ (デフォルト: ./downloads)"
    )
    
    args = parser.parse_args()
    
    if args.model:
        download_model(args.model, f"{args.save_dir}/models")
    
    if args.dataset:
        download_dataset(args.dataset, f"{args.save_dir}/datasets")
    
    if not args.model and not args.dataset:
        print("エラー: --model または --dataset を指定してください")
        parser.print_help()

if __name__ == "__main__":
    main()

# 使用方法
# python model_download.py --model Qwen/Qwen3-Next-80B-A3B-Instruct
# python model_download.py --dataset squad