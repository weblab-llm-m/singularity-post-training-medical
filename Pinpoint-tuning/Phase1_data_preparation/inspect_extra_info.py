#!/usr/bin/env python3
"""
Inspect Extra Info
parquetファイルのextra_infoカラムを検査
"""

import pandas as pd
import json
import argparse


def inspect_extra_info(parquet_file: str, num_samples: int = 5):
    """
    parquetファイルのextra_infoカラムを検査

    Args:
        parquet_file: parquetファイルのパス
        num_samples: 表示するサンプル数
    """
    print("="*60)
    print("Inspecting Extra Info Column")
    print("="*60 + "\n")

    # データをロード
    print(f"Loading: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}\n")

    # extra_infoカラムを検査
    if 'extra_info' not in df.columns:
        print("ERROR: 'extra_info' column not found!")
        return

    print(f"Inspecting first {num_samples} samples of 'extra_info' column:")
    print("="*60)

    for idx in range(min(num_samples, len(df))):
        print(f"\n--- Sample {idx+1} ---")
        extra_info = df.iloc[idx]['extra_info']

        if isinstance(extra_info, dict):
            print(f"Type: dict")
            print(f"Keys: {list(extra_info.keys())}")

            for key, value in extra_info.items():
                if isinstance(value, str):
                    # 長い文字列は省略
                    if len(value) > 200:
                        print(f"  {key}: {value[:200]}... (length: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"Type: {type(extra_info)}")
            print(f"Value: {extra_info}")

    print("\n" + "="*60)

    # 統計情報
    print("\nStatistics:")
    print(f"  Total rows: {len(df)}")

    if 'extra_info' in df.columns:
        # extra_infoの型分布
        type_counts = df['extra_info'].apply(lambda x: type(x).__name__).value_counts()
        print(f"  Extra info types:")
        for dtype, count in type_counts.items():
            print(f"    {dtype}: {count}")

        # 辞書型の場合、キーの統計
        dict_samples = df[df['extra_info'].apply(lambda x: isinstance(x, dict))]
        if len(dict_samples) > 0:
            # 全てのキーを集計
            all_keys = set()
            for extra_info in dict_samples['extra_info']:
                if isinstance(extra_info, dict):
                    all_keys.update(extra_info.keys())

            print(f"  Common keys in extra_info dict: {sorted(list(all_keys))}")

    print("\n✓ Inspection completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Extra Info Column")
    parser.add_argument("--input_file", type=str, required=True, help="Input parquet file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to display")

    args = parser.parse_args()

    inspect_extra_info(
        parquet_file=args.input_file,
        num_samples=args.num_samples
    )
