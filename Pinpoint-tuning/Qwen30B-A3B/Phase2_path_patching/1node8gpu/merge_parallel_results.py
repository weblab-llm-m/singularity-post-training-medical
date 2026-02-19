#!/usr/bin/env python3
"""
8並列Path Patching結果を統合するスクリプト

各プロセスのresults.ptはサンプル数で平均化済みなので、
サンプル数による重み付け平均で統合する。
サンプル数はログファイルから自動取得。
"""

import argparse
import re
import torch
from pathlib import Path


def get_sample_count(results_dir: Path, process_id: int) -> int:
    """ログファイルからサンプル数を取得"""
    log_path = results_dir / f"process_{process_id}.log"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            m = re.search(r"Using samples:\s+(\d+)", line)
            if m:
                return int(m.group(1))
    raise ValueError(f"Cannot determine sample count for process {process_id}")


def merge_results(results_dir: str, num_processes: int = 8):
    results_dir = Path(results_dir)

    combined = None
    total_samples = 0

    for i in range(num_processes):
        pt_path = results_dir / f"process_{i}" / "results.pt"
        if not pt_path.exists():
            print(f"  WARNING: {pt_path} not found, skipping")
            continue

        r = torch.load(pt_path, weights_only=True)
        n = get_sample_count(results_dir, i)
        print(f"  Process {i}: {n} samples, shape {r.shape}")

        if combined is None:
            combined = torch.zeros_like(r)
        combined += r * n
        total_samples += n

    combined /= total_samples
    print(f"\n  Total samples: {total_samples}")
    print(f"  Result shape: {combined.shape}")

    # 保存
    out_path = results_dir / "results_combined.pt"
    torch.save(combined, out_path)
    print(f"  Saved: {out_path}")

    # Top heads表示
    indices = torch.topk(combined.flatten(), k=16, largest=False).indices.numpy()
    num_heads = combined.shape[1]
    print(f"\n  Top 16 Attention Heads (Most Impactful):")
    for rank, idx in enumerate(indices):
        layer, head = idx // num_heads, idx % num_heads
        print(f"    {rank+1:2d}. Layer {layer:2d}, Head {head:2d}: {combined[layer, head].item():+.4f}%")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel path patching results")
    parser.add_argument("results_dir", type=str,
                        help="Directory containing process_*/results.pt")
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("Merging Parallel Path Patching Results")
    print("=" * 60)
    merge_results(args.results_dir, args.num_processes)
    print("=" * 60)
