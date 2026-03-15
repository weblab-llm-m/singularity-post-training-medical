#!/usr/bin/env python3
"""
64並列Path Patching結果を統合するスクリプト

8ノード × 8GPU = 64プロセスの結果を重み付け平均で統合。
サンプル数はログファイルから自動取得。
"""

import argparse
import re
import json
import torch
from pathlib import Path
from datetime import datetime


def get_sample_count(results_dir: Path, proc_id: str) -> int:
    """ログファイルからサンプル数を取得"""
    log_path = results_dir / f"process_{proc_id}.log"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            m = re.search(r"Using samples:\s+(\d+)", line)
            if m:
                return int(m.group(1))

    # フォールバック: chunk_data.jsonlの行数
    chunk_path = results_dir / "data_chunks" / f"chunk_{proc_id}.jsonl"
    if chunk_path.exists():
        return sum(1 for _ in open(chunk_path))

    raise ValueError(f"Cannot determine sample count for process_{proc_id}")


def merge_results(results_dir: str, num_processes: int = 64):
    results_dir = Path(results_dir)

    combined = None
    total_samples = 0
    success = 0
    failed = []
    per_sample_list = []

    for i in range(num_processes):
        proc_id = f"{i:02d}"
        pt_path = results_dir / f"process_{proc_id}" / "results.pt"

        if not pt_path.exists():
            print(f"  WARNING: process_{proc_id}/results.pt not found, skipping")
            failed.append(proc_id)
            continue

        r = torch.load(pt_path, weights_only=True)
        n = get_sample_count(results_dir, proc_id)
        print(f"  Process {proc_id}: {n} samples, shape {r.shape}")

        if combined is None:
            combined = torch.zeros_like(r)
        combined += r * n
        total_samples += n
        success += 1

        # Per-sample results
        ps_path = results_dir / f"process_{proc_id}" / "results_per_sample.pt"
        if ps_path.exists():
            ps = torch.load(ps_path, weights_only=True)
            per_sample_list.append(ps)
            print(f"           per-sample: {ps.shape}")

    if combined is None:
        print("ERROR: No results found!")
        return None

    combined /= total_samples

    print(f"\n  Processes: {success}/{num_processes} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Total samples: {total_samples}")
    print(f"  Result shape: {combined.shape}")

    # 保存 (averaged)
    out_path = results_dir / "results_combined.pt"
    torch.save(combined, out_path)
    print(f"  Saved: {out_path}")

    # 保存 (per-sample)
    if per_sample_list:
        per_sample_combined = torch.cat(per_sample_list, dim=0)
        ps_out_path = results_dir / "results_per_sample_combined.pt"
        torch.save(per_sample_combined, ps_out_path)
        print(f"  Saved: {ps_out_path} (shape: {per_sample_combined.shape})")
    else:
        per_sample_combined = None
        print("  WARNING: No per-sample results found")

    # Top heads表示
    num_heads = combined.shape[1]
    num_layers = combined.shape[0]

    print(f"\n  Top 20 Attention Heads (Most Negative Impact):")
    indices = torch.topk(combined.flatten(), k=20, largest=False).indices.numpy()
    top_heads = []
    for rank, idx in enumerate(indices):
        layer, head = int(idx // num_heads), int(idx % num_heads)
        impact = combined[layer, head].item()
        top_heads.append({'rank': rank + 1, 'layer': layer, 'head': head, 'impact': impact})
        print(f"    {rank+1:2d}. Layer {layer:2d}, Head {head:2d}: {impact:+.4f}%")

    print(f"\n  Top 10 Attention Heads (Most Positive Impact):")
    indices_pos = torch.topk(combined.flatten(), k=10, largest=True).indices.numpy()
    for rank, idx in enumerate(indices_pos):
        layer, head = int(idx // num_heads), int(idx % num_heads)
        impact = combined[layer, head].item()
        print(f"    {rank+1:2d}. Layer {layer:2d}, Head {head:2d}: {impact:+.4f}%")

    # サマリーJSON保存
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_processes': num_processes,
        'succeeded': success,
        'failed_processes': failed,
        'total_samples': total_samples,
        'result_shape': list(combined.shape),
        'per_sample_shape': list(per_sample_combined.shape) if per_sample_combined is not None else None,
        'top_negative_heads': top_heads,
        'global_stats': {
            'mean': float(combined.mean()),
            'std': float(combined.std()),
            'min': float(combined.min()),
            'max': float(combined.max()),
        }
    }
    if per_sample_combined is not None:
        # Per-sample統計: 各ヘッドのサンプル間分散
        per_head_std = per_sample_combined.std(dim=0)  # (num_layers, num_heads)
        summary['per_sample_stats'] = {
            'mean_std_across_heads': float(per_head_std.mean()),
            'max_std_head': {
                'layer': int(per_head_std.argmax() // per_head_std.shape[1]),
                'head': int(per_head_std.argmax() % per_head_std.shape[1]),
                'std': float(per_head_std.max()),
            },
        }
    summary_path = results_dir / "merge_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge 64-GPU parallel path patching results")
    parser.add_argument("results_dir", type=str,
                        help="Directory containing process_XX/results.pt")
    parser.add_argument("--num_processes", type=int, default=64)
    args = parser.parse_args()

    print("=" * 60)
    print("Merging 64-GPU Parallel Path Patching Results")
    print("=" * 60)
    merge_results(args.results_dir, args.num_processes)
    print("=" * 60)
