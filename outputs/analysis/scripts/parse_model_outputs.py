"""Phase 4-1: wandb JSON → pandas DataFrame 変換 + 全モデル統合"""
import json
from pathlib import Path
import sys

MODEL_OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "model_outputs"

MODELS = {
    "base-Qwen3-30B-A3B-Instruct-2507.json": ("Base 30B", "30B"),
    "positive-pinpoint-Qwen3-30B-A3B-Instruct-2507.json": ("Pos PP SFT", "30B"),
    "negative-pinpoint-Qwen3-30B-A3B-Instruct-2507.json": ("Neg PP SFT", "30B"),
    "base-Qwen3-Next-80B-A3B-Instruct.json": ("Base 80B", "80B"),
    "grpo-Qwen3-Next-80B-A3B-Instruct.json": ("GRPO", "80B"),
    "gspo-Qwen3-Next-80B-A3B-Instruct.json": ("GSPO", "80B"),
    "chord_Qwen3-Next-80B-A3B-Instruct.json": ("CHORD", "80B"),
}


def load_model_output(filepath):
    """wandb table JSONをdictのリストに変換"""
    with open(filepath) as f:
        data = json.load(f)
    cols = data["columns"]
    return [dict(zip(cols, row)) for row in data["data"]]


def load_all():
    """全モデルをロードし、model_name/groupカラムを追加して返す"""
    all_rows = []
    for fname, (display_name, group) in MODELS.items():
        fp = MODEL_OUTPUTS_DIR / fname
        if not fp.exists():
            print(f"WARN: {fp} not found, skipping", file=sys.stderr)
            continue
        rows = load_model_output(fp)
        for r in rows:
            r["model_display"] = display_name
            r["model_group"] = group
        all_rows.extend(rows)
        print(f"Loaded {fname}: {len(rows)} rows -> {display_name}", file=sys.stderr)
    return all_rows


if __name__ == "__main__":
    rows = load_all()
    print(f"\nTotal rows: {len(rows)}")
    # Quick summary
    from collections import Counter
    model_counts = Counter((r["model_display"], r["dataset_name"]) for r in rows)
    for (m, d), c in sorted(model_counts.items()):
        print(f"  {m:20s} | {d:30s} | {c:5d}")
