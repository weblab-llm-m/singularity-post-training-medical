"""Phase 1-1: summary_result JSONから全体比較テーブルを生成"""
import json
import os
from pathlib import Path

SUMMARY_DIR = Path(__file__).resolve().parent.parent.parent / "summary_result"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

# モデル定義（表示名, ファイル名, グループ）
MODELS = [
    ("Base 30B", "base-Qwen3-30B-A3B-Instruct-2507.json", "30B"),
    ("Positive Pinpoint SFT", "positive-pinpoint-Qwen3-30B-A3B-Instruct-2507.json", "30B"),
    ("Negative Pinpoint SFT", "negative-pinpoint-Qwen3-30B-A3B-Instruct-2507.json", "30B"),
    ("Base 80B", "bse-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
    ("GRPO", "grpo-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
    ("GSPO", "gspo-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
    ("CHORD", "chord-Qwen3-Next-80B-A3B-Instruct.json", "80B"),
]


def load_summary(filepath):
    with open(filepath) as f:
        data = json.load(f)
    cols = data["columns"]
    rows = []
    for row in data["data"]:
        rows.append(dict(zip(cols, row)))
    return rows


def get_metric(rows, dataset, category, year, metric):
    for r in rows:
        if r["dataset"] == dataset and r["category"] == category and r["year"] == year:
            return r.get(metric)
    return None


def fmt(val, base_val=None):
    if val is None:
        return "-"
    s = f"{val*100:.1f}"
    if base_val is not None and base_val != val:
        diff = (val - base_val) * 100
        sign = "+" if diff > 0 else ""
        s += f" ({sign}{diff:.1f})"
    return s


def main():
    all_data = {}
    for display_name, filename, group in MODELS:
        filepath = SUMMARY_DIR / filename
        if filepath.exists():
            all_data[display_name] = (load_summary(filepath), group)

    lines = []

    # === Table 1: 全体比較 ===
    lines.append("# Phase 1-1: 全体比較テーブル\n")

    # --- igakuqa ---
    lines.append("## 1. 医師国家試験 (igakuqa)\n")
    lines.append("| Model | Group | 2023 | 2024 | 2025 | **All** | All(subset) |")
    lines.append("|-------|-------|------|------|------|---------|-------------|")

    base_vals = {"30B": {}, "80B": {}}
    for display_name, filename, group in MODELS:
        if display_name not in all_data:
            continue
        rows, g = all_data[display_name]
        all_acc = get_metric(rows, "igakuqa", "all", "all", "accuracy")
        all_sub = get_metric(rows, "igakuqa", "all", "all", "subset_accuracy")
        y2023 = get_metric(rows, "igakuqa", "igakuqa", "2023", "accuracy")
        y2024 = get_metric(rows, "igakuqa", "igakuqa", "2024", "accuracy")
        y2025 = get_metric(rows, "igakuqa", "igakuqa", "2025", "accuracy")

        if display_name.startswith("Base"):
            base_vals[g] = {"all": all_acc, "sub": all_sub}

        bv = base_vals[g].get("all")
        bsv = base_vals[g].get("sub")

        lines.append(
            f"| {display_name} | {g} | {fmt(y2023)} | {fmt(y2024)} | {fmt(y2025)} "
            f"| **{fmt(all_acc, bv if not display_name.startswith('Base') else None)}** "
            f"| {fmt(all_sub, bsv if not display_name.startswith('Base') else None)} |"
        )

    # --- specialist_exam_test_v2 ---
    lines.append("\n## 2. 専門医試験 (specialist_exam_test_v2)\n")
    lines.append("| Model | Group | **All** | All(subset) |")
    lines.append("|-------|-------|---------|-------------|")

    base_spec = {"30B": {}, "80B": {}}
    for display_name, filename, group in MODELS:
        if display_name not in all_data:
            continue
        rows, g = all_data[display_name]
        all_acc = get_metric(rows, "specialist_exam_test_v2", "all", "all", "accuracy")
        all_sub = get_metric(rows, "specialist_exam_test_v2", "all", "all", "subset_accuracy")

        if display_name.startswith("Base"):
            base_spec[g] = {"all": all_acc, "sub": all_sub}

        bv = base_spec[g].get("all")
        bsv = base_spec[g].get("sub")

        lines.append(
            f"| {display_name} | {g} "
            f"| **{fmt(all_acc, bv if not display_name.startswith('Base') else None)}** "
            f"| {fmt(all_sub, bsv if not display_name.startswith('Base') else None)} |"
        )

    # --- specialist by category ---
    lines.append("\n## 3. 専門医試験 診療科別 (specialist_exam_test_v2)\n")

    categories = [
        "naika", "geka", "sanfuqa", "seikeigekaqa", "seishin",
        "shinkeinaika", "shinzogeka", "masui", "kyukyu",
        "kanzo", "syokaki", "ganka", "zibika",
    ]
    cat_jp = {
        "naika": "内科", "geka": "外科", "sanfuqa": "産婦人科",
        "seikeigekaqa": "整形外科", "seishin": "精神科",
        "shinkeinaika": "神経内科", "shinzogeka": "心臓外科",
        "masui": "麻酔科", "kyukyu": "救急", "kanzo": "肝臓",
        "syokaki": "消化器", "ganka": "眼科", "zibika": "耳鼻科",
    }

    header = "| 診療科 |"
    sep = "|--------|"
    for display_name, _, _ in MODELS:
        if display_name not in all_data:
            continue
        short = display_name.replace("Pinpoint SFT", "PP").replace("Positive ", "Pos ").replace("Negative ", "Neg ")
        header += f" {short} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for cat in categories:
        row_str = f"| {cat_jp[cat]} |"
        for display_name, _, group in MODELS:
            if display_name not in all_data:
                continue
            rows, g = all_data[display_name]
            acc = get_metric(rows, "specialist_exam_test_v2", cat, "all", "accuracy")
            row_str += f" {fmt(acc)} |"
        lines.append(row_str)

    # --- guideline_wrong_filtered ---
    lines.append("\n## 4. 診療ガイドライン問題 (guideline_wrong_filtered)\n")
    lines.append("| Model | Group | All |")
    lines.append("|-------|-------|-----|")

    for display_name, filename, group in MODELS:
        if display_name not in all_data:
            continue
        rows, g = all_data[display_name]
        all_acc = get_metric(rows, "guideline_wrong_filtered", "all", "all", "accuracy")
        if all_acc is not None:
            lines.append(f"| {display_name} | {g} | {fmt(all_acc)} |")

    # --- Delta table ---
    lines.append("\n## 5. Baseモデルからの変化量 (accuracy, all/all)\n")
    lines.append("| Model | igakuqa | specialist | guideline |")
    lines.append("|-------|---------|------------|-----------|")

    for display_name, filename, group in MODELS:
        if display_name.startswith("Base"):
            continue
        if display_name not in all_data:
            continue
        rows, g = all_data[display_name]
        base_rows = all_data[f"Base {g}"][0]

        igaku = get_metric(rows, "igakuqa", "all", "all", "accuracy")
        igaku_b = get_metric(base_rows, "igakuqa", "all", "all", "accuracy")
        spec = get_metric(rows, "specialist_exam_test_v2", "all", "all", "accuracy")
        spec_b = get_metric(base_rows, "specialist_exam_test_v2", "all", "all", "accuracy")
        guide = get_metric(rows, "guideline_wrong_filtered", "all", "all", "accuracy")
        guide_b = get_metric(base_rows, "guideline_wrong_filtered", "all", "all", "accuracy")

        def delta(v, b):
            if v is None or b is None:
                return "-"
            d = (v - b) * 100
            return f"{'+' if d > 0 else ''}{d:.1f}"

        lines.append(f"| {display_name} | {delta(igaku, igaku_b)} | {delta(spec, spec_b)} | {delta(guide, guide_b)} |")

    output_path = OUTPUT_DIR / "phase1_overall.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {output_path}")

    # Also print to stdout
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
