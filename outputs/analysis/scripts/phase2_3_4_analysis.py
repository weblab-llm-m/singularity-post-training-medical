"""Phase 2/3/4: 統合分析スクリプト
- Phase 4-3: フォーマット遵守率
- Phase 2: RL系手法の統計検定 (McNemar + Bootstrap CI)
- Phase 3: Pinpoint SFT 劣化分析
- Phase 4-2: 正誤パターン分析
"""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

MODEL_OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "model_outputs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

MODELS_30B = {
    "base-Qwen3-30B-A3B-Instruct-2507.json": "Base 30B",
    "positive-pinpoint-Qwen3-30B-A3B-Instruct-2507.json": "Pos PP SFT",
    "negative-pinpoint-Qwen3-30B-A3B-Instruct-2507.json": "Neg PP SFT",
}
MODELS_80B = {
    "base-Qwen3-Next-80B-A3B-Instruct.json": "Base 80B",
    "grpo-Qwen3-Next-80B-A3B-Instruct.json": "GRPO",
    "gspo-Qwen3-Next-80B-A3B-Instruct.json": "GSPO",
    "chord_Qwen3-Next-80B-A3B-Instruct.json": "CHORD",
}


def load_model(fname):
    fp = MODEL_OUTPUTS_DIR / fname
    with open(fp) as f:
        data = json.load(f)
    cols = data["columns"]
    return [dict(zip(cols, row)) for row in data["data"]]


def build_qid_map(rows):
    """question_id -> row のマップを作成"""
    return {r["question_id"]: r for r in rows}


# ============================================================
#  Phase 4-3: フォーマット遵守率
# ============================================================
def format_compliance(lines):
    lines.append("\n# Phase 4-3: フォーマット遵守率検証\n")
    lines.append("cleaned_predictionがNone（[ans]...[/ans]パース失敗）の割合\n")
    lines.append("| Model | Dataset | Total | Parse Fail | Fail Rate |")
    lines.append("|-------|---------|-------|------------|-----------|")

    all_models = {**MODELS_30B, **MODELS_80B}
    for fname, name in all_models.items():
        rows = load_model(fname)
        by_ds = defaultdict(lambda: [0, 0])
        for r in rows:
            ds = r["dataset_name"]
            by_ds[ds][0] += 1
            if r["cleaned_prediction"] is None or r["cleaned_prediction"] == [] or r["cleaned_prediction"] == "":
                by_ds[ds][1] += 1
        for ds in sorted(by_ds.keys()):
            total, fail = by_ds[ds]
            rate = fail / total * 100 if total > 0 else 0
            marker = " **" if rate > 5 else ""
            lines.append(f"| {name} | {ds} | {total} | {fail} | {rate:.1f}%{marker} |")


# ============================================================
#  Phase 2: RL系手法の統計検定
# ============================================================
def mcnemar_test(a_correct, b_correct):
    """McNemar検定 (chi-squared approximation)
    a_correct, b_correct: 同一問題セットでのbool配列
    Returns: (b01, b10, chi2, p_value)
    """
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)  # A wrong, B right
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)  # A right, B wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n  # with continuity correction
    # approximate p-value from chi2(1)
    p = chi2_sf(chi2)
    return b01, b10, chi2, p


def chi2_sf(x):
    """Survival function for chi-squared distribution with df=1 (approximation)"""
    if x <= 0:
        return 1.0
    # Using complementary error function approximation
    z = math.sqrt(x)
    return 2 * (1 - normal_cdf(z))


def normal_cdf(x):
    """Standard normal CDF approximation (Abramowitz & Stegun)"""
    if x < 0:
        return 1 - normal_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-x * x / 2.0) * (t * (0.319381530 + t * (-0.356563782 +
        t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
    return 1.0 - p


def bootstrap_ci(correct_arr, n_bootstrap=2000, ci=0.95):
    """Bootstrap confidence interval for accuracy"""
    n = len(correct_arr)
    accs = []
    for _ in range(n_bootstrap):
        sample = [correct_arr[random.randint(0, n - 1)] for _ in range(n)]
        accs.append(sum(sample) / n)
    accs.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap)
    return accs[lo_idx], accs[hi_idx]


def rl_comparison(lines):
    lines.append("\n# Phase 2: RL系手法の統計検定付き比較\n")

    base_rows = load_model("base-Qwen3-Next-80B-A3B-Instruct.json")
    grpo_rows = load_model("grpo-Qwen3-Next-80B-A3B-Instruct.json")
    gspo_rows = load_model("gspo-Qwen3-Next-80B-A3B-Instruct.json")
    chord_rows = load_model("chord_Qwen3-Next-80B-A3B-Instruct.json")

    base_map = build_qid_map(base_rows)
    grpo_map = build_qid_map(grpo_rows)
    gspo_map = build_qid_map(gspo_rows)
    chord_map = build_qid_map(chord_rows)

    for ds_name in ["igakuqa", "specialist_exam_test_v2"]:
        lines.append(f"\n## {ds_name}\n")

        # 共通問題IDの特定
        base_qids = {r["question_id"] for r in base_rows if r["dataset_name"] == ds_name}
        grpo_qids = {r["question_id"] for r in grpo_rows if r["dataset_name"] == ds_name}
        gspo_qids = {r["question_id"] for r in gspo_rows if r["dataset_name"] == ds_name}
        chord_qids = {r["question_id"] for r in chord_rows if r["dataset_name"] == ds_name}
        common_qids = sorted(base_qids & grpo_qids & gspo_qids & chord_qids)

        lines.append(f"共通問題数: {len(common_qids)}\n")

        # 正答配列
        results = {}
        for name, qmap in [("Base 80B", base_map), ("GRPO", grpo_map),
                           ("GSPO", gspo_map), ("CHORD", chord_map)]:
            correct = [bool(qmap[qid]["is_correct"]) for qid in common_qids]
            acc = sum(correct) / len(correct)
            lo, hi = bootstrap_ci(correct)
            results[name] = correct
            lines.append(f"- **{name}**: {acc*100:.1f}% (95% CI: [{lo*100:.1f}%, {hi*100:.1f}%])")

        # McNemar: 各手法 vs Base
        lines.append(f"\n### McNemar検定 (vs Base 80B)\n")
        lines.append("| Comparison | Base→X gained | X→Base lost | chi2 | p-value | sig |")
        lines.append("|------------|--------------|-------------|------|---------|-----|")
        for name in ["GRPO", "GSPO", "CHORD"]:
            b01, b10, chi2, p = mcnemar_test(results["Base 80B"], results[name])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            lines.append(f"| Base vs {name} | {b01} | {b10} | {chi2:.2f} | {p:.4f} | {sig} |")

        # McNemar: 手法間
        lines.append(f"\n### McNemar検定 (手法間)\n")
        lines.append("| Comparison | A→B gained | B→A lost | chi2 | p-value | sig |")
        lines.append("|------------|-----------|---------|------|---------|-----|")
        for a_name, b_name in [("GRPO", "GSPO"), ("GRPO", "CHORD"), ("GSPO", "CHORD")]:
            b01, b10, chi2, p = mcnemar_test(results[a_name], results[b_name])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            lines.append(f"| {a_name} vs {b_name} | {b01} | {b10} | {chi2:.2f} | {p:.4f} | {sig} |")


# ============================================================
#  Phase 3: Pinpoint SFT 劣化分析
# ============================================================
def pinpoint_analysis(lines):
    lines.append("\n# Phase 3: Pinpoint SFT 劣化分析\n")

    base_rows = load_model("base-Qwen3-30B-A3B-Instruct-2507.json")
    pos_rows = load_model("positive-pinpoint-Qwen3-30B-A3B-Instruct-2507.json")
    neg_rows = load_model("negative-pinpoint-Qwen3-30B-A3B-Instruct-2507.json")

    base_map = build_qid_map(base_rows)
    pos_map = build_qid_map(pos_rows)
    neg_map = build_qid_map(neg_rows)

    for ds_name in ["igakuqa", "specialist_exam_test_v2"]:
        lines.append(f"\n## {ds_name}\n")

        common_qids = sorted(
            {r["question_id"] for r in base_rows if r["dataset_name"] == ds_name} &
            {r["question_id"] for r in pos_rows if r["dataset_name"] == ds_name} &
            {r["question_id"] for r in neg_rows if r["dataset_name"] == ds_name}
        )

        # McNemar: Pinpoint vs Base
        lines.append(f"### McNemar検定 (vs Base 30B, N={len(common_qids)})\n")
        lines.append("| Comparison | Base→X gained | X→Base lost | Net | chi2 | p-value | sig |")
        lines.append("|------------|--------------|-------------|-----|------|---------|-----|")
        for name, qmap in [("Pos PP SFT", pos_map), ("Neg PP SFT", neg_map)]:
            base_c = [bool(base_map[q]["is_correct"]) for q in common_qids]
            other_c = [bool(qmap[q]["is_correct"]) for q in common_qids]
            b01, b10, chi2, p = mcnemar_test(base_c, other_c)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            lines.append(f"| Base vs {name} | {b01} | {b10} | {b01-b10:+d} | {chi2:.2f} | {p:.6f} | {sig} |")

        # 劣化パターンの診療科別分析 (specialist のみ)
        if ds_name == "specialist_exam_test_v2":
            lines.append(f"\n### 診療科別の劣化パターン (Base正解→SFT不正解)\n")
            lines.append("| 診療科 | Base正答 | Pos壊れ | Pos壊れ率 | Neg壊れ | Neg壊れ率 |")
            lines.append("|--------|---------|---------|----------|---------|----------|")

            cat_stats = defaultdict(lambda: {"base_correct": 0, "pos_broke": 0, "neg_broke": 0, "total": 0})
            for qid in common_qids:
                cat = base_map[qid].get("category", "unknown")
                cat_stats[cat]["total"] += 1
                base_ok = bool(base_map[qid]["is_correct"])
                pos_ok = bool(pos_map[qid]["is_correct"])
                neg_ok = bool(neg_map[qid]["is_correct"])
                if base_ok:
                    cat_stats[cat]["base_correct"] += 1
                    if not pos_ok:
                        cat_stats[cat]["pos_broke"] += 1
                    if not neg_ok:
                        cat_stats[cat]["neg_broke"] += 1

            for cat in sorted(cat_stats.keys()):
                s = cat_stats[cat]
                bc = s["base_correct"]
                if bc == 0:
                    continue
                pr = s["pos_broke"] / bc * 100
                nr = s["neg_broke"] / bc * 100
                lines.append(f"| {cat} | {bc} | {s['pos_broke']} | {pr:.1f}% | {s['neg_broke']} | {nr:.1f}% |")


# ============================================================
#  Phase 4-2: 正誤パターン分析
# ============================================================
def error_pattern_analysis(lines):
    lines.append("\n# Phase 4-2: 正誤パターン分析\n")

    # 80B Group
    lines.append("## 80B Group: Base → RL手法の正誤変化\n")
    base_rows = load_model("base-Qwen3-Next-80B-A3B-Instruct.json")
    grpo_rows = load_model("grpo-Qwen3-Next-80B-A3B-Instruct.json")
    gspo_rows = load_model("gspo-Qwen3-Next-80B-A3B-Instruct.json")
    chord_rows = load_model("chord_Qwen3-Next-80B-A3B-Instruct.json")

    base_map = build_qid_map(base_rows)
    model_maps = {"GRPO": build_qid_map(grpo_rows), "GSPO": build_qid_map(gspo_rows),
                  "CHORD": build_qid_map(chord_rows)}

    for ds_name in ["igakuqa", "specialist_exam_test_v2"]:
        lines.append(f"\n### {ds_name}\n")
        common_qids = sorted(
            {r["question_id"] for r in base_rows if r["dataset_name"] == ds_name} &
            {r["question_id"] for r in grpo_rows if r["dataset_name"] == ds_name} &
            {r["question_id"] for r in gspo_rows if r["dataset_name"] == ds_name} &
            {r["question_id"] for r in chord_rows if r["dataset_name"] == ds_name}
        )

        lines.append("| Pattern | Count | % | Description |")
        lines.append("|---------|-------|---|-------------|")

        # 全モデルの正誤パターン
        patterns = Counter()
        for qid in common_qids:
            base_ok = bool(base_map[qid]["is_correct"])
            others_ok = tuple(bool(model_maps[m][qid]["is_correct"]) for m in ["GRPO", "GSPO", "CHORD"])
            patterns[(base_ok, others_ok)] += 1

        n = len(common_qids)
        # 主要パターンを整理
        all_right = patterns.get((True, (True, True, True)), 0)
        all_wrong = patterns.get((False, (False, False, False)), 0)
        base_right_all_wrong = patterns.get((True, (False, False, False)), 0)
        base_wrong_all_right = patterns.get((False, (True, True, True)), 0)

        # Base正解→全RL不正解
        lines.append(f"| Base○ RL全○ | {all_right} | {all_right/n*100:.1f}% | 全モデル正解（安定） |")
        lines.append(f"| Base× RL全× | {all_wrong} | {all_wrong/n*100:.1f}% | 全モデル不正解（困難問題） |")
        lines.append(f"| Base× → RL全○ | {base_wrong_all_right} | {base_wrong_all_right/n*100:.1f}% | RL学習で全手法が獲得 |")
        lines.append(f"| Base○ → RL全× | {base_right_all_wrong} | {base_right_all_wrong/n*100:.1f}% | RL学習で全手法が喪失 |")

        # 一部のRLのみ改善/劣化
        partial_gain = sum(v for (b, o), v in patterns.items()
                          if not b and any(o) and not all(o))
        partial_loss = sum(v for (b, o), v in patterns.items()
                          if b and any(not x for x in o) and not all(not x for x in o))
        lines.append(f"| Base× → 一部RL○ | {partial_gain} | {partial_gain/n*100:.1f}% | 手法間で差が出た問題 |")
        lines.append(f"| Base○ → 一部RL× | {partial_loss} | {partial_loss/n*100:.1f}% | 一部手法で劣化 |")

        # RL手法別の「独自に正解」した問題数
        lines.append(f"\n**各RL手法の独自改善/独自劣化:**\n")
        lines.append("| 手法 | Baseから改善 | うち独自改善 | Baseから劣化 | うち独自劣化 |")
        lines.append("|------|-----------|------------|-----------|------------|")
        for i, m_name in enumerate(["GRPO", "GSPO", "CHORD"]):
            gained = sum(1 for qid in common_qids
                        if not base_map[qid]["is_correct"] and model_maps[m_name][qid]["is_correct"])
            lost = sum(1 for qid in common_qids
                      if base_map[qid]["is_correct"] and not model_maps[m_name][qid]["is_correct"])
            unique_gain = sum(1 for qid in common_qids
                            if not base_map[qid]["is_correct"]
                            and model_maps[m_name][qid]["is_correct"]
                            and all(not model_maps[o][qid]["is_correct"]
                                   for o in ["GRPO", "GSPO", "CHORD"] if o != m_name))
            unique_loss = sum(1 for qid in common_qids
                            if base_map[qid]["is_correct"]
                            and not model_maps[m_name][qid]["is_correct"]
                            and all(model_maps[o][qid]["is_correct"]
                                   for o in ["GRPO", "GSPO", "CHORD"] if o != m_name))
            lines.append(f"| {m_name} | {gained} | {unique_gain} | {lost} | {unique_loss} |")

    # 30B Group
    lines.append("\n## 30B Group: Base → Pinpoint SFTの正誤変化\n")
    base30_rows = load_model("base-Qwen3-30B-A3B-Instruct-2507.json")
    pos_rows = load_model("positive-pinpoint-Qwen3-30B-A3B-Instruct-2507.json")
    neg_rows = load_model("negative-pinpoint-Qwen3-30B-A3B-Instruct-2507.json")

    base30_map = build_qid_map(base30_rows)
    pos_map = build_qid_map(pos_rows)
    neg_map = build_qid_map(neg_rows)

    for ds_name in ["igakuqa", "specialist_exam_test_v2"]:
        lines.append(f"\n### {ds_name}\n")
        common_qids = sorted(
            {r["question_id"] for r in base30_rows if r["dataset_name"] == ds_name} &
            {r["question_id"] for r in pos_rows if r["dataset_name"] == ds_name} &
            {r["question_id"] for r in neg_rows if r["dataset_name"] == ds_name}
        )
        n = len(common_qids)

        lines.append("| 手法 | Baseから改善 | Baseから劣化 | Net | 改善率 | 劣化率 |")
        lines.append("|------|-----------|-----------|-----|--------|--------|")
        for name, qmap in [("Pos PP SFT", pos_map), ("Neg PP SFT", neg_map)]:
            gained = sum(1 for q in common_qids
                        if not base30_map[q]["is_correct"] and qmap[q]["is_correct"])
            lost = sum(1 for q in common_qids
                      if base30_map[q]["is_correct"] and not qmap[q]["is_correct"])
            base_correct = sum(1 for q in common_qids if base30_map[q]["is_correct"])
            base_wrong = n - base_correct
            lines.append(f"| {name} | {gained} ({gained/base_wrong*100:.1f}% of Base不正解) "
                        f"| {lost} ({lost/base_correct*100:.1f}% of Base正解) "
                        f"| {gained-lost:+d} | {gained/n*100:.1f}% | {lost/n*100:.1f}% |")


def main():
    lines = []
    print("Running Phase 4-3: Format compliance...")
    format_compliance(lines)
    print("Running Phase 2: RL comparison...")
    rl_comparison(lines)
    print("Running Phase 3: Pinpoint analysis...")
    pinpoint_analysis(lines)
    print("Running Phase 4-2: Error patterns...")
    error_pattern_analysis(lines)

    output_path = OUTPUT_DIR / "phase2_3_4_analysis.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten to {output_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
