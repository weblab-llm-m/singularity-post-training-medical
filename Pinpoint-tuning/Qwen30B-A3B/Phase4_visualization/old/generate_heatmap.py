#!/usr/bin/env python3
"""
Phase 4: Visualization - Plotly Heatmap Generator
Qwen3-30B-A3B: 48 layers × 32 heads = 1,536 heads

14Bと同じplotlyベースのヒートマップを生成
"""

import json
import argparse
import os
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def generate_heatmap(
    patching_results: np.ndarray,
    classification_results: dict,
    output_dir: str,
    num_samples: int = 256,
):
    """
    Path Patching結果 + Head分類をplotlyで可視化

    Args:
        patching_results: [num_layers, num_heads] 影響度
        classification_results: 分類結果dict
        output_dir: 出力ディレクトリ
        num_samples: サンプル数（タイトル用）
    """
    num_layers, num_heads = patching_results.shape

    # カウント
    medical_heads = classification_results.get('medical_term_heads', [])
    profile_heads = classification_results.get('profile_indicator_heads', [])
    reasoning_heads = classification_results.get('reasoning_flow_heads', [])
    n_med = len(medical_heads)
    n_prof = len(profile_heads)
    n_reas = len(reasoning_heads)

    # ベースのヒートマップ
    fig = px.imshow(
        patching_results,
        title=(
            f"Head Classification with Path Patching Impact ({num_samples} Samples)<br>"
            f"Blue=Medical Term ({n_med}) | Green=Profile Indicator ({n_prof}) | Red=Reasoning ({n_reas})"
        ),
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Impact (%)",
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=400,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
    )

    # Medical Term Heads: 青四角
    if medical_heads:
        fig.add_scatter(
            x=[h for l, h in medical_heads],
            y=[l for l, h in medical_heads],
            mode='markers',
            marker=dict(color='blue', size=10, symbol='square-open', line=dict(width=2)),
            name=f'Medical Term Heads ({n_med})',
        )

    # Profile Indicator Heads: 緑四角
    if profile_heads:
        fig.add_scatter(
            x=[h for l, h in profile_heads],
            y=[l for l, h in profile_heads],
            mode='markers',
            marker=dict(color='green', size=10, symbol='square-open', line=dict(width=2)),
            name=f'Profile Indicator Heads ({n_prof})',
        )

    # Reasoning Flow Heads: 赤丸
    if reasoning_heads:
        fig.add_scatter(
            x=[h for l, h in reasoning_heads],
            y=[l for l, h in reasoning_heads],
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle-open', line=dict(width=2)),
            name=f'Reasoning Heads ({n_reas})',
        )

    fig.update_layout(
        yaxis_title="Layer",
        xaxis_title="Head",
        xaxis_range=[-0.5, num_heads - 0.5],
        showlegend=True,
        legend=dict(x=1.1, y=1.0),
        width=1200,
        height=900,
    )
    fig.update_yaxes(range=[num_layers - 0.5, -0.5], autorange=False)

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join(output_dir, f"head_classification_heatmap_{num_samples}samples.html")
    fig.write_html(html_path)
    print(f"HTML saved: {html_path}")

    png_path = os.path.join(output_dir, f"head_classification_heatmap_{num_samples}samples.png")
    try:
        fig.write_image(png_path, scale=2)
        print(f"PNG saved: {png_path}")
    except Exception as e:
        print(f"PNG export failed ({e}), generating with matplotlib instead...")
        _generate_png_matplotlib(patching_results, classification_results, png_path,
                                 num_layers, num_heads, num_samples)

    return fig


def _generate_png_matplotlib(patching_results, classification_results, output_path,
                              num_layers, num_heads, num_samples):
    """matplotlibフォールバックでplotly風のPNGを生成"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    medical_heads = classification_results.get('medical_term_heads', [])
    profile_heads = classification_results.get('profile_indicator_heads', [])
    reasoning_heads = classification_results.get('reasoning_flow_heads', [])
    n_med, n_prof, n_reas = len(medical_heads), len(profile_heads), len(reasoning_heads)

    fig, ax = plt.subplots(figsize=(16, 12))

    vmax = max(abs(patching_results.min()), abs(patching_results.max()))
    im = ax.imshow(patching_results, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='Impact (%)', shrink=0.8, pad=0.02)

    # Medical Term: 青四角
    if medical_heads:
        ax.scatter([h for l, h in medical_heads], [l for l, h in medical_heads],
                   facecolors='none', edgecolors='blue', marker='s', s=80, linewidths=2,
                   label=f'Medical Term Heads ({n_med})', zorder=3)

    # Profile Indicator: 緑四角
    if profile_heads:
        ax.scatter([h for l, h in profile_heads], [l for l, h in profile_heads],
                   facecolors='none', edgecolors='green', marker='s', s=80, linewidths=2,
                   label=f'Profile Indicator Heads ({n_prof})', zorder=3)

    # Reasoning: 赤丸
    if reasoning_heads:
        ax.scatter([h for l, h in reasoning_heads], [l for l, h in reasoning_heads],
                   facecolors='none', edgecolors='red', marker='o', s=80, linewidths=2,
                   label=f'Reasoning Heads ({n_reas})', zorder=3)

    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(
        f'Head Classification with Path Patching Impact ({num_samples} Samples)\n'
        f'Blue=Medical Term ({n_med}) | Green=Profile Indicator ({n_prof}) | Red=Reasoning ({n_reas})',
        fontsize=13
    )
    ax.set_xticks(range(0, num_heads, 2))
    ax.set_yticks(range(0, num_layers, 2))
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, -0.05), ncol=3, fontsize=10)

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"PNG saved (matplotlib): {output_path}")


def generate_statistical_report(
    patching_results: np.ndarray,
    classification_results: dict,
    output_dir: str,
    num_samples: int = 256,
):
    """統計レポートをJSON + Markdownで生成"""
    num_layers, num_heads = patching_results.shape
    total_heads = num_layers * num_heads

    head_types = {
        'medical_term_heads': 'Medical Term',
        'profile_indicator_heads': 'Profile Indicator',
        'reasoning_flow_heads': 'Reasoning Flow',
        'unclassified': 'Unclassified',
    }

    # 各タイプのimpact統計
    impact_stats = {}
    for key, label in head_types.items():
        heads = classification_results.get(key, [])
        if not heads:
            continue
        impacts = [patching_results[l, h] for l, h in heads if l < num_layers and h < num_heads]
        if impacts:
            impact_stats[key] = {
                'label': label,
                'count': len(impacts),
                'percentage': f"{len(impacts)/total_heads*100:.1f}%",
                'mean_impact': float(np.mean(impacts)),
                'std_impact': float(np.std(impacts)),
                'median_impact': float(np.median(impacts)),
                'min_impact': float(np.min(impacts)),
                'max_impact': float(np.max(impacts)),
            }

    # レイヤーグループ分析
    group_size = num_layers // 3
    groups = {
        'early': (0, group_size),
        'middle': (group_size, group_size * 2),
        'late': (group_size * 2, num_layers),
    }
    layer_group_stats = {}
    for gname, (start, end) in groups.items():
        layer_group_stats[gname] = {'layers': f"{start}-{end-1}"}
        for key in ['medical_term_heads', 'profile_indicator_heads', 'reasoning_flow_heads']:
            heads = classification_results.get(key, [])
            layer_group_stats[gname][key] = sum(1 for l, h in heads if start <= l < end)

    # Top impactful heads
    flat = []
    for key in ['medical_term_heads', 'profile_indicator_heads', 'reasoning_flow_heads']:
        for l, h in classification_results.get(key, []):
            if l < num_layers and h < num_heads:
                flat.append((l, h, patching_results[l, h], head_types[key]))
    flat.sort(key=lambda x: x[2])
    top_negative = flat[:10]
    top_positive = flat[-5:][::-1]

    report = {
        'model': 'Qwen3-30B-A3B-Instruct-2507',
        'num_layers': num_layers,
        'num_heads': num_heads,
        'total_heads': total_heads,
        'num_samples': num_samples,
        'impact_stats': impact_stats,
        'layer_group_analysis': layer_group_stats,
        'top_negative_heads': [
            {'layer': int(l), 'head': int(h), 'impact': float(v), 'type': t}
            for l, h, v, t in top_negative
        ],
        'top_positive_heads': [
            {'layer': int(l), 'head': int(h), 'impact': float(v), 'type': t}
            for l, h, v, t in top_positive
        ],
    }

    json_path = os.path.join(output_dir, 'statistical_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Statistical report saved: {json_path}")

    # Markdownレポート
    md = f"""# Phase 4: Activation Patching 分析レポート
## Qwen3-30B-A3B-Instruct-2507

**サンプル数**: {num_samples}
**総ヘッド数**: {total_heads} ({num_layers}層 × {num_heads}ヘッド)

---

## ヘッド分類結果

| カテゴリ | 数 | 割合 | Mean Impact | Std | Median |
|---------|-----|------|-------------|-----|--------|
"""
    for key, label in head_types.items():
        s = impact_stats.get(key)
        if s:
            md += f"| {label} | {s['count']} | {s['percentage']} | {s['mean_impact']:.4f}% | {s['std_impact']:.4f}% | {s['median_impact']:.4f}% |\n"

    md += f"""
---

## レイヤーグループ別分析

| グループ | レイヤー | Medical | Profile | Reasoning | 合計 |
|---------|---------|---------|---------|-----------|------|
"""
    for gname in ['early', 'middle', 'late']:
        s = layer_group_stats[gname]
        total = s['medical_term_heads'] + s['profile_indicator_heads'] + s['reasoning_flow_heads']
        md += f"| {gname.title()} | {s['layers']} | {s['medical_term_heads']} | {s['profile_indicator_heads']} | {s['reasoning_flow_heads']} | {total} |\n"

    md += f"""
---

## Top 10 Most Negative Impact Heads

| Rank | Layer | Head | Impact (%) | Category |
|------|-------|------|------------|----------|
"""
    for i, item in enumerate(top_negative):
        l, h, v, t = item
        md += f"| {i+1} | {l} | {h} | {v:.4f} | {t} |\n"

    md += """
---

## 生成ファイル

- `head_classification_heatmap_*samples.html`: インタラクティブヒートマップ
- `head_classification_heatmap_*samples.png`: 静的ヒートマップ (300dpi)
- `statistical_report.json`: 統計データ (JSON)
- `PHASE4_ANALYSIS_REPORT.md`: 本レポート
"""

    md_path = os.path.join(output_dir, 'PHASE4_ANALYSIS_REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Markdown report saved: {md_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Visualization")
    parser.add_argument("--patching_results", type=str, required=True,
                        help="Path patching results (.pt)")
    parser.add_argument("--classification_results", type=str, required=True,
                        help="Head classification results (.json)")
    parser.add_argument("--output_dir", type=str, default="Phase4_visualization")
    parser.add_argument("--num_samples", type=int, default=256)

    args = parser.parse_args()

    # データ読み込み
    print(f"Loading patching results: {args.patching_results}")
    patching_tensor = torch.load(args.patching_results, map_location='cpu')
    patching_results = patching_tensor.numpy()
    num_layers, num_heads = patching_results.shape
    print(f"  Shape: [{num_layers}, {num_heads}]")

    print(f"Loading classification results: {args.classification_results}")
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    for key in ['medical_term_heads', 'profile_indicator_heads', 'reasoning_flow_heads', 'unclassified']:
        n = len(classification_results.get(key, []))
        print(f"  {key}: {n}")

    # ヒートマップ生成
    print("\n=== Generating Heatmap ===")
    generate_heatmap(patching_results, classification_results, args.output_dir, args.num_samples)

    # 統計レポート生成
    print("\n=== Generating Statistical Report ===")
    generate_statistical_report(patching_results, classification_results, args.output_dir, args.num_samples)

    print("\n=== Phase 4 Complete! ===")


if __name__ == '__main__':
    main()
