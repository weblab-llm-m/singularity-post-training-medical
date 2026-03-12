#!/usr/bin/env python3
"""
Report Generator for MoE Models
統合レポート生成（Markdown形式）

Qwen3-30B-A3B対応版:
- MoE構造の情報を含む
- 48レイヤー × 32ヘッドの分析レポート
"""

import json
import argparse
import os
from datetime import datetime


def generate_markdown_report(
    classification_results: dict,
    statistical_report: dict,
    output_path: str,
    num_layers: int = 48,
    num_heads: int = 32
):
    """
    Markdownレポート生成

    Args:
        classification_results: 分類結果
        statistical_report: 統計レポート
        output_path: 出力パス
        num_layers: レイヤー数
        num_heads: ヘッド数
    """
    total_heads = num_layers * num_heads

    # ヘッドカウント
    medical_count = len(classification_results.get('medical_term_heads', []))
    profile_count = len(classification_results.get('profile_indicator_heads', []))
    reasoning_count = len(classification_results.get('reasoning_flow_heads', []))
    unclassified_count = len(classification_results.get('unclassified', []))

    report = f"""# 産婦人科データ Path Patching 分析レポート
## Qwen3-30B-A3B-Instruct-2507 (MoE)

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## モデル情報

| パラメータ | 値 |
|-----------|-----|
| モデルタイプ | qwen3_moe |
| レイヤー数 | {num_layers} |
| Attentionヘッド数/層 | {num_heads} |
| 総ヘッド数 | {total_heads} |
| アーキテクチャ | Mixture-of-Experts |

---

## ヘッド分類結果サマリー

| カテゴリ | 数 | 割合 |
|---------|-----|------|
| Medical Term Heads | {medical_count} | {medical_count/total_heads*100:.1f}% |
| Profile Indicator Heads | {profile_count} | {profile_count/total_heads*100:.1f}% |
| Reasoning Flow Heads | {reasoning_count} | {reasoning_count/total_heads*100:.1f}% |
| Unclassified | {unclassified_count} | {unclassified_count/total_heads*100:.1f}% |
| **合計** | **{total_heads}** | **100%** |

---

## レイヤーグループ別分析

"""

    # レイヤーグループ分析を追加
    if 'layer_group_analysis' in statistical_report:
        report += "| グループ | レイヤー | Medical | Profile | Reasoning | 合計 |\n"
        report += "|---------|---------|---------|---------|-----------|------|\n"

        for group, stats in statistical_report['layer_group_analysis'].items():
            total = (stats.get('medical_term_heads', 0) +
                    stats.get('profile_indicator_heads', 0) +
                    stats.get('reasoning_flow_heads', 0))
            report += f"| {group.title()} | {stats['layers']} | "
            report += f"{stats.get('medical_term_heads', 0)} | "
            report += f"{stats.get('profile_indicator_heads', 0)} | "
            report += f"{stats.get('reasoning_flow_heads', 0)} | "
            report += f"{total} |\n"

    report += """
---

## Path Patching Impact分析

"""

    # Impact統計を追加
    if 'impact_correlations' in statistical_report:
        report += "| Head Type | Count | Mean Impact | Std | Median |\n"
        report += "|-----------|-------|-------------|-----|--------|\n"

        for head_type, stats in statistical_report['impact_correlations'].items():
            report += f"| {head_type.replace('_heads', '').replace('_', ' ').title()} | "
            report += f"{stats.get('count', 0)} | "
            report += f"{stats.get('mean_impact', 0):.2f}% | "
            report += f"{stats.get('std_impact', 0):.2f}% | "
            report += f"{stats.get('median_impact', 0):.2f}% |\n"

    report += """
---

## Pinpoint Tuning 推奨事項

### 優先度の高いヘッド

1. **Medical Term Heads** (優先度: 最高)
   - 医療用語の理解に重要
   - Pinpoint Tuning対象として最優先

2. **Profile Indicator Heads** (優先度: 高)
   - 患者属性（年齢・性別・既往歴等）の認識に対応
   - チューニング対象として推奨

3. **Reasoning Flow Heads** (優先度: 中)
   - 推論フローに寄与
   - 選択的にチューニング

### MoE固有の考慮事項

- **Router層 (mlp.gate)**: freeze推奨（エキスパート選択の安定性維持）
- **Expert層**: 基本的にfreeze（まずはAttentionヘッドのみ学習）
- **メモリ**: 30Bモデルのため、Gradient CheckpointingやDeepSpeed ZeRO-3を推奨

---

## 次のステップ

1. `Phase3_attention_analysis/generate_trainable_heads.py` でtrainable_heads.jsonを生成
2. `Phase5_pinpoint_tuning/` でSPT (Supervised Pinpoint Tuning) を実行
3. 評価データセットでの性能検証

---

## 生成ファイル

- `classification_heatmap.png`: ヘッド分類のヒートマップ
- `layer_distribution.png`: レイヤー別分布
- `statistical_report.json`: 統計分析結果
- `medical_head_analysis_report.md`: 本レポート

"""

    # 保存
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Report Generator for MoE Models")
    parser.add_argument("--classification_results", type=str, required=True,
                       help="Classification results JSON file")
    parser.add_argument("--statistical_report", type=str, required=True,
                       help="Statistical report JSON file")
    parser.add_argument("--output_path", type=str,
                       default="Phase4_visualization/medical_head_analysis_report.md",
                       help="Output report path")
    parser.add_argument("--num_layers", type=int, default=48,
                       help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=32,
                       help="Number of heads per layer")

    args = parser.parse_args()

    # データ読み込み
    print(f"Loading classification results from: {args.classification_results}")
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    print(f"Loading statistical report from: {args.statistical_report}")
    with open(args.statistical_report, 'r') as f:
        statistical_report = json.load(f)

    # レポート生成
    generate_markdown_report(
        classification_results,
        statistical_report,
        args.output_path,
        args.num_layers,
        args.num_heads
    )


if __name__ == '__main__':
    main()
