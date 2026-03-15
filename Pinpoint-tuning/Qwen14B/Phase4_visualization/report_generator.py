#!/usr/bin/env python3
"""
Report Generator
統合レポート生成（Markdown/HTML）
"""

import json
import argparse
import os
from datetime import datetime


def generate_markdown_report(classification_results, statistical_report, output_path):
    """Markdownレポート生成"""

    report = f"""# 産婦人科データ Path Patching 分析レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## サマリー

### ヘッド分類結果

| カテゴリ | 数 | 割合 |
|---------|---|-----|
| Medical Term Heads | {len(classification_results['medical_term_heads'])} | {len(classification_results['medical_term_heads'])/16:.1%} |
| Guideline Indicator Heads | {len(classification_results['guideline_indicator_heads'])} | {len(classification_results['guideline_indicator_heads'])/16:.1%} |
| Reasoning Flow Heads | {len(classification_results['reasoning_flow_heads'])} | {len(classification_results['reasoning_flow_heads'])/16:.1%} |
| Unclassified | {len(classification_results['unclassified'])} | {len(classification_results['unclassified'])/16:.1%} |

### Path Patching Impact

"""

    # Impact統計を追加
    if 'impact_correlations' in statistical_report:
        report += "\n| Head Type | Mean Impact | Std Impact |\n|-----------|------------|------------|\n"

        for head_type, stats in statistical_report['impact_correlations'].items():
            report += f"| {head_type} | {stats['mean_impact']:.4f} | {stats['std_impact']:.4f} |\n"

    report += """

## 推奨事項

1. **Medical Term Heads**: 医療用語理解に重要 → Pinpoint Tuning対象として優先
2. **Guideline Indicator Heads**: ガイドライン参照に重要 → チューニング対象
3. **Reasoning Flow Heads**: 推論フローに寄与 → 選択的にチューニング

## 次のステップ

- Phase 5でトレーニング対象ヘッドを選択
- SPT (Supervised Pinpoint Tuning) を実行
"""

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Report Generator")
    parser.add_argument("--classification_results", type=str, required=True)
    parser.add_argument("--statistical_report", type=str, required=True)
    parser.add_argument("--output_path", type=str,
                       default="Phase4_visualization/medical_head_analysis_report.md")

    args = parser.parse_args()

    # データ読み込み
    with open(args.classification_results, 'r') as f:
        classification_results = json.load(f)

    with open(args.statistical_report, 'r') as f:
        statistical_report = json.load(f)

    # レポート生成
    generate_markdown_report(classification_results, statistical_report, args.output_path)


if __name__ == '__main__':
    main()
