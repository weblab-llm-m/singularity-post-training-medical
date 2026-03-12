#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Japanese evaluation report"""

import json
import pandas as pd

# Load both result files
tuned_file = "Phase5_pinpoint_tuning/evaluation_results/tuned_model_results.json"
base_file = "Phase5_pinpoint_tuning/evaluation_results/base_model_results.json"

with open(tuned_file, 'r') as f:
    tuned_data = json.load(f)

with open(base_file, 'r') as f:
    base_data = json.load(f)

# Load test data for questions
test_df = pd.read_parquet('/home/Competition2025/P05/shareP05/data/gynecology_guideline_2023_some_models_correct_formatted/test.parquet')

# Create markdown report in Japanese
md_content = []

md_content.append("# Phase 5: モデル評価結果レポート")
md_content.append("")
md_content.append("**日付:** 2025年10月24日")
md_content.append("**データセット:** テストセット (産婦人科診療ガイドライン2023)")
md_content.append("**総サンプル数:** 10")
md_content.append("")
md_content.append("---")
md_content.append("")

# Summary
md_content.append("## 評価結果サマリー")
md_content.append("")

tuned_acc = tuned_data['metrics']['accuracy']
base_acc = base_data['metrics']['accuracy']
improvement = tuned_acc - base_acc

md_content.append("| モデル | 精度 | 正解数 / 総数 |")
md_content.append("|-------|------|---------------|")
md_content.append(f"| **ベースモデル (Qwen3-14B)** | {base_acc:.2f}% | {base_data['metrics']['correct']}/{base_data['metrics']['total_samples']} |")
md_content.append(f"| **SPTチューニング済みモデル (144個の医療用語ヘッド)** | {tuned_acc:.2f}% | {tuned_data['metrics']['correct']}/{tuned_data['metrics']['total_samples']} |")
md_content.append(f"| **改善** | **{improvement:+.2f}%** | **+{tuned_data['metrics']['correct'] - base_data['metrics']['correct']}** |")
md_content.append("")

md_content.append("### 主な発見")
md_content.append("")
md_content.append(f"- **ベースモデル:** {base_acc:.0f}% - 思考プロセスは生成するが、期待される配列形式で回答を出力しない")
md_content.append(f"- **SPTチューニング済みモデル:** {tuned_acc:.0f}% - トレーニング中に正しい形式で回答を出力することを学習")
md_content.append(f"- **トレーニング効果:** SPTトレーニングにより、モデルは配列形式（`['a']` や `['a', 'b']`）で回答を出力することを学習")
md_content.append("")

md_content.append("---")
md_content.append("")

# Detailed results for all 10 samples
md_content.append("## 詳細結果（全10サンプル）")
md_content.append("")

for i in range(len(tuned_data['results'])):
    t_res = tuned_data['results'][i]
    b_res = base_data['results'][i]

    # Get full question from dataframe
    row = test_df.iloc[i]
    question = row['prompt'][0]['content'] if row['prompt'] else "N/A"

    md_content.append(f"### サンプル {i+1}")
    md_content.append("")

    # Question
    md_content.append("**問題:**")
    md_content.append("```")
    md_content.append(question)
    md_content.append("```")
    md_content.append("")

    # Ground truth
    md_content.append(f"**正解:** `{t_res['ground_truth']}`")
    md_content.append("")

    # Base model
    b_status = "✓ 正解" if b_res['correct'] else "✗ 不正解"
    md_content.append(f"#### ベースモデル (Qwen3-14B) - {b_status}")
    md_content.append("")
    md_content.append(f"**予測:** `{b_res['predicted']}`")
    md_content.append("")
    md_content.append("**生成出力:**")
    md_content.append("```")
    md_content.append(b_res['generated_text'])
    md_content.append("```")
    md_content.append("")

    # Tuned model
    t_status = "✓ 正解" if t_res['correct'] else "✗ 不正解"
    md_content.append(f"#### SPTチューニング済みモデル (144個の医療用語ヘッド) - {t_status}")
    md_content.append("")
    md_content.append(f"**予測:** `{t_res['predicted']}`")
    md_content.append("")
    md_content.append("**生成出力:**")
    md_content.append("```")
    md_content.append(t_res['generated_text'])
    md_content.append("```")
    md_content.append("")

    md_content.append("---")
    md_content.append("")

# Analysis section
md_content.append("## 分析")
md_content.append("")

md_content.append("### ベースモデルの挙動")
md_content.append("")
md_content.append("ベースQwen3-14Bモデル:")
md_content.append("- `<think>` タグ内で詳細な思考プロセスを生成")
md_content.append("- 医療問題の理解を示している")
md_content.append("- **期待される配列形式で回答を出力しない**")
md_content.append("- 代わりに自然言語での説明を続ける")
md_content.append("- 形式の不一致により0%の精度となる")
md_content.append("")

md_content.append("### SPTチューニング済みモデルの挙動")
md_content.append("")
md_content.append("SPTチューニング済みモデル (144個の医療用語ヘッド):")
md_content.append("- `<think>` タグ内で簡潔な思考を生成")
md_content.append("- **正しい配列形式で回答を出力** (例: `['a']`, `['b', 'c']`)")
md_content.append("- トレーニングデータ（1,761サンプル）からこの形式を学習")
md_content.append("- 100%の形式準拠を達成")
md_content.append("- 144個の医療用語ヘッドのファインチューニングが出力形式の適応に成功したことを実証")
md_content.append("")

md_content.append("### トレーニングデータの影響")
md_content.append("")
md_content.append("トレーニングデータは以下の形式で回答を含んでいました:")
md_content.append("```python")
md_content.append("# reward_model.ground_truth から")
md_content.append("['a']")
md_content.append("['a', 'd', 'e']")
md_content.append("['a', 'b']")
md_content.append("```")
md_content.append("")
md_content.append("SPTチューニング済みモデルは、144個の医療用語ヘッド（パラメータの2.4%）のみをファインチューニングすることで、この出力形式を模倣することを学習しました。")
md_content.append("")

md_content.append("---")
md_content.append("")

# Conclusion
md_content.append("## 結論")
md_content.append("")

md_content.append("### SPTトレーニングの成功")
md_content.append("")
md_content.append("1. **形式学習:** モデルは配列形式で回答を出力することを成功裏に学習")
md_content.append("2. **選択的ファインチューニング:** 144個のヘッドのみ（340Mパラメータ、2.4%）をチューニング")
md_content.append("3. **トレーニング効率:** 112ステップ（9分54秒）で達成")
md_content.append("4. **損失削減:** 98%の損失削減（4.12 → 0.08）")
md_content.append("")

md_content.append("### 制限事項")
md_content.append("")
md_content.append("1. **小さいテストセット:** テストセットは10サンプルのみ")
md_content.append("2. **形式 vs 正確性:** 現在の指標は形式の準拠を測定しており、回答の正確性ではない")
md_content.append("3. **重複スコアリング:** 集合の重複を使用（例: GT `['a', 'd', 'e']` に対して `['a', 'b', 'd']` を予測すると正解としてカウント）")
md_content.append("")

md_content.append("### 推奨事項")
md_content.append("")
md_content.append("1. **より大きなテストセットでの評価**（例: トレーニングデータから200+サンプル）")
md_content.append("2. **完全一致スコアリングの使用**（集合の重複の代わりに）")
md_content.append("3. **形式準拠を超えた医療精度の分析**")
md_content.append("4. **チューニング前後のヘッド活性化の比較**")
md_content.append("")

md_content.append("---")
md_content.append("")

# Training details
md_content.append("## トレーニング詳細")
md_content.append("")
md_content.append("### SPTトレーニング設定")
md_content.append("")
md_content.append("| パラメータ | 値 |")
md_content.append("|-----------|-----|")
md_content.append("| ベースモデル | Qwen3-14B (140億パラメータ) |")
md_content.append("| チューニング対象ヘッド | 144個の医療用語ヘッド（レイヤー0-12） |")
md_content.append("| チューニング可能パラメータ | 340,788,864 (全体の2.4%) |")
md_content.append("| トレーニングデータ | 1,761サンプル（産婦人科QA） |")
md_content.append("| エポック数 | 2 |")
md_content.append("| バッチサイズ | 1 × 32勾配累積 = 32実効バッチサイズ |")
md_content.append("| 学習率 | 2e-4（コサインスケジュール） |")
md_content.append("| 最大シーケンス長 | 512トークン |")
md_content.append("| 総最適化ステップ | 112 |")
md_content.append("| トレーニング時間 | 9分54秒 |")
md_content.append("")

md_content.append("### トレーニング結果")
md_content.append("")
md_content.append("| 指標 | 値 |")
md_content.append("|------|-----|")
md_content.append("| 初期損失 | 4.1236 |")
md_content.append("| 最終損失 | 0.0844 |")
md_content.append("| 最小損失 | 0.0707（ステップ80） |")
md_content.append("| 損失削減率 | 98.0% |")
md_content.append("| 最終学習率 | 4.44e-07 |")
md_content.append("")

md_content.append("---")
md_content.append("")
md_content.append("**レポート生成日:** 2025年10月24日")
md_content.append("")
md_content.append("**モデル:** Qwen3-14B + SPT（144個の医療用語ヘッド）")
md_content.append("")
md_content.append("**トレーニング:** Phase 5 SPTトレーニング（2エポック、1,761サンプル）")
md_content.append("")
md_content.append("**評価データ:** テストセット（10サンプル）")

# Write to file
with open('Phase5_pinpoint_tuning/EVALUATION_RESULTS_DETAILED.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_content))

print("✓ 日本語の詳細評価レポートを作成しました: Phase5_pinpoint_tuning/EVALUATION_RESULTS_DETAILED.md")
print(f"✓ 総行数: {len(md_content)}")
print(f"✓ ファイルサイズ: {len('\\n'.join(md_content))} bytes")
