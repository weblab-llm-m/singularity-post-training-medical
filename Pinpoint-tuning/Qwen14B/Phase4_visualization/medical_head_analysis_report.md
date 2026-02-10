# 産婦人科データ Path Patching 分析レポート

**生成日時**: 2025-10-23 19:54:20

## サマリー

### ヘッド分類結果

| カテゴリ | 数 | 割合 |
|---------|---|-----|
| Medical Term Heads | 1 | 6.2% |
| Guideline Indicator Heads | 794 | 4962.5% |
| Reasoning Flow Heads | 0 | 0.0% |
| Unclassified | 805 | 5031.2% |

### Path Patching Impact


| Head Type | Mean Impact | Std Impact |
|-----------|------------|------------|
| medical_term_heads | 1.3245 | 0.0000 |
| guideline_indicator_heads | -0.9038 | 1.6267 |
| reasoning_flow_heads | nan | nan |


## 推奨事項

1. **Medical Term Heads**: 医療用語理解に重要 → Pinpoint Tuning対象として優先
2. **Guideline Indicator Heads**: ガイドライン参照に重要 → チューニング対象
3. **Reasoning Flow Heads**: 推論フローに寄与 → 選択的にチューニング

## 次のステップ

- Phase 5でトレーニング対象ヘッドを選択
- SPT (Supervised Pinpoint Tuning) を実行
