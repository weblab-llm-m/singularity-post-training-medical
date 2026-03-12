# Phase 4: Activation Patching 分析レポート
## Qwen3-30B-A3B-Instruct-2507

**サンプル数**: 256
**総ヘッド数**: 1536 (48層 × 32ヘッド)

---

## ヘッド分類結果

| カテゴリ | 数 | 割合 | Mean Impact | Std | Median |
|---------|-----|------|-------------|-----|--------|
| Medical Term | 143 | 9.3% | -0.0039% | 0.0221% | 0.0002% |
| Profile Indicator | 205 | 13.3% | -0.0086% | 0.0215% | -0.0080% |
| Reasoning Flow | 469 | 30.5% | -0.0086% | 0.0262% | -0.0079% |
| Unclassified | 719 | 46.8% | -0.0075% | 0.0258% | -0.0073% |

---

## レイヤーグループ別分析

| グループ | レイヤー | Medical | Profile | Reasoning | 合計 |
|---------|---------|---------|---------|-----------|------|
| Early | 0-15 | 39 | 65 | 158 | 262 |
| Middle | 16-31 | 29 | 64 | 182 | 275 |
| Late | 32-47 | 75 | 76 | 129 | 280 |

---

## Top 10 Most Negative Impact Heads

| Rank | Layer | Head | Impact (%) | Category |
|------|-------|------|------------|----------|
| 1 | 47 | 28 | -0.1140 | Reasoning Flow |
| 2 | 0 | 6 | -0.0929 | Reasoning Flow |
| 3 | 1 | 27 | -0.0903 | Reasoning Flow |
| 4 | 0 | 11 | -0.0872 | Reasoning Flow |
| 5 | 1 | 18 | -0.0788 | Reasoning Flow |
| 6 | 47 | 26 | -0.0767 | Reasoning Flow |
| 7 | 4 | 26 | -0.0760 | Medical Term |
| 8 | 23 | 15 | -0.0727 | Reasoning Flow |
| 9 | 8 | 11 | -0.0725 | Profile Indicator |
| 10 | 12 | 14 | -0.0703 | Reasoning Flow |

---

## 生成ファイル

- `head_classification_heatmap_*samples.html`: インタラクティブヒートマップ
- `head_classification_heatmap_*samples.png`: 静的ヒートマップ (300dpi)
- `statistical_report.json`: 統計データ (JSON)
- `PHASE4_ANALYSIS_REPORT.md`: 本レポート
