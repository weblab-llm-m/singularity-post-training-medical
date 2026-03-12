
# Phase 4-3: フォーマット遵守率検証

cleaned_predictionがNone（[ans]...[/ans]パース失敗）の割合

| Model | Dataset | Total | Parse Fail | Fail Rate |
|-------|---------|-------|------------|-----------|
| Base 30B | guideline_wrong_filtered | 5178 | 0 | 0.0% |
| Base 30B | igakuqa | 1122 | 0 | 0.0% |
| Base 30B | specialist_exam_test_v2 | 3757 | 0 | 0.0% |
| Pos PP SFT | guideline_wrong_filtered | 5178 | 0 | 0.0% |
| Pos PP SFT | igakuqa | 1122 | 0 | 0.0% |
| Pos PP SFT | specialist_exam_test_v2 | 3757 | 0 | 0.0% |
| Neg PP SFT | guideline_wrong_filtered | 5178 | 0 | 0.0% |
| Neg PP SFT | igakuqa | 1122 | 0 | 0.0% |
| Neg PP SFT | specialist_exam_test_v2 | 3757 | 0 | 0.0% |
| Base 80B | guideline_wrong_filtered | 5178 | 0 | 0.0% |
| Base 80B | igakuqa | 1122 | 0 | 0.0% |
| Base 80B | specialist_exam_test_v2 | 3757 | 0 | 0.0% |
| GRPO | igakuqa | 1122 | 0 | 0.0% |
| GRPO | specialist_exam_test_v2 | 3757 | 0 | 0.0% |
| GRPO | specialist_exam_v2 | 6910 | 0 | 0.0% |
| GSPO | igakuqa | 1122 | 0 | 0.0% |
| GSPO | specialist_exam_test_v2 | 3757 | 0 | 0.0% |
| CHORD | igakuqa | 1122 | 0 | 0.0% |
| CHORD | specialist_exam_test_v2 | 3757 | 0 | 0.0% |
| CHORD | specialist_exam_v2 | 6910 | 0 | 0.0% |

# Phase 2: RL系手法の統計検定付き比較


## igakuqa

共通問題数: 1122

- **Base 80B**: 89.5% (95% CI: [87.6%, 91.3%])
- **GRPO**: 90.3% (95% CI: [88.5%, 92.0%])
- **GSPO**: 90.2% (95% CI: [88.5%, 91.9%])
- **CHORD**: 90.4% (95% CI: [88.7%, 92.2%])

### McNemar検定 (vs Base 80B)

| Comparison | Base→X gained | X→Base lost | chi2 | p-value | sig |
|------------|--------------|-------------|------|---------|-----|
| Base vs GRPO | 41 | 32 | 0.88 | 0.3491 | n.s. |
| Base vs GSPO | 38 | 30 | 0.72 | 0.3960 | n.s. |
| Base vs CHORD | 40 | 30 | 1.16 | 0.2821 | n.s. |

### McNemar検定 (手法間)

| Comparison | A→B gained | B→A lost | chi2 | p-value | sig |
|------------|-----------|---------|------|---------|-----|
| GRPO vs GSPO | 30 | 31 | 0.00 | 1.0000 | n.s. |
| GRPO vs CHORD | 36 | 35 | 0.00 | 1.0000 | n.s. |
| GSPO vs CHORD | 38 | 36 | 0.01 | 0.9075 | n.s. |

## specialist_exam_test_v2

共通問題数: 3757

- **Base 80B**: 69.0% (95% CI: [67.5%, 70.4%])
- **GRPO**: 71.1% (95% CI: [69.7%, 72.5%])
- **GSPO**: 70.9% (95% CI: [69.5%, 72.4%])
- **CHORD**: 70.0% (95% CI: [68.5%, 71.6%])

### McNemar検定 (vs Base 80B)

| Comparison | Base→X gained | X→Base lost | chi2 | p-value | sig |
|------------|--------------|-------------|------|---------|-----|
| Base vs GRPO | 341 | 261 | 10.37 | 0.0013 | ** |
| Base vs GSPO | 342 | 269 | 8.48 | 0.0036 | ** |
| Base vs CHORD | 317 | 278 | 2.43 | 0.1193 | n.s. |

### McNemar検定 (手法間)

| Comparison | A→B gained | B→A lost | chi2 | p-value | sig |
|------------|-----------|---------|------|---------|-----|
| GRPO vs GSPO | 217 | 224 | 0.08 | 0.7751 | n.s. |
| GRPO vs CHORD | 234 | 275 | 3.14 | 0.0762 | n.s. |
| GSPO vs CHORD | 218 | 252 | 2.32 | 0.1280 | n.s. |

# Phase 3: Pinpoint SFT 劣化分析


## igakuqa

### McNemar検定 (vs Base 30B, N=1122)

| Comparison | Base→X gained | X→Base lost | Net | chi2 | p-value | sig |
|------------|--------------|-------------|-----|------|---------|-----|
| Base vs Pos PP SFT | 57 | 123 | -66 | 23.47 | 0.000001 | *** |
| Base vs Neg PP SFT | 58 | 136 | -78 | 30.56 | 0.000000 | *** |

## specialist_exam_test_v2

### McNemar検定 (vs Base 30B, N=3757)

| Comparison | Base→X gained | X→Base lost | Net | chi2 | p-value | sig |
|------------|--------------|-------------|-----|------|---------|-----|
| Base vs Pos PP SFT | 305 | 905 | -600 | 296.53 | 0.000000 | *** |
| Base vs Neg PP SFT | 337 | 813 | -476 | 196.20 | 0.000000 | *** |

### 診療科別の劣化パターン (Base正解→SFT不正解)

| 診療科 | Base正答 | Pos壊れ | Pos壊れ率 | Neg壊れ | Neg壊れ率 |
|--------|---------|---------|----------|---------|----------|
| ganka | 138 | 59 | 42.8% | 64 | 46.4% |
| geka | 202 | 82 | 40.6% | 71 | 35.1% |
| kanzo | 129 | 59 | 45.7% | 41 | 31.8% |
| kyukyu | 182 | 83 | 45.6% | 61 | 33.5% |
| masui | 204 | 96 | 47.1% | 85 | 41.7% |
| naika | 282 | 59 | 20.9% | 56 | 19.9% |
| sanfuqa | 231 | 69 | 29.9% | 81 | 35.1% |
| seikeigekaqa | 115 | 49 | 42.6% | 38 | 33.0% |
| seishin | 231 | 87 | 37.7% | 82 | 35.5% |
| shinkeinaika | 116 | 55 | 47.4% | 47 | 40.5% |
| shinzogeka | 157 | 67 | 42.7% | 63 | 40.1% |
| syokaki | 146 | 63 | 43.2% | 55 | 37.7% |
| zibika | 154 | 77 | 50.0% | 69 | 44.8% |

# Phase 4-2: 正誤パターン分析

## 80B Group: Base → RL手法の正誤変化


### igakuqa

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| Base○ RL全○ | 935 | 83.3% | 全モデル正解（安定） |
| Base× RL全× | 58 | 5.2% | 全モデル不正解（困難問題） |
| Base× → RL全○ | 20 | 1.8% | RL学習で全手法が獲得 |
| Base○ → RL全× | 6 | 0.5% | RL学習で全手法が喪失 |
| Base× → 一部RL○ | 40 | 3.6% | 手法間で差が出た問題 |
| Base○ → 一部RL× | 63 | 5.6% | 一部手法で劣化 |

**各RL手法の独自改善/独自劣化:**

| 手法 | Baseから改善 | うち独自改善 | Baseから劣化 | うち独自劣化 |
|------|-----------|------------|-----------|------------|
| GRPO | 41 | 5 | 32 | 18 |
| GSPO | 38 | 7 | 30 | 15 |
| CHORD | 40 | 9 | 30 | 19 |

### specialist_exam_test_v2

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| Base○ RL全○ | 2116 | 56.3% | 全モデル正解（安定） |
| Base× RL全× | 659 | 17.5% | 全モデル不正解（困難問題） |
| Base× → RL全○ | 174 | 4.6% | RL学習で全手法が獲得 |
| Base○ → RL全× | 98 | 2.6% | RL学習で全手法が喪失 |
| Base× → 一部RL○ | 333 | 8.9% | 手法間で差が出た問題 |
| Base○ → 一部RL× | 377 | 10.0% | 一部手法で劣化 |

**各RL手法の独自改善/独自劣化:**

| 手法 | Baseから改善 | うち独自改善 | Baseから劣化 | うち独自劣化 |
|------|-----------|------------|-----------|------------|
| GRPO | 341 | 65 | 261 | 79 |
| GSPO | 342 | 58 | 269 | 71 |
| CHORD | 317 | 65 | 278 | 90 |

## 30B Group: Base → Pinpoint SFTの正誤変化


### igakuqa

| 手法 | Baseから改善 | Baseから劣化 | Net | 改善率 | 劣化率 |
|------|-----------|-----------|-----|--------|--------|
| Pos PP SFT | 57 (36.8% of Base不正解) | 123 (12.7% of Base正解) | -66 | 5.1% | 11.0% |
| Neg PP SFT | 58 (37.4% of Base不正解) | 136 (14.1% of Base正解) | -78 | 5.2% | 12.1% |

### specialist_exam_test_v2

| 手法 | Baseから改善 | Baseから劣化 | Net | 改善率 | 劣化率 |
|------|-----------|-----------|-----|--------|--------|
| Pos PP SFT | 305 (20.7% of Base不正解) | 905 (39.6% of Base正解) | -600 | 8.1% | 24.1% |
| Neg PP SFT | 337 (22.9% of Base不正解) | 813 (35.5% of Base正解) | -476 | 9.0% | 21.6% |