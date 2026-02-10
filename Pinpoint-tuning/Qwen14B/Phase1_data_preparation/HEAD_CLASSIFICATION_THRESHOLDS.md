# Head分類の閾値と計算方法

## 概要

本ドキュメントは、Phase3のHead分類で使用される閾値の具体的な計算方法を記録する。

### 参考文献

| 論文 | 内容 |
|------|------|
| Zhang et al. (2025) arXiv:2502.11806 | 翻訳メカニズム論文。3種類のHeads分類の概念的基盤 |
| Wang et al. (2022) arXiv:2211.00593 | Path Patching原論文（GPT-2） |
| Chen et al. (2024) arXiv:2409.01658 | Sycophancy論文 |

### 概念の対応関係

| Zhang et al. (翻訳タスク) | 医療QA実装 |
|---------------------------|-----------|
| Source Heads | Medical Term Heads |
| Indicator Heads | Guideline Indicator Heads |
| Positional Heads | Reasoning Flow Heads |

**重要**: Zhang et al.の論文は定性的な記述のみ（「スパイク状」「均一な注意」など）で、具体的な閾値は記載されていない。本実装の閾値は独自に設計されたもの。

---

## 7カテゴリの用途

### 辞書カテゴリとポジションの対応

| カテゴリ | 集約先 |
|---------|--------|
| `diseases` | `medical_term_positions` |
| `diagnostic_methods` | `medical_term_positions` |
| `biomarkers` | `medical_term_positions` |
| `treatments` | `medical_term_positions` |
| `anatomical_terms` | `medical_term_positions` |
| `guidelines` | `guideline_indicator_positions` |
| `reasoning_keywords` | `reasoning_keyword_positions` |

### Phase3での用途

| Head タイプ | 使用するポジション | 役割 |
|------------|------------------|------|
| Medical Term Heads | `medical_term_positions` | 医療用語の意味抽出 |
| Guideline Indicator Heads | `guideline_indicator_positions` | 質問タイプ（正誤判定等）の認識 |
| Reasoning Flow Heads | `reasoning_keyword_positions` | 論理的推論フローの管理 |

---

## 入力データ構造

### 注意パターン

```python
attention_patterns: Dict[int, torch.Tensor]
# {layer_idx: [batch, num_heads, seq_len]}
```

### 前処理：サンプル平均化

各レイヤー・各ヘッドについて、全サンプルで平均化した注意パターン `[seq_len]` を使用：

```python
def _get_average_attention_pattern(self, layer: int, head: int) -> torch.Tensor:
    # [batch, num_heads, seq_len] → [batch, seq_len]
    layer_patterns = self.attention_patterns[layer]
    head_patterns = layer_patterns[:, head, :]  # [batch, seq_len]

    # バッチ平均
    avg_pattern = head_patterns.mean(dim=0)  # [seq_len]

    return avg_pattern
```

---

## 1. Medical Term Heads

### 概念

翻訳論文のSource Headsに相当。医療用語トークンに対して高い平均注意を示すヘッド。

### 計算方法

```python
def is_medical_term_head(self, attn_pattern: torch.Tensor) -> bool:
    # 全サンプルの医療用語トークン位置を集計
    all_medical_positions = []
    for annotation in self.annotation_data:
        positions = annotation.get('medical_term_positions', [])
        all_medical_positions.extend(positions)

    # シーケンス長を超えない有効な位置のみ使用
    seq_len = len(attn_pattern)
    valid_positions = [p for p in all_medical_positions if p < seq_len]

    # 医療用語位置への注意値の平均を計算
    medical_attention_score = attn_pattern[valid_positions].mean().item()

    # 閾値判定
    threshold = 0.002  # 0.2%
    return medical_attention_score > threshold
```

### 閾値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `threshold` | 0.002 | 医療用語位置への平均注意が0.2%以上 |

### 「平均注意」の意味

- `attn_pattern[seq_len]`: 各トークン位置への注意重み（softmax後、合計≈1.0）
- 医療用語位置（例: トークン[5,6,12,13,20]）の注意値を取り出して平均

### 図解

```
注意パターン [seq_len] の例:
位置:    [0]    [1]    [2]    [3]    [4]    [5]    ...
注意値:  0.01   0.02   0.15   0.03   0.01   0.08   ...
                 ↑                          ↑
              medical                    medical
              term位置                   term位置

計算: (0.02 + 0.08) / 2 = 0.05 → 0.002より大きい → Medical Term Head
```

---

## 2. Guideline Indicator Heads

### 概念

翻訳論文のIndicator Headsに相当。ガイドライン指示語（「正しいものを選べ」「CQ」等）に対してスパイク状の注意パターンを示すヘッド。

### 計算方法

```python
def is_guideline_indicator_head(self, attn_pattern: torch.Tensor) -> bool:
    # 全サンプルのガイドライン位置を集計
    all_guideline_positions = []
    for annotation in self.annotation_data:
        positions = annotation.get('guideline_indicator_positions', [])
        all_guideline_positions.extend(positions)

    seq_len = len(attn_pattern)
    valid_positions = [p for p in all_guideline_positions if p < seq_len]

    # ガイドライン位置への最大注意
    max_attn_to_guideline = attn_pattern[valid_positions].max().item()

    # 他の位置（ガイドライン以外）への平均注意
    other_positions = [i for i in range(seq_len) if i not in valid_positions]
    mean_other_attn = attn_pattern[other_positions].mean().item()

    # スパイク比率
    spike_ratio = max_attn_to_guideline / (mean_other_attn + 1e-10)

    # 閾値判定（AND条件）
    spike_threshold = 0.005  # 0.5%
    spike_ratio_threshold = 2.0

    return (max_attn_to_guideline > spike_threshold and
            spike_ratio > spike_ratio_threshold)
```

### 閾値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `spike_threshold` | 0.005 | 最大注意値が0.5%以上（絶対閾値） |
| `spike_ratio` | 2.0 | スパイク比率（max/mean_other）が2.0以上（相対閾値） |

### 2段階フィルタリングの理由

- **絶対閾値のみ**: 全体的に低い注意でもノイズを検出してしまう
- **相対閾値のみ**: 全体的に高い注意の場合も検出してしまう
- **AND条件**: 真のスパイクパターンのみを検出

### 図解

```
スパイクパターンの例:
位置:    [0]    [1]    [2]    [3]    [4]    [5]    ...
注意値:  0.01   0.01   0.01   0.08   0.01   0.01   ...
                              ↑
                           guideline
                           位置（スパイク）

計算:
  max_attn = 0.08
  mean_other = (0.01 × 5) / 5 = 0.01
  spike_ratio = 0.08 / 0.01 = 8.0

判定:
  0.08 > 0.005 (✓) AND 8.0 > 2.0 (✓) → Guideline Indicator Head
```

---

## 3. Reasoning Flow Heads

### 概念

翻訳論文のPositional Headsに相当。推論キーワード（`<think>`、「したがって」等）と隣接トークンへの均一な注意分布を示すヘッド。

### 計算方法

```python
def is_reasoning_flow_head(self, attn_pattern: torch.Tensor) -> bool:
    # 全サンプルの推論キーワード位置を集計
    all_reasoning_positions = []
    for annotation in self.annotation_data:
        positions = annotation.get('reasoning_keyword_positions', [])
        all_reasoning_positions.extend(positions)

    seq_len = len(attn_pattern)

    # 末尾の隣接トークン（窓サイズ=3）
    adjacent_window = 3
    adjacent_positions = list(range(max(0, seq_len - adjacent_window), seq_len))

    # 推論位置と隣接位置を結合
    relevant_positions = list(set(all_reasoning_positions + adjacent_positions))
    relevant_positions = [p for p in relevant_positions if p < seq_len]

    # 均一性を測定
    relevant_attention = attn_pattern[relevant_positions]
    attention_std = relevant_attention.std().item()
    attention_mean = relevant_attention.mean().item()

    # 相対的均一性
    relative_std = attention_std / (attention_mean + 1e-10)

    # 閾値判定
    uniformity_threshold = 0.002  # 標準偏差 < 0.2%
    mean_threshold = 0.0005       # 平均注意 > 0.05%
    relative_std_threshold = 1.2  # 相対標準偏差 < 1.2

    # 絶対基準 OR 相対基準
    absolute_criteria = (attention_std < uniformity_threshold and
                        attention_mean > mean_threshold)
    relative_criteria = (relative_std < relative_std_threshold and
                        attention_mean > mean_threshold)

    return absolute_criteria or relative_criteria
```

### 閾値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `uniformity_threshold` | 0.002 | 標準偏差が0.2%以下（絶対的均一性） |
| `attention_mean_threshold` | 0.0005 | 平均注意が0.05%以上（最小注意レベル） |
| `relative_std_threshold` | 1.2 | std/mean < 1.2（相対的均一性） |
| `adjacent_window` | 3 | 末尾から3トークンを隣接位置として追加 |

### OR条件の理由

- **絶対基準**: 注意値が全体的に低い場合でも均一性を検出
- **相対基準**: 注意値が高い場合でも相対的な均一性を検出

### 図解

```
均一パターンの例:
位置:    [0]    [1]    [2]    [3]    [4]    [5]    ...  [n-2]  [n-1]
注意値:  0.01   0.01   0.02   0.01   0.01   0.02   ...  0.02   0.01
                 ↑             ↑                         ↑      ↑
              reasoning     reasoning                 adjacent adjacent

relevant_positions = [1, 3, n-2, n-1]
relevant_attention = [0.01, 0.01, 0.02, 0.01]

計算:
  std = 0.005
  mean = 0.0125
  relative_std = 0.005 / 0.0125 = 0.4

判定:
  絶対基準: 0.005 < 0.002 (✗) AND 0.0125 > 0.0005 (✓) → ✗
  相対基準: 0.4 < 1.2 (✓) AND 0.0125 > 0.0005 (✓) → ✓
  → Reasoning Flow Head
```

---

## 設定ファイル

閾値は `configs/head_classification_params.yaml` で管理：

```yaml
classification_criteria:
  medical_term:
    threshold: 0.002

  guideline_indicator:
    spike_threshold: 0.005
    spike_ratio: 2.0

  reasoning_flow:
    uniformity_threshold: 0.002
    attention_mean_threshold: 0.0005
    relative_std_threshold: 1.2
    adjacent_window: 3
```

---

## 分類の優先順位

ヘッドは以下の優先順位で分類される（排他的）：

1. Medical Term Heads
2. Guideline Indicator Heads
3. Reasoning Flow Heads
4. Unclassified

```python
if self.is_medical_term_head(avg_attention_pattern):
    return 'medical_term_heads'
elif self.is_guideline_indicator_head(avg_attention_pattern):
    return 'guideline_indicator_heads'
elif self.is_reasoning_flow_head(avg_attention_pattern):
    return 'reasoning_flow_heads'
else:
    return 'unclassified'
```

---

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| `Phase1_data_preparation/medical_term_annotator.py` | ポジションのアノテーション |
| `Phase1_data_preparation/medical_terms_dictionary.json` | 7カテゴリの用語辞書 |
| `Phase3_attention_analysis/head_classifier.py` | Head分類の実装 |
| `configs/head_classification_params.yaml` | 閾値設定 |

---

**作成日**: 2026-02-05
**最終更新**: 2026-02-05
