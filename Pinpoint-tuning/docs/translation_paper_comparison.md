# 翻訳論文とMedical Path Patchingの比較分析

**比較対象:**
- 論文: Zhang et al. (2025) "Exploring Translation Mechanism of Large Language Models" arXiv:2502.11806
- 実装: `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching`

**作成日:** 2025-10-26

---

## 1. Impact（影響度）計算の比較

### Zhang et al. (2025) - 翻訳タスク

**計算式:**
```
δⱼ = (ỹf - yf) / (yf + ε)
```

**説明:**
- `ỹf`: パッチング後のロジット値
- `yf`: 元のロジット値
- `ε`: ゼロ除算を防ぐための微小値
- **結果は百分率**として解釈される
- **閾値**: >1.0%のロジット変化で「重要なヘッド（crucial head）」と定義

**特徴:**
- 単一のロジット値を使用
- パッチングによる**絶対的なロジット変化**を測定
- データセット全体で集約して重要性を評価

### Medical Path Patching - 医療QAタスク

**計算式:**
```python
impact % = ((patched_logit_diff - default_logit_diff) / default_logit_diff) × 100
```

**説明:**
- `patched_logit_diff`: パッチング後の（正解ロジット - 誤答ロジット）
- `default_logit_diff`: 元の（正解ロジット - 誤答ロジット）
- **結果は百分率**として解釈される
- **閾値**: 正のimpact（>0）のヘッドを「効果的なヘッド」として使用（Phase5_v5では24ヘッド）

**特徴:**
- **ロジット差分**を使用（正解と誤答の相対的な差）
- パッチングによる**答えの確信度の変化**を測定
- サンプル全体で平均化して各ヘッドのimpactを算出

### 共通点

1. **Path Patching手法**: 両者とも同じ根本的アプローチ（Wang et al. 2022の手法）
2. **百分率表現**: 両者とも元の値で正規化し、百分率で表現
3. **因果関係測定**: ヘッドをパッチングすることで、そのヘッドの因果的影響を定量化
4. **閾値ベース選択**: 重要なヘッドを選択するために閾値を使用

### 相違点

| 観点 | Zhang et al. (2025) | Medical Path Patching |
|------|---------------------|----------------------|
| **測定対象** | 単一ロジット値の変化 | ロジット差分（正解-誤答）の変化 |
| **解釈** | ターゲットトークン生成確率の変化 | 正答確信度の変化 |
| **閾値** | >1.0% | >0%（正のimpact） |
| **選択率** | <5%のヘッドが重要 | 7.5%のヘッド（24/321）が正のimpact |
| **タスク特性** | 翻訳（生成タスク） | 多肢選択QA（分類タスク） |

---

## 2. Source Heads vs Medical Term Heads

### Zhang et al. (2025) - Source Heads

**定義:**
> "demonstrate concentrated attention on source-language tokens, specializing in cross-lingual alignment"
> （ソース言語トークンに集中した注意を示し、言語間アライメントに特化）

**検出方法:**
- 注意重み分布の可視化により特定
- ソース入力トークンへの「集中した注意パターン」を識別
- 入力シーケンス位置における注意重みの分布を測定

**特徴:**
- 翻訳における**ソース言語の意味抽出**を担当
- 中間層（layers 12-20）と最終2層に集中
- クロスリンガルな意味アライメントが主要機能

### Medical Path Patching - Medical Term Heads

**定義:**
> 医学用語トークンに対して、高い平均注意を示すヘッド

**検出方法:**
```python
def is_medical_term_head(self, attn_pattern: torch.Tensor) -> bool:
    # 医学用語位置への注意を抽出
    medical_attention_score = attn_pattern[medical_term_positions].mean().item()
    threshold = 0.20
    return medical_attention_score > threshold
```

**計算プロセス:**
1. **サンプル間平均化**: 438サンプルの注意パターンを平均
2. **位置間平均化**: 医学用語位置の注意重みを平均
3. **閾値判定**: 平均値 > 0.20で Medical Term Head と分類

**特徴:**
- 医療QAにおける**医学用語の意味抽出**を担当
- 質問文中の医学用語（疾患名、治療法など）に注目
- ドメイン特化語彙の認識が主要機能

### 共通点

1. **意味的特徴抽出**: 両者とも入力の重要な意味的要素を抽出
2. **集中型注意パターン**: 特定のトークン群に注意が集中
3. **タスク特化**: Source=ソース言語、Medical=医学用語という、各タスクに特化した対象
4. **クロスドメインアライメント**: Source=言語間、Medical=一般語彙↔専門語彙のアライメント

### 相違点

| 観点 | Source Heads | Medical Term Heads |
|------|-------------|-------------------|
| **対象トークン** | ソース言語全体 | 医学用語のみ |
| **検出基準** | 可視化による定性的判断 | 平均注意 >0.20（定量的閾値） |
| **計算方法** | 記載なし（可視化ベース） | 2段階平均化（サンプル→位置） |
| **機能** | 言語間意味転送 | ドメイン特化語彙認識 |
| **層分布** | 中間層+最終2層 | 全層（Phase3で40層×40ヘッド分析） |

---

## 3. Indicator Heads vs Guideline Indicator Heads

### Zhang et al. (2025) - Indicator Heads

**定義:**
> "exhibit spike-shaped attention patterns on translation-specific indicators (e.g., language identifiers like 'English' or '中文', and structural cues like colons)"
> （翻訳特有の指示子に対してスパイク状の注意パターンを示す。例：言語識別子「English」「中文」、構造的手がかり「:」など）

**検出方法:**
- 指示子トークン位置での注意値分布を観察
- **「スパイク状」の集中パターン**を識別
- 言語マーカーや句読点への焦点を追跡

**特徴:**
- タスク指示の解釈を担当
- 「Translate to English:」などのプロンプト構造を認識
- **離散的な指示子トークン**への極端な注意集中

### Medical Path Patching - Guideline Indicator Heads

**定義:**
> 診療ガイドライン指示語（「正しいものを選べ」「誤っているものはどれか」など）に対して、スパイク状の注意パターンを示すヘッド

**検出方法:**
```python
def is_guideline_indicator_head(self, attn_pattern: torch.Tensor) -> bool:
    # ガイドライン指示語への最大注意
    max_attn_to_guideline = attn_pattern[guideline_positions].max().item()
    # その他位置への平均注意
    mean_other_attn = attn_pattern[other_positions].mean().item()
    # スパイク比率
    spike_ratio = max_attn_to_guideline / (mean_other_attn + 1e-10)

    # 2段階フィルタリング
    return (max_attn_to_guideline > 0.15 and  # 絶対閾値
            spike_ratio > 3.0)                  # 相対閾値
```

**計算プロセス:**
1. **絶対閾値**: 指示語への最大注意 > 0.15（ノイズ除外）
2. **相対閾値**: スパイク比率 > 3.0（真のスパイクパターン検証）
3. **AND条件**: 両方を満たす場合のみ Guideline Indicator Head と分類

**特徴:**
- 質問タイプ（正誤判定、数量指定など）の認識を担当
- 「正しい」「誤り」「1個選べ」「3個選べ」などの指示語に注目
- **2段階フィルタリング**により偽陽性を防止

### 共通点

1. **スパイクパターン**: 両者とも特定の指示子トークンへの極端な注意集中
2. **タスク指示解釈**: タスクの実行方法を指示する要素に注目
3. **構造的手がかり**: 言語マーカー（翻訳）/質問タイプマーカー（医療QA）の認識
4. **機能的役割**: 「何をすべきか」の理解を担当

### 相違点

| 観点 | Indicator Heads | Guideline Indicator Heads |
|------|----------------|--------------------------|
| **対象指示子** | 言語識別子（English, 中文, :） | 質問指示語（正しい, 誤り, 選べ） |
| **検出基準** | 定性的（可視化でスパイク確認） | 定量的（max>0.15 AND ratio>3.0） |
| **閾値** | 記載なし | 絶対0.15 + 相対3.0 |
| **フィルタリング段階** | 不明 | 2段階（絶対+相対）のAND条件 |
| **機能** | 言語方向指定 | 質問タイプ（正誤/数量）認識 |
| **論文での明示度** | 「スパイク状」のみ記載 | 具体的な閾値・計算式あり |

**重要な独自性（Medical Path Patching）:**
- **スパイク比率（spike_ratio = max / mean_other）**の導入は、Zhang et al.の論文には明示されていない
- これは実装者（Claude Code）が「スパイク状」という概念を定量化するために独自に設計した手法
- 絶対閾値のみでは不十分（全体的に高い注意の場合も検出してしまう）ため、相対的な突出度を測る必要性から生まれた

---

## 4. Positional Heads vs Reasoning Flow Heads

### Zhang et al. (2025) - Positional Heads

**定義:**
> "predominately attend to adjacent tokens, managing contextual dependencies and resolving grammatical agreement"
> （主に隣接トークンに注意を向け、文脈依存関係の管理と文法的一致の解決を担当）

**検出方法:**
- 「入力文脈全体にわたる均一な注意」を示す注意分布を分析
- 集中型ではなく、**一貫した隣接性パターン**を識別
- 注意されるトークンの位置的近接性を測定

**特徴:**
- 構文的・位置的依存関係の処理を担当
- 隣接トークン間の関係性構築
- **均一な注意分布**が特徴（集中ではなく分散）

### Medical Path Patching - Reasoning Flow Heads

**定義:**
> 推論フロー全体に対して、均一な注意分布を示し、論理的推論の流れを管理するヘッド

**検出方法:**
```python
def is_reasoning_flow_head(self, attn_pattern: torch.Tensor) -> bool:
    # 質問+選択肢領域の注意を抽出
    relevant_attention = attn_pattern[question_and_options_positions]

    # 均一性の測定（標準偏差）
    attention_std = relevant_attention.std().item()
    # 平均注意レベル
    attention_mean = relevant_attention.mean().item()

    # 2条件判定
    return (attention_std < 0.05 and      # 均一性閾値
            attention_mean > 0.01)         # 最小注意閾値
```

**計算プロセス:**
1. **均一性評価**: 標準偏差 < 0.05（低い変動 = 均一な分布）
2. **平均注意レベル**: 平均 > 0.01（注意が十分に存在）
3. **AND条件**: 両方を満たす場合のみ Reasoning Flow Head と分類

**特徴:**
- 論理的推論の全体フローの管理を担当
- 質問文と全選択肢を均等に参照
- **低い標準偏差**が特徴（スパイクの逆）

### 共通点

1. **均一な注意分布**: 両者とも集中型ではなく分散型の注意パターン
2. **文脈的処理**: 局所的ではなく広範囲の文脈を考慮
3. **構造的役割**: 個別要素ではなく全体構造の管理
4. **統合機能**: 他の要素からの情報を統合して処理

### 相違点

| 観点 | Positional Heads | Reasoning Flow Heads |
|------|-----------------|---------------------|
| **注意対象** | 隣接トークン（局所的） | 質問+選択肢全体（大域的） |
| **検出基準** | 均一な注意分布（定性的） | 標準偏差<0.05（定量的） |
| **測定指標** | 位置的近接性 | 標準偏差 + 平均注意 |
| **機能** | 文法的一致・構文処理 | 論理的推論フロー管理 |
| **スコープ** | トークン隣接関係 | 問題全体の論理構造 |
| **論文での明示度** | 「均一な注意」のみ | 具体的な閾値・計算式あり |

**重要な独自性（Medical Path Patching）:**
- **標準偏差による均一性定量化**は、Zhang et al.の論文には明示されていない
- 「均一な注意」という概念を **std < 0.05** という具体的な指標に変換
- 多肢選択QAの特性（質問+複数選択肢）に合わせた設計

---

## 5. 総合比較：方法論の共通点と相違点

### 5.1 共通する基礎理論

両プロジェクトは同じ理論的基盤を共有：

1. **Path Patching手法** (Wang et al. 2022)
   - アテンションヘッドの因果的影響を定量化
   - パッチング前後のロジット変化を測定

2. **機能的ヘッド分類**
   - スパース性: わずかなヘッドが重要（<5% vs 7.5%）
   - 機能分化: 異なる役割を持つヘッドが存在
   - 3カテゴリ構造: 入力特徴/指示解釈/推論処理

3. **タスク特化型適応**
   - Zhang: 翻訳タスクに特化
   - Medical: 医療QAに特化

### 5.2 主要な方法論的相違点

| 観点 | Zhang et al. (2025) | Medical Path Patching |
|------|---------------------|----------------------|
| **研究アプローチ** | 定性的分析主体（可視化+解釈） | 定量的分析主体（閾値+自動分類） |
| **ヘッド分類方法** | 手動による可視化と解釈 | 自動分類アルゴリズム |
| **閾値の明示性** | 低い（>1.0%のみ） | 高い（全カテゴリで具体的閾値） |
| **計算式の詳細度** | 中程度 | 高い（全てコード化） |
| **再現性** | 中程度（定性的要素あり） | 高い（完全自動化可能） |
| **適用タスク** | 翻訳（生成タスク） | 医療QA（分類タスク） |

### 5.3 Medical Path Patchingの独自貢献

Phase3のhead_classifier.pyにおける独自の方法論的貢献：

1. **定量的閾値の確立**
   - Medical: 平均注意 > 0.20
   - Guideline: 最大注意 > 0.15 AND スパイク比率 > 3.0
   - Reasoning: 標準偏差 < 0.05 AND 平均注意 > 0.01

2. **2段階フィルタリング（Guideline）**
   - 絶対閾値: ノイズ除外
   - 相対閾値: 真のスパイク検証
   - Zhang論文では「スパイク状」の定性的記述のみ

3. **標準偏差による均一性定量化（Reasoning）**
   - 「均一な注意」を std < 0.05 という数値指標に変換
   - Zhang論文では「均一な注意分布」の定性的記述のみ

4. **自動分類システム**
   - 438サンプル × 40層 × 40ヘッド = 701,440パターンを自動分類
   - 人手による可視化・解釈なしで実行可能

### 5.4 適応の理由：タスクの違い

| タスク特性 | 翻訳 | 医療QA |
|-----------|-----|-------|
| **入力構造** | ソース文のみ | 質問+複数選択肢 |
| **出力形式** | ターゲット文生成 | 選択肢記号（a, b, c...） |
| **評価指標** | BLEU, 生成品質 | 正答率（分類精度） |
| **重要トークン** | ソース語彙全体 | 医学用語、指示語、選択肢 |
| **論理構造** | 線形（文の順序） | 階層的（質問→選択肢群） |

これらの違いにより、同じ理論的枠組み（3種類の機能的ヘッド）を異なる方法で実装：

- **Source → Medical**: 「ソース言語」から「医学用語」へのドメイン適応
- **Indicator → Guideline**: 「言語識別子」から「質問タイプ指示語」への適応
- **Positional → Reasoning**: 「隣接トークン」から「問題全体の論理フロー」への適応

---

## 6. 結論

### Zhang et al. (2025)からの理論的継承

Medical Path Patchingプロジェクトは、Zhang et al.の翻訳メカニズム研究から以下を継承：

1. **Path Patching手法**: 因果的影響の定量化
2. **3カテゴリのヘッド分類**: 入力特徴/指示解釈/推論処理
3. **スパース重要性**: わずかなヘッドが支配的
4. **機能的特化**: 各ヘッドが特定の役割を担当

### 医療QAタスクへの独自適応

一方で、以下の独自の方法論的発展を実現：

1. **定量的分類基準の確立**: 全カテゴリで明確な数値閾値
2. **自動分類アルゴリズム**: 大規模データで再現可能
3. **2段階フィルタリング**: 偽陽性防止の工夫（Guideline）
4. **標準偏差による均一性測定**: 定性的概念の定量化（Reasoning）
5. **タスク特化設計**: 医療QAの構造（質問+選択肢）に最適化

### 学術的位置づけ

本プロジェクトは：
- **理論的には**: Zhang et al. (2025)の翻訳メカニズム研究の直接的な応用
- **方法論的には**: 定性的分析を定量的・自動化可能な手法に拡張
- **タスク的には**: 生成タスク（翻訳）から分類タスク（医療QA）への適応

したがって、Zhang et al.の研究成果を基盤としながら、医療QAという新しいドメインに適応し、より再現性の高い自動分類手法を開発した**適応的拡張研究**と位置づけられる。

---

## 参考文献

1. **Zhang et al. (2025)**: "Exploring Translation Mechanism of Large Language Models", arXiv:2502.11806
2. **Wang et al. (2022)**: "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small", arXiv:2211.00593
3. **Chen et al. (2024)**: "Understanding Sycophancy in Language Models", arXiv:2409.01658
4. **Medical Path Patching**: Phase2 (path_patching_medical.py), Phase3 (head_classifier.py)

---

**ファイル参照:**
- `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase2_path_patching/path_patching_medical.py` - Impact計算実装
- `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase3_attention_analysis/head_classifier.py` - ヘッド分類実装
- `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/README.md` - プロジェクト概要
