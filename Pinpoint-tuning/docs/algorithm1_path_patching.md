# Algorithm 1: Path Patchingによる重要コンポーネント検出 - 詳細解説

**出典:** Zhang et al. (2025) "Exploring Translation Mechanism of Large Language Models" arXiv:2502.11806

---

## アルゴリズム概要

**目的:** モデル内の各コンポーネント（アテンションヘッドやMLP）が予測にどれだけ因果的に影響しているかを定量化する

**手法:** Path Patching - コンポーネントの活性化を「反事実的データ」から「事実的データ」にパッチング（置き換え）し、出力の変化を測定

---

## 入力・出力

### Require（入力）
- **Dataset D**: 事実的/反事実的ペアのデータセット `(X_f, X_cf)`
  - `X_f`: 事実的入力（factual input）- 正常な入力データ
  - `X_cf`: 反事実的入力（counterfactual input）- 意図的に変更された入力データ
- **Model F**: 分析対象のモデル
- **Components C**: モデル内のコンポーネント集合（アテンションヘッド、MLPなど）

### Ensure（出力）
- **Node importance scores Δ**: 各コンポーネントの重要度スコア `δ_1, ..., δ_m`

---

## アルゴリズム詳細解説

### フェーズ1: データペアごとの処理（Lines 1-11）

#### Line 1: データペアのループ開始
```
for each data pair (X_f^(i), X_cf^(i)) ∈ D do
```

**解説:**
- データセット内の各ペアを順番に処理
- `i`: データペアのインデックス
- 例（翻訳タスク）:
  - `X_f`: "Translate to English: 你好" → 正常な翻訳プロンプト
  - `X_cf`: "Translate to French: 你好" → 言語を変更した反事実的プロンプト

---

#### Line 2: 参照活性化の計算
```
Compute reference activations H_f ← F(X_f^(i))
```

**解説:**
- **事実的入力**をモデルに通し、全層・全コンポーネントの内部活性化を記録
- `H_f`: 参照となる「正常な」活性化パターン
- これが「ベースライン」となる

**具体例:**
```python
# 疑似コード
with torch.no_grad():
    outputs = model(X_f, output_hidden_states=True, output_attentions=True)
    H_f = {
        'layer_0_head_0': attention_output_0_0,
        'layer_0_head_1': attention_output_0_1,
        ...
        'layer_39_head_39': attention_output_39_39,
        'mlp_0': mlp_output_0,
        ...
    }
```

---

#### Line 3: 対照的活性化の計算
```
Compute contrastive activations H_cf ← F(X_cf^(i))
```

**解説:**
- **反事実的入力**をモデルに通し、全層・全コンポーネントの内部活性化を記録
- `H_cf`: 「改変された」活性化パターン
- これが「パッチング元」のデータとなる

**具体例:**
```python
# 疑似コード
with torch.no_grad():
    outputs_cf = model(X_cf, output_hidden_states=True, output_attentions=True)
    H_cf = {
        'layer_0_head_0': attention_output_cf_0_0,
        'layer_0_head_1': attention_output_cf_0_1,
        ...
    }
```

---

#### Line 4: コンポーネントごとのループ開始
```
for each component c^(j) ∈ C do
```

**解説:**
- モデル内の各コンポーネントを順番にテスト
- 例: Qwen3-14Bの場合
  - 40層 × 40ヘッド = 1,600アテンションヘッド
  - 40層のMLPブロック = 40 MLPs
  - 合計約1,640コンポーネント

---

#### Lines 5-6: ハイブリッド活性化マップの作成
```
Create hybrid activation map H̃_f where:
H̃_f ← {
    H_cf^k  if k = c^(j)   # このコンポーネントだけ反事実的活性化
    H_f^k   otherwise       # 他は全て事実的活性化
}
```

**解説:**
- **これがPath Patchingの核心部分**
- 1つのコンポーネント`c^(j)`だけを反事実的活性化`H_cf`に置き換え
- 他のコンポーネントは全て正常な活性化`H_f`を維持
- 「もしこのコンポーネントだけが異なる入力を見たらどうなるか」を実験

**視覚的イメージ:**
```
正常な処理フロー:
X_f → [Head 0] → [Head 1] → [Head 2] → ... → 出力
        ↓H_f      ↓H_f       ↓H_f

パッチング（Head 1のみテスト）:
X_f → [Head 0] → [Head 1] → [Head 2] → ... → 出力
        ↓H_f      ↓H_cf!     ↓H_f
                  ^^^^
                  ここだけ置き換え
```

**具体例（Phase2実装）:**
```python
# medical_path_patching/Phase2_path_patching/path_patching_medical.py 参照
def patch_head_hook(module, input, output):
    """特定のヘッドの出力を置き換えるフック"""
    if should_patch_this_head:
        # H_cf から該当ヘッドの活性化を取得して置き換え
        output[:, :, head_idx, :] = counterfactual_activation
    return output
```

---

#### Line 7: 元のロジットの計算
```
Obtain original logit y_f ← F(X_f; H_f)
```

**解説:**
- 事実的入力`X_f`を、全て正常な活性化`H_f`で処理した場合の出力ロジット
- これが「ベースライン」の予測

**具体例:**
```python
# 翻訳タスク
y_f = model.lm_head(H_f[-1])  # 最終層の出力
# y_f["Hello"] = 8.5  ← "Hello"トークンのロジット

# 医療QAタスク（Phase2実装）
logits = model(X_f).logits
y_f = logits[0, -1, answer_token_id]  # 正解トークン "d" のロジット
```

---

#### Line 8: パッチング後のロジットの計算
```
Obtain patched logit ỹ_f ← F(X_f; H̃_f)
```

**解説:**
- 事実的入力`X_f`を、**1つだけパッチングした**活性化`H̃_f`で処理した場合の出力ロジット
- 「このコンポーネントが異なる活性化を持ったら、出力はどう変わるか」を測定

**具体例:**
```python
# パッチング後のモデル実行
with patched_forward_hooks:  # H̃_f を使用
    logits_patched = model(X_f).logits
    ỹ_f = logits_patched[0, -1, answer_token_id]
```

---

#### Line 9: パッチング効果の計算
```
Calculate patched effect: δ_j^(i) ← (ỹ_f - y_f) / (y_f + ε)
```

**解説:**
- **このコンポーネントの因果的影響**を定量化
- 分子: パッチングによるロジットの変化量
- 分母: 元のロジット値で正規化（百分率変化率）
- `ε`: ゼロ除算を防ぐ微小値（例: 1e-10）

**解釈:**
- **δ > 0**: パッチングで出力が増加 → このコンポーネントは元々出力を抑制していた
- **δ < 0**: パッチングで出力が減少 → このコンポーネントは元々出力を促進していた
- **δ ≈ 0**: パッチングの影響なし → このコンポーネントは重要でない

**具体例:**
```python
# 元の出力: y_f = 8.5
# パッチング後: ỹ_f = 8.7
δ = (8.7 - 8.5) / (8.5 + 1e-10) = 0.2 / 8.5 ≈ 0.0235 = 2.35%

# 解釈: このヘッドをパッチングすると出力が2.35%増加
# → 元々このヘッドは出力を2.35%抑制していた
```

**医療QAでの応用（Phase2実装）:**
```python
# Phase2では「ロジット差分」を使用
default_logit_diff = correct_logit - incorrect_logit  # 正解と誤答の差
patched_logit_diff = correct_logit_patched - incorrect_logit_patched

impact = ((patched_logit_diff - default_logit_diff) / default_logit_diff) * 100
```

---

#### Line 10-11: ループ終了
```
end for  # コンポーネントループ終了
end for  # データペアループ終了
```

**結果:**
- この時点で、各データペア`i`の各コンポーネント`j`について `δ_j^(i)` が計算済み
- データ構造: `δ[i][j]` = データペア`i`でのコンポーネント`j`の影響度

---

### フェーズ2: データセット全体での集約（Lines 12-14）

#### Lines 12-14: 重要度スコアの集約
```
for each importance score δ_i ∈ Δ do
    Aggregate across dataset: δ_i ← (1/|D|) Σ_{j=1}^{|D|} δ_i^(j)
end for
```

**解説:**
- 各コンポーネント`i`について、全データペアでの影響度を平均化
- `|D|`: データセットのサンプル数
- これにより、**データセット全体で一貫して重要なコンポーネント**を特定

**具体例:**
```python
# 疑似コード
importance_scores = {}
for component_j in all_components:
    # 全データペアでの影響度を集める
    effects = [δ[i][j] for i in range(len(dataset))]
    # 平均を計算
    importance_scores[component_j] = np.mean(effects)

# 結果例:
# importance_scores = {
#     'layer_38_head_37': 0.0156,  # 1.56%の平均影響度
#     'layer_33_head_3': 0.0089,
#     'layer_0_head_0': 0.0002,    # ほぼ影響なし
#     ...
# }
```

---

#### Line 15: 結果を返す
```
return Node importance scores Δ
```

**解説:**
- 各コンポーネントの重要度スコア `Δ = {δ_1, δ_2, ..., δ_m}` を返す
- これを使って重要なコンポーネントを選別

**閾値適用（Zhang et al.）:**
```python
# >1.0%の変化を示すコンポーネントを「重要」とする
crucial_components = [c for c, score in importance_scores.items()
                      if abs(score) > 0.01]
```

**閾値適用（Medical Path Patching Phase5_v5）:**
```python
# 正のimpact（>0）を示すヘッドのみを選択
positive_impact_heads = [c for c, score in importance_scores.items()
                         if score > 0]
# 結果: 321ヘッド中24ヘッドが正のimpact
```

---

## アルゴリズムの流れ図

```
入力データ準備
└─ D = {(X_f^1, X_cf^1), (X_f^2, X_cf^2), ..., (X_f^n, X_cf^n)}

↓

[外側ループ: 各データペア i]
│
├─ 1. 事実的活性化を計算: H_f ← F(X_f^i)
│
├─ 2. 反事実的活性化を計算: H_cf ← F(X_cf^i)
│
├─ [内側ループ: 各コンポーネント j]
│  │
│  ├─ 3. ハイブリッド活性化を作成: H̃_f
│  │    └─ コンポーネントj だけ H_cf を使用、他は H_f
│  │
│  ├─ 4. 元のロジット計算: y_f ← F(X_f; H_f)
│  │
│  ├─ 5. パッチング後ロジット計算: ỹ_f ← F(X_f; H̃_f)
│  │
│  └─ 6. 影響度計算: δ_j^i ← (ỹ_f - y_f) / (y_f + ε)
│
└─ 結果: δ^i = {δ_1^i, δ_2^i, ..., δ_m^i}

↓

[集約フェーズ]
各コンポーネントjについて:
    δ_j = (1/n) Σ_{i=1}^{n} δ_j^i

↓

出力: Δ = {δ_1, δ_2, ..., δ_m}
```

---

## 計算量の分析

### 時間計算量
- データペア数: `n = |D|`
- コンポーネント数: `m = |C|`
- 各フォワードパスの計算量: `O(F)`

**合計:** `O(n × m × F)`

**具体例（Phase2実装）:**
- データペア数: 438サンプル
- アテンションヘッド数: 40層 × 40ヘッド = 1,600
- 合計フォワードパス: 438 × 1,600 = **700,800回**

これが、Phase2の計算に長時間かかる理由です。

### メモリ計算量
- 各サンプルで `H_f` と `H_cf` を保存: `O(n × 層数 × 隠れ次元)`
- Phase2では、計算とメモリのトレードオフのため、サンプルごとに順次処理

---

## Medical Path Patchingでの実装対応

### Phase2実装との対応

| Algorithm 1 | Phase2実装 (path_patching_medical.py) |
|------------|--------------------------------------|
| `X_f` | 正解を持つ質問文 |
| `X_cf` | 全選択肢を "incorrect" にした質問文 |
| `H_f` | `cache_default` - 正常時の活性化 |
| `H_cf` | `cache_corrupted` - 改変時の活性化 |
| `H̃_f` | PyTorch hookで1ヘッドだけ置き換え |
| `y_f` | `default_logit_diff` (正解-誤答) |
| `ỹ_f` | `cur_logit_diff` (パッチング後の正解-誤答) |
| `δ_j^i` | `impact[layer][head][sample]` |
| 集約 | `results[layer][head] += mean(impacts)` |

### Phase2のコード（Lines 189-193）

```python
# Line 9 に相当
results[source_layer][source_head_idx] += (
    (cur_logit_diff - default_logit_diff) / default_logit_diff
).mean(dim=0)

# Line 13-14 に相当（最後に百分率化）
results *= 100
```

---

## Zhang et al. とMedical Path Patchingの違い

| 観点 | Zhang et al. (2025) | Medical Path Patching |
|------|---------------------|----------------------|
| **反事実的入力** | 言語方向変更（En→Fr） | 全選択肢をincorrectに |
| **測定対象** | 単一ロジット値 | ロジット差分（正解-誤答） |
| **集約方法** | データセット平均 | サンプル平均→百分率化 |
| **閾値** | >1.0% | >0%（正のimpact） |
| **選択率** | <5% | 7.5%（24/321ヘッド） |

---

## まとめ

### Algorithm 1の本質

**Path Patching = 「外科的介入」による因果性測定**

1. **正常状態**と**改変状態**の両方でモデルを実行
2. **1つのコンポーネントだけ**を改変状態からの活性化に置き換え
3. **出力の変化**を測定することで、そのコンポーネントの因果的影響を定量化
4. **全データで平均化**して、一貫して重要なコンポーネントを特定

### 重要な洞察

- **スパース性**: わずかなコンポーネントのみが重要（<5-10%）
- **因果性**: 相関ではなく因果的影響を測定
- **解釈可能性**: どのコンポーネントがどう影響しているかを定量化

### Medical Path Patchingへの応用

Phase2-Phase3-Phase5の流れ：
1. **Phase2**: Algorithm 1を実装し、全1,600ヘッドのimpactを計算
2. **Phase3**: Impactが高いヘッドの注意パターンを分析して機能分類
3. **Phase5**: 正のimpactを持つ24ヘッドのみを選択的にファインチューニング

これにより、翻訳タスクの研究を医療QAタスクに適応し、効率的なモデル改善を実現しています。

---

**参考実装:**
- `/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase2_path_patching/path_patching_medical.py`
- `/home/Competition2025/P08/P08U023/model_analyze/sycophancy-interpretability/path_patching/path_patching_hf.py`
