# 医療LLM Post-Training 分析計画

## 1. 研究概要

医療ドメインにおけるLLMの後段学習（Post-Training）手法の比較研究。
Attention Head Impact分析に基づくPinpoint SFTと、RL系手法（GRPO/GSPO/CHORD）の有効性を検証する。

---

## 2. 実験設定

### 2.1 ベースモデル

| モデル | パラメータ | アーキテクチャ |
|--------|-----------|---------------|
| Qwen3-30B-A3B-Instruct-2507 | 30B (A3B MoE) | qwen3_moe |
| Qwen3-Next-80B-A3B-Instruct | 80B (A3B MoE) | qwen3_next |

### 2.2 学習手法

#### A. Pinpoint SFT（30Bモデル上, Qwen3-30B-A3B-Instruct-2507）

Attention Head Impact分析（PatchPatching）により、医学用語をマスクした際の正解率変動を測定し、
特定のLayer/Headのみを選択的にfine-tuneする手法。

**PatchPatching によるヘッド選択:**

各ヘッドについて2種類のimpactスコアを計測:
- `medical_impact`: 医学用語マスク時の正解率変動
- `reasoning_impact`: 推論プロセスへの影響度

両スコアが**同符号**のヘッドを選択:
| 手法 | ヘッド選択基準 | 意味 |
|------|--------------|------|
| **Positive Pinpoint** | medical_impact > 0 **かつ** reasoning_impact > 0 | 医学用語マスク時に正解率が**上がる**ヘッド → 正答を**妨害**している → 修正対象 |
| **Negative Pinpoint** | medical_impact < 0 **かつ** reasoning_impact < 0 | 医学用語マスク時に正解率が**下がる**ヘッド → 医療タスクと**正の関連**がある → 正常なヘッド |

**PatchPatchingのimpactスコア統計:**

| | Positive (378 heads) | Negative (457 heads) |
|---|---------|----------|
| medical_impact 範囲 | +0.0000 ~ +0.0494 | -0.6238 ~ -0.0000 |
| medical_impact 平均 | +0.0061 | -0.0075 |
| reasoning_impact 範囲 | +0.0000 ~ +0.1291 | -0.1965 ~ -0.0000 |
| reasoning_impact 平均 | +0.0067 | -0.0076 |

> Negativeヘッドの方がimpactの絶対値が大きい（特にLayer9:Head3のmedical_impact=-0.624は突出）。
> 一方Positiveヘッドは個々のimpactが小さく、多数のヘッドの蓄積で影響を及ぼしている。

**Pinpoint設定の詳細:**

| | Positive | Negative |
|---|---------|----------|
| 対象Layer数 | 46/48層 (Layer 43,45 除外) | 48/48層 (全層) |
| 対象Head数 | 378/1536 (24.6%) | 457/1536 (29.8%) |
| Head集中層 (Top5) | L24:18, L22:17, L1:16, L0:15, L26:14 | L45:23, L47:21, L44:20, L16:20, L43:19 |
| 浅い層(0-15)のHead | 多い (Layer0:15, Layer1:16) | 少ない (Layer0:3, Layer1:1) |
| 深い層(38-47)のHead | 少ない (1-6 heads/layer) | 多い (12-23 heads/layer) |
| Expert指定 | なし（コメントアウト） | なし（コメントアウト） |

**重要な構造的差異**: Positiveヘッドは浅い〜中間層に集中、Negativeヘッドは深い層に集中。
これはモデル内部で、浅い層が医学知識の「干渉」、深い層が医学タスクの「推論」を担うことを示唆。

**データソース**: `both_positive_heads.json`, `both_negative_heads.json` (pinpoint_sft/)

- 共通設定: lr=1e-4, max_epochs=1, global_batch_size=32, max_length=4096
- Freeze対象: router, shared_expert, embed_lm_head
- Freeze対象外: MLP, attention（指定head以外も同一layerなら学習対象）
- データ: `sft_igakuqa_v2.jsonl`（過去の医師国家試験 + 正解付き回答, messages形式）

#### B. RL系手法（80Bモデル上, Qwen3-Next-80B-A3B-Instruct）

| 手法 | 特徴 | 主要ハイパーパラメータ |
|------|------|---------------------|
| **GRPO** | 標準GRPO | beta=0.1 (※Wandb記録による), lr=1e-6, reward_weights=[ophtho:1.5, chinese:1.0], steps_per_gen=5 |
| **GSPO** | Importance Sampling付きGRPO | beta=0.1, epsilon=3e-4/4e-4, reward_weights=[ophtho:1.5, chinese:0.3], steps_per_gen=3 |
| **CHORD** | GRPO + SFT hybrid | beta=0.1, mu_peak=0.1, mu_valley=0.01, mu_decay_steps=500, reward_weights=[ophtho:1.5, chinese:1.0], SFTデータ併用 |

> **注記**: GRPOスクリプト(train_grpo.sh)にはbeta=0.05と記載があるが、
> Wandbのモデル名`grpo_reward_chinese_1.0_beta_0.1_5epochs`が実際の実行値。
> スクリプトはresume時に更新された可能性あり。

> **注記**: GSPOとCHORDでreward_weightsのchinese比重が異なる(0.3 vs 1.0)。
> 手法間比較の際、この交絡因子に留意が必要。

- 共通: max_epochs=5, global_batch_size=512, num_generations=16, temperature=0.9
- データ: `igakuqa.jsonl`（GRPO形式, messages + answer）
- CHORDのみ追加: `sft_igakuqa.jsonl`（SFTデータ, chord_sft_per_device_train_batch_size=2）

**報酬関数の実装:**

| 名前 | 実際の動作 | 備考 |
|------|-----------|------|
| `ophtho` | **汎用正解判定**: `[ans]...[/ans]` を抽出し、gold answerと完全一致で1.0, 不一致で0.0 | 名前は眼科由来だが実装は汎用 |
| `chinese` | **中国語ペナルティ**: 「。」区切りの文でひらがな/カタカナを含まない文が2文連続したら0.0 | 日本語での回答を促す |

### 2.3 評価データセット

| データセット | 内容 | 全問題数 | テキストのみ(subset) | 備考 |
|-------------|------|---------|---------------------|------|
| igakuqa | 医師国家試験 | 1,122問 | 807問 | 2023/2024/2025年度 |
| specialist_exam_test_v2 | 専門医試験 | 3,757問 | 2,793問 | 13診療科 |
| specialist_exam_v2 | 専門医試験（年度別） | 6,910問 | 5,055問 | GRPO/CHORDのみ |
| guideline_wrong_filtered | 診療ガイドライン問題 | 5,178問 | N/A | 各診療科, 30Bグループ+80B Base |

**subset_accuracyの定義:**
- `is_subset=True`: 画像・図表を参照しない**テキストのみ**の問題
- `is_subset=False`: 「別冊」「図」「写真」等の画像参照を含む問題（LLMでは画像を見られない）
- `subset_accuracy`: テキストのみ問題での正解率 → **LLMの公正な評価指標**
- guideline_wrong_filteredは全問テキストのみのため、subsetの区分なし (subset_total=0)

### 2.4 学習データと評価データの分離

- 学習データ: 医師国家試験の**過去問**（2022年以前）
- 評価データ: 医師国家試験 **2023/2024/2025年度**
- specialist_exam: 学習データに含まれない（学習は igakuqa のみ）
- guideline_wrong_filtered: 学習データに含まれない
- → **データリークなし**を確認済み

---

## 3. 分析計画

### Phase 1: データ整理・可視化

#### 1-1. 全体比較テーブル作成
- 全7モデル × 3データセットの accuracy / subset_accuracy を1つの表に集約
- **accuracy(全問)** と **subset_accuracy(テキストのみ)** を並記し、画像問題の影響を明示
- **出力**: Markdownテーブル

#### 1-2. 診療科別ヒートマップ
- specialist_exam_test_v2 の13科 × 7モデルの accuracy マトリクス
- ベースモデルからの改善幅（差分）のヒートマップも作成
- **出力**: matplotlib/seaborn による図

#### 1-3. igakuqa 年度別推移
- 2023/2024/2025年の accuracy / subset_accuracy を手法別に折れ線グラフ化
- **出力**: matplotlib による図

### Phase 2: RL系手法の比較（80Bグループ内）

#### 2-1. GRPO vs GSPO vs CHORD 全体比較
- Base 80B → 各手法への改善幅を算出
- igakuqa / specialist 両方での改善量の比較
- 統計検定: **McNemar検定**（同一問題セットでの paired comparison）
- 信頼区間: **Bootstrap CI**（1000回リサンプリングで95%CI算出）

#### 2-2. 診療科別の手法優劣
- 13科それぞれで最も改善が大きい手法を特定
- CHORD のSFTハイブリッドが特定科で効くかの検証
- GSPO の importance sampling が効果を発揮する科の特定

#### 2-3. 交絡因子の整理
以下の要因が手法間比較を困難にしていることを明記:

| 要因 | GRPO | GSPO | CHORD |
|------|------|------|-------|
| beta | 0.1 | 0.1 | 0.1 |
| chinese reward weight | 1.0 | **0.3** | 1.0 |
| steps_per_generation | 5 | **3** | 5 |
| SFTデータ併用 | No | No | **Yes** |

→ 純粋な手法効果の分離は困難。結果は「設定込みでの比較」として記述する。

### Phase 3: Pinpoint SFT の分析（30Bグループ内）

#### 3-1. Positive vs Negative の比較
- 両方ともBase 30Bから**大幅劣化**している事実の分析
  - igakuqa: 86.2% → 80.3%(Pos) / 79.2%(Neg) ... **-5.9pt / -7.0pt**
  - specialist: 60.9% → 44.9%(Pos) / 48.2%(Neg) ... **-16.0pt / -12.7pt**
- 劣化原因の仮説（優先度順）:
  1. **学習率が高すぎる**: lr=1e-4 はRL系(1e-6)の100倍。選択的Headの学習でも、同一Layerの他のパラメータ(MLP等, `pinpoint_freeze_mlp=false`)に波及し、既存知識を破壊した可能性
  2. **学習対象が広すぎる**: 全48層中46-48層、全Headの25-30%が対象。「Pinpoint」の名に反してかなり広範囲
  3. **Catastrophic Forgetting**: 医師国家試験データのみでのSFTにより、specialist領域の知識が上書きされた
  4. **回答フォーマットの変化**: SFTにより`[ans]...[/ans]`の出力形式が変わり、パース失敗が増えた可能性 → model_outputsで要検証

#### 3-2. Positive vs Negative の構造的差異と結果の関係
- **Positiveが浅い層を多く学習** → 汎用的な表現を変更 → igakuqaでは若干マシ(80.3%)
- **Negativeが深い層を多く学習** → タスク固有の推論を変更 → specialistでは若干マシ(48.2%)
- この「層の深さ×タスク種別」の交互作用を定量的に検証

#### 3-3. guideline_wrong_filtered の分析
- 全モデルで10-16%台と低い
- Baseモデル(15.7%)よりPinpoint SFTが低い(10.8%/11.7%) → SFTによる汎化性の低下
- 80B Base(16.3%)も同程度 → モデルサイズでも改善しないタスク
- タスク特性の検討: ガイドライン改訂内容の正誤判断はLLMのcutoff問題か？

### Phase 4: Model Output 詳細分析（スクリプトベース）

#### 4-1. 分析スクリプト作成
`model_outputs/` の個別回答データ（wandb table JSON, 各~10,000問）を使い分析を自動化:

```
outputs/analysis/
├── scripts/
│   ├── parse_model_outputs.py     # wandb JSON → pandas DataFrame 変換
│   ├── compare_models.py          # モデル間の正誤パターン比較
│   ├── error_analysis.py          # 誤答パターンの分類
│   └── visualize.py               # 可視化スクリプト
└── results/
    ├── figures/                    # 生成された図
    └── tables/                    # 生成されたテーブル
```

**model_outputsのカラム構成:**
`question_id, year, problem_text, raw_answer, cleaned_prediction, correct_answer, is_correct, is_subset, question_text, choices, explanation, subject, tags, dataset_name, category, run_id, code_version, problem_version`

#### 4-2. 正誤パターン分析
- **Base正解 → SFT/RL不正解**: 学習により「壊れた」問題の特定
- **Base不正解 → SFT/RL正解**: 学習により「獲得した」問題の特定
- 上記の診療科・問題形式(is_subset)別の分布

#### 4-3. フォーマット遵守率の検証
- `cleaned_prediction` が None（パース失敗）の割合をモデル間比較
- Pinpoint SFTモデルで [ans]...[/ans] 形式の出力率が変化していないか確認
- これが accuracy 低下の一因かを定量化

#### 4-4. Pinpoint SFT 劣化の原因特定
- Base 30B で正解していたがPinpoint SFTで不正解になった問題の傾向分析
- 特に specialist_exam で16pt低下した問題群の特徴抽出
- 回答テキストの質的変化（回答の長さ、推論ステップ数、選択肢の偏り）

#### 4-5. RL系手法の改善パターン
- Base 80B で不正解だった問題のうち、GRPO/GSPO/CHORDで正解になったもの
- 3手法すべてで改善した問題 vs 特定手法でのみ改善した問題
- 改善問題の診療科分布 → 報酬関数の効果が科によって異なるか

### Phase 5: 考察・レポート作成

#### 5-1. 主要な知見のまとめ
1. RL系手法（GRPO/GSPO/CHORD）は80B Base上で安定した改善を達成（+1-2pt）
2. Pinpoint SFTは30B Base上で性能劣化 → lr, 学習範囲, 手法設計の問題
3. 30Bグループ内・80Bグループ内でそれぞれ手法を比較
4. 診療科別の難易度格差と学習手法との関係

#### 5-2. 考察の軸
- **RL vs SFT**: RL系(lr=1e-6, KL制約)がSFT(lr=1e-4, 制約なし)より安定する構造的理由
- **Pinpoint仮説の検証**: Head Impact分析の方向性（浅い層=干渉, 深い層=推論）は発見として有効。ただしSFTの適用設定（lr, freeze範囲）に改善余地
- **MoEモデル固有の課題**: Expert/Head単位の学習制御の粒度と効果
- **医療ドメインの特殊性**: 診療科間の知識転移と干渉、ガイドライン問題の根本的な困難さ

#### 5-3. Limitations
- 30B Pinpoint SFT vs 80B RL は直接比較不可（モデルサイズ・手法・lr全て異なる）
- RL系3手法間の比較にも交絡因子あり（reward weight, steps_per_gen）
- specialist_exam_v2（年度別）はGRPO/CHORDのみ → 全手法での年度別比較は不可
- Pinpoint SFTの最適lr・freeze設定の探索は未実施（1設定のみ）
- SFTのepoch数(1) vs RLのepoch数(5)の差も交絡因子

#### 5-4. 最終レポート構成（MD）
```
ANALYSIS_REPORT.md
├── 1. Introduction
├── 2. Experimental Setup
│   ├── 2.1 Models
│   ├── 2.2 Training Methods
│   ├── 2.3 Evaluation
│   └── 2.4 Metrics (accuracy / subset_accuracy の使い分け)
├── 3. Results
│   ├── 3.1 Overall Performance Comparison
│   ├── 3.2 RL Methods Analysis (80B: GRPO/GSPO/CHORD)
│   ├── 3.3 Pinpoint SFT Analysis (30B: Positive/Negative)
│   ├── 3.4 Format Compliance Analysis
│   └── 3.5 Detailed Error Analysis
├── 4. Discussion
│   ├── 4.1 Why RL Improves and SFT Degrades
│   ├── 4.2 Pinpoint Head Distribution: Shallow vs Deep
│   ├── 4.3 Cross-specialty Knowledge Transfer
│   └── 4.4 The Guideline Problem
├── 5. Limitations & Future Work
└── Appendix: Full Results Tables & Figures
```

---

## 4. 実行順序

| Step | タスク | 依存 | 成果物 |
|------|--------|------|--------|
| 1 | summary_result JSONの整理・全体テーブル作成 | - | テーブル（MD） |
| 2 | 可視化スクリプト作成（ヒートマップ・折れ線） | Step 1 | 図（PNG） |
| 3 | model_outputs パーサー作成 | - | parse_model_outputs.py |
| 4 | フォーマット遵守率検証 | Step 3 | format_compliance結果 |
| 5 | モデル間正誤比較スクリプト | Step 3 | compare_models.py |
| 6 | RL系手法の詳細比較・統計検定・考察 | Step 1,2,5 | レポートSection 3.2 |
| 7 | Pinpoint SFT の詳細分析・考察 | Step 1,2,4,5 | レポートSection 3.3 |
| 8 | 誤答パターン分析 | Step 5 | レポートSection 3.5 |
| 9 | Discussion・全体レポート統合 | Step 6,7,8 | ANALYSIS_REPORT.md |

---

## 5. 注意事項

- 30B と 80B は**別グループとして独立に分析**。グループ間の直接比較は行わない
- specialist_exam_v2（年度別詳細）はGRPOとCHORDのみ存在 → GSPO/Baseとの年度別比較は不可
- guideline_wrong_filtered は 30Bグループ + 80B Base のみ → RL系モデルでの追加評価は将来課題
- model_outputs ファイルは大きい（各~10,000行） → pandas で行単位処理可能なサイズ
- 統計検定は McNemar検定 (paired) + Bootstrap 95% CI を基本とする
