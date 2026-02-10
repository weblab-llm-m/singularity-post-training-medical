# Qwen3-14B から Qwen3-30B-A3B-Instruct-2507 への移行計画

## 概要

本ドキュメントは、`Pinpoint-tuning/Qwen14B` のモデル分析・学習スクリプトを `Qwen/Qwen3-30B-A3B-Instruct-2507` 向けに作り変える計画をまとめたものです。

## 1. モデルアーキテクチャの比較

### Qwen3-14B（Dense モデル）

| パラメータ | 値 |
|-----------|-----|
| 総パラメータ数 | 14B |
| レイヤー数 | 40 |
| Attention Heads (Query) | 40 |
| Attention Heads (KV) | 8 (GQA) |
| Head Dimension | 128 |
| Hidden Size | 5120 |
| アーキテクチャ | Dense Transformer |
| model_type | `qwen2` |

### Qwen3-30B-A3B-Instruct-2507（MoE モデル）

| パラメータ | 値 |
|-----------|-----|
| 総パラメータ数 | 30.5B |
| アクティブパラメータ | 3.3B |
| レイヤー数 | 48 |
| Attention Heads (Query) | 32 |
| Attention Heads (KV) | 4 (GQA) |
| Total Experts | 128 |
| Activated Experts | 8 |
| Context Length | 262,144 (最大1M) |
| アーキテクチャ | Mixture-of-Experts |
| model_type | `qwen3_moe` (推定) |

## 2. 主要な変更点

### 2.1 アーキテクチャの違い

1. **MoE構造**: MLP層がエキスパートベースになる
   - 各トークンに対して128個のエキスパートから8個を選択
   - ルーター（Router）がエキスパート選択を制御
   - Shared Expertsの有無を確認する必要あり

2. **Attention構造の違い**:
   - Query Heads: 40 → 32
   - KV Heads: 8 → 4
   - num_key_value_groups: 5 → 8

3. **レイヤー数の増加**: 40 → 48

### 2.2 Path Patching への影響

Path Patching では以下の修正が必要:

1. **モジュール名の変更**:
   - Dense: `model.layers.{i}.mlp`
   - MoE: `model.layers.{i}.mlp.experts.{j}` または `model.layers.{i}.mlp.shared_expert`

2. **ヘッド分析の対象**:
   - Attention Heads の数が変わるため、分析対象が変更
   - 全ヘッド数: 40×40 = 1600 → 48×32 = 1536

### 2.3 Pinpoint Tuning への影響

1. **freeze_modules の修正**:
   - MoE の mlp.experts への対応が必要
   - ルーター層の freeze/unfreeze 戦略の検討

2. **GQA の比率変更**:
   - `num_key_value_groups` の計算ロジックを更新

## 3. ディレクトリ構造（予定）

```
Qwen30B-A3B/
├── README.md                           # プロジェクト概要
├── migration_plan.md                   # 本計画書
├── ARCHITECTURE_NOTES.md               # MoEアーキテクチャ詳細メモ
│
├── configs/                            # 設定ファイル
│   ├── medical_config.yaml             # モデル設定（MoE対応版）
│   └── head_classification_params.yaml # ヘッド分類パラメータ
│
├── Phase1_data_preparation/            # データ準備（Qwen14Bから流用可能）
│   ├── medical_terms_dictionary.json
│   ├── medical_term_annotator.py
│   ├── counterfactual_generator.py
│   └── path_patching_data_builder.py
│
├── Phase2_path_patching/               # Path Patching（MoE対応必要）
│   ├── configs/
│   │   └── qwen3_moe.json              # MoE用モジュール名設定
│   ├── dataset.py
│   ├── hook_functions.py
│   ├── hook_functions_moe.py           # 【新規】MoE用フック関数
│   ├── path_patching_medical.py        # 修正版
│   └── utils.py
│
├── Phase3_attention_analysis/          # 注意分析（修正必要）
│   ├── attention_extractor.py          # ヘッド数の変更対応
│   ├── head_classifier.py
│   └── medical_pattern_detector.py
│
├── Phase4_visualization/               # 可視化（ほぼ流用可能）
│   ├── heatmap_generator.py
│   ├── statistical_analyzer.py
│   └── report_generator.py
│
├── Phase5_pinpoint_tuning/             # Pinpoint Tuning（MoE対応必要）
│   ├── model/
│   │   ├── model_hf.py
│   │   ├── model_peft.py
│   │   └── __init__.py
│   ├── trainer/
│   │   ├── trainer_hf.py
│   │   ├── callbacks.py
│   │   └── __init__.py
│   ├── dataset/
│   │   ├── dataset_medical.py
│   │   ├── dataset_json.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── arguments.py                # MoEパラメータ追加
│   │   ├── utils_spt.py                # freeze_modules のMoE対応
│   │   └── __init__.py
│   ├── configs/
│   │   └── lora_config.json
│   ├── run_spt_medical.py
│   ├── run_spt_acs_8gpu.sh             # 修正版（モデルパス変更等）
│   ├── evaluate_model_fixed.py
│   └── trainable_heads.json            # 分析結果から生成
│
├── utils_common/                       # 共通ユーティリティ
│   ├── tokenizer_utils.py
│   ├── medical_nlp_utils.py
│   └── visualization_helpers.py
│
└── scripts/                            # 実行スクリプト
    ├── run_phase1.sh
    ├── run_phase2.sh
    ├── run_phase3.sh
    ├── run_phase5_training.sh
    └── run_full_pipeline.sh
```

## 4. 実装タスク一覧

### Phase 1: 設定・準備（優先度: 高）

| タスク | 説明 | 難易度 |
|--------|------|--------|
| 4.1.1 | `configs/medical_config.yaml` の作成（モデルパス、レイヤー数等を更新） | 低 |
| 4.1.2 | `configs/head_classification_params.yaml` のコピー・調整 | 低 |
| 4.1.3 | Phase1 スクリプトのコピー（変更不要） | 低 |

### Phase 2: Path Patching のMoE対応（優先度: 高）

| タスク | 説明 | 難易度 |
|--------|------|--------|
| 4.2.1 | `configs/qwen3_moe.json` の作成（モジュール名定義） | 中 |
| 4.2.2 | `path_patching_medical.py` のmodel_type判定追加 | 中 |
| 4.2.3 | MoEレイヤーのモジュール構造調査 | 高 |
| 4.2.4 | `hook_functions_moe.py` の作成（MoE用フック） | 高 |

### Phase 3: 注意分析の調整（優先度: 中）

| タスク | 説明 | 難易度 |
|--------|------|--------|
| 4.3.1 | `attention_extractor.py` のヘッド数対応 | 中 |
| 4.3.2 | `head_classifier.py` のパラメータ調整 | 低 |

### Phase 4: 可視化（優先度: 低）

| タスク | 説明 | 難易度 |
|--------|------|--------|
| 4.4.1 | `heatmap_generator.py` のコピー・調整 | 低 |
| 4.4.2 | レポートテンプレートの調整 | 低 |

### Phase 5: Pinpoint Tuning のMoE対応（優先度: 最高）

| タスク | 説明 | 難易度 |
|--------|------|--------|
| 4.5.1 | `utils/utils_spt.py` の `freeze_modules` MoE対応 | 高 |
| 4.5.2 | `utils/arguments.py` にMoE関連引数追加 | 中 |
| 4.5.3 | `run_spt_acs_8gpu.sh` のモデルパス・パラメータ修正 | 低 |
| 4.5.4 | MoEのエキスパート選択的freeze戦略の設計 | 高 |
| 4.5.5 | メモリ最適化（30Bモデル対応） | 高 |

## 5. 技術的詳細

### 5.1 freeze_modules のMoE対応方針

```python
def freeze_modules_moe(model, path_patching_path, precise_level, train_topk, train_kv, train_experts=False):
    """
    MoEモデル用のfreeze関数

    追加パラメータ:
    - train_experts: エキスパート層も学習するか
    """
    # 1. 全パラメータをfreeze
    for param in model.parameters():
        param.requires_grad = False

    # 2. Attentionヘッドのunfreeze（既存ロジック適用）
    # ... (Qwen14B版と同様)

    # 3. MoE固有の処理
    if train_experts:
        for layer in selected_layers:
            # エキスパート層のunfreeze
            for name, param in layer.mlp.named_parameters():
                if 'experts' in name:
                    param.requires_grad = True
            # ルーター層は基本的にfreeze（安定性のため）
```

### 5.2 GQA比率の計算

```python
# Qwen3-14B
num_key_value_groups = 40 // 8 = 5  # 5つのQueryヘッドが1つのKVヘッドを共有

# Qwen3-30B-A3B
num_key_value_groups = 32 // 4 = 8  # 8つのQueryヘッドが1つのKVヘッドを共有
```

### 5.3 メモリ要件の見積もり

| 設定 | Qwen3-14B | Qwen3-30B-A3B |
|------|-----------|---------------|
| BF16推論 | ~28GB | ~62GB（全パラメータ）, ~7GB（アクティブ）|
| 学習（バッチ1） | ~56GB | ~120GB+ |
| 推奨GPU | 8× A100 40GB | 8× A100 80GB または H100 |

**注意**: MoEモデルは全エキスパートをメモリにロードする必要があるため、アクティブパラメータ数（3.3B）に比してメモリ使用量が大きい。

### 5.4 学習スクリプトの主要変更点

```bash
# run_spt_acs_8gpu.sh の変更例

# モデルパス変更
MODEL_PATH="/path/to/Qwen3-30B-A3B-Instruct-2507"

# パラメータ調整
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION=32  # メモリ制約のため増加
MAX_SEQ_LENGTH=1024       # メモリ制約のため削減検討

# MoE固有オプション（将来的に追加）
--train_experts false
--freeze_router true
```

## 6. リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| MoEのルーター層への影響 | 高 | ルーター層はfreezeを基本とする |
| メモリ不足 | 高 | Gradient Checkpointing、バッチサイズ削減、DeepSpeed ZeRO-3の検討 |
| Attention構造の違い | 中 | GQA比率を正確に計算してフック関数を修正 |
| エキスパート選択の不均衡 | 中 | Load Balancing Lossの監視 |
| transformersバージョン互換性 | 中 | 最新版transformersを使用（Qwen3-MoE対応版） |

## 7. 検証計画

### 7.1 単体テスト

1. モデルロードの確認
2. Attentionヘッド数の確認
3. MoEレイヤー構造の確認
4. freeze_modulesの動作確認

### 7.2 統合テスト

1. Path Patching の少量データ実行
2. ヘッド分類の動作確認
3. SPT学習の短時間実行
4. 評価スクリプトの動作確認

### 7.3 性能テスト

1. 産婦人科データでの精度評価
2. 学習前後のベースライン比較
3. メモリ使用量の監視

## 8. スケジュール目安

| フェーズ | 内容 | 目安 |
|----------|------|------|
| Week 1 | 設定ファイル作成、MoEアーキテクチャ調査 | |
| Week 2 | Phase2 Path Patchingの修正・テスト | |
| Week 3 | Phase5 freeze_modulesの修正・テスト | |
| Week 4 | 統合テスト、性能評価 | |

## 9. 参考資料

- [Qwen3-30B-A3B-Instruct-2507 - Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Qwen3-30B-A3B アーキテクチャ詳細](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- 翻訳メカニズム論文 (arXiv:2502.11806)
- Chen et al. (2024) "From Yes-Men to Truth-Tellers" arXiv:2409.01658

## 10. 次のステップ

### 完了済み (2026-02-03)

1. ✅ **モデル構造確認**: Qwen3-30B-A3B のモデル構造を分析
   - `analyze_model_structure.py` 作成・実行
   - `ARCHITECTURE_NOTES.md` にアーキテクチャ詳細を記録

2. ✅ **Phase1準備**: configs のコピーと修正
   - `configs/medical_config.yaml` (MoE対応版)
   - `configs/head_classification_params.yaml` (パラメータ調整版)
   - `configs/qwen3_moe.json` (モジュール名定義)

3. ✅ **MoE調査**: モジュール名を特定
   - model.safetensors.index.json から構造を解析
   - Router: `model.layers.{i}.mlp.gate`
   - Experts: `model.layers.{i}.mlp.experts.{j}.{gate_proj,up_proj,down_proj}`
   - Shared Expert: なし

4. ✅ **freeze_modules**: MoE対応版の設計・実装
   - `Phase5_pinpoint_tuning/utils/utils_spt.py` 作成
   - `freeze_modules_moe()` 関数実装
   - Router/Expert のfreeze制御オプション追加

5. ✅ **Phase2 Path Patching**: MoE対応版の実装
   - `Phase2_path_patching/hook_functions_moe.py` 作成
     - MoE用フック関数（Router, Expert対応）
     - `MoEPathPatchingHooks` クラス
   - `Phase2_path_patching/path_patching_medical.py` 作成
     - model_type自動判定（qwen2, qwen3_moe対応）
     - MoE固有の設定読み込み
   - `Phase2_path_patching/configs/qwen3_moe.json` 作成
   - 共通ファイル（dataset.py, utils.py）コピー

6. ✅ **Phase1 データ準備**: Qwen14Bからスクリプトをコピー
   - `Phase1_data_preparation/` ディレクトリ作成
   - スクリプトファイルコピー:
     - `medical_term_annotator.py`
     - `counterfactual_generator.py`
     - `path_patching_data_builder.py`
     - `generate_medical_dict_improved.py`
     - `merge_dictionaries.py`
     - `utils_common/` (tokenizer_utils, medical_nlp_utils)
   - データファイル（シンボリックリンク）:
     - `medical_terms_dictionary.json`
     - `annotated_medical_data_full.jsonl` (1761サンプル)
     - `path_patching_strategy2.jsonl` (1404サンプル)
     - `counterfactual_strategy2_full.jsonl` (1404サンプル)

7. ✅ **Phase3 Attention分析**: MoE対応版の実装
   - `Phase3_attention_analysis/attention_extractor.py` 作成
     - 48レイヤー、32ヘッド対応
     - GQA情報表示
   - `Phase3_attention_analysis/head_classifier.py` 作成
     - 1536ヘッド分類対応
   - `Phase3_attention_analysis/medical_pattern_detector.py` コピー
   - `Phase3_attention_analysis/generate_trainable_heads.py` 新規作成
     - Path PatchingとHead Classification結果からSPT用ヘッド選択

8. ✅ **Phase4 可視化**: MoE対応版の実装
   - `Phase4_visualization/heatmap_generator.py` 作成
     - 48×32ヒートマップ生成
     - レイヤー分布プロット追加
   - `Phase4_visualization/statistical_analyzer.py` 作成
     - レイヤーグループ分析（Early/Middle/Late）
   - `Phase4_visualization/report_generator.py` 作成
     - MoE情報を含むMarkdownレポート
   - `utils_common/visualization_helpers.py` コピー

9. ✅ **検証**: 実際にモデルをロードして動作確認
   - `venv/` 環境作成（torch, transformers, accelerate, pyyaml）
   - `verify_model_structure.py` 作成・実行
   - 全テスト合格:
     - モデル設定検証 ✓
     - トークナイザー検証 ✓
     - モデル構造検証（メタデバイス） ✓
     - モジュール名アクセス検証 ✓
     - hook_functions_moe インポート ✓
     - freeze_modules_moe インポート ✓

10. ✅ **Phase5 Pinpoint Tuning スクリプトの完成**
    - `Phase5_pinpoint_tuning/` ディレクトリ構造作成
    - モジュール作成:
      - `model/__init__.py`, `model_hf.py`, `model_peft.py`
      - `dataset/__init__.py`, `dataset_medical.py`
      - `trainer/__init__.py`, `trainer_hf.py`, `callbacks.py`
      - `utils/__init__.py` (更新), `arguments.py` (MoE引数追加)
    - メインスクリプト:
      - `run_spt_medical.py` (MoE対応メイン学習スクリプト)
      - `run_spt_moe.sh` (シングルGPU用)
      - `run_spt_moe_8gpu.sh` (8 GPU DeepSpeed ZeRO-3用)
      - `evaluate_model_fixed.py` (MoE対応評価スクリプト)

### 次のタスク

11. **実際のPath Patching実行**
    - 少量データでのPath Patching実行
    - `trainable_heads.json` の生成

12. **統合テスト**
    - SPT学習の短時間実行
    - 評価の実行

---

**作成日**: 2026-02-03
**更新履歴**:
- 初版作成
- 2026-02-03: 次のステップ 1-6 完了
- 2026-02-03: 次のステップ 7-9 完了（Phase3, Phase4, 検証）
- 2026-02-03: 次のステップ 10 完了（Phase5 Pinpoint Tuningスクリプト）
