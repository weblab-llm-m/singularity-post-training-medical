# Medical Path Patching for Gynecology QA

産婦人科診療ガイドラインデータに対するPath Patching分析システム

## 概要

このプロジェクトは、翻訳メカニズム論文（arXiv:2502.11806）の手法を応用し、Qwen3-14Bモデルの注意ヘッドを機能的に分類します。

### 3種類のヘッド分類

1. **Medical Term Heads**: 医療用語に注目するヘッド
2. **Guideline Indicator Heads**: ガイドライン指示語にスパイク注意するヘッド
3. **Reasoning Flow Heads**: 推論キーワードに均一注意するヘッド

## ディレクトリ構造

```
medical_path_patching/
├── Phase1_data_preparation/       # データ準備
│   ├── medical_terms_dictionary.json
│   ├── medical_term_annotator.py
│   ├── counterfactual_generator.py
│   └── path_patching_data_builder.py
├── Phase2_path_patching/          # Path Patching実行（要実装）
├── Phase3_attention_analysis/     # 注意パターン解析
│   ├── attention_extractor.py
│   ├── head_classifier.py
│   └── medical_pattern_detector.py
├── Phase4_visualization/          # 可視化
│   ├── heatmap_generator.py
│   ├── statistical_analyzer.py
│   └── report_generator.py
├── Phase5_pinpoint_tuning/        # Pinpoint Tuning
│   └── select_trainable_heads.py
├── configs/                       # 設定ファイル
│   ├── medical_config.yaml
│   └── head_classification_params.yaml
├── utils_common/                  # 共通ユーティリティ
│   ├── tokenizer_utils.py
│   ├── medical_nlp_utils.py
│   └── visualization_helpers.py
└── scripts/                       # 実行スクリプト
    ├── run_phase1.sh
    ├── run_phase3.sh
    └── run_full_pipeline.sh
```

## 実装済みファイル一覧

### Phase 1: データ準備 ✅
- [x] medical_terms_dictionary.json
- [x] medical_term_annotator.py
- [x] counterfactual_generator.py
- [x] path_patching_data_builder.py

### Phase 2: Path Patching ⚠️
- [x] attention_extractor.py（Phase 3に配置）
- [ ] path_patching_medical.py（要実装 - sycophancy-interpretabilityから改変）
- [ ] utils.py（要実装 - sycophancy-interpretabilityから改変）

### Phase 3: 注意分析 ✅
- [x] head_classifier.py
- [x] medical_pattern_detector.py

### Phase 4: 可視化 ✅
- [x] heatmap_generator.py
- [x] statistical_analyzer.py
- [x] report_generator.py

### Phase 5: Pinpoint Tuning ✅
- [x] select_trainable_heads.py

### 共通ユーティリティ ✅
- [x] tokenizer_utils.py
- [x] medical_nlp_utils.py
- [x] visualization_helpers.py

### 設定・実行スクリプト ✅
- [x] medical_config.yaml
- [x] head_classification_params.yaml
- [x] run_phase1.sh
- [x] run_phase3.sh
- [x] run_full_pipeline.sh

## 使用方法

### Phase 1: データ準備（実行可能）

```bash
cd /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching
bash scripts/run_phase1.sh
```

このフェーズは以下を生成します:
- アノテーション済みデータ
- Counterfactualデータ
- Path Patching用データセット

### Phase 2: Path Patching（要実装）

Phase 2を実行するには、以下のファイルが必要です:

```bash
# sycophancy-interpretabilityから必要ファイルをコピー
cp ../sycophancy-interpretability/path_patching/hook_functions.py Phase2_path_patching/
cp ../sycophancy-interpretability/path_patching/dataset.py Phase2_path_patching/

# path_patching_medical.pyを実装（設計書参照）
# utils.pyを改変（設計書参照）
```

### Phase 3: ヘッド分類

```bash
bash scripts/run_phase3.sh
```

### Phase 4 & 5: 可視化とPinpoint Tuning

設計書の「実行手順」セクションを参照してください。

## 依存パッケージ

```bash
pip install torch transformers pandas numpy scipy matplotlib plotly pyyaml seaborn
```

## 設定ファイルのカスタマイズ

### ヘッド分類基準の調整

`configs/head_classification_params.yaml`で閾値を調整できます:

```yaml
classification_criteria:
  medical_term:
    threshold: 0.30  # 医療用語への平均注意
  guideline_indicator:
    spike_threshold: 0.70  # スパイク閾値
    spike_ratio: 5.0       # スパイク比率
  reasoning_flow:
    uniformity_threshold: 0.10  # 均一性
    attention_mean_threshold: 0.40  # 平均注意
```

## トラブルシューティング

### OOMエラー
```bash
# batch_sizeを削減
# configs/medical_config.yaml内:
path_patching:
  batch_size: 1  # 2 → 1に削減
```

### 医療用語が検出されない
```json
// Phase1_data_preparation/medical_terms_dictionary.jsonに
// 実際のデータに出現する用語を追加
```

### ヘッド分類が偏る
```yaml
# configs/head_classification_params.yamlの閾値を緩和
classification_criteria:
  medical_term:
    threshold: 0.20  # 0.30 → 0.20に緩和
```

## 参考文献

1. Zhang et al. (2025) "Exploring Translation Mechanism of Large Language Models" arXiv:2502.11806
2. Chen et al. (2024) "From Yes-Men to Truth-Tellers" arXiv:2409.01658
3. 産婦人科診療ガイドライン婦人科外来編2023

## ライセンス

このプロジェクトは研究目的で作成されました。

## 著者

Claude Code (2025-10-23)

## 実装状況

**Phase 1**: ✅ 完全実装・テスト可能
**Phase 2**: ⚠️ 一部実装（path_patching_medical.py等が必要）
**Phase 3**: ✅ 完全実装
**Phase 4**: ✅ 完全実装
**Phase 5**: ✅ 完全実装

**次のステップ**: Phase 2の実装完了後、全パイプラインが実行可能になります。
