# Pinpoint Tuning — Qwen3-14B (Dense)

産婦人科診療ガイドラインデータに対する Path Patching 分析・Pinpoint Tuning システム。
翻訳メカニズム論文（arXiv:2502.11806）の手法を Qwen3-14B（Dense, 40層×40ヘッド）に適用。

## モデル仕様

| パラメータ | 値 |
|-----------|-----|
| モデル | Qwen3-14B |
| アーキテクチャ | Dense Transformer |
| レイヤー数 | 40 |
| Attention Heads | 40 |
| Hidden Size | 5120 |

## 3種類のヘッド分類

1. **Medical Term Heads** — 医療用語に注目するヘッド
2. **Guideline Indicator Heads** — ガイドライン指示語にスパイク注意するヘッド
3. **Reasoning Flow Heads** — 推論キーワードに均一注意するヘッド

## ディレクトリ構造

```
Qwen14B/
├── Phase1_data_preparation/       # データ準備・アノテーション
├── Phase2_path_patching/          # Path Patching 実行
├── Phase3_attention_analysis/     # 注意パターン解析・ヘッド分類
├── Phase4_visualization/          # ヒートマップ・統計レポート
├── Phase5_pinpoint_tuning/        # Pinpoint Tuning（LoRA fine-tuning）
│   └── experiments/               # 実験バリアント（v1〜v5）
├── configs/                       # 設定ファイル
├── utils_common/                  # Qwen14B固有ユーティリティ（クラスAPI）
└── scripts/                       # 実行スクリプト
```

## 共通コード（shared/）

Phase2/3/5 の一部モジュールはモデル間で共通化され、`../shared/` に配置されている。
各ファイルは re-export ラッパーとして残されており、既存の import パスは維持される。

| ローカルファイル | 共通モジュール |
|----------------|---------------|
| `Phase2_path_patching/dataset.py` | `shared/phase2/dataset.py` |
| `Phase2_path_patching/utils.py` | `shared/phase2/utils.py` |
| `Phase3_attention_analysis/attention_extractor.py` | `shared/phase3/attention_extractor.py` |
| `Phase3_attention_analysis/head_classifier.py` | `shared/phase3/head_classifier.py` |
| `Phase3_attention_analysis/medical_pattern_detector.py` | `shared/phase3/medical_pattern_detector.py` |
| `Phase5_pinpoint_tuning/run_spt_medical.py` | `shared/phase5/run_spt_medical.py` |

詳細: [shared/README.md](../shared/README.md)

## 実装状況

| Phase | 状態 | 内容 |
|-------|------|------|
| Phase 1 | 実装済み | 医療用語アノテーション・Counterfactual生成 |
| Phase 2 | 実装済み | Path Patching による因果的重要度測定 |
| Phase 3 | 実装済み | ヘッド分類（3種） |
| Phase 4 | 実装済み | ヒートマップ・統計分析・レポート |
| Phase 5 | 実装済み | 選定ヘッドの LoRA fine-tuning（v1〜v5） |

## Phase5 実験バリアント

| バリアント | 概要 |
|-----------|------|
| v1_base | ベースライン実装 |
| v2_megatron | Megatron設定 |
| v3_acs_data | ACSデータ使用 |
| v4_acs_8gpu | 8GPU訓練 |
| v5_mk2_8gpu | 正のimpactヘッド24個に絞った改良版 |

## 参考文献

1. Zhang et al. (2025) "Exploring Translation Mechanism of Large Language Models" arXiv:2502.11806
2. Chen et al. (2024) "From Yes-Men to Truth-Tellers" arXiv:2409.01658
