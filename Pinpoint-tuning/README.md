# Pinpoint Tuning

医療LLMの注意ヘッド分析に基づく選択的ファインチューニングフレームワーク。

翻訳論文の手法（Source / Indicator / Positional Heads）を医療QAタスクに適用し、
重要な注意ヘッドのみを対象とした効率的な LoRA fine-tuning を実現する。

## パイプライン概要

```
Phase1: データ準備        → 医療用語アノテーション
Phase2: Path Patching     → 注意ヘッドの因果的重要度測定
Phase3: Attention Analysis → ヘッド分類（Medical Term / Indicator / Reasoning Flow）
Phase4: Visualization     → ヒートマップ・統計レポート生成
Phase5: Pinpoint Tuning   → 選定ヘッドのみ LoRA fine-tuning
```

## ディレクトリ構造

```
Pinpoint-tuning/
├── Qwen14B/              # Qwen3-14B (Dense, 40層×40ヘッド)
├── Qwen30B-A3B/          # Qwen3-30B-A3B (MoE, 48層×32ヘッド, 128 experts top-8)
├── shared/               # モデル間共通コード（Phase2/3/5）
├── docs/
│   ├── architecture/     # 全体アーキテクチャ解説
│   └── plans/            # 未実装モデルの計画書（GLM-5, Qwen3-97B-A17B）
└── README.md             # このファイル
```

## 対象モデル

| モデル | アーキテクチャ | レイヤー | ヘッド | 状態 |
|--------|---------------|----------|--------|------|
| Qwen3-14B | Dense | 40 | 40 | 実装済み |
| Qwen3-30B-A3B | MoE (128 experts, top-8) | 48 | 32 | 実装済み |
| GLM-5 | — | — | — | 計画段階 |
| Qwen3-97B-A17B | — | — | — | 計画段階 |

## shared/ — 共通モジュール

モデル間で共通化されたコードを格納。各モデルディレクトリからは re-export ラッパー経由でインポートされるため、既存の import パスは維持される。

| モジュール | 内容 |
|------------|------|
| `shared/phase2/dataset.py` | Path Patching 用データセットクラス |
| `shared/phase2/utils.py` | Path Patching ユーティリティ |
| `shared/phase3/attention_extractor.py` | 注意パターン抽出（Dense/MoE・GQA自動検出） |
| `shared/phase3/head_classifier.py` | ヘッド分類（`indicator_type` で guideline/profile 切替） |
| `shared/phase3/medical_pattern_detector.py` | 医療パターン検出 |
| `shared/phase5/run_spt_medical.py` | 訓練スクリプト（`model_type` で Dense/MoE 切替） |

## Dense vs MoE の違い

共通化されていないファイルは、Dense/MoE 間でロジックが本質的に異なるもの:

- **Phase1** `medical_term_annotator.py` — データソース・アノテーションロジックが異なる
- **Phase2** `hook_functions.py` — MoE版は Router/Expert hook を追加（103行 vs 509行）
- **Phase4** 可視化・レポート — MoE固有の分析セクション（Expert分散、ルーティング分析）
- **utils_common** — Qwen14B: クラスAPI / Qwen30B: 関数API

## 使い方

各モデルの README を参照:
- [Qwen14B/README.md](Qwen14B/README.md)
- [Qwen30B-A3B/README.md](Qwen30B-A3B/README.md)

詳細なアーキテクチャ解説:
- [docs/architecture/README.md](docs/architecture/README.md)
