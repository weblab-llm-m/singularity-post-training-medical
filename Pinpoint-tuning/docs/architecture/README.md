# Pinpoint Tuning - アーキテクチャ概要

## 概要

Pinpoint Tuning は Path Patching によって特定された重要な Attention ヘッドに対してのみ
LoRA Fine-tuning を適用する手法。医療QAタスクにおけるモデルの性能を効率的に向上させる。

## パイプライン

```
Phase1: データ準備
  ↓ 医療用語アノテーション + Counterfactual データ生成
Phase2: Path Patching
  ↓ 各 Attention ヘッドの影響度スコアを測定
Phase3: Attention 分析
  ↓ ヘッドを機能別に分類（Medical Term / Profile / Reasoning Flow）
Phase4: 可視化
  ↓ ヒートマップ・統計レポート生成
Phase5: Pinpoint Tuning
  → 選定ヘッドに LoRA を適用して Fine-tuning
```

## 対象モデル

| モデル | アーキテクチャ | ヘッド数 | 状態 |
|--------|--------------|---------|------|
| Qwen3-14B | Dense | 1,200 (40層×40ヘッド) | Phase1-5 完了 |
| Qwen3-30B-A3B | MoE (128 experts, top-8) | 1,536 (48層×32ヘッド) | Phase1-5 完了 |
| GLM-5 | Dense | - | 計画段階 ([plan](../plans/glm5_phase2_path_patching.md)) |
| Qwen3-97B-A17B | MoE | - | 計画段階 ([plan](../plans/qwen397b_phase2_path_patching.md)) |

## ディレクトリ構造

```
Pinpoint-tuning/
├── shared/                  # モデル間で共有するコード
│   ├── phase2/              # Path Patching 共通モジュール
│   └── phase3/              # Attention 分析 共通モジュール
├── Qwen14B/                 # Dense モデル実装
│   ├── Phase1-4/            # 分析パイプライン
│   └── Phase5_pinpoint_tuning/
│       ├── model/dataset/utils/trainer/  # 共通コード
│       └── experiments/     # v1-v5 の実験設定
├── Qwen30B-A3B/             # MoE モデル実装
│   ├── Phase1-4/            # MoE対応分析パイプライン
│   └── Phase5_pinpoint_tuning/  # MoE対応 Pinpoint Tuning
└── docs/                    # ドキュメント・計画
    ├── plans/               # 未実装モデルの計画書
    └── architecture/        # 本ドキュメント
```

## Dense vs MoE の主な違い

| 要素 | Dense (Qwen14B) | MoE (Qwen30B-A3B) |
|------|-----------------|-------------------|
| Phase2 Hook | `hook_functions.py` | `hook_functions_moe.py` (Router/Expert hook 追加) |
| Phase3 分類 | `guideline_indicator_heads` | `profile_indicator_heads` |
| Phase5 Freeze | `freeze_modules` | `freeze_modules_moe` (Expert/Router 凍結対応) |
| Phase5 引数 | - | `--freeze_router`, `--freeze_experts`, `--train_selected_experts` |
