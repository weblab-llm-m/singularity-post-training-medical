# singularity-post-training-medical

医療ドメインにおけるLLMの事後学習（Post-Training）手法の実験コード・分析パイプライン一式。
**Ramen Team** による研究プロジェクト。

## 概要

以下の2系統のアプローチを実装・実験・評価した。

1. **RL系手法**（GRPO / GSPO / CHORD）：Qwen3-Next-80B-A3B-Instruct に対する強化学習ベースの事後学習
2. **Pinpoint Tuning**：Qwen3-30B-A3B-Instruct-2507 に対し、Path Patching法でAttention Headの影響度を分析し、選択的にFine-tuneする手法

## 学習済みモデル（HuggingFace）

| モデル | 手法 |
|--------|------|
| [Ramen-GRPO-Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/weblab-LLM-M/Ramen-GRPO-Qwen3-Next-80B-A3B-Instruct) | GRPO |
| [Ramen-GSPO-Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/weblab-LLM-M/Ramen-GSPO-Qwen3-Next-80B-A3B-Instruct) | GSPO |
| [Ramen-CHORD-Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/weblab-LLM-M/Ramen-CHORD-Qwen3-Next-80B-A3B-Instruct) | CHORD |
| [Ramen-PinPointTuning-Positive-Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/weblab-LLM-M/Ramen-PinPointTuning-Positive-Qwen3-30B-A3B-Instruct-2507) | Pinpoint Tuning (Positive) |
| [Ramen-PinPointTuning-Negative-Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/weblab-LLM-M/Ramen-PinPointTuning-Negative-Qwen3-30B-A3B-Instruct-2507) | Pinpoint Tuning (Negative) |

## ディレクトリ構成

```
singularity-post-training-medical/
├── Pinpoint-tuning/              # Attention Head分析・選択的Fine-tuning パイプライン（Phase1-5）
│   ├── Qwen14B/                  #   Dense モデル（Qwen3-14B）向け実装
│   ├── Qwen30B-A3B/              #   MoE モデル（Qwen3-30B-A3B）向け実装
│   └── shared/                   #   Dense/MoE 共通モジュール
├── ms-swift-megatron_v3.8.1/     # ms-swift v3.8.1 事後学習（SFT / DFT / DPO）
├── ms-swift-megatron_v3.9.3/     # ms-swift v3.9.3 事後学習（GRPO / GSPO / CHORD / Pinpoint SFT）
├── dataset_sft/                  # SFT・CHORD用データセット生成パイプライン
├── tools/                        # 補助ツール（データ管理、モデルダウンロード）
├── outputs/                      # 分析結果・評価レポート
│   ├── ANALYSIS_REPORT.md        #   手法比較 統合レポート
│   ├── ANALYSIS_REPORT_PinPointTuning.md
│   └── ANALYSIS_REPORT_RL.md
└── README.md
```

各サブディレクトリにREADMEがあるため、詳細はそちらを参照。

## 主要成果

| モデル | 手法 | 医師国試 accuracy | 専門医試験 accuracy | 統計的有意性 |
|--------|------|:-----------------:|:------------------:|:------------|
| Base 80B | — | 89.5% | 69.0% | — |
| **GRPO** | RL | **90.3% (+0.8)** | **71.1% (+2.1)** | specialist: p=0.0013 ** |
| **GSPO** | RL | **90.2% (+0.7)** | **70.9% (+1.9)** | specialist: p=0.0036 ** |
| **CHORD** | RL | **90.4% (+0.9)** | 70.0% (+1.0) | specialist: n.s. |
| Base 30B | — | 86.2% | 60.9% | — |
| Pinpoint (Pos) | SFT | 80.3% (−5.9) | 44.9% (−16.0) | p<0.000001 *** |
| Pinpoint (Neg) | SFT | 79.2% (−7.0) | 48.2% (−12.7) | p<0.000001 *** |

詳細な分析結果は [outputs/ANALYSIS_REPORT.md](outputs/ANALYSIS_REPORT.md) を参照。