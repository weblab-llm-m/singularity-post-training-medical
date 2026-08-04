# singularity-post-training-medical

本リポジトリは、NEDO（国立研究開発法人新エネルギー・産業技術総合開発機構）「AIの安全性確保に関する研究開発・検証等の推進事業 / 日本語版医療特化型LLMの社会実装に向けた安全性検証・実証」において開発された医療特化型大規模言語モデルに関する、東京大学開発チーム内の **Ramen Team** または **医療PJ Team** による事後学習（Post-Training）手法の実験コード・分析パイプライン一式です。

## 概要

Ramen Teamとしては、以下の3系統のアプローチを実装・実験・評価した。
1. **RL系手法**（GRPO / GSPO / CHORD）：Qwen3-Next-80B-A3B-Instruct に対する強化学習ベースの事後学習
2. **Pinpoint Tuning**：Qwen3-30B-A3B-Instruct-2507 に対し、Path Patching法でAttention Headの影響度を分析し、選択的にFine-tuneする手法

医療PJ Teamsとしては、以下の系統のアプローチを実装・実験・評価した。
1. **事後学習手法**（DPO / SFT / DFT）：Qwen3-235B-A22B-thinkingとQwen3-235B-A22B-Instructに対する強化学習ベースの事後学習

## 学習済みモデル（HuggingFace）

Ramen Team
| モデル | 手法 |
|--------|------|
| [Ramen-GRPO-Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/weblab-LLM-M/Ramen-GRPO-Qwen3-Next-80B-A3B-Instruct) | GRPO |
| [Ramen-GSPO-Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/weblab-LLM-M/Ramen-GSPO-Qwen3-Next-80B-A3B-Instruct) | GSPO |
| [Ramen-CHORD-Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/weblab-LLM-M/Ramen-CHORD-Qwen3-Next-80B-A3B-Instruct) | CHORD |
| [Ramen-PinPointTuning-Positive-Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/weblab-LLM-M/Ramen-PinPointTuning-Positive-Qwen3-30B-A3B-Instruct-2507) | Pinpoint Tuning (Positive) |
| [Ramen-PinPointTuning-Negative-Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/weblab-LLM-M/Ramen-PinPointTuning-Negative-Qwen3-30B-A3B-Instruct-2507) | Pinpoint Tuning (Negative) |

医療PJ Team
| モデル | 手法 |
|--------|------|
| [Weblab-MedLLM-Qwen3-235B-Instruct](https://huggingface.co/weblab-LLM-M/Weblab-MedLLM-Qwen3-235B-Instruct) | DFT |
| [Weblab-MedLLM-Qwen3-235B-Thinking](https://huggingface.co/weblab-LLM-M/Weblab-MedLLM-Qwen3-235B-Thinking) | DFT |
| [medicalpj-Qwen3-235B-A22B-thinking-exp8-Instruction-dft](https://huggingface.co/weblab-LLM-M/medicalpj-Qwen3-235B-A22B-thinking-exp8-Instruction-dft) | DFT |
| [medicalpj-Qwen3-235B-A22B-Instruct-exp8-Instruction-dft](https://huggingface.co/weblab-LLM-M/medicalpj-Qwen3-235B-A22B-Instruct-exp8-Instruction-dft) | DFT |
| [medicalpj-Qwen3-235B-A22B-Instruct-exp10-Base-Instruction-dft](https://huggingface.co/weblab-LLM-M/medicalpj-Qwen3-235B-A22B-Instruct-exp10-Base-Instruction-dft) | DFT |

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

Ramen Team
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

医療PJ Team  
以下に主要ベンチマークの性能を示します。括弧内はベースモデルからの変化幅です。

| モデル | 医師国試 | ガイドライン |
|-----------|------|------|
| Weblab-MedLLM-Qwen3-235B-Thinking | 82.7% (+1.7) | 60.6%(+4.1) |
| Weblab-MedLLM-Qwen3-235B-Instruct | 79.6% (+1.3) | 59.5%(+1.4) |
| medicalpj-Qwen3-235B-A22B-Instruct-exp8-Instruction-dft | 80.0% (+1.7) | 59.4%(+1.3) |
| medicalpj-Qwen3-235B-A22B-thinking-exp8-Instruction-dft | 81.5% (+0.2) | 56.3%(+0.2) |
| medicalpj-Qwen3-235B-A22B-Instruct-exp10-Base-Instruction-dft | 79.3% (+1.0) | 58.7%(+0.6) |



## 免責事項

本リポジトリは研究目的で公開しています。内容の正確性や完全性について保証するものではありません。本リポジトリに含まれるコード・学習済みモデル・データの利用によって生じたいかなる損害についても、作成者は一切の責任を負いません。また、動作に関する不具合や修正依頼に応じることはできません。

本リポジトリで公開している学習済みモデルは医療ドメインを対象とした事後学習の研究成果であり、臨床診断・治療判断など医療現場での実利用を意図したものではありません。

## 謝辞

この成果は、NEDO（国立研究開発法人新エネルギー・産業技術総合開発機構）の 委託業務（JPNP25006）の結果得られたものです。

This paper is based on results obtained from a project, JPNP25006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).

## References

本リポジトリの実装で参考にした論文・フレームワークを以下に示す。コードの直接借用は行っておらず、いずれもアイデア・アルゴリズム・API仕様の参照である。

### Papers（手法）

| 手法 | 論文 |
|------|------|
| GRPO | Shao et al. (2024) *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| GSPO | Zheng et al. (2025) *Group Sequence Policy Optimization*. [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) |
| CHORD | Xie et al. (2025) *On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting*. [arXiv:2508.11408](https://arxiv.org/abs/2508.11408) |
| Path Patching | Wang et al. (2022) *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593) |
| Pinpoint Tuning | Chen et al. (2024) *From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models*. [arXiv:2409.01658](https://arxiv.org/abs/2409.01658) |
| Path Patching メトリクス設計（counterfactual設計・impactメトリクス） | Zhang et al. (2025) *Exploring Translation Mechanism of Large Language Models*. [arXiv:2502.11806](https://arxiv.org/abs/2502.11806) |
| DFT | Wu et al. (2025) *On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification*. [arXiv:2508.05629](https://arxiv.org/abs/2508.05629) |

### Frameworks / Implementations（実装土台・参考実装）

| リポジトリ | ライセンス | 用途 |
|------------|-----------|------|
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | Apache-2.0 | 事後学習フレームワーク本体（GRPO/GSPO/CHORDのplugin実装土台） |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | BSD-3-Clause系 | Megatron backend（公式イメージを未改変で使用） |
| [agentscope-ai/Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) ([examples/mix_chord](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/mix_chord)) | Apache-2.0 | CHORD原著者らによる公式実装。パラメータ命名・デフォルト値の参照元 |