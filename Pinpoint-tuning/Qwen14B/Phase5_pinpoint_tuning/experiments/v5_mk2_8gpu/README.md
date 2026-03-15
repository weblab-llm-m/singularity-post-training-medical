# Phase5_v5: Pinpoint Tuning MK2 (8 GPU) - Positive Impact Heads Only

## Overview

Phase5_v5は、Phase5_v4をベースに作成され、**impact%がプラスのヘッドのみ**を学習対象とするバージョンです。

Phase3の元データ（head_classification_results_438samples.json）を再分析した結果、321ヘッド中、**プラスのimpactを持つのは24ヘッドのみ**であることが判明しました。

## 分析結果（Phase3データの再分析）

### Impact分析
- **Total classified heads**: 321
  - Medical term: 144
  - Guideline: 141
  - Reasoning flow: 36
- **Positive impact heads**: 24 (7.5%)
  - Medical term: 7/144 (4.9%)
  - Guideline: 8/141 (5.7%)
  - Reasoning flow: 9/36 (25.0%)
- **Negative/Zero impact heads**: 297 (92.5%)

### Impact範囲（正のimpactのみ）
- **最小impact**: 0.206514 (Layer 0, Head 14 - medical_term)
- **最大impact**: 0.440244 (Layer 38, Head 37 - reasoning_flow)
- **平均impact**: 0.256833

### Priority分布
- **High (>0.4)**: 1 head
- **Medium (>0.2)**: 23 heads
- **Low (≤0.2)**: 0 heads

### 全24ヘッド一覧（impact降順）

| Rank | Layer | Head | Impact | Type | Priority |
|------|-------|------|--------|------|----------|
| 1 | 38 | 37 | 0.440244 | reasoning_flow | high |
| 2 | 33 | 3 | 0.356804 | reasoning_flow | medium |
| 3 | 29 | 7 | 0.300245 | reasoning_flow | medium |
| 4 | 28 | 26 | 0.294892 | reasoning_flow | medium |
| 5 | 16 | 6 | 0.290904 | guideline | medium |
| 6 | 18 | 27 | 0.273556 | guideline | medium |
| 7 | 14 | 30 | 0.266556 | guideline | medium |
| 8 | 33 | 1 | 0.266206 | reasoning_flow | medium |
| 9 | 32 | 22 | 0.262062 | reasoning_flow | medium |
| 10 | 10 | 2 | 0.252759 | medical_term | medium |
| 11 | 29 | 6 | 0.248043 | reasoning_flow | medium |
| 12 | 8 | 19 | 0.245552 | medical_term | medium |
| 13 | 20 | 22 | 0.236838 | guideline | medium |
| 14 | 1 | 0 | 0.235257 | medical_term | medium |
| 15 | 17 | 21 | 0.232929 | guideline | medium |
| 16 | 25 | 1 | 0.231207 | guideline | medium |
| 17 | 11 | 20 | 0.230670 | medical_term | medium |
| 18 | 32 | 0 | 0.228704 | reasoning_flow | medium |
| 19 | 16 | 16 | 0.218442 | guideline | medium |
| 20 | 29 | 27 | 0.214790 | reasoning_flow | medium |
| 21 | 4 | 39 | 0.211004 | medical_term | medium |
| 22 | 8 | 37 | 0.210032 | medical_term | medium |
| 23 | 21 | 8 | 0.209781 | guideline | medium |
| 24 | 0 | 14 | 0.206514 | medical_term | medium |

## Configuration

### Dataset
- **Source**: ACS_data_v1
- **Path**: `/home/Competition2025/P05/shareP05/data/ACS_data_v1`
- **Training samples**: 9,200

### Model
- **Base model**: Qwen3-14B
- **Trainable heads**: 24 (positive impact only)
- **PRECISE_LEVEL**: 3 (qkv_proj + o_proj)

### Training Parameters
- **Learning rate**: 2e-5
- **Batch size**: 1 (per device)
- **Gradient accumulation**: 16 (8 GPUs)
- **Global batch size**: 128
- **Epochs**: 1
- **Max sequence length**: 2048
- **LR scheduler**: cosine with 0.1 warmup
- **Dtype**: bfloat16
- **Gradient checkpointing**: enabled

## Directory Structure

```
Phase5_v5_pinpoint_tuning_mk2_8gpu/
├── trainable_heads_all_321.json         # 学習対象ヘッド (24ヘッド - positive impact only)
├── trainable_heads_positive_impact.json # 同じファイルのコピー
├── filter_positive_impact_heads.py      # フィルタリングスクリプト
├── run_spt_acs_8gpu.sh                  # 8GPU学習スクリプト
├── run_spt_acs_training.sh              # シングルGPU学習スクリプト
├── run_spt_321heads_acs.sh              # 代替学習スクリプト
├── monitor_training.sh                   # 進捗監視スクリプト
├── run_evaluation*.sh                   # 評価スクリプト群
├── evaluate_model_fixed.py              # 評価用Pythonスクリプト
├── run_spt_medical.py                   # トレーニングコード
├── trainer/                              # トレーナーモジュール
├── model/                                # モデルモジュール
├── dataset/                              # データセットモジュール
├── utils/                                # ユーティリティ
├── spt_output/                           # 学習出力ディレクトリ
├── cache/                                # キャッシュディレクトリ
└── README.md                             # このファイル
```

## Usage

### Training (8 GPUs)
```bash
cd /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v5_pinpoint_tuning_mk2_8gpu
nohup bash run_spt_acs_8gpu.sh > training_8gpu_nohup.log 2>&1 &
```

### Training (Single GPU)
```bash
cd /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v5_pinpoint_tuning_mk2_8gpu
nohup bash run_spt_acs_training.sh > training_nohup.log 2>&1 &
```

### Monitor Training
```bash
./monitor_training.sh
# or
tail -f spt_output/training.log
```

### Evaluation
```bash
# Both base and trained models
./run_evaluation.sh

# Trained model only
./run_evaluation_trained_only.sh

# Base model only
./run_evaluation_base_only.sh
```

## Differences from Phase5_v4

| Aspect | Phase5_v4 | Phase5_v5 |
|--------|-----------|-----------|
| Data Source | trainable_heads_all_321.json | Phase3 head_classification_results_438samples.json |
| Head Selection Criteria | All 321 heads | **Positive impact only** |
| Actual Trainable Heads | 321 | **24** |
| Medical term heads | 144 | **7** |
| Guideline heads | 141 | **8** |
| Reasoning flow heads | 36 | **9** |
| Impact Range | 0.206-0.633 (as stored) | **0.207-0.440 (verified positive)** |

## Key Findings

### Previous Assumption (Phase5_v4)
- trainable_heads_all_321.jsonに記録されていたimpact値は、全て**正の値として記録**されていました
- これは**絶対値**または**符号反転された値**であった可能性があります

### Correct Analysis (Phase5_v5)
- **元データ（Phase3）を再確認**した結果：
  - all_head_impactsには**負の値**が多数含まれている
  - 321ヘッド中、**真にプラスのimpactを持つのは24ヘッド（7.5%）のみ**
  - Reasoning flowヘッドは25%がプラス（他カテゴリーより高い）

### Impact
Phase5_v5では、**本当に効果的な24ヘッドのみ**に集中して学習することで：
- 学習パラメータ数が大幅に削減（321→24ヘッド、93%削減）
- より精密なファインチューニングが可能
- 過学習リスクの低減

## Regeneration

フィルタリングスクリプトを再実行する場合：
```bash
python3 filter_positive_impact_heads.py
```

これにより、trainable_heads_positive_impact.jsonが再生成されます。
