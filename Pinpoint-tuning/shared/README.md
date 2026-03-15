# shared/ — モデル間共通モジュール

Qwen14B（Dense）と Qwen30B-A3B（MoE）で共通のコードを集約したディレクトリ。
各モデルディレクトリには re-export ラッパーが配置されており、既存の import パスは維持される。

## 構造

```
shared/
├── phase2/                          # Path Patching 共通モジュール
│   ├── dataset.py                   # PathPatchingDataset クラス
│   └── utils.py                     # 可視化・ユーティリティ関数
├── phase3/                          # Attention Analysis 共通モジュール
│   ├── attention_extractor.py       # 注意パターン抽出（Dense/MoE・GQA自動検出）
│   ├── head_classifier.py           # ヘッド3種分類（indicator_type切替対応）
│   └── medical_pattern_detector.py  # 医療パターン検出
└── phase5/                          # Pinpoint Tuning 共通モジュール
    └── run_spt_medical.py           # 訓練スクリプト（model_type切替対応）
```

## re-export の仕組み

各モデルディレクトリの元のファイルは以下の形式のラッパーに置き換えられている:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'phase3'))
from attention_extractor import *
```

これにより `from Phase3_attention_analysis.attention_extractor import AttentionExtractor` のような既存の import パスがそのまま動作する。

## モデル固有パラメータ

共通モジュールはハードコードされたデフォルト値を持たず、呼び出し時にモデル固有の値を指定する:

| パラメータ | Qwen14B | Qwen30B-A3B |
|------------|---------|-------------|
| `num_layers` | 40 | 48 |
| `num_heads` | 40 | 32 |
| `num_kv_heads` | — | 4 (GQA) |
| `indicator_type` | `guideline` | `profile` |
| `model_type` | — | `qwen3_moe` |
