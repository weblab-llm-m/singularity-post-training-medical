# Pinpoint Tuning - Qwen3-30B-A3B

Qwen3-30B-A3B-Instruct-2507 (MoEモデル) に対する Pinpoint Tuning パイプライン。

## モデル概要

| パラメータ | 値 |
|-----------|-----|
| model_type | `qwen3_moe` |
| num_hidden_layers | 48 |
| num_attention_heads | 32 (GQA: KV=4) |
| num_experts | 128 (top-8) |
| 全Attentionヘッド数 | 1,536 (48 × 32) |

アーキテクチャの詳細は [ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md) を参照。

## ディレクトリ構造

```
Qwen30B-A3B/
├── Phase1_data_preparation/     # データ準備（医療用語アノテーション、Counterfactual生成）
│   ├── dictionary_260212/       # 拡張辞書セット
│   └── utils_common/            # 共通ユーティリティ
├── Phase2_path_patching/        # Path Patching（Attentionヘッド影響度測定）
│   ├── 1node8gpu/               # 1ノード8GPU版
│   ├── 8node64gpu/              # 8ノード64GPU版
│   ├── 8node64gpu_260212/       # 拡張辞書版
│   └── test_8samples/           # テスト用
├── Phase3_attention_analysis/   # Attention分析（ヘッド機能分類）
├── Phase4_visualization/        # 可視化・統計レポート
├── Phase5_pinpoint_tuning/      # MoE対応 Pinpoint Tuning
└── configs/                     # モデル設定
```

## 実行順序

1. **Phase1**: `medical_term_annotator.py` → `path_patching_data_builder.py`
2. **Phase2**: `path_patching_medical.py`（1node8gpu/ または 8node64gpu/）
3. **Phase3**: `attention_extractor.py` → `head_classifier.py`
4. **Phase4**: `heatmap_generator.py` → `report_generator.py`
5. **Phase5**: `run_spt_medical.py`（MoE対応版）

## 注意事項

- Phase2 の `hook_functions_moe.py` は MoE 固有の Router/Expert ホックを含む
- Phase5 の `run_spt_medical.py` は Dense 版と異なり MoE エキスパート選択をサポート
- 共通コード（Phase2 dataset.py, utils.py 等）は `../shared/` を参照
