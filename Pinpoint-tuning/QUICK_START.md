# Quick Start Guide

## Phase 1: データ準備（すぐに実行可能）

```bash
cd /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching
bash scripts/run_phase1.sh
```

このコマンドで以下が生成されます:
- アノテーション済みデータ
- Counterfactualデータ
- Path Patching用データセット

## Phase 2の準備（次のステップ）

```bash
# 1. 既存ファイルをコピー
cd /home/Competition2025/P08/P08U023/model_analyze
cp sycophancy-interpretability/path_patching/*.py medical_path_patching/Phase2_path_patching/

# 2. スクリプト構造設計書.mdの「3. 既存改変スクリプト詳細」セクションを参照して改変
```

## 全体フロー

```
Phase 1: データ準備 ✅
  ↓
Phase 2: Path Patching ⚠️（要実装）
  ↓
Phase 3: ヘッド分類 ✅
  ↓
Phase 4: 可視化 ✅
  ↓
Phase 5: Pinpoint Tuning ✅
```

詳細は `README.md` と `IMPLEMENTATION_STATUS.md` を参照してください。
