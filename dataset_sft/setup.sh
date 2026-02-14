#!/bin/bash
set -e

echo "=== SFT DataGen 環境セットアップ ==="

source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n sft_datagen python=3.12 -y
conda activate sft_datagen

pip install --no-cache-dir \
    "vllm>=0.10.0" \
    "openai>=1.79.0" \
    "datasets>=3.6.0" \
    "hydra-core>=1.3.2" \
    "tqdm>=4.66.0" \
    "huggingface_hub>=0.30.0" \
    "hf-transfer>=0.1.9" \
    "python-dotenv>=1.0.0"

echo ""
echo "=== セットアップ完了 ==="
echo "  conda activate sft_datagen"
echo "  hf auth login  # HFトークン設定"
echo "  python prepare_data.py  # データ前処理"
echo "  sbatch run_1node.sh     # 1ノード実行"
echo "  sbatch run_8node.sh     # 8ノード並列実行"