# 概要
環境構築の方法を述べる。


# 0. Clone
```
git clone swift-RLリポジトリ(git@github.com:weblab-llm-m/swift-RL.git)
git clone このリポジトリ
```

# 1. ディレクトリ配置（推奨）
```
swift-RL/
  ├─ containers/
  │   ├─ ms-swift/                 # SWIFT の最新版ソースツリー
  │   ├─ swift3.9.3.sif            # Singularity イメージ
  │   └─ megatron-lm-core_r0.14.0/ # NVIDIA Megatron-LM の Git リポジトリ（training 用）
  ├─ dataset/
  │   └─ dataset_info_*.json
  ├─ outputs/
  └─ .cache_home/                  # HF/Triton/torch_extensions のキャッシュ
singularity-post-training-medical/
  ├─ ms-swift-megatron_v3.8.1/      # 省略
  ├─ ms-swift-megatron_v3.9.3/
      ├─ chord/
      │   └─ train_chord.sh
      ├─ grpo/
      │   └─train_grpo.sh
      ├─ gspo/
          └─train_gspo.sh

```

# 2. Singularityを用意する
```
cd swift-RL/containers
singularity pull swift3.9.3.sif docker:modelscope-registry
```

# 3. ms-swiftを用意する
```
cd swift-RL/containers
git clone https://github.com/modelscope/ms-swift.git
cp swift-RL/src/swift/patch_promptid.py swift-RL/containers/ms-swift/examples/train/grpo/plugin
cp swift-RL/src/swift/igaku_plugin.py swift-RL/containers/ms-swift/examples/train/grpo/plugin
cp swift-RL/src/swift/qwen3_next.py swift-RL/containers/ms-swift/swift/megatron/model/gpt/qwen3_next.py
```

# 4. Megatron-LM（training）を用意する。
Megatron GRPO は **`megatron.training`** を import します。pip の `megatron-core` だけでは不足するため、**Megatron-LM リポジトリ（core_r0.14.0）** を clone してパスを通します。

```bash
cd swift-RL/containers
git clone --branch core_r0.14.0 https://github.com/NVIDIA/Megatron-LM.git megatron-lm-core_r0.14.0
```

起動時に **コンテナへ bind** し、**`MEGATRON_LM_PATH`** と **`PYTHONPATH`** に追加します。(実装済み)

# 5. Dataset作成（IgakuQA過去問）
```
python Haraのtoolのもの
```

# 6. envファイル設定
singularity-post-training-medical以下に作成
形式
```
WANDB_API_KEY="api_key"
HF_TOKEN="api_key"
```

# 7. 実行
```
cd singularity-post-training-medical/grpo
sbatch train_*.sh
```