# 概要
環境構築の方法を述べる。


# 0. Clone
[Swift-RLリポジトリ](https://github.com/weblab-llm-m/swift-RL/tree/main)
```bash
cd
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
```bash
cd ~/swift-RL/containers
singularity pull swift3.9.3.sif docker:modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.3
```

# 3. Megatron-LM（training）を用意する。
Megatron GRPO は **`megatron.training`** を import します。pip の `megatron-core` だけでは不足するため、**Megatron-LM リポジトリ（core_r0.14.0）** を clone してパスを通します。

```bash
cd ~/swift-RL/containers
git clone --branch core_r0.14.0 https://github.com/NVIDIA/Megatron-LM.git megatron-lm-core_r0.14.0
```

# 4. ms-swiftを用意する
```bash
cd ~/swift-RL/containers
git clone --branch release/3.11 https://github.com/modelscope/ms-swift.git
# ms-swiftのpatchやplugin（拡張機能）を追加
cp $HOME/swift-RL/src/swift/patch_promptid.py $HOME/swift-RL/containers/ms-swift/examples/train/grpo/plugin
cp $HOME/singularity-post-training-medical/ms-swift-megatron_v3.9.3/plugin/reward_ophtho_plugin.py $HOME/swift-RL/containers/ms-swift/examples/train/grpo/plugin
cp $HOME/singularity-post-training-medical/ms-swift-megatron_v3.9.3/plugin/reward_chinese_plugin.py $HOME/swift-RL/containers/ms-swift/examples/train/grpo/plugin
cp $HOME/swift-RL/src/swift/qwen3_next.py $HOME/swift-RL/containers/ms-swift/swift/megatron/model/gpt/qwen3_next.py
# CHORDの学習を追加
cp $HOME/singularity-post-training-medical/ms-swift-megatron_v3.9.3/plugin/grpo_trainer.py $HOME/swift-RL/containers/ms-swift/swift/megatron/trainers/grpo_trainer.py
cp $HOME/singularity-post-training-medical/ms-swift-megatron_v3.9.3/plugin/megatron_args.py $HOME/swift-RL/containers/ms-swift/swift/megatron/argument/megatron_args.py
cp $HOME/singularity-post-training-medical/ms-swift-megatron_v3.9.3/plugin/qwen3_next.py $HOME/swift-RL/containers/ms-swift/swift/megatron/model/gpt/qwen3_next.py
# resumeを行うためにassertを解除するだけ
cp $HOME/singularity-post-training-medical/ms-swift-megatron_v3.9.3/plugin/checkpointing.py $HOME/swift-RL/containers/megatron-lm-core_r0.14.0/megatron/training/checkpointing.py
```

起動時に **コンテナへ bind** し、**`MEGATRON_LM_PATH`** と **`PYTHONPATH`** に追加します。(実装済み)

# 5. envファイル設定
singularity-post-training-medical以下に`.env`作成
形式
```env
WANDB_API_KEY="api_key"
HF_TOKEN="api_key"
```

# 6. Dataset作成（IgakuQA過去問）年度を絞っている（評価で使用するため）
```
singularity shell --nv \
  -B "$HOME/swift-RL:$HOME/swift-RL" \
  -B "/dev/shm:/dev/shm" \
  -B "$HOME/singularity-post-training-medical:$HOME/singularity-post-training-medical" \
  $HOME/swift-RL/containers/swift3.9.3.sif
python $HOME/singularity-post-training-medical/tools/dataset.py
```

# 7. Modelダウンロード
```
# Singularity環境入った状態で
python $HOME/singularity-post-training-medical/tools/model_download.py --model Qwen/Qwen3-Next-80B-A3B-Instruct
# Singularityから出る
exit
```

# 8. 実行
- train_*.sh：最初の学習スクリプト（途中で止めてもよい）
- train_*_resume.sh：途中から再開する学習スクリプト（WandbのIDや学習途中パス）
- train_*_oneshot.sh：途中で止めることを考えずに行う学習スクリプト（HFに直す必要がない）

```
cd ~/singularity-post-training-medical/ms-swift-megatron_v3.9.3/grpo
sbatch train_*.sh
```

# 9. mcoreからHF変換
```
cd ~/singularity-post-training-medical/ms-swift-megatron_v3.9.3/
sbatch convert_mcore_to_hf.sh
```