# singularity-post-training-medical

ms-swift3.8.1のsingularityの実験コードおよびテンプレートを提供します。  
実験管理のため、一定のディレクトリ構造や命名規則に従って構成されています。

## 1.ディレクトリ構成
全体の構成は以下のようになっています。
```
singularity-post-training-medical
├── dft　　　　　　　　　　       dft用のコードディレクトリ
|　　├──dft_multinode.sh　　　　 dftのsingularityのマルチノード実行スクリプト
|　　├──dft_multinode_exec.sh　  dftのマルチノード用の実行パラメータスクリプト(dft_multinode.shを平行して実行)
|　　├──dft_singlenode.sh        dftのsingularityのシングルノード実行スクリプト
|    └──dft_singlenode_exec.sh　 dftのシングルノード用の実行パラメータスクリプト(dft_singlenode.shを平行して実行)
├── dpo
|　　├──dpo_multinode.sh　　　　  dpoのsingularityのマルチノード実行スクリプト
|　　├──dpo_multinode_exec.sh　   dpoのマルチノード用の実行パラメータスクリプト(dpo_multinode.shを平行して実行)
|　　├──dpo_singlenode.sh         dpoのsingularityのシングルノード実行スクリプト
|    └──dpo_singlenode_exec.sh　  dpoのシングルノード用の実行パラメータスクリプト(dpo_singlenode.shを平行して実行)
├── sft
|　　├──sft_multinode.sh　　　　   sftのsingularityのマルチノード実行スクリプト
|　　├──sft_multinode_exec.sh　    sftのマルチノード用の実行パラメータスクリプト(sft_multinode.shを平行して実行)
|　　├──sft_singlenode.sh          sftのsingularityのシングルノード実行スクリプト
|    └──sft_singlenode_exec.sh　   sftのシングルノード用の実行パラメータスクリプト(sft_singlenode.shを平行して実行)
├── Dockerfile                     ms-swift3.8.1のDockerfile
├── megatron_export_auto.sh        megatron変換用の自動実行スクリプト  
├── megatron_export.sh　　　　　    megatron変換用の手動実行スクリプト
├── merge_full_megatron_hf_auto.sh 全量のHF形式変換用自動スクリプト
├── merge_full_megatron_hf.sh      全量のHF形式変換用手動スクリプト
├── merge_megatron_hf_auto.sh      loraマージのHF形式変換用自動スクリプト
├── merge_megatron_hf.sh           loraマージのHF形式変換用手動スクリプト
└── READEME.md                     README
```

## 2.SingularityイメージのビルドとDockerHubからの取得

### 前提

- DockerHubアカウントが必要です。
- Step 1はDockerが使える環境（ローカルPCまたはDocker対応サーバー）で実行します。
- Step 2以降は計算クラスター（Singularity/Apptainerが入っているサーバー）で実行します。

---

### Step 1: DockerイメージをビルドしてDockerHubにプッシュ

```bash
# リポジトリルートで実行
cd (singularity-post-training-medicalのパス)/ms-swift-megatron_v3.8.1

# Dockerイメージをビルド（YOUR_USERNAMEを自分のDockerHubユーザー名に変更）
docker build -t YOUR_USERNAME/ms-swift-megatron:v3.8.1 .

# DockerHubにログイン
docker login

# DockerHubにプッシュ
docker push YOUR_USERNAME/ms-swift-megatron:v3.8.1
```

> **注意**: ベースイメージが `nvcr.io/nvidia/pytorch:25.05-py3` なのでビルドに30〜60分程度かかります。

---

### Step 2: SingularityでDockerHubからインストール

```bash
# DockerHubから直接SIFファイルを作成
SINGULARITY_CACHEDIR=(任意のディレクトリ) build -s （インストールディレクトリ） docker://YOUR_USERNAME/ms-swift-megatron:v3.8.1
singularity shell -w --nv -B /home　（インストールディレクトリ）
# 通信が出来ているか確認する
curl -I https://wandb.ai
```

## 3.HF形式からmegatron
## 実行ファイル

```bash
megatron_export.sh ：megatronの変換スクリプト
megatron_export_auto.sh　：　megatron_export.shの自動スクリプト
```

## 手順

①singularity-post-training-medicalに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/megatron-Huggingface
```

②megatronの変換スクリプト(megatron_export.sh)を編集する

```bash
swift export \
    --model （モデルパス）/${MODEL_NAME} \　←こちら編集
    --model_type qwen3_moe \　←こちら編集
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir （モデルパス）/${MODEL_NAME}-mcore \　←こちら編集
```

③自動スクリプト(megatron_export_auto.sh)のノードを編集する

```bash
#!/bin/bash
#SBATCH --job-name=megatron_exprot
#SBATCH --partition=P08317
#SBATCH --nodelist=osk-gpu38
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
```

## 手動スクリプト実行

①計算ノードに入る

```bash
srun --partition=xxxxx --nodelist=xxx-gpuxx  --gres=gpu:8 --nodes=1 --time=03:00:00 --pty /bin/bash
```

②singularityに入る

```bash
singularity shell -w --nv -B /home (②singularityに入るパス)/ms-swift-megatron_v3.8.1
```
②megatronの変換スクリプト(megatron_export.sh)を実行する

```bash
bash megatron_export.sh
```

## 自動スクリプト実行

①自動スクリプトを実行する

```bash
sbatch megatron_export_auto.sh
```
## 4.megatronからHF形式に変換
## 実行ファイル

```bash
merge_full_megatron_hf_auto.sh  ：全量のHF形式変換用自動スクリプト
merge_full_megatron_hf.sh　：　全量のHF形式変換用手動スクリプト
```

## 手順

①singularity-post-training-medicalに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical
```

②全量のHF形式変換用手動スクリプト(merge_full_megatron_hf.sh)を編集する

```bash
numactl --interleave=all \
swift export \
    --mcore_model （元モデルのパス） \　←こちらを編集
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir （エクスポートパス）/ \　←こちらを編集
```

③自動スクリプト(merge_full_megatron_hf_auto.sh)のノードを編集する

```bash
#!/bin/bash
#SBATCH --job-name=megatron_exprot
#SBATCH --partition=xxxxx
#SBATCH --nodelist=xxx-xxxx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
```

## 手動スクリプト実行

①計算ノードに入る

```bash
srun --partition=xxxx --nodelist=xxx-gpuxx  --gres=gpu:8 --nodes=1 --time=03:00:00 --pty /bin/bash
```

②singularityに入る

```bash
singularity shell -w --nv -B /home /home/matsuolab/nishimae/singularity/ms-swift-megatron_v3.8.1
```

③全量のHF形式変換用手動スクリプト(merge_full_megatron_hf.sh)を実行する

```bash
bash megatron_export.sh
```

## 自動スクリプト実行

①自動スクリプトを実行する

```bash
sbatch merge_full_megatron_hf_auto.sh
```
## 5.megatronのloraマージからHF形式に変換
## 実行ファイル

```bash
merge_megatron_hf_auto.sh  ：  loraのHF形式変換用自動スクリプト
merge_megatron_hf.sh　     ：　loraのHF形式変換用手動スクリプト
```

## 手順

①singularity-post-training-medicalに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/megatron-Huggingface
```

②loraのHF形式変換用手動スクリプト(merge_megatron_hf.sh)を編集する

```bash
numactl --interleave=all \
swift export \
    --mcore_model (mcoreの変換モデル)/ \ ←こちらを編集
    --mcore_adapters （学習したモデルパス） \　←こちらを編集
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir （出力先）/ \　←こちらを編集
```

③自動スクリプト(merge_megatron_hf_auto.sh)のノードを編集する

```bash
#!/bin/bash
#SBATCH --job-name=megatron_exprot
#SBATCH --partition=xxxxxx
#SBATCH --nodelist=xxxxx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
```

## 手動スクリプト実行

①計算ノードに入る

```bash
srun --partition=xxxxx --nodelist=xxxx  --gres=gpu:8 --nodes=1 --time=03:00:00 --pty /bin/bash
```

②singularityに入る

```bash
singularity shell -w --nv -B /home （singularityのイメージパス）/ms-swift-megatron_v3.8.1
```

③loraのHF形式変換用手動スクリプト(merge_full_megatron_hf.sh)を実行する

```bash
bash merge_megatron_hf.sh
```

## 自動スクリプト実行

①自動スクリプトを実行する

```bash
sbatch merge_megatron_hf_auto.sh
```

## 5.SFT実行
## 実行ファイル

```bash

(1)マルチノード用
sft_multinode.sh 　　　　：　マルチノードのsingularityを起動するスクリプト
sft_multinode_exec.sh　　：　マルチノード用のSFTパラメータ設定コードのスクリプト

(2)シングルノード用
sft_singlenode.sh 　　　　：　シングルノードのsingularityを起動するスクリプト
sft_singlenode_exec.sh　　：　シングルノード用のSFTパラメータ設定コードのスクリプト
```

## マルチノード用

①singularity-post-training-medicalのsftに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/post-training/sft
```

②「sft_multinode.sh」を編集する

```bash
#編集の抜粋
#!/bin/bash
#SBATCH --job-name=sft_megatron_multinode
#SBATCH --partition=xxxx
#SBATCH --nodelist=xxx[xx-xx]
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

-------------------------------------------------------
export NNODES=4
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

-------------------------------------------------------

# singularityのパスを編集する
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home （singularityのイメージパス）/ms-swift-megatron_v3.8.1 \
bash sft_multinode_exec.sh ${MODEL} ${MASTER_PORT}
```

③「sft_multinode_exec.sh」を編集する

```bash
#該当するパラメータを修正する
USE_HF=1  MASTER_PORT=${MASTER_PORT} megatron sft \
    --load ${MODEL} \
    --dataset 'team-suzuki/SFT_006_origin_1' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --lazy_tokenize true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --moe_expert_capacity_factor 1.0 \
    --moe_token_dispatcher_type alltoall \
    --sequence_parallel true \
    --micro_batch_size 4 \
    --global_batch_size 32 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/multinode/${MODEL} \
    --eval_interval 200 \
    --save_interval 400 \
    --max_length 16384 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --wandb_exp_name sft_4node_16k \
    --wandb_project sft_megatron_235B \
    --wandb_save_dir wandb_logs
```

④logsディレクトリを作成する

```bash
mkdir -p (logsディレクトリ)
```

⑤以下のスクリプト実行させて、モデル学習を開始する

```bash
sbatch sft_multinode.sh　(追加学習対象モデル)
```

## シングルノード用

①singularity-post-training-medicalのsftに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/post-training/sft
```

②「sft_singlenode.sh」を編集する

```bash
#編集の抜粋
#!/bin/bash
#SBATCH --job-name=sft_megatron_multinode
#SBATCH --partition=xxxx
#SBATCH --nodelist=xxxx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

-------------------------------------------------------
export NNODES=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

-------------------------------------------------------

# singularityのパスを編集する
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home (singularityのイメージパス)/ms-swift-megatron_v3.8.1 \
bash sft_singlenode_exec.sh ${MODEL} ${MASTER_PORT}
```

③「sft_singlenode_exec.sh」を編集する

```bash
#該当するパラメータを修正する
USE_HF=1  MASTER_PORT=${MASTER_PORT} megatron sft \
    --load ${MODEL} \
    --dataset 'team-suzuki/SFT_006_origin_1' \　←学習データを設定する
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --lazy_tokenize true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/multinode/${MODEL} \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 16384 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --wandb_exp_name test_multinode \
    --wandb_project sft_megatron \
    --wandb_save_dir wandb_logs
```

④logsディレクトリを作成する

```bash
mkdir -p (logsディレクトリ)
```

⑤以下のスクリプト実行させて、モデル学習を開始する

```bash
sbatch sft_singlenode.sh　(追加学習対象モデル)
```

## 6.DPO実行
## 実行ファイル

```bash

(1)マルチノード用
dpo_multinode.sh 　　　　：　マルチノードのsingularityを起動するスクリプト
dpo_multinode_exec.sh　　：　マルチノード用のDPOパラメータ設定コードのスクリプト

(2)シングルノード用
dpo_singlenode.sh 　　　　：　シングルノードのsingularityを起動するスクリプト
dpo_singlenode_exec.sh　　：　シングルノード用のDPOパラメータ設定コードのスクリプト
```

## マルチノード用

①singularity-post-training-medicalのdpoに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/post-training/dpo
```

②「dpo_multinode.sh」を編集する

```bash
#編集の抜粋
#!/bin/bash
#SBATCH --job-name=sft_megatron_multinode
#SBATCH --partition=xxxxx
#SBATCH --nodelist=xxxx[xx-xx]
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

-------------------------------------------------------
export NNODES=4
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

-------------------------------------------------------

# singularityのパスを編集する
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home (singularityのイメージパス)/ms-swift-megatron_v3.8.1 \
bash dpo_multinode_exec.sh ${MODEL} ${MASTER_PORT}
```

④「dpo_multinode_exec.sh」を編集する

```bash
#該当するパラメータを修正する
USE_HF=1 MASTER_PORT=${MASTER_PORT} megatron rlhf \
    --rlhf_type dpo \
    --load ${MODEL} \
    --dataset team-suzuki/DPO_006_1_withloop \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.05 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 5e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/dpo/Qwen3-235B-A22B-Thinking-2507/006_1_withloop \
    --eval_interval 100 \
    --save_interval 400 \
    --max_length 16384 \
    --num_workers 4 \
    --dataset_num_proc 4 \
    --truncation_strategy right \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --beta 0.1 \
    --loss_type sigmoid \
    --loss_scale ignore_empty_think \
    --wandb_exp_name 006_1_withloop \
    --wandb_project dpo_235b \
    --wandb_save_dir wandb_logs
```

⑤logsディレクトリを作成する

```bash
mkdir -p (logsディレクトリ)
```

⑥以下のスクリプト実行させて、モデル学習を開始する

```bash
sbatch dpo_multinode.sh　(追加学習対象モデル)
```

## シングルノード用

①singularity-post-training-medicalのsftに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/post-training/dpo
```

②「dpo_singlenode.sh」を編集する

```bash
#編集の抜粋
#!/bin/bash
#SBATCH --job-name=dpo_megatron_multinode
#SBATCH --partition=xxxxxx
#SBATCH --nodelist=xxxxx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

-------------------------------------------------------
export NNODES=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

-------------------------------------------------------

# singularityのパスを編集する
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home (singularityのイメージパス)/ms-swift-megatron_v3.8.1 \
bash dpo_singlenode_exec.sh ${MODEL} ${MASTER_PORT}
```

③「dpo_singlenode_exec.sh」を編集する

```bash
#該当するパラメータを修正する
USE_HF=1 MASTER_PORT=${MASTER_PORT} megatron rlhf \
    --rlhf_type dpo \
    --load ${MODEL} \
    --dataset team-suzuki/DPO_006_1 \
    --columns '{"chosen":"messages","rejected":"rejected_messages"}' \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.1 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 2 \
    --global_batch_size 32 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/dpo/Qwen3-30B-A3B-Thinking-2507/006_1 \
    --eval_interval 100 \
    --save_interval 400 \
    --max_length 16384 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --truncation_strategy right \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --beta 0.1 \
    --loss_type sigmoid \
    --loss_scale ignore_empty_think \
    --wandb_exp_name 006_1 \
    --wandb_project dpo_30b \
    --wandb_save_dir wandb_logs
```

④logsディレクトリを作成する

```bash
mkdir -p (logsディレクトリ)
```

⑤以下のスクリプト実行させて、モデル学習を開始する

```bash
sbatch dpo_singlenode.sh (追加学習対象モデル)
```

## 7.DFT実行
## 実行ファイル

```bash

(1)マルチノード用
dft_multinode.sh 　　　　：　マルチノードのsingularityを起動するスクリプト
dft_multinode_exec.sh　　：　マルチノード用のDFTパラメータ設定コードのスクリプト

(2)シングルノード用
dft_singlenode.sh 　　　　：　シングルノードのsingularityを起動するスクリプト
dft_singlenode_exec.sh　　：　シングルノード用のDFTパラメータ設定コードのスクリプト
```

## マルチノード用

①singularity-post-training-medicalのdftに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/post-training/dft
```

②「dft_multinode.sh」を編集する

```bash
#編集の抜粋
#!/bin/bash
#SBATCH --job-name=dft_megatron_multinode_1
#SBATCH --partition=xxxxxx
#SBATCH --nodelist=xxxx[xx-xx]
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

-------------------------------------------------------
export NNODES=4
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

-------------------------------------------------------

# singularityのパスを編集する
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home (singularityのイメージパス)/ms-swift-megatron_v3.8.1 \
bash dft_multinode_exec.sh (追加学習対象モデル)
```

④「dft_multinode_exec.sh」を編集する

```bash
#該当するパラメータを修正する
USE_HF=1  MASTER_PORT=${MASTER_PORT} megatron sft \
    --load ${MODEL} \
    --dataset team-suzuki/DFT_235B-Thinking_006_origin_1 \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --lazy_tokenize true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --moe_expert_capacity_factor 1.0 \
    --moe_token_dispatcher_type alltoall \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 2e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/multinode/${MODEL} \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 16384 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --wandb_exp_name dft_4node \
    --wandb_project dft_megatron \
    --enable_dft_loss true \
    --wandb_save_dir wandb_logs
```

⑤logsディレクトリを作成する

```bash
mkdir -p (logsディレクトリ)
```

⑥以下のスクリプト実行させて、モデル学習を開始する

```bash
sbatch dft_multinode.sh　(追加学習対象モデル)
```

## シングルノード用

①singularity-post-training-medicalのsftに移動する

```bash
cd (singularity-post-training-medicalのパス)/singularity-post-training-medical/dft
```

②「dft_singlenode.sh」を編集する

```bash
#編集の抜粋
#!/bin/bash
#SBATCH --job-name=sft_megatron_multinode
#SBATCH --partition=xxxxx
#SBATCH --nodelist=xxxx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

-------------------------------------------------------
export NNODES=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
-------------------------------------------------------

# singularityのパスを編集する
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home (singularityのイメージパス)/ms-swift-megatron_v3.8.1 \
bash dft_singlenode_exec.sh ${MODEL} ${MASTER_PORT}
```

③「dft_singlenode_exec.sh」を編集する

```bash
#該当するパラメータを修正する
USE_HF=1  MASTER_PORT=${MASTER_PORT} megatron sft \
    --load ${MODEL} \
    --dataset team-suzuki/DFT_235B-Thinking_006_origin_1 \
    --split_dataset_ratio 0.01 \
    --enable_dft_loss true \
    --train_type lora \
    --lazy_tokenize true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/multinode/${MODEL} \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --wandb_exp_name test_multinode \
    --wandb_project sft_megatron_30b \
    --wandb_save_dir wandb_logs
```

④logsディレクトリを作成する

```bash
mkdir -p (logsディレクトリ)
```

⑤以下のスクリプト実行させて、モデル学習を開始する

```bash
sbatch dft_singlenode.sh　(追加学習対象モデル)
```