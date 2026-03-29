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
