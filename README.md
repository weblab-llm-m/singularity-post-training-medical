# singularity-post-training-medical

ms-swift3.8.1のsingularityの実験コードおよびテンプレートを提供します。  
実験管理のため、一定のディレクトリ構造や命名規則に従って構成されています。

## Overview
### ディレクトリ構成
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