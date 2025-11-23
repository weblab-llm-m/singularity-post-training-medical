#!/bin/bash
ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

# singularity shell -w --nv -B /home {singularityのパス}　でsingularityに入る。
# mcore_adaptersのパスのみ変えてください。
# bash で実行してください。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

numactl --interleave=all \
swift export \
    --mcore_model /home/matsuolab/nishimae/singularity/model/Qwen3-235B-A22B-thinking-exp8-Instruction-mcore/ \
    --mcore_adapters /home/matsuolab/nishimae/singularity/singularity-post-training-medical/sft/megatron_output/multinode/home/matsuolab/nishimae/singularity/model/Qwen3-235B-A22B-thinking-exp8-Instruction-mcore/v1-20251119-214254 \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir ./hf_output/Qwen3-235B-A22B-thinking-exp8-Instruction-mcore/sft_igakuqa_lora_235B_17 \