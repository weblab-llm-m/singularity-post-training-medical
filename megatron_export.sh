CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

export MODEL_NAME=Qwen3-30B-A3B-Instruct-2507
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# megatronのモデルに変換する
numactl --interleave=all \
swift export \
    --model /home/matsuolab/nishimae/singularity/model/${MODEL_NAME} \
    --model_type qwen3_moe \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /home/matsuolab/nishimae/singularity/model/${MODEL_NAME}-mcore \