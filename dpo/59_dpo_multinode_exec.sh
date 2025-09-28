# For more information on multi-node training launch methods, refer to:
# https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node

set -eu

# Debug information
echo "=== Debug Info ==="
echo "MEGATRON_LM_PATH: $MEGATRON_LM_PATH"
echo "MODELSCOPE_CACHE: $MODELSCOPE_CACHE"
ls -la /workspace/Megatron-LM/ 2>/dev/null || echo "Megatron-LM directory not found"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
echo "=================="

# Set CUDA environment variables explicitly
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export NNODES=${NNODES}
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NPROC_PER_NODE=${NPROC_PER_NODE}
export NCCL_DEBUG=INFO

echo "=== Debug Info ==="
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "=================="


ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

export WANDB_ENTITY=llm-m_wandb-weblab

MODEL=${1}
MASTER_PORT=${2}

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