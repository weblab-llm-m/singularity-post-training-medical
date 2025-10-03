#!/bin/bash
#SBATCH --job-name=megatron_merge_lora_hf
#SBATCH --partition=P08317
#SBATCH --nodelist=osk-gpu38
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "start job"
# Multinodeの場合nnodesを増やす（nodelistの番号変更とnodesの値変更）
export NNODES=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo "start srun"
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home /home/matsuolab/nishimae/singularity/ms-swift-megatron_v3.8.1 \
bash merge_megatron_hf.sh