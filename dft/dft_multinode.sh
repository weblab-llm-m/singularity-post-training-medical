#!/bin/bash
#SBATCH --job-name=dft_megatron_multinode_1
#SBATCH --partition=P08317
#SBATCH -nodelist=osk-gpu[40-43]
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "start job"
export NNODES=4
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

MODEL=${1}
echo "start srun"
#srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home /home/Competition2025/P05/shareP05/share_envs/ms-swift-megatron_v3.7.3 \
srun --jobid $SLURM_JOBID --gpus-per-node=${NPROC_PER_NODE}  singularity run -w --nv -B /home /home/Competition2025/P03/P03U011/ms-swift-megatron \
bash dft_multinode_exec.sh ${MODEL}