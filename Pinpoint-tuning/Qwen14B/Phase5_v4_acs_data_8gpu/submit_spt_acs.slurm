#!/bin/bash
#SBATCH --job-name=spt_acs_v1
#SBATCH --output=/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v4_acs_data_8gpu/spt_output/slurm_%j.out
#SBATCH --error=/home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v4_acs_data_8gpu/spt_output/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

echo "=========================================================================="
echo "SLURM Job: SPT Training - ACS Data v1"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Run training script
bash /home/Competition2025/P08/P08U023/model_analyze/medical_path_patching/Phase5_v4_acs_data_8gpu/run_spt_acs_training.sh

echo ""
echo "End time: $(date)"
echo "=========================================================================="
