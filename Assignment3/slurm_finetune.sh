#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

mkdir -p logs

source .venv/bin/activate

echo "Starting LLM finetuning"

python finetune.py

echo "LLM finetuning completed"