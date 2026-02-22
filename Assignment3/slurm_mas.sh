#!/bin/bash

#SBATCH --job-name=mas_label
#SBATCH --output=logs/mas_label_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

mkdir -p logs

source .venv/bin/activate

echo "Running MAS labeling script..."

python mas_label.py

echo "MAS labeling completed."