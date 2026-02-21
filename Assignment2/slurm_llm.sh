#!/bin/bash

#SBATCH --job-name=llm_label
#SBATCH --output=logs/llm_label_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

mkdir -p logs

source .venv/bin/activate

echo "Starting LLM labeling"

python llm_label.py

echo "LLM labeling completed"