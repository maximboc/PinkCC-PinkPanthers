#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out.log
#SBATCH --partition=ird_gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=250G

# Path to save logs
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"


# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Running on: $SLURM_NODELIST"

# Preprocess the data
echo "Started eval"
ovseg_inference /home/user-data_challenge-33/data/raw_data/EVAL --models pagnoux
