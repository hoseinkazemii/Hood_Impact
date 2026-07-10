#!/bin/bash -l
#SBATCH --job-name=hood_temporal_deeponet
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4                   # GH200 partition (matches your interactive srun)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192g
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err

set -euo pipefail

# --- environment -----------------------------------------------------------
# `bash -l` sources your profile so conda is already initialized, exactly like
# the `--pty bash -l` interactive session.
conda activate hood_impact

cd "$SLURM_SUBMIT_DIR"

# Weights & Biases: run `wandb login` once on a login node,
# or uncomment the next line to disable online logging.
# export WANDB_MODE=offline

echo "Node: $(hostname) | GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- run -------------------------------------------------------------------
# data_format / num_samples / split / batch_size are all driven by Config
# (industrylike is the default; split = 45 train / 5 val / 10 test).
python temporal_deeponet.py

echo "Done."
