#!/usr/bin/env bash
#SBATCH --job-name=cvae_perturb
#SBATCH --account=pi-xuanyao
#SBATCH --partition=xuanyao-hm
#SBATCH --qos=xuanyao
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=/project/xuanyao/jiaming/deeplearning/results/slurm_%j.out
#SBATCH --error=/project/xuanyao/jiaming/deeplearning/results/slurm_%j.err

cd /project/xuanyao/jiaming/deeplearning

CONDA_PREFIX="/project/xuanyao/jiaming/miniconda3/envs/cvae_perturb"
PYTHON="${CONDA_PREFIX}/bin/python"

# Purge system modules so the old system anaconda cannot inject its libstdc++.
module purge 2>/dev/null || true

# Drop all existing library paths and use ONLY the conda env's lib directory.
# This prevents any system libstdc++ from shadowing the newer one in the conda env.
unset LD_PRELOAD
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"

echo "=== Step 1: Train CVAE (100 epochs) ==="
"${PYTHON}" 01_train_cvae.py \
  --latent_dim 64 \
  --beta 1.0 \
  --epochs 100 \
  --batch_size 512 \
  --lr 1e-3 \
  --patience 15 \
  --num_workers 4

echo "=== Step 2: Analysis & Figures ==="
"${PYTHON}" 02_analysis.py \
  --checkpoint model_best.pt \
  --latent_dim 64 \
  --n_top_genes 40 \
  --batch_size 512

echo "Done!"
