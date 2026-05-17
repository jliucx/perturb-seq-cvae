#!/usr/bin/env bash
# Parameter sweep for Papalexi dataset CVAE — phase excluded from covariates.
# z is free to capture cell-cycle variation instead of having it conditioned out.
# All 20 result subdirectories are placed under results_papalexi_nophase/.
# Usage: bash sweep_papalexi_nophase.sh

CONDA_PREFIX="/project/xuanyao/jiaming/miniconda3/envs/cvae_perturb"
PYTHON="${CONDA_PREFIX}/bin/python"
BASE_DIR="/project/xuanyao/jiaming/deeplearning"
SWEEP_DIR="${BASE_DIR}/results_papalexi_nophase"

BETAS=(1e-4 1e-3 0.01 0.1 1.0)
DIMS=(32 64 128 256)

mkdir -p "${SWEEP_DIR}"

for BETA in "${BETAS[@]}"; do
  for DIM in "${DIMS[@]}"; do
    TAG="beta${BETA}_dim${DIM}"
    RESULTS_DIR="${SWEEP_DIR}/${TAG}"
    mkdir -p "${RESULTS_DIR}"

    sbatch --job-name="cvae_papalexi_nophase_${TAG}" \
      --account=pi-xuanyao \
      --partition=xuanyao-hm \
      --qos=xuanyao \
      --time=04:00:00 \
      --cpus-per-task=8 \
      --mem=60G \
      --output="${RESULTS_DIR}/slurm_%j.out" \
      --error="${RESULTS_DIR}/slurm_%j.err" \
      --wrap="
cd ${BASE_DIR}
module purge 2>/dev/null || true
unset LD_PRELOAD; unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH='${CONDA_PREFIX}/lib'

echo '=== Train: beta=${BETA} latent_dim=${DIM} (no phase) ==='
${PYTHON} 01_train_cvae.py \
  --data_dir      data \
  --results_dir   ${RESULTS_DIR} \
  --latent_dim    ${DIM} \
  --beta          ${BETA} \
  --epochs        100 \
  --batch_size    512 \
  --lr            1e-3 \
  --patience      15 \
  --num_workers   4 \
  --exclude_covs  phase

echo '=== Analyze ==='
${PYTHON} 02_analysis.py \
  --data_dir    data \
  --results_dir ${RESULTS_DIR} \
  --checkpoint  model_best.pt \
  --latent_dim  ${DIM} \
  --batch_size  512 \
  --ntc_prefix  NTg

echo 'Done: ${TAG}'
"
    echo "Submitted: ${TAG}"
  done
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs finish, compare silhouette scores:"
echo "  for d in ${SWEEP_DIR}/beta*; do"
echo "    echo \"\$(basename \$d): \$(cat \$d/silhouette.txt 2>/dev/null || echo N/A)\""
echo "  done | sort -t: -k2 -n -r"
