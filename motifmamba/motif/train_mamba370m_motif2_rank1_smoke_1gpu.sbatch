#!/bin/bash
#SBATCH -J m370m_smoke_1g
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -o m370m_smoke_%j.out

set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-/workspace}}
SCRIPT_DIR="${ROOT}/mamba-main/motif"

mkdir -p "${ROOT}/logs/pq_motif_pile"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export DIST_STRATEGY=ddp

bash "${SCRIPT_DIR}/run_train_mamba370m_motif2_rank1_smoke_notest.sh" \
  2>&1 | tee "${ROOT}/logs/pq_motif_pile/smoke_370m_${SLURM_JOB_ID}.log"
