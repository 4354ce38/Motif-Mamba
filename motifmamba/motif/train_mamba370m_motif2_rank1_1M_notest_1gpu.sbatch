#!/bin/bash
#SBATCH -J m370m_motif2_r1_1M_1g
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -o /data/run01/user/hch/logs/pq_motif_pile/%x-%j.out
#SBATCH --open-mode=append

set -eEuo pipefail
trap 'code=$?; echo "[exit] code=${code} at $(date "+%F %T")"' EXIT

ROOT=${ROOT:-/data/run01/user/hch}
CONDA_BIN=${CONDA_BIN:-/home/user/run/hch/miniconda3/bin/conda}
RUN_SCRIPT=${RUN_SCRIPT:-${ROOT}/mamba-main/motif/run_train_mamba370m_motif2_rank1_1M_notest_1gpu.sh}
RUN_LOG=${RUN_LOG:-${ROOT}/logs/pq_motif_pile/train_370m_1gpu_${SLURM_JOB_ID}.log}

mkdir -p "${ROOT}/logs/pq_motif_pile"
cd "${ROOT}"

echo "[start] job=${SLURM_JOB_ID} host=$(hostname) time=$(date '+%F %T')"
"${CONDA_BIN}" run -n motifmamba python -c "import numpy, torch; print('env ok:', __import__('sys').executable)"

PY=$("${CONDA_BIN}" run -n motifmamba python -c "import sys; print(sys.executable)")
PY="${PY}" bash "${RUN_SCRIPT}" 2>&1 | tee "${RUN_LOG}"

echo "[done] run_log=${RUN_LOG}"
