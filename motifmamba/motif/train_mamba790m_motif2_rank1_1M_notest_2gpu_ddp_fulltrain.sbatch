#!/bin/bash
#SBATCH -J m790m_motif2_r1_1M_2g_kpq
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o /home/user/run/hch/logs/pq_motif_pile/%x-%j.out
#SBATCH --open-mode=append

set -eEuo pipefail
trap 'code=$?; echo "[exit] code=${code} at $(date "+%F %T")"' EXIT

ROOT=/home/user/run/hch
CONDA_BIN=/home/user/run/hch/miniconda3/bin/conda
SCRIPT=${ROOT}/mamba-main/motif/train_kpq_motif_pile_lm.py
PRETRAINED_DIR=${ROOT}/mamba-790m
PILE_ROOT=${ROOT}/datasets/pile_standard_pythia_10M
OUT_DIR=${ROOT}/logs/pq_motif_pile/mamba790m_motif2_rank1_1M_notest_2gpu_ddp_kpq
RUN_LOG=${ROOT}/logs/pq_motif_pile/train_${SLURM_JOB_ID}.log

mkdir -p "${ROOT}/logs/pq_motif_pile" "${OUT_DIR}"
cd "${ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY_CMD=("${CONDA_BIN}" run -n motifmamba python)

echo "[start] job=${SLURM_JOB_ID} host=$(hostname) time=$(date '+%F %T')"
"${PY_CMD[@]}" -c "import numpy, torch; print('env ok:', __import__('sys').executable)"

CUDA_VISIBLE_DEVICES=0,1 "${PY_CMD[@]}" -m torch.distributed.run \
  --nproc_per_node 2 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  "${SCRIPT}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --dtype bfloat16 \
  --seed 42 \
  --dist-strategy ddp \
  --pq-rank 1 \
  --no-pq-per-dim \
  --pq-k-init 1e-6 \
  --train-pq-k \
  --no-train-base-model \
  --motif-class 2 \
  --motif-coef 0.1 \
  --motif-amplitude 2e4 \
  --motif-bias 1e-4 \
  --warmup-lr 1e-4 \
  --warmup-coef 1e4 \
  --warmup-ratio 0.3 \
  --warmup-max-steps 5000 \
  --warmup-log-every 20 \
  --no-warmup-strict \
  --train-lr 5e-5 \
  --train-steps 0 \
  --train-epochs 1.0 \
  --max-train-tokens 1000000 \
  --min-train-tokens 0 \
  --batch-size 1 \
  --seq-len 256 \
  --grad-accum-steps 1 \
  --eval-every-steps 0 \
  --eval-tokens 200000 \
  --disable-early-stop \
  --grad-clip 0.3 \
  --nan-backoff-retries 12 \
  --nan-backoff-scale-k 0.25 \
  --nan-backoff-scale-pq 0.5 \
  --log-every 10 \
  --save-every 200 \
  2>&1 | tee "${RUN_LOG}"

echo "[done] log=${RUN_LOG}"
