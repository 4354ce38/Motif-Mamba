#!/bin/bash
#SBATCH -J m1p4b_motif2_r1_resume9M_4g
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o /data/run01/user/hch/logs/pq_motif_pile/%x-%j.out
#SBATCH --open-mode=append

set -eEuo pipefail
trap 'code=$?; echo "[exit] code=${code} at $(date "+%F %T")"' EXIT

ROOT=/home/user/run/hch
CONDA_BIN=/home/user/run/hch/miniconda3/bin/conda
SCRIPT=${ROOT}/mamba-main/motif/train_kpq_motif_pile_lm.py
PRETRAINED_DIR=${ROOT}/mamba-1.4b
PILE_ROOT=${ROOT}/datasets/pile_standard_pythia_10M

# 1M tokens checkpoint to continue from.
INIT_ADAPTER=/home/user/run/hch/logs/pq_motif_pile/mamba1p4b_motif2_rank1_1M_notest_512/pq_adapter_latest.pt

# Write resumed run to a new output dir.
OUT_DIR=${ROOT}/logs/pq_motif_pile/mamba1p4b_motif2_rank1_resume9M_from1M
RUN_LOG=${ROOT}/logs/pq_motif_pile/train_${SLURM_JOB_ID}.log

mkdir -p "${ROOT}/logs/pq_motif_pile" "${OUT_DIR}"
cd "${ROOT}"

PY_CMD=("${CONDA_BIN}" run -n motifmamba python)

echo "[start] job=${SLURM_JOB_ID} host=$(hostname) time=$(date '+%F %T')"
"${PY_CMD[@]}" -c "import numpy, torch; print('env ok:', __import__('sys').executable)"

echo "[resume] init_adapter=${INIT_ADAPTER}"

CUDA_VISIBLE_DEVICES=0,1,2,3 "${PY_CMD[@]}" -m torch.distributed.run \
  --nproc_per_node 4 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  "${SCRIPT}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --init-adapter "${INIT_ADAPTER}" \
  --device cuda \
  --dtype float16 \
  --seed 42 \
  --dist-strategy ddp \
  --pq-rank 1 \
  --no-pq-per-dim \
  --pq-k-init 1e-4 \
  --train-pq-k \
  --no-train-base-model \
  --motif-class 2 \
  --motif-coef 0.1 \
  --motif-amplitude 1e5 \
  --motif-bias 5e-5 \
  --warmup-lr 5e-4 \
  --warmup-coef 1e6 \
  --warmup-ratio 0.1 \
  --warmup-max-steps 0 \
  --warmup-log-every 20 \
  --no-warmup-strict \
  --train-lr 1e-4 \
  --train-steps 0 \
  --train-epochs 1.0 \
  --max-train-tokens 9000000 \
  --min-train-tokens 0 \
  --batch-size 1 \
  --seq-len 2048 \
  --grad-accum-steps 1 \
  --eval-every-steps 0 \
  --eval-tokens 200000 \
  --disable-early-stop \
  --grad-clip 0.5 \
  --log-every 10 \
  --save-every 200 \
  2>&1 | tee "${RUN_LOG}"

echo "[done] log=${RUN_LOG}"
