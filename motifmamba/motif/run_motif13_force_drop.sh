#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
OUT_DIR=${OUT_DIR:-${ROOT}/motif/logs/motif13_force_drop_$(date +%Y%m%d_%H%M%S)}

DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}
PQ_RANK=${PQ_RANK:-1}
TRAIN_ONLY_LAYER=${TRAIN_ONLY_LAYER:-0}

mkdir -p "${OUT_DIR}"
cd "${ROOT}/mamba-main"

# Aggressive settings for motif13 tiny-gradient issue
PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" MPLCONFIGDIR=/tmp/mpl \
"${PY}" -u motif/train_kpq_motif_pile_lm.py \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype float32 \
  --seed "${SEED}" \
  --motif-class 13 \
  --pq-rank "${PQ_RANK}" \
  --no-pq-per-dim \
  --train-only-layer "${TRAIN_ONLY_LAYER}" \
  --motif-loss-only-layer "${TRAIN_ONLY_LAYER}" \
  --train-pq-k \
  --pq-k-init 1.0 \
  --warmup-only \
  --no-warmup-strict \
  --warmup-max-steps 3000 \
  --warmup-ratio 0.1 \
  --warmup-lr 3e-2 \
  --warmup-coef 1e7 \
  --motif-amplitude 300 \
  --motif-bias 0.0 \
  --motif-loss-form log_mse \
  --motif-log-eps 1e-18 \
  --motif-loss-weights "1,1,1,1,1,1,1,1,1,1,1,1,200" \
  --motif-per-dim-mode sampled_channel \
  --motif-channel-samples 16 \
  --max-train-tokens 0 \
  --min-train-tokens 0 \
  --batch-size 1 \
  --seq-len 2048

echo "[done] out_dir=${OUT_DIR}"
echo "[done] warmup_summary=${OUT_DIR}/warmup_summary.json"
echo "[done] train_log=${OUT_DIR}/train_log.csv"
