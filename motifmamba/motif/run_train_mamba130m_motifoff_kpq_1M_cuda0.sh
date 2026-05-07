#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
OUT_DIR=${OUT_DIR:-${ROOT}/motif/logs/pile_mamba130m_motifoff_rank2_kpq_1M_cuda1}

DEVICE=${DEVICE:-cuda:1}
DTYPE=${DTYPE:-float16}
PQ_RANK=${PQ_RANK:-2}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_TOKENS=${MAX_TOKENS:-1000000}
MIN_TOKENS=${MIN_TOKENS:-0}

mkdir -p "${OUT_DIR}"
cd "${ROOT}/mamba-main"

echo "[run] device=${DEVICE}, motif=off, max_tokens=${MAX_TOKENS}"
echo "[run] out_dir=${OUT_DIR}"

PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" \
"${PY}" motif/train_kpq_motif_pile_lm.py \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --pq-rank "${PQ_RANK}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-tokens "${MAX_TOKENS}" \
  --min-train-tokens "${MIN_TOKENS}" \
  --motif-class 2 \
  --motif-coef 0 \
  --warmup-max-steps 0 \
  --no-warmup-strict

