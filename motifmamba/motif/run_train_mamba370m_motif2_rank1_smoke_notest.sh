#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke run:
# - base model: mamba-370m
# - motif class: 2
# - pq rank: 1
# - no mid-training eval
# - tiny training budget to quickly verify end-to-end run

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAMBA_MAIN=$(cd "${SCRIPT_DIR}/.." && pwd)
ROOT=$(cd "${MAMBA_MAIN}/.." && pwd)

PY=${PY:-python}
SCRIPT=${SCRIPT:-${SCRIPT_DIR}/train_kpq_motif_pile_lm.py}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-370m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia_10M}
OUT_DIR=${OUT_DIR:-${ROOT}/logs/pq_motif_pile/mamba370m_motif2_rank1_smoke_notest}

DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
SEED=${SEED:-42}

PQ_RANK=${PQ_RANK:-1}
PQ_K_INIT=${PQ_K_INIT:-1e-4}

MOTIF_CLASS=${MOTIF_CLASS:-2}
MOTIF_COEF=${MOTIF_COEF:-0.1}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-1e5}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}

WARMUP_LR=${WARMUP_LR:-5e-4}
WARMUP_COEF=${WARMUP_COEF:-1e6}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-1}
WARMUP_LOG_EVERY=${WARMUP_LOG_EVERY:-1}

TRAIN_LR=${TRAIN_LR:-1e-4}
TRAIN_STEPS=${TRAIN_STEPS:-1}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1024}
MIN_TRAIN_TOKENS=${MIN_TRAIN_TOKENS:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-0}
EVAL_TOKENS=${EVAL_TOKENS:-1024}
GRAD_CLIP=${GRAD_CLIP:-0.5}
LOG_EVERY=${LOG_EVERY:-1}
SAVE_EVERY=${SAVE_EVERY:-1}

mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1

echo "[run] script=${SCRIPT}"
echo "[run] model=${PRETRAINED_DIR}"
echo "[run] data=${PILE_ROOT}"
echo "[run] out_dir=${OUT_DIR}"
echo "[run] smoke: warmup_max_steps=${WARMUP_MAX_STEPS}, train_steps=${TRAIN_STEPS}, max_train_tokens=${MAX_TRAIN_TOKENS}"

"${PY}" -u "${SCRIPT}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  --pq-rank "${PQ_RANK}" \
  --no-pq-per-dim \
  --pq-k-init "${PQ_K_INIT}" \
  --train-pq-k \
  --no-train-base-model \
  --motif-class "${MOTIF_CLASS}" \
  --motif-coef "${MOTIF_COEF}" \
  --motif-amplitude "${MOTIF_AMPLITUDE}" \
  --motif-bias "${MOTIF_BIAS}" \
  --warmup-lr "${WARMUP_LR}" \
  --warmup-coef "${WARMUP_COEF}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --warmup-max-steps "${WARMUP_MAX_STEPS}" \
  --warmup-log-every "${WARMUP_LOG_EVERY}" \
  --no-warmup-strict \
  --train-lr "${TRAIN_LR}" \
  --train-steps "${TRAIN_STEPS}" \
  --max-train-tokens "${MAX_TRAIN_TOKENS}" \
  --min-train-tokens "${MIN_TRAIN_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --eval-tokens "${EVAL_TOKENS}" \
  --disable-early-stop \
  --grad-clip "${GRAD_CLIP}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo "[done] log=${OUT_DIR}/train.log"
echo "[done] csv=${OUT_DIR}/train_log.csv"
echo "[done] adapter=${OUT_DIR}/pq_adapter_latest.pt"
