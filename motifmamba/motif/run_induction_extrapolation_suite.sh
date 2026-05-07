#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
SCRIPT=${SCRIPT:-${ROOT}/mamba-main/motif/induction_extrapolation_benchmark.py}

DEVICE=${DEVICE:-cuda:0}
OUT_DIR=${OUT_DIR:-${ROOT}/motif/logs/induction_extrapolation}
MODELS=${MODELS:-transformer,mamba,motifmamba}

TRAIN_LEN=${TRAIN_LEN:-256}
TEST_LENS=${TEST_LENS:-64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576}
TRAIN_STEPS=${TRAIN_STEPS:-100000}

# <=0 means auto by memory
BATCH_SIZE=${BATCH_SIZE:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-0}
EVAL_BATCHES=${EVAL_BATCHES:-64}
EVAL_TOKENS_BUDGET=${EVAL_TOKENS_BUDGET:-131072}
MAX_EVAL_BATCH_SIZE=${MAX_EVAL_BATCH_SIZE:-64}
MAX_TRANSFORMER_EVAL_LEN=${MAX_TRANSFORMER_EVAL_LEN:-8192}

LR=${LR:-1e-3}
WD=${WD:-0.0}
GRAD_CLIP=${GRAD_CLIP:-1.0}
AMP=${AMP:-bf16}
SEED=${SEED:-42}
ENABLE_EARLY_STOP=${ENABLE_EARLY_STOP:-1}
EARLY_STOP_MIN_STEPS=${EARLY_STOP_MIN_STEPS:-2000}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-1200}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-5e-4}
EARLY_STOP_EMA_BETA=${EARLY_STOP_EMA_BETA:-0.95}
EARLY_STOP_STABILITY_WINDOW=${EARLY_STOP_STABILITY_WINDOW:-400}
EARLY_STOP_STABILITY_TOL=${EARLY_STOP_STABILITY_TOL:-2e-3}

# Small-model regime (~tens/hundreds K params). transformer/rwkv can auto-match to mamba.
D_MODEL=${D_MODEL:-64}
N_LAYERS=${N_LAYERS:-2}
MAX_HEADS=${MAX_HEADS:-8}
FFN_MULT=${FFN_MULT:-2.0}
D_STATE=${D_STATE:-16}
D_CONV=${D_CONV:-4}
EXPAND=${EXPAND:-2}

PQ_RANK=${PQ_RANK:-2}
PQ_K_INIT=${PQ_K_INIT:-1e-4}
PQ_GRAD_MODE=${PQ_GRAD_MODE:-kernel}

cd "${ROOT}/mamba-main"
if [[ "${ENABLE_EARLY_STOP}" == "1" ]]; then
  EARLY_STOP_ARG="--enable-early-stop"
else
  EARLY_STOP_ARG="--disable-early-stop"
fi

PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" \
MOTIFMAMBA_PQ_GRAD_MODE="${PQ_GRAD_MODE}" \
"${PY}" "${SCRIPT}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}" \
  --models "${MODELS}" \
  --train-len "${TRAIN_LEN}" \
  --test-lens "${TEST_LENS}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-batches "${EVAL_BATCHES}" \
  --eval-tokens-budget "${EVAL_TOKENS_BUDGET}" \
  --max-eval-batch-size "${MAX_EVAL_BATCH_SIZE}" \
  --max-transformer-eval-len "${MAX_TRANSFORMER_EVAL_LEN}" \
  --lr "${LR}" \
  --weight-decay "${WD}" \
  --grad-clip "${GRAD_CLIP}" \
  --amp "${AMP}" \
  "${EARLY_STOP_ARG}" \
  --early-stop-min-steps "${EARLY_STOP_MIN_STEPS}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
  --early-stop-ema-beta "${EARLY_STOP_EMA_BETA}" \
  --early-stop-stability-window "${EARLY_STOP_STABILITY_WINDOW}" \
  --early-stop-stability-tol "${EARLY_STOP_STABILITY_TOL}" \
  --d-model "${D_MODEL}" \
  --n-layers "${N_LAYERS}" \
  --max-heads "${MAX_HEADS}" \
  --ffn-mult "${FFN_MULT}" \
  --d-state "${D_STATE}" \
  --d-conv "${D_CONV}" \
  --expand "${EXPAND}" \
  --pq-rank "${PQ_RANK}" \
  --pq-k-init "${PQ_K_INIT}" \
  --auto-match-params
