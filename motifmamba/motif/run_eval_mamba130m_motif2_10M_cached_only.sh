#!/usr/bin/env bash
set -euo pipefail

# Evaluate one adapter on core LM tasks using existing lm-eval cache only.
# It will NOT rebuild context; if cache file is missing, it exits.

ROOT=${ROOT:-/workspace}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}

ADAPTER=${ADAPTER:-${ROOT}/motif/logs/pile_mamba130m_motif2_rank2_fulltrain_10M_seq2048_cuda0/pq_adapter_latest.pt}
BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}

TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,swde,squadv2,fda,triviaqa,nq_open,drop}
TAG=${TAG:-mamba130m_motif2_rank2_fulltrain_10M_seq2048_cached_only}
USE_CACHE=${USE_CACHE:-${ROOT}/.cache/lm_eval/core14_cache.sqlite}

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not executable: ${EVAL_SCRIPT}"
  exit 1
fi
if [[ ! -f "${ADAPTER}" ]]; then
  echo "[error] adapter not found: ${ADAPTER}"
  exit 1
fi
if [[ ! -f "${USE_CACHE}" ]]; then
  echo "[error] cache file not found: ${USE_CACHE}"
  echo "[hint] run once with CACHE_REQUESTS=refresh to build cache."
  exit 1
fi

echo "[run] adapter=${ADAPTER}"
echo "[run] use_cache=${USE_CACHE} (cache-only)"

ADAPTER="${ADAPTER}" \
ROOT="${ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
HARNESS_ROOT="${HARNESS_ROOT}" \
WRAPPER="${WRAPPER}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_LENGTH="${MAX_LENGTH}" \
DTYPE="${DTYPE}" \
CACHE_REQUESTS=true \
USE_CACHE="${USE_CACHE}" \
TASKS="${TASKS}" \
TAG="${TAG}" \
bash "${EVAL_SCRIPT}"
