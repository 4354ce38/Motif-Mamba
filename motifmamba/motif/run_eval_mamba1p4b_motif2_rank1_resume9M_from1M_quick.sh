#!/usr/bin/env bash
set -euo pipefail

# Quick lm-eval run for mamba-1.4b + motif2 rank1 PQ adapter.
# Default tasks are a small, commonly used LM subset.

ROOT=${ROOT:-/workspace}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}

ADAPTER=${ADAPTER:-/workspace/motif/logs/mamba1p4b_motif2_rank1_resume9M_from1M/pq_adapter_latest.pt}
BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-1.4b}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}

TASKS=${TASKS:-lambada_openai,hellaswag,piqa,arc_easy,winogrande}
TAG=${TAG:-mamba1p4b_motif2_rank1_resume9M_from1M_quick}

# Optional knobs:
# - LIMIT=100 for faster smoke evaluation on each task.
# - NUM_FEWSHOT=0/5/... to override task default few-shot.
LIMIT=${LIMIT:-0}
NUM_FEWSHOT=${NUM_FEWSHOT:-}

USE_CACHE=${USE_CACHE:-${ROOT}/.cache/lm_eval/${TAG}.sqlite}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not executable: ${EVAL_SCRIPT}"
  exit 1
fi
if [[ ! -f "${ADAPTER}" ]]; then
  echo "[error] adapter not found: ${ADAPTER}"
  exit 1
fi

mkdir -p "$(dirname "${USE_CACHE}")"

echo "[run] base_model=${BASE_MODEL}"
echo "[run] adapter=${ADAPTER}"
echo "[run] tasks=${TASKS}"
echo "[run] device=${DEVICE}, batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}, dtype=${DTYPE}"
if [[ -n "${NUM_FEWSHOT}" ]]; then
  echo "[run] num_fewshot=${NUM_FEWSHOT}"
fi
if [[ "${LIMIT}" != "0" ]]; then
  echo "[run] limit=${LIMIT}"
fi
echo "[run] cache_requests=${CACHE_REQUESTS}"
echo "[run] use_cache=${USE_CACHE}"

ROOT="${ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
HARNESS_ROOT="${HARNESS_ROOT}" \
WRAPPER="${WRAPPER}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
ADAPTER="${ADAPTER}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_LENGTH="${MAX_LENGTH}" \
DTYPE="${DTYPE}" \
TASKS="${TASKS}" \
TAG="${TAG}" \
LIMIT="${LIMIT}" \
NUM_FEWSHOT="${NUM_FEWSHOT}" \
CACHE_REQUESTS="${CACHE_REQUESTS}" \
USE_CACHE="${USE_CACHE}" \
bash "${EVAL_SCRIPT}"
