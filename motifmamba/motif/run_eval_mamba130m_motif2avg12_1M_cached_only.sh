#!/usr/bin/env bash
set -euo pipefail

# Evaluate motif2avg12 1M checkpoint with lm-eval cache reuse.
# If cache file is missing, it will rebuild once and reuse it next time.

ROOT=${ROOT:-/workspace}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}

MODEL_DIR=${MODEL_DIR:-${ROOT}/motif/logs/pile_mamba130m_motif2avg12_rank2_fulltrain_1M_seq2048_20260415_095438}
FULL_CKPT=${FULL_CKPT:-${MODEL_DIR}/full_model_latest.pt}
ADAPTER=${ADAPTER:-${MODEL_DIR}/pq_adapter_latest.pt}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}

TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande}
TAG=${TAG:-mamba130m_motif2avg12_rank2_fulltrain_1M_seq2048_cached_only}
USE_CACHE=${USE_CACHE:-${ROOT}/.cache/lm_eval/${TAG}.sqlite}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not executable: ${EVAL_SCRIPT}"
  exit 1
fi
if [[ ! -f "${FULL_CKPT}" ]]; then
  echo "[error] full checkpoint not found: ${FULL_CKPT}"
  exit 1
fi
if [[ ! -f "${ADAPTER}" ]]; then
  echo "[error] adapter not found: ${ADAPTER}"
  exit 1
fi

mkdir -p "$(dirname "${USE_CACHE}")"

echo "[run] full_ckpt=${FULL_CKPT}"
echo "[run] adapter=${ADAPTER}"
echo "[run] cache_requests=${CACHE_REQUESTS}"
echo "[run] use_cache=${USE_CACHE}"

ROOT="${ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
HARNESS_ROOT="${HARNESS_ROOT}" \
WRAPPER="${WRAPPER}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
FULL_CKPT="${FULL_CKPT}" \
ADAPTER="${ADAPTER}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_LENGTH="${MAX_LENGTH}" \
DTYPE="${DTYPE}" \
CACHE_REQUESTS="${CACHE_REQUESTS}" \
USE_CACHE="${USE_CACHE}" \
TASKS="${TASKS}" \
TAG="${TAG}" \
bash "${EVAL_SCRIPT}"
