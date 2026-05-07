#!/usr/bin/env bash
set -euo pipefail

# Evaluate motif2 + motif12 adapters on core14 tasks with shared lm-eval cache.
# First run refreshes cache if missing; subsequent runs reuse cache automatically.

ROOT=${ROOT:-/workspace}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}
BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}
USE_CACHE=${USE_CACHE:-${ROOT}/.cache/lm_eval/core14_cache.sqlite}

ADAPTER_M2=${ADAPTER_M2:-${ROOT}/motif/logs/pile_mamba130m_motif2_rank2_fulltrain_1B/pq_adapter_latest.pt}
ADAPTER_M12=${ADAPTER_M12:-${ROOT}/motif/logs/pile_mamba130m_motif12_rank2_fulltrain_1B/pq_adapter_latest.pt}

DEVICE_M2=${DEVICE_M2:-cuda:1}
DEVICE_M12=${DEVICE_M12:-cuda:2}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}
TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,swde,squadv2,fda,triviaqa,nq_open,drop}

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not executable: ${EVAL_SCRIPT}"
  exit 1
fi

if [[ ! -f "${ADAPTER_M2}" ]]; then
  echo "[error] missing adapter: ${ADAPTER_M2}"
  exit 1
fi
if [[ ! -f "${ADAPTER_M12}" ]]; then
  echo "[error] missing adapter: ${ADAPTER_M12}"
  exit 1
fi

echo "[run] shared cache: ${USE_CACHE}"

# motif2
ADAPTER="${ADAPTER_M2}" \
ROOT="${ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
HARNESS_ROOT="${HARNESS_ROOT}" \
WRAPPER="${WRAPPER}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
DEVICE="${DEVICE_M2}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_LENGTH="${MAX_LENGTH}" \
DTYPE="${DTYPE}" \
CACHE_REQUESTS=auto \
USE_CACHE="${USE_CACHE}" \
TASKS="${TASKS}" \
TAG="mamba130m_motif2_adapter_core14_cached" \
bash "${EVAL_SCRIPT}"

# motif12
ADAPTER="${ADAPTER_M12}" \
ROOT="${ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
HARNESS_ROOT="${HARNESS_ROOT}" \
WRAPPER="${WRAPPER}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
DEVICE="${DEVICE_M12}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_LENGTH="${MAX_LENGTH}" \
DTYPE="${DTYPE}" \
CACHE_REQUESTS=auto \
USE_CACHE="${USE_CACHE}" \
TASKS="${TASKS}" \
TAG="mamba130m_motif12_adapter_core14_cached" \
bash "${EVAL_SCRIPT}"

echo "[done] both evaluations completed"
