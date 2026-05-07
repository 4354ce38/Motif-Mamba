#!/usr/bin/env bash
set -euo pipefail

# Compare two 130m models with a unified eval setting.
# Default: 0-shot, max_length=2048, core7 tasks.

ROOT=${ROOT:-/workspace}
GRID_SCRIPT=${GRID_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_bestof_grid.sh}

CANDIDATES=${CANDIDATES:-${ROOT}/motif/logs/pile_mamba130m_motif2avg12_rank2_fulltrain_1M_seq2048_20260415_095438,${ROOT}/motif/logs/pile_mamba130m_motifFRP_rank2_fulltrain_1M_seq2048_20260416_141153}

DEVICE=${DEVICE:-cuda:1}
BATCH_SIZE=${BATCH_SIZE:-8}
DTYPE=${DTYPE:-float16}
BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande}
FEWSHOTS=${FEWSHOTS:-0}
MAX_LENGTHS=${MAX_LENGTHS:-2048}

USE_CACHE=${USE_CACHE:-off}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}
LIMIT=${LIMIT:-0}
GRID_TAG=${GRID_TAG:-mamba130m_motif2avg12_vs_motifFRP}

if [[ ! -f "${GRID_SCRIPT}" ]]; then
  echo "[error] grid script not found: ${GRID_SCRIPT}"
  exit 1
fi

CANDIDATES="${CANDIDATES}" \
ROOT="${ROOT}" \
BASE_MODEL="${BASE_MODEL}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
DTYPE="${DTYPE}" \
TASKS="${TASKS}" \
FEWSHOTS="${FEWSHOTS}" \
MAX_LENGTHS="${MAX_LENGTHS}" \
USE_CACHE="${USE_CACHE}" \
CACHE_REQUESTS="${CACHE_REQUESTS}" \
LIMIT="${LIMIT}" \
GRID_TAG="${GRID_TAG}" \
bash "${GRID_SCRIPT}"
