#!/usr/bin/env bash
set -euo pipefail

# Batch eval for motif2 / motif2E / motif12 models.
# Update MODEL_2 / MODEL_2E / MODEL_12 to paths on your other machine.
#
# Candidate path can be:
# - a directory containing pq_adapter_latest.pt and/or full_model_latest.pt
# - a direct .pt file path

ROOT=${ROOT:-/workspace}
GRID_SCRIPT=${GRID_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_bestof_grid.sh}

MODEL_2=${MODEL_2:-/path/to/pile_mamba130m_motif2_rank2_xxx}
MODEL_2E=${MODEL_2E:-/path/to/pile_mamba130m_motif2E_rank2_xxx}
MODEL_12=${MODEL_12:-/path/to/pile_mamba130m_motif12_rank2_xxx}

DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande}

# Keep paper-like default first. You can override to sweep, e.g. FEWSHOTS=0,5 MAX_LENGTHS=2048,3072
FEWSHOTS=${FEWSHOTS:-0}
MAX_LENGTHS=${MAX_LENGTHS:-2048}

DTYPE=${DTYPE:-float16}
BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}
USE_CACHE=${USE_CACHE:-off}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}
LIMIT=${LIMIT:-0}
GRID_TAG=${GRID_TAG:-mamba130m_motif2_2E_12_batch}

if [[ ! -f "${GRID_SCRIPT}" ]]; then
  echo "[error] grid script not found: ${GRID_SCRIPT}"
  exit 1
fi

for p in "${MODEL_2}" "${MODEL_2E}" "${MODEL_12}"; do
  if [[ "${p}" == /path/to/* ]]; then
    echo "[error] please set MODEL_2 / MODEL_2E / MODEL_12 first."
    echo "[hint] current placeholder: ${p}"
    exit 1
  fi
done

CANDIDATES="${MODEL_2},${MODEL_2E},${MODEL_12}" \
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
