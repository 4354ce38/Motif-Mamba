#!/usr/bin/env bash
set -euo pipefail

# Evaluate MAVGF model on another machine.
# Default root points to /workspace.

ROOT=${ROOT:-/workspace}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}

# Set this to your MAVGF training log directory on the other machine.
MODEL_DIR=${MODEL_DIR:-/path/to/pile_mamba130m_mavgf_rank2_xxx}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}

TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande}
TAG=${TAG:-mamba130m_mavgf_other_machine}
USE_CACHE=${USE_CACHE:-off}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}
LIMIT=${LIMIT:-0}

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not found: ${EVAL_SCRIPT}"
  exit 1
fi
if [[ "${MODEL_DIR}" == /path/to/* ]]; then
  echo "[error] please set MODEL_DIR to your MAVGF model directory."
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[error] model directory not found: ${MODEL_DIR}"
  exit 1
fi

FULL_CKPT="${FULL_CKPT:-${MODEL_DIR}/full_model_latest.pt}"
ADAPTER="${ADAPTER:-${MODEL_DIR}/pq_adapter_latest.pt}"

if [[ ! -f "${FULL_CKPT}" ]]; then
  FULL_CKPT=""
fi
if [[ ! -f "${ADAPTER}" ]]; then
  ADAPTER=""
fi
if [[ -z "${FULL_CKPT}" && -z "${ADAPTER}" ]]; then
  echo "[error] neither full_model_latest.pt nor pq_adapter_latest.pt found in ${MODEL_DIR}"
  exit 1
fi

echo "[run] model_dir=${MODEL_DIR}"
if [[ -n "${FULL_CKPT}" ]]; then
  echo "[run] full_ckpt=${FULL_CKPT}"
fi
if [[ -n "${ADAPTER}" ]]; then
  echo "[run] adapter=${ADAPTER}"
fi

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
NUM_FEWSHOT="${NUM_FEWSHOT}" \
CACHE_REQUESTS="${CACHE_REQUESTS}" \
USE_CACHE="${USE_CACHE}" \
LIMIT="${LIMIT}" \
TASKS="${TASKS}" \
TAG="${TAG}" \
bash "${EVAL_SCRIPT}"
