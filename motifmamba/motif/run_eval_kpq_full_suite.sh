#!/usr/bin/env bash
set -euo pipefail

# Evaluate a PQ-adapted Mamba model on the requested full suite:
# LAMBADA(openai+standard), HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, OBQA,
# SWDE, SQuADv2, FDA, TriviaQA, NQ-open, DROP, NIAH single 1/2/3.

PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
ROOT=${ROOT:-/workspace}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-1.4b}
ADAPTER=${ADAPTER:-}
FULL_CKPT=${FULL_CKPT:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

DEVICE=${DEVICE:-cuda:1}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}
LIMIT=${LIMIT:-0}
NUM_FEWSHOT=${NUM_FEWSHOT:-}
OFFLINE=${OFFLINE:-0}
CACHE_REQUESTS=${CACHE_REQUESTS:-auto}
USE_CACHE=${USE_CACHE:-${ROOT}/.cache/lm_eval/core14_cache.sqlite}

TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,swde,squadv2,fda,triviaqa,nq_open,drop,niah_single_1,niah_single_2,niah_single_3}

OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/lm_eval}
TAG=${TAG:-kpq_full_suite}
OUT_DIR="${OUT_ROOT}/${TAG}_$(date +%Y%m%d_%H%M%S)"

# Avoid cross-model cache collisions when callers leave USE_CACHE at default.
if [[ "${USE_CACHE}" == "${ROOT}/.cache/lm_eval/core14_cache.sqlite" ]]; then
  USE_CACHE="${ROOT}/.cache/lm_eval/${TAG}.sqlite"
fi

mkdir -p "${OUT_DIR}" "${ROOT}/.cache/huggingface/datasets"
export PYTHONPATH="${ROOT}/mamba-main:${HARNESS_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT}/.cache/huggingface/datasets}"
mkdir -p "$(dirname "${USE_CACHE}")"

if [[ "${OFFLINE}" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
else
  unset TRANSFORMERS_OFFLINE || true
  unset HF_DATASETS_OFFLINE || true
fi

EXTRA_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${NUM_FEWSHOT}" ]]; then
  EXTRA_ARGS+=(--num_fewshot "${NUM_FEWSHOT}")
fi
if [[ "${CACHE_REQUESTS}" == "auto" ]]; then
  # true: use request cache if available, otherwise build once and save it.
  CACHE_REQUESTS="true"
fi
if [[ -n "${CACHE_REQUESTS}" && "${CACHE_REQUESTS}" != "off" ]]; then
  EXTRA_ARGS+=(--cache_requests "${CACHE_REQUESTS}")
fi
if [[ -n "${USE_CACHE}" && "${USE_CACHE}" != "off" ]]; then
  EXTRA_ARGS+=(--use_cache "${USE_CACHE}")
fi

echo "[run] adapter=${ADAPTER}"
if [[ -n "${FULL_CKPT}" ]]; then
  echo "[run] full_ckpt=${FULL_CKPT}"
fi
echo "[run] tasks=${TASKS}"
echo "[run] device=${DEVICE} batch=${BATCH_SIZE} max_length=${MAX_LENGTH} dtype=${DTYPE}"
if [[ -n "${NUM_FEWSHOT}" ]]; then
  echo "[run] num_fewshot=${NUM_FEWSHOT}"
fi
if [[ -n "${CACHE_REQUESTS}" ]]; then
  echo "[run] cache_requests=${CACHE_REQUESTS}"
fi
if [[ -n "${USE_CACHE}" ]]; then
  echo "[run] use_cache=${USE_CACHE}"
fi
echo "[run] output=${OUT_DIR}"

MODEL_ARGS="pretrained=${BASE_MODEL},tokenizer=${TOKENIZER_PATH},max_length=${MAX_LENGTH},dtype=${DTYPE}"
if [[ -n "${FULL_CKPT}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},full_ckpt=${FULL_CKPT}"
fi
if [[ -n "${ADAPTER}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},pq_adapter=${ADAPTER}"
fi

cd "${HARNESS_ROOT}"
"${PY}" "${WRAPPER}" run \
  --model mamba_ssm_pq \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/run.log"

echo "[done] ${OUT_DIR}"
