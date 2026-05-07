#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
ROOT=${ROOT:-/workspace}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-1.4b}
TOKENIZER_PATH=${TOKENIZER_PATH:-/home/user/.cache/huggingface/hub/models--EleutherAI--gpt-neox-20b/snapshots/<snapshot_id>}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
OFFLINE=${OFFLINE:-0}

MODEL_DIR_1=${MODEL_DIR_1:-${ROOT}/motif/logs/pile_mamba1.4b_motif2_kpq}
MODEL_DIR_2=${MODEL_DIR_2:-${ROOT}/motif/logs/pile_mamba1.4b_motif12_kpq_cuda1}
TASKS=${TASKS:-lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande}
OUT_ROOT=${OUT_ROOT:-${ROOT}/logs/recall_harness}

export PYTHONPATH="${ROOT}/mamba-main:${HARNESS_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT}/.cache/huggingface/datasets}"
mkdir -p "${OUT_ROOT}" "${HF_DATASETS_CACHE}"

if [[ "${OFFLINE}" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
else
  unset TRANSFORMERS_OFFLINE || true
  unset HF_DATASETS_OFFLINE || true
fi

run_one () {
  local model_dir="$1"
  local tag="$2"
  local out_dir="${OUT_ROOT}/${tag}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${out_dir}"
  echo "[run] tag=${tag}"
  echo "[run] model_dir=${model_dir}"
  echo "[run] output=${out_dir}"
  (
    cd "${HARNESS_ROOT}"
    "${PY}" "${WRAPPER}" run \
      --model mamba_ssm_pq \
      --model_args "pretrained=${BASE_MODEL},pq_adapter=${model_dir},tokenizer=${TOKENIZER_PATH},max_length=${MAX_LENGTH},dtype=float16" \
      --tasks "${TASKS}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE}" \
      --output_path "${out_dir}" \
      2>&1 | tee "${out_dir}/run.log"
  )
  echo "[done] ${tag}: ${out_dir}"
}

run_one "${MODEL_DIR_1}" "mamba1p4b_motif2_latest_core6"
run_one "${MODEL_DIR_2}" "mamba1p4b_motif12_latest_core6"
