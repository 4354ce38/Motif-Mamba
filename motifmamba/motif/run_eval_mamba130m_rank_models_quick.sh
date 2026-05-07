#!/usr/bin/env bash
set -euo pipefail

# Batch quick lm-eval for all rank models under motif/logs/rank.
# It scans */pq_adapter_latest.pt and runs a unified LM task subset.

ROOT=${ROOT:-/workspace}
RANK_ROOT=${RANK_ROOT:-${ROOT}/motif/logs/rank}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}

TASKS=${TASKS:-lambada_openai,hellaswag,piqa,arc_easy,winogrande}
LIMIT=${LIMIT:-0}
NUM_FEWSHOT=${NUM_FEWSHOT:-}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}

OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/lm_eval}
BATCH_TAG=${BATCH_TAG:-mamba130m_rank_quick}
STAMP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="${OUT_ROOT}/${BATCH_TAG}_${STAMP}"
RUNS_CSV="${BATCH_DIR}/runs.csv"
MODEL_SUMMARY_CSV="${BATCH_DIR}/model_summary.csv"

# Optional: limit number of adapters for smoke test, e.g. MAX_MODELS=3
MAX_MODELS=${MAX_MODELS:-0}

if [[ ! -d "${RANK_ROOT}" ]]; then
  echo "[error] rank directory not found: ${RANK_ROOT}"
  exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not found: ${EVAL_SCRIPT}"
  exit 1
fi

mkdir -p "${BATCH_DIR}"
echo "model,adapter,task,metric,value,results_json" > "${RUNS_CSV}"
echo "model,mean_primary_metric,results_json" > "${MODEL_SUMMARY_CSV}"

mapfile -t ADAPTERS < <(find "${RANK_ROOT}" -maxdepth 2 -type f -name 'pq_adapter_latest.pt' | sort)
if [[ ${#ADAPTERS[@]} -eq 0 ]]; then
  echo "[error] no pq_adapter_latest.pt found under ${RANK_ROOT}"
  exit 1
fi

count=0
for adapter in "${ADAPTERS[@]}"; do
  count=$((count + 1))
  if [[ "${MAX_MODELS}" != "0" && "${count}" -gt "${MAX_MODELS}" ]]; then
    break
  fi

  model_dir="$(dirname "${adapter}")"
  model_name="$(basename "${model_dir}")"
  tag="${BATCH_TAG}_${model_name}"
  use_cache="${ROOT}/.cache/lm_eval/${tag}.sqlite"

  echo "[run ${count}] ${model_name}"

  ROOT="${ROOT}" \
  BASE_MODEL="${BASE_MODEL}" \
  HARNESS_ROOT="${HARNESS_ROOT}" \
  WRAPPER="${WRAPPER}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  ADAPTER="${adapter}" \
  DEVICE="${DEVICE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  DTYPE="${DTYPE}" \
  TASKS="${TASKS}" \
  TAG="${tag}" \
  OUT_ROOT="${OUT_ROOT}" \
  LIMIT="${LIMIT}" \
  NUM_FEWSHOT="${NUM_FEWSHOT}" \
  CACHE_REQUESTS="${CACHE_REQUESTS}" \
  USE_CACHE="${use_cache}" \
  bash "${EVAL_SCRIPT}"

  out_dir="$(ls -dt "${OUT_ROOT}/${tag}"_* 2>/dev/null | head -n1 || true)"
  if [[ -z "${out_dir}" ]]; then
    echo "[warn] no output directory found for ${model_name}, skip summary"
    continue
  fi

  results_json="$(find "${out_dir}" -type f -name 'results_*.json' | head -n1 || true)"
  if [[ -z "${results_json}" ]]; then
    echo "[warn] no results json found for ${model_name}, skip summary"
    continue
  fi

  python - "${results_json}" "${model_name}" "${adapter}" "${RUNS_CSV}" "${MODEL_SUMMARY_CSV}" <<'PY'
import csv
import json
import math
import sys

results_json, model_name, adapter, runs_csv, summary_csv = sys.argv[1:6]
d = json.load(open(results_json, "r"))
res = d.get("results", {})

preferred = ["acc,none", "exact_match,none", "em,none", "f1,none"]
vals = []
rows = []

for task, td in res.items():
    metric = None
    value = None
    for k in preferred:
        v = td.get(k)
        if isinstance(v, (int, float)):
            metric, value = k, float(v)
            break
    if metric is None:
        for k, v in td.items():
            if "_stderr" in k:
                continue
            if isinstance(v, (int, float)):
                metric, value = k, float(v)
                break
    if metric is None:
        continue
    vals.append(value)
    rows.append((model_name, adapter, task, metric, value, results_json))

with open(runs_csv, "a", newline="") as f:
    w = csv.writer(f)
    for r in rows:
        w.writerow(r)

mean_v = sum(vals) / len(vals) if vals else float("nan")
with open(summary_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([model_name, mean_v if math.isfinite(mean_v) else "nan", results_json])
PY
done

echo "[done] batch_dir=${BATCH_DIR}"
echo "[done] runs_csv=${RUNS_CSV}"
echo "[done] model_summary_csv=${MODEL_SUMMARY_CSV}"
