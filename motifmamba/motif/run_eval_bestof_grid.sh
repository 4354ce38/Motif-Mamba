#!/usr/bin/env bash
set -euo pipefail

# Grid-evaluate candidates over few-shot and max_length, then report best configs.
#
# Candidate can be:
# 1) a training log dir containing pq_adapter_latest.pt and/or full_model_latest.pt
# 2) a direct path to pq_adapter_latest.pt
# 3) a direct path to full_model_latest.pt
#
# Example:
#   CANDIDATES="${ROOT}/motif/logs/runA,${ROOT}/motif/logs/runB" \
#   FEWSHOTS="0,5" \
#   MAX_LENGTHS="2048,3072" \
#   bash run_eval_bestof_grid.sh

ROOT=${ROOT:-/workspace}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}

# Comma-separated list of candidate paths.
CANDIDATES=${CANDIDATES:-${ROOT}/motif/logs/pile_mamba130m_motifoff_rank2_kpq_1M_cuda1}
FEWSHOTS=${FEWSHOTS:-0,5}
MAX_LENGTHS=${MAX_LENGTHS:-2048,3072}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}

DEVICE=${DEVICE:-cuda:1}
BATCH_SIZE=${BATCH_SIZE:-8}
DTYPE=${DTYPE:-float16}
TASKS=${TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande}

# Response cache is model-agnostic in lm-eval. Keep it off by default to avoid cross-model leakage.
USE_CACHE=${USE_CACHE:-off}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}
LIMIT=${LIMIT:-0}

OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/lm_eval}
GRID_TAG=${GRID_TAG:-mamba130m_bestof_grid}
STAMP=$(date +%Y%m%d_%H%M%S)
GRID_DIR="${OUT_ROOT}/${GRID_TAG}_${STAMP}"
RUNS_CSV="${GRID_DIR}/runs.csv"
BEST_TASK_CSV="${GRID_DIR}/best_by_task.csv"
BEST_OVERALL_JSON="${GRID_DIR}/best_overall.json"
README_TXT="${GRID_DIR}/README.txt"

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not executable: ${EVAL_SCRIPT}"
  exit 1
fi

mkdir -p "${GRID_DIR}"

split_csv() {
  local value="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "${value}"
}

resolve_candidate() {
  local cand="$1"
  local full_ckpt=""
  local adapter=""

  if [[ -d "${cand}" ]]; then
    if [[ -f "${cand}/full_model_latest.pt" ]]; then
      full_ckpt="${cand}/full_model_latest.pt"
    fi
    if [[ -f "${cand}/pq_adapter_latest.pt" ]]; then
      adapter="${cand}/pq_adapter_latest.pt"
    fi
  elif [[ -f "${cand}" ]]; then
    case "$(basename "${cand}")" in
      *full_model*.pt) full_ckpt="${cand}" ;;
      *pq_adapter*.pt) adapter="${cand}" ;;
      *)
        # fallback: treat unknown .pt as full checkpoint
        full_ckpt="${cand}"
        ;;
    esac
  else
    echo "[error] candidate not found: ${cand}" >&2
    return 1
  fi

  if [[ -z "${full_ckpt}" && -z "${adapter}" ]]; then
    echo "[error] no usable checkpoint in candidate: ${cand}" >&2
    return 1
  fi

  printf '%s\t%s\n' "${full_ckpt}" "${adapter}"
}

json_metric_extract_py() {
  local results_json="$1"
  python - "${results_json}" <<'PY'
import json, os, sys

path = sys.argv[1]
data = json.load(open(path, "r"))
results = data.get("results", {})

preferred = ["acc,none", "exact_match,none", "em,none", "f1,none"]
rows = []
vals = []

for task, td in results.items():
    chosen_key = None
    chosen_val = None
    for k in preferred:
        v = td.get(k)
        if isinstance(v, (int, float)):
            chosen_key, chosen_val = k, float(v)
            break
    if chosen_key is None:
        for k, v in td.items():
            if "_stderr" in k:
                continue
            if isinstance(v, (int, float)):
                chosen_key, chosen_val = k, float(v)
                break
    if chosen_key is None:
        continue
    rows.append((task, chosen_key, chosen_val))
    vals.append(chosen_val)

overall = sum(vals) / len(vals) if vals else float("nan")
print(f"OVERALL_MEAN\t{overall}")
for task, metric, val in rows:
    print(f"TASK\t{task}\t{metric}\t{val}")
PY
}

mapfile -t CANDS_ARR < <(printf "%s\n" "${CANDIDATES}" | tr ',' '\n' | sed '/^\s*$/d')
split_csv "${FEWSHOTS}" FEWSHOT_ARR
split_csv "${MAX_LENGTHS}" MAXLEN_ARR

if [[ ${#CANDS_ARR[@]} -eq 0 ]]; then
  echo "[error] no candidates specified"
  exit 1
fi

echo "run_id,candidate,full_ckpt,adapter,fewshot,max_length,task,metric,value,overall_mean,results_json" > "${RUNS_CSV}"

run_id=0
for cand in "${CANDS_ARR[@]}"; do
  resolved="$(resolve_candidate "${cand}")"
  full_ckpt="$(printf '%s' "${resolved}" | cut -f1)"
  adapter="$(printf '%s' "${resolved}" | cut -f2)"

  for fewshot in "${FEWSHOT_ARR[@]}"; do
    fewshot="$(echo "${fewshot}" | xargs)"
    for maxlen in "${MAXLEN_ARR[@]}"; do
      maxlen="$(echo "${maxlen}" | xargs)"
      run_id=$((run_id + 1))

      run_tag="${GRID_TAG}_r${run_id}_fs${fewshot}_ml${maxlen}"
      if [[ -n "${adapter}" ]]; then
        run_tag="${run_tag}_$(basename "$(dirname "${adapter}")")"
      elif [[ -n "${full_ckpt}" ]]; then
        run_tag="${run_tag}_$(basename "$(dirname "${full_ckpt}")")"
      fi

      echo "[run ${run_id}] candidate=${cand} fewshot=${fewshot} max_length=${maxlen}"

      ROOT="${ROOT}" \
      BASE_MODEL="${BASE_MODEL}" \
      HARNESS_ROOT="${HARNESS_ROOT}" \
      WRAPPER="${WRAPPER}" \
      TOKENIZER_PATH="${TOKENIZER_PATH}" \
      FULL_CKPT="${full_ckpt}" \
      ADAPTER="${adapter}" \
      DEVICE="${DEVICE}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      MAX_LENGTH="${maxlen}" \
      DTYPE="${DTYPE}" \
      LIMIT="${LIMIT}" \
      NUM_FEWSHOT="${fewshot}" \
      CACHE_REQUESTS="${CACHE_REQUESTS}" \
      USE_CACHE="${USE_CACHE}" \
      TASKS="${TASKS}" \
      TAG="${run_tag}" \
      OUT_ROOT="${OUT_ROOT}" \
      bash "${EVAL_SCRIPT}"

      out_dir="$(ls -dt "${OUT_ROOT}/${run_tag}"_* 2>/dev/null | head -n1 || true)"
      if [[ -z "${out_dir}" ]]; then
        echo "[error] cannot find output directory for ${run_tag}"
        exit 1
      fi

      results_json="$(find "${out_dir}" -type f -name 'results_*.json' | head -n1 || true)"
      if [[ -z "${results_json}" ]]; then
        echo "[error] cannot find results json in ${out_dir}"
        exit 1
      fi

      overall_mean="nan"
      while IFS=$'\t' read -r tag c1 c2 c3; do
        if [[ "${tag}" == "OVERALL_MEAN" ]]; then
          overall_mean="${c1}"
        elif [[ "${tag}" == "TASK" ]]; then
          task="${c1}"
          metric="${c2}"
          value="${c3}"
          echo "${run_id},${cand},${full_ckpt},${adapter},${fewshot},${maxlen},${task},${metric},${value},${overall_mean},${results_json}" >> "${RUNS_CSV}"
        fi
      done < <(json_metric_extract_py "${results_json}")
    done
  done
done

python - "${RUNS_CSV}" "${BEST_TASK_CSV}" "${BEST_OVERALL_JSON}" <<'PY'
import csv, json, math, sys

runs_csv, best_task_csv, best_overall_json = sys.argv[1:4]

rows = list(csv.DictReader(open(runs_csv, "r")))
if not rows:
    raise SystemExit("No rows in runs.csv")

# Best by task (max value).
best_task = {}
for r in rows:
    task = r["task"]
    try:
        v = float(r["value"])
    except ValueError:
        continue
    if task not in best_task or v > best_task[task][0]:
        best_task[task] = (v, r)

with open(best_task_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "best_value", "metric", "run_id", "candidate", "fewshot", "max_length", "results_json"])
    for task in sorted(best_task):
        v, r = best_task[task]
        w.writerow([task, v, r["metric"], r["run_id"], r["candidate"], r["fewshot"], r["max_length"], r["results_json"]])

# Best overall by run_id (all rows in one run share overall_mean).
best = None
seen = set()
for r in rows:
    rid = r["run_id"]
    if rid in seen:
        continue
    seen.add(rid)
    try:
        m = float(r["overall_mean"])
    except ValueError:
        continue
    if not math.isfinite(m):
        continue
    if best is None or m > best[0]:
        best = (m, r)

if best is None:
    payload = {"error": "No finite overall_mean found"}
else:
    m, r = best
    payload = {
        "best_overall_mean": m,
        "run_id": r["run_id"],
        "candidate": r["candidate"],
        "full_ckpt": r["full_ckpt"],
        "adapter": r["adapter"],
        "fewshot": int(r["fewshot"]),
        "max_length": int(r["max_length"]),
        "results_json": r["results_json"],
    }

json.dump(payload, open(best_overall_json, "w"), indent=2)
print(json.dumps(payload, indent=2))
PY

cat > "${README_TXT}" <<EOF
Best-of grid finished.

Outputs:
- ${RUNS_CSV}
- ${BEST_TASK_CSV}
- ${BEST_OVERALL_JSON}

Notes:
- overall_mean is the arithmetic mean over each task's primary scalar metric.
- by default USE_CACHE=off to avoid cross-model response-cache leakage.
- request cache for prompt/context build is controlled by CACHE_REQUESTS=${CACHE_REQUESTS}.
EOF

echo "[done] ${GRID_DIR}"
