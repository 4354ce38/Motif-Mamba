#!/usr/bin/env bash
set -euo pipefail

# Evaluate base mamba130m vs motifmamba130m on Winogrande (winogr).

PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
ROOT=${ROOT:-/workspace}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
PQ_WRAPPER=${PQ_WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
MOTIF_ADAPTER=${MOTIF_ADAPTER:-${ROOT}/motif/logs/mamba130m_motif2_rank1_seq256_20M_kpq_1gpu}
TOKENIZER_PATH=${TOKENIZER_PATH:-/home/user/.cache/huggingface/hub/models--EleutherAI--gpt-neox-20b/snapshots/<snapshot_id>}

DEVICE=${DEVICE:-cuda:1}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}
LIMIT=${LIMIT:-0}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}

TASKS=${TASKS:-winogrande}
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/winogr_compare}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/mamba130m_vs_motif130m_winogr_${STAMP}"

BASE_OUT="${RUN_DIR}/base_mamba130m"
MOTIF_OUT="${RUN_DIR}/motifmamba130m"

mkdir -p "${BASE_OUT}" "${MOTIF_OUT}" "${ROOT}/.cache/lm_eval"
export PYTHONPATH="${ROOT}/mamba-main:${HARNESS_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT}/.cache/huggingface/datasets}"

EXTRA_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${NUM_FEWSHOT}" ]]; then
  EXTRA_ARGS+=(--num_fewshot "${NUM_FEWSHOT}")
fi

echo "[run] output_root=${RUN_DIR}"
echo "[run] task=${TASKS}"
echo "[run] device=${DEVICE}, batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}, dtype=${DTYPE}"

cd "${HARNESS_ROOT}"

echo "[run] base mamba130m ..."
"${PY}" -m lm_eval run \
  --model mamba_ssm \
  --model_args "pretrained=${BASE_MODEL},tokenizer=${TOKENIZER_PATH},max_length=${MAX_LENGTH},dtype=${DTYPE}" \
  --tasks "${TASKS}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --output_path "${BASE_OUT}" \
  --cache_requests true \
  --use_cache "${ROOT}/.cache/lm_eval/winogr_base_mamba130m.sqlite" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${BASE_OUT}/run.log"

echo "[run] motifmamba130m ..."
"${PY}" "${PQ_WRAPPER}" run \
  --model mamba_ssm_pq \
  --model_args "pretrained=${BASE_MODEL},pq_adapter=${MOTIF_ADAPTER},tokenizer=${TOKENIZER_PATH},max_length=${MAX_LENGTH},dtype=${DTYPE}" \
  --tasks "${TASKS}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --output_path "${MOTIF_OUT}" \
  --cache_requests true \
  --use_cache "${ROOT}/.cache/lm_eval/winogr_motifmamba130m.sqlite" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${MOTIF_OUT}/run.log"

COMPARE_METRIC=${COMPARE_METRIC:-acc_norm}
echo "[post] compare metric=${COMPARE_METRIC} (base wrong + motif right)"
"${PY}" - "${BASE_OUT}" "${MOTIF_OUT}" "${RUN_DIR}" "${COMPARE_METRIC}" <<'PY'
import csv
import glob
import json
import os
import sys

base_out, motif_out, run_dir, metric_pref = sys.argv[1:5]

def find_one_samples(out_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(out_dir, "**", "samples_*.jsonl"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No samples_*.jsonl found under: {out_dir}")
    return files[-1]

def pick_metric(d: dict, preferred: str) -> str:
    for m in (preferred, "acc_norm", "acc"):
        v = d.get(m)
        if isinstance(v, (int, float)):
            return m
    raise KeyError("No acc/acc_norm metric found in sample line")

def pred_idx_from_resps(d: dict) -> int:
    vals = []
    for x in d.get("filtered_resps", []):
        try:
            vals.append(float(x[0]))
        except Exception:
            vals.append(float("-inf"))
    if not vals:
        return -1
    return max(range(len(vals)), key=lambda i: vals[i])

def gold_idx_from_doc(d: dict) -> int:
    doc = d.get("doc", {})
    for k in ("gold", "label"):
        if k in doc:
            try:
                return int(doc[k])
            except Exception:
                pass
    try:
        return int(d.get("target"))
    except Exception:
        return -1

base_file = find_one_samples(base_out)
motif_file = find_one_samples(motif_out)

base_rows = {}
with open(base_file, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        base_rows[int(d["doc_id"])] = d

motif_rows = {}
with open(motif_file, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        motif_rows[int(d["doc_id"])] = d

common_ids = sorted(set(base_rows.keys()) & set(motif_rows.keys()))
rows = []
for doc_id in common_ids:
    b = base_rows[doc_id]
    m = motif_rows[doc_id]
    metric = pick_metric(b, metric_pref)
    b_ok = float(b.get(metric, 0.0))
    m_ok = float(m.get(metric, 0.0))
    if b_ok == 0.0 and m_ok == 1.0:
        q = b.get("doc", {}).get("query") or b.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
        choices = b.get("doc", {}).get("choices") or b.get("doc", {}).get("endings") or []
        gold = gold_idx_from_doc(b)
        b_pred = pred_idx_from_resps(b)
        m_pred = pred_idx_from_resps(m)
        rows.append({
            "doc_id": doc_id,
            "metric": metric,
            "query": q,
            "gold_idx": gold,
            "gold_choice": choices[gold] if 0 <= gold < len(choices) else "",
            "base_pred_idx": b_pred,
            "base_pred_choice": choices[b_pred] if 0 <= b_pred < len(choices) else "",
            "motif_pred_idx": m_pred,
            "motif_pred_choice": choices[m_pred] if 0 <= m_pred < len(choices) else "",
            "choices": choices,
        })

jsonl_out = os.path.join(run_dir, "base_wrong_motif_right.jsonl")
csv_out = os.path.join(run_dir, "base_wrong_motif_right.csv")
summary_out = os.path.join(run_dir, "compare_summary.txt")

with open(jsonl_out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(csv_out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "doc_id", "metric", "query", "gold_idx", "gold_choice",
        "base_pred_idx", "base_pred_choice", "motif_pred_idx", "motif_pred_choice",
    ])
    for r in rows:
        w.writerow([
            r["doc_id"], r["metric"], r["query"], r["gold_idx"], r["gold_choice"],
            r["base_pred_idx"], r["base_pred_choice"], r["motif_pred_idx"], r["motif_pred_choice"],
        ])

with open(summary_out, "w", encoding="utf-8") as f:
    f.write(f"base_samples={len(base_rows)}\n")
    f.write(f"motif_samples={len(motif_rows)}\n")
    f.write(f"common_samples={len(common_ids)}\n")
    f.write(f"base_wrong_motif_right={len(rows)}\n")
    f.write(f"metric={metric_pref}\n")
    f.write(f"base_file={base_file}\n")
    f.write(f"motif_file={motif_file}\n")
    f.write(f"jsonl_out={jsonl_out}\n")
    f.write(f"csv_out={csv_out}\n")

print(f"[post] common={len(common_ids)} base_wrong_motif_right={len(rows)}")
print(f"[post] jsonl={jsonl_out}")
print(f"[post] csv={csv_out}")
print(f"[post] summary={summary_out}")
PY

echo "[done] ${RUN_DIR}"
