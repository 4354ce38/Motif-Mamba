#!/usr/bin/env bash
set -euo pipefail

# Evaluate base mamba130m vs motifmamba130m on LAMBADA (LAMB).

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

TASKS=${TASKS:-lambada_openai}
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/lamb_compare}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUT_ROOT}/mamba130m_vs_motif130m_lamb_${STAMP}"

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
  --use_cache "${ROOT}/.cache/lm_eval/lamb_base_mamba130m.sqlite" \
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
  --use_cache "${ROOT}/.cache/lm_eval/lamb_motifmamba130m.sqlite" \
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

def get_scores(d: dict) -> list[float]:
    vals = []
    for x in d.get("filtered_resps", []):
        try:
            vals.append(float(x[0]))
        except Exception:
            vals.append(float("-inf"))
    return vals

def pred_idx_from_scores(vals: list[float]) -> int:
    if not vals:
        return -1
    return max(range(len(vals)), key=lambda i: vals[i])

def extract_choices(doc: dict) -> tuple[list[str], list[str]]:
    # labels, texts
    if "sol1" in doc and "sol2" in doc:
        return ["0", "1"], [str(doc.get("sol1", "")), str(doc.get("sol2", ""))]
    ch = doc.get("choices")
    if isinstance(ch, dict):
        labels = [str(x) for x in ch.get("label", [])]
        texts = [str(x) for x in ch.get("text", [])]
        if texts:
            if not labels or len(labels) != len(texts):
                labels = [str(i) for i in range(len(texts))]
            return labels, texts
    if isinstance(ch, list):
        if ch and isinstance(ch[0], dict):
            labels = [str(x.get("label", i)) for i, x in enumerate(ch)]
            texts = [str(x.get("text", "")) for x in ch]
            return labels, texts
        return [str(i) for i in range(len(ch))], [str(x) for x in ch]
    endings = doc.get("endings")
    if isinstance(endings, list):
        return [str(i) for i in range(len(endings))], [str(x) for x in endings]
    opt = []
    for k in ("option1", "option2", "option3", "option4", "option5"):
        if k in doc:
            opt.append(str(doc[k]))
    if opt:
        return [str(i) for i in range(len(opt))], opt
    return [], []

def extract_question(doc: dict, sample: dict) -> str:
    q = doc.get("goal")
    if isinstance(q, str) and q.strip():
        return q.strip()
    q = doc.get("text")
    if isinstance(q, str) and q.strip():
        return q.strip()
    q = doc.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()
    q = doc.get("sentence")
    if isinstance(q, str) and q.strip():
        return q.strip()
    q = doc.get("question")
    if isinstance(q, dict):
        stem = q.get("stem")
        if isinstance(stem, str) and stem.strip():
            return stem.strip()
    if isinstance(q, str) and q.strip():
        return q.strip()
    arg0 = sample.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
    if isinstance(arg0, str):
        return arg0.strip()
    return ""

def extract_gold_idx(doc: dict, sample: dict, labels: list[str], n_choices: int) -> int:
    for k in ("gold", "label"):
        if k in doc:
            try:
                v = int(doc[k])
                if 0 <= v < n_choices:
                    return v
            except Exception:
                pass
    ans = doc.get("answer")
    if ans is not None:
        s = str(ans).strip()
        if s.isdigit():
            v = int(s)
            if 0 <= v < n_choices:
                return v
            if 1 <= v <= n_choices:
                return v - 1
    ans_key = doc.get("answerKey")
    if ans_key is not None and labels:
        s = str(ans_key).strip()
        if s in labels:
            return labels.index(s)
    tgt = sample.get("target")
    if isinstance(tgt, str) and labels and tgt.strip() in labels:
        return labels.index(tgt.strip())
    try:
        v = int(sample.get("target"))
        if 0 <= v < n_choices:
            return v
        if 1 <= v <= n_choices:
            return v - 1
    except Exception:
        pass
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
        doc = b.get("doc", {})
        labels, choices = extract_choices(doc)
        q = extract_question(doc, b)
        gold = extract_gold_idx(doc, b, labels, len(choices))
        b_scores = get_scores(b)
        m_scores = get_scores(m)
        b_pred = pred_idx_from_scores(b_scores)
        m_pred = pred_idx_from_scores(m_scores)
        rows.append({
            "doc_id": doc_id,
            "metric": metric,
            "query": q,
            "choices": choices,
            "gold_idx": gold,
            "gold_choice": choices[gold] if 0 <= gold < len(choices) else str(b.get("target", "")).strip(),
            "base_pred_idx": b_pred,
            "base_pred_choice": choices[b_pred] if 0 <= b_pred < len(choices) else "",
            "motif_pred_idx": m_pred,
            "motif_pred_choice": choices[m_pred] if 0 <= m_pred < len(choices) else "",
            "base_scores": b_scores,
            "motif_scores": m_scores,
            "target_text": str(b.get("target", "")).strip(),
            "base_acc": b_ok,
            "motif_acc": m_ok,
        })

jsonl_out = os.path.join(run_dir, "base_wrong_motif_right.jsonl")
csv_out = os.path.join(run_dir, "base_wrong_motif_right.csv")
detail_jsonl = os.path.join(run_dir, "base_wrong_motif_right_detailed.jsonl")
detail_csv = os.path.join(run_dir, "base_wrong_motif_right_detailed.csv")
detail_txt = os.path.join(run_dir, "base_wrong_motif_right_detailed.txt")
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

with open(detail_jsonl, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

max_n = max((len(r["choices"]) for r in rows), default=0)
with open(detail_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    header = [
        "doc_id", "metric", "query",
        "target_text",
        "gold_idx", "gold_choice",
        "base_pred_idx", "base_pred_choice",
        "motif_pred_idx", "motif_pred_choice",
        "base_acc", "motif_acc",
    ]
    header += [f"choice_{i}" for i in range(max_n)]
    header += [f"base_score_{i}" for i in range(max_n)]
    header += [f"motif_score_{i}" for i in range(max_n)]
    w.writerow(header)
    for r in rows:
        c = r["choices"]
        bs = r["base_scores"]
        ms = r["motif_scores"]
        row = [
            r["doc_id"], r["metric"], r["query"],
            r["target_text"],
            r["gold_idx"], r["gold_choice"],
            r["base_pred_idx"], r["base_pred_choice"],
            r["motif_pred_idx"], r["motif_pred_choice"],
            r["base_acc"], r["motif_acc"],
        ]
        row += c + [""] * (max_n - len(c))
        row += [bs[i] if i < len(bs) else "" for i in range(max_n)]
        row += [ms[i] if i < len(ms) else "" for i in range(max_n)]
        w.writerow(row)

with open(detail_txt, "w", encoding="utf-8") as f:
    for k, r in enumerate(rows, 1):
        f.write(f"[{k}] doc_id={r['doc_id']}\n")
        f.write(f"Q: {r['query']}\n")
        if r["choices"]:
            for i, ch in enumerate(r["choices"]):
                f.write(f"  {i}) {ch}\n")
            f.write(f"Gold: {r['gold_idx']} -> {r['gold_choice']}\n")
            f.write(f"Mamba: {r['base_pred_idx']} -> {r['base_pred_choice']} (scores: {r['base_scores']}, acc={r['base_acc']})\n")
            f.write(f"MotifMamba: {r['motif_pred_idx']} -> {r['motif_pred_choice']} (scores: {r['motif_scores']}, acc={r['motif_acc']})\n\n")
        else:
            f.write(f"Target: {r['target_text']}\n")
            f.write(f"Mamba: score={r['base_scores'][0] if r['base_scores'] else ''}, acc={r['base_acc']}\n")
            f.write(f"MotifMamba: score={r['motif_scores'][0] if r['motif_scores'] else ''}, acc={r['motif_acc']}\n\n")

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
    f.write(f"detail_jsonl={detail_jsonl}\n")
    f.write(f"detail_csv={detail_csv}\n")
    f.write(f"detail_txt={detail_txt}\n")

print(f"[post] common={len(common_ids)} base_wrong_motif_right={len(rows)}")
print(f"[post] jsonl={jsonl_out}")
print(f"[post] csv={csv_out}")
print(f"[post] detail_jsonl={detail_jsonl}")
print(f"[post] detail_csv={detail_csv}")
print(f"[post] detail_txt={detail_txt}")
print(f"[post] summary={summary_out}")
PY

echo "[done] ${RUN_DIR}"
