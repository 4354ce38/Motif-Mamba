#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
SCRIPT_DIR="${ROOT}/mamba-main/motif"
BENCH="${SCRIPT_DIR}/induction_extrapolation_benchmark.py"

DEVICE=${DEVICE:-cuda:0}
MODELS=${MODELS:-lstm,rwkv}
TRAIN_LEN=${TRAIN_LEN:-256}
TEST_LENS=${TEST_LENS:-64,128,256,512,1024,2048,4096,8192,16384}
TRAIN_STEPS=${TRAIN_STEPS:-100000}
SEED=${SEED:-42}
AMP=${AMP:-bf16}

# <=0 means auto
BATCH_SIZE=${BATCH_SIZE:-0}
EVAL_BATCHES=${EVAL_BATCHES:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-0}
EVAL_TOKENS_BUDGET=${EVAL_TOKENS_BUDGET:-131072}
MAX_EVAL_BATCH_SIZE=${MAX_EVAL_BATCH_SIZE:-64}
MAX_TRANSFORMER_EVAL_LEN=${MAX_TRANSFORMER_EVAL_LEN:-16384}

OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/induction_extrapolation}
mkdir -p "${OUT_ROOT}"

cd "${SCRIPT_DIR}"
PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" \
"${PY}" -u "${BENCH}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --out-dir "${OUT_ROOT}" \
  --models "${MODELS}" \
  --train-len "${TRAIN_LEN}" \
  --test-lens "${TEST_LENS}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batches "${EVAL_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-tokens-budget "${EVAL_TOKENS_BUDGET}" \
  --max-eval-batch-size "${MAX_EVAL_BATCH_SIZE}" \
  --max-transformer-eval-len "${MAX_TRANSFORMER_EVAL_LEN}" \
  --amp "${AMP}" \
  --auto-match-params

RUN_DIR=$(ls -dt "${OUT_ROOT}"/induction_short"${TRAIN_LEN}"_steps"${TRAIN_STEPS}"_* 2>/dev/null | head -n 1 || true)
if [[ -z "${RUN_DIR}" ]]; then
  echo "[error] cannot find run dir under ${OUT_ROOT}"
  exit 1
fi

CSV_PATH="${RUN_DIR}/accuracy_by_length.csv"
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[error] missing ${CSV_PATH}"
  exit 1
fi

MPLCONFIGDIR=/tmp/mpl \
"${PY}" - "${CSV_PATH}" <<'PY'
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

csv_path = Path(sys.argv[1]).resolve()
df = pd.read_csv(csv_path)
df = df[df["status"] == "ok"].copy()
if df.empty:
    raise RuntimeError(f"No rows with status=ok in {csv_path}")

df["test_len"] = df["test_len"].astype(int)
df["acc"] = df["acc"].astype(float)

out_png = csv_path.with_name("accuracy_by_length_plot.png")
out_svg = csv_path.with_name("accuracy_by_length_plot.svg")

order = ["lstm", "rwkv"]
palette = {"lstm": "#9467bd", "rwkv": "#d62728"}

plt.figure(figsize=(9, 5), dpi=160)
for model in order:
    sub = df[df["model"] == model].sort_values("test_len")
    if sub.empty:
        continue
    plt.plot(
        sub["test_len"],
        sub["acc"],
        marker="o",
        linewidth=2,
        markersize=5,
        label=model,
        color=palette.get(model),
    )

for model in sorted(set(df["model"]) - set(order)):
    sub = df[df["model"] == model].sort_values("test_len")
    plt.plot(sub["test_len"], sub["acc"], marker="o", linewidth=2, markersize=5, label=model)

xticks = sorted(df["test_len"].unique())
plt.xscale("log", base=2)
plt.xticks(xticks, [str(x) for x in xticks])
plt.ylim(0.0, 1.05)
plt.xlabel("Sequence Length (log2 scale)")
plt.ylabel("Induction Accuracy")
plt.title("Induction Extrapolation: LSTM vs RWKV")
plt.grid(True, which="both", linestyle="--", alpha=0.35)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(out_png)
plt.savefig(out_svg)

print(f"[plot] csv={csv_path}")
print(f"[plot] png={out_png}")
print(f"[plot] svg={out_svg}")
PY

echo "[done] run_dir=${RUN_DIR}"
echo "[done] csv=${CSV_PATH}"
echo "[done] png=${RUN_DIR}/accuracy_by_length_plot.png"
echo "[done] svg=${RUN_DIR}/accuracy_by_length_plot.svg"
