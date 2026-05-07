#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
SCRIPT_DIR="${ROOT}/mamba-main/motif"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_induction_checkpoints.py"

# Comma-separated absolute run dirs (each should contain config.json + model_*_latest.pt)
RUN_DIRS=${RUN_DIRS:-/workspace/motif/logs/induction_extrapolation/induction_short256_steps100000_20260422_145651,/workspace/motif/logs/induction_extrapolation/induction_short256_steps100000_20260422_162647,/workspace/motif/logs/induction_extrapolation/induction_short256_steps100000_20260422_185256}

DEVICE=${DEVICE:-cuda:0}
TEST_LENS=${TEST_LENS:-64,128,256,512,1024,2048,4096,8192,16384}
EVAL_BATCHES=${EVAL_BATCHES:-16}
MAX_EVAL_BATCH_SIZE=${MAX_EVAL_BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-0}
EVAL_TOKENS_BUDGET=${EVAL_TOKENS_BUDGET:-131072}
AMP=${AMP:-bf16}

OUT_DIR=${OUT_DIR:-${ROOT}/motif/logs/induction_extrapolation}
TS=$(date +%Y%m%d_%H%M%S)
OUT_CSV=${OUT_CSV:-${OUT_DIR}/retest_accuracy_by_length_to16384_${TS}.csv}

mkdir -p "${OUT_DIR}"

cd "${SCRIPT_DIR}"
PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" \
"${PY}" -u "${EVAL_SCRIPT}" \
  --run-dirs "${RUN_DIRS}" \
  --device "${DEVICE}" \
  --test-lens "${TEST_LENS}" \
  --eval-batches "${EVAL_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-tokens-budget "${EVAL_TOKENS_BUDGET}" \
  --max-eval-batch-size "${MAX_EVAL_BATCH_SIZE}" \
  --amp "${AMP}" \
  --out-csv "${OUT_CSV}"

MPLCONFIGDIR=/tmp/mpl \
"${PY}" - "${OUT_CSV}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path(sys.argv[1]).resolve()
out_png = csv_path.with_suffix(".png")
out_svg = csv_path.with_suffix(".svg")

df = pd.read_csv(csv_path)
df = df[df["status"] == "ok"].copy()
if df.empty:
    raise RuntimeError(f"No rows with status=ok in {csv_path}")

df["test_len"] = df["test_len"].astype(int)
df["acc"] = df["acc"].astype(float)

model_order = ["transformer", "mamba", "motifmamba", "rwkv"]
palette = {
    "transformer": "#1f77b4",
    "mamba": "#ff7f0e",
    "motifmamba": "#2ca02c",
    "rwkv": "#d62728",
}

plt.figure(figsize=(9, 5), dpi=160)
for model in model_order:
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

for model in sorted(set(df["model"]) - set(model_order)):
    sub = df[df["model"] == model].sort_values("test_len")
    plt.plot(sub["test_len"], sub["acc"], marker="o", linewidth=2, markersize=5, label=model)

xticks = sorted(df["test_len"].unique())
plt.xscale("log", base=2)
plt.xticks(xticks, [str(x) for x in xticks])
plt.ylim(0.0, 1.05)
plt.xlabel("Sequence Length (log2 scale)")
plt.ylabel("Induction Accuracy")
plt.title("Induction Extrapolation (up to 16384)")
plt.grid(True, which="both", linestyle="--", alpha=0.35)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(out_png)
plt.savefig(out_svg)

print(f"[plot] csv={csv_path}")
print(f"[plot] png={out_png}")
print(f"[plot] svg={out_svg}")
PY

echo "[done] csv=${OUT_CSV}"
echo "[done] png=${OUT_CSV%.csv}.png"
echo "[done] svg=${OUT_CSV%.csv}.svg"
