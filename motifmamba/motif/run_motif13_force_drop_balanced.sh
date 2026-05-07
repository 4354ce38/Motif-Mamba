#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
OUT_DIR=${OUT_DIR:-${ROOT}/motif/logs/motif13_force_drop_balanced_$(date +%Y%m%d_%H%M%S)}

DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}
PQ_RANK=${PQ_RANK:-1}
TRAIN_ONLY_LAYER=${TRAIN_ONLY_LAYER:-0}
MOTIF13_TARGET=${MOTIF13_TARGET:-0.25}

mkdir -p "${OUT_DIR}"
cd "${ROOT}/mamba-main"

# 1) short probe to get initial motif_1..13 observation as baseline
PROBE_DIR="${OUT_DIR}/_probe_init"
mkdir -p "${PROBE_DIR}"
PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" MPLCONFIGDIR=/tmp/mpl \
"${PY}" -u motif/train_kpq_motif_pile_lm.py \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${PROBE_DIR}" \
  --device "${DEVICE}" \
  --dtype float32 \
  --seed "${SEED}" \
  --motif-class 13 \
  --pq-rank "${PQ_RANK}" \
  --no-pq-per-dim \
  --train-only-layer "${TRAIN_ONLY_LAYER}" \
  --motif-loss-only-layer "${TRAIN_ONLY_LAYER}" \
  --warmup-only \
  --no-warmup-strict \
  --warmup-max-steps 1 \
  --warmup-ratio 0.1 \
  --warmup-lr 1e-3 \
  --max-train-tokens 0 \
  --min-train-tokens 0 \
  --batch-size 1 \
  --seq-len 2048 >/dev/null 2>&1 || true

export PROBE_DIR
export MOTIF13_TARGET
TARGETS=$("${PY}" - <<'PY'
import pandas as pd
from pathlib import Path
p=Path("/workspace")
# this path is passed from env by shell substitution below
import os
probe_dir=os.environ['PROBE_DIR']
motif13_target=float(os.environ['MOTIF13_TARGET'])
log=Path(probe_dir)/'train_log.csv'
df=pd.read_csv(log)
row=df.iloc[0]
vals=[]
for i in range(1,13):
    vals.append(float(row[f'motif_{i}']))
vals.append(motif13_target)
print(','.join(f'{v:.12g}' for v in vals))
PY
)

# Keep 1..12 stability, strongly push 13
WEIGHTS=${WEIGHTS:-"8,8,8,8,8,8,8,8,8,8,8,8,80"}

echo "[run] targets=${TARGETS}"
echo "[run] weights=${WEIGHTS}"

# 2) balanced run
PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" MPLCONFIGDIR=/tmp/mpl \
"${PY}" -u motif/train_kpq_motif_pile_lm.py \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype float32 \
  --seed "${SEED}" \
  --motif-target-values "${TARGETS}" \
  --motif-loss-weights "${WEIGHTS}" \
  --motif-loss-form log_mse \
  --motif-log-eps 1e-18 \
  --pq-rank "${PQ_RANK}" \
  --no-pq-per-dim \
  --train-only-layer "${TRAIN_ONLY_LAYER}" \
  --motif-loss-only-layer "${TRAIN_ONLY_LAYER}" \
  --train-pq-k \
  --pq-k-init 1.0 \
  --warmup-only \
  --no-warmup-strict \
  --warmup-max-steps 3000 \
  --warmup-ratio 0.1 \
  --warmup-lr 3e-2 \
  --warmup-coef 1e7 \
  --motif-amplitude 300 \
  --motif-bias 0.0 \
  --motif-per-dim-mode sampled_channel \
  --motif-channel-samples 16 \
  --max-train-tokens 0 \
  --min-train-tokens 0 \
  --batch-size 1 \
  --seq-len 2048

echo "[done] out_dir=${OUT_DIR}"
