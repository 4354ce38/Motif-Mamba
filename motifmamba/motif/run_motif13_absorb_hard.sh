#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
DEVICE=${DEVICE:-cuda:0}

# We optimize absolute motif_13 observation, not just loss ratio.
TARGET_OBS=${TARGET_OBS:-0.20}   # set 0.25 if you want stricter goal
RANK_LIST=${RANK_LIST:-1,2,4,8}
SEEDS=${SEEDS:-42,43,44,45,46}

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/motif13_absorb_hard_${TS}}
mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/summary.csv"
echo "rank,seed,best_motif13,last_motif13,run_dir" > "${SUMMARY}"

cd "${ROOT}/mamba-main"

best_global_obs=0
best_global_dir=""
best_global_rank=""
best_global_seed=""

IFS=',' read -r -a RANK_ARR <<< "${RANK_LIST}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

for rank in "${RANK_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    RUN_DIR="${OUT_ROOT}/rank${rank}_seed${seed}"
    mkdir -p "${RUN_DIR}"

    echo "[run] rank=${rank}, seed=${seed}, out=${RUN_DIR}"

    # Single-objective motif13 push
    PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" MPLCONFIGDIR=/tmp/mpl \
    "${PY}" -u motif/train_kpq_motif_pile_lm.py \
      --pretrained-dir "${PRETRAINED_DIR}" \
      --pile-root "${PILE_ROOT}" \
      --out-dir "${RUN_DIR}" \
      --device "${DEVICE}" \
      --dtype float32 \
      --seed "${seed}" \
      --motif-class 13 \
      --motif-target 0.25 \
      --pq-rank "${rank}" \
      --no-pq-per-dim \
      --train-only-layer 0 \
      --motif-loss-only-layer 0 \
      --train-pq-k \
      --pq-k-init 1.0 \
      --warmup-only \
      --no-warmup-strict \
      --warmup-max-steps 3000 \
      --warmup-ratio 0.01 \
      --warmup-lr 5e-2 \
      --warmup-coef 1e7 \
      --motif-amplitude 300 \
      --motif-bias 0.0 \
      --motif-loss-form log_mse \
      --motif-log-eps 1e-18 \
      --motif-loss-weights "1,1,1,1,1,1,1,1,1,1,1,1,300" \
      --motif-per-dim-mode sampled_channel \
      --motif-channel-samples 16 \
      --max-train-tokens 0 \
      --min-train-tokens 0 \
      --batch-size 1 \
      --seq-len 2048 > "${RUN_DIR}/run.log" 2>&1 || true

    read -r best_obs last_obs <<EOF2
$(${PY} - "${RUN_DIR}/train_log.csv" <<'PY'
import sys
import pandas as pd
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print('0 0')
    raise SystemExit

df=pd.read_csv(p)
if 'phase' in df.columns:
    df=df[df['phase']=='warmup']
if df.empty or 'motif_13' not in df.columns:
    print('0 0')
else:
    y=df['motif_13'].astype(float)
    print(float(y.max()), float(y.iloc[-1]))
PY
)
EOF2

    echo "${rank},${seed},${best_obs},${last_obs},${RUN_DIR}" >> "${SUMMARY}"
    echo "[obs] best_motif13=${best_obs}, last_motif13=${last_obs}"

    awk_best=$(awk -v a="$best_obs" -v b="$best_global_obs" 'BEGIN{print (a>b)?1:0}')
    if [[ "$awk_best" == "1" ]]; then
      best_global_obs="$best_obs"
      best_global_dir="$RUN_DIR"
      best_global_rank="$rank"
      best_global_seed="$seed"
    fi

    hit=$(awk -v a="$best_obs" -v t="$TARGET_OBS" 'BEGIN{print (a>=t)?1:0}')
    if [[ "$hit" == "1" ]]; then
      echo "[hit] reached TARGET_OBS=${TARGET_OBS} at rank=${rank}, seed=${seed}"
      echo "[best] dir=${RUN_DIR}"
      echo "[best] summary=${SUMMARY}"
      exit 0
    fi
  done

done

echo "[done] no run reached TARGET_OBS=${TARGET_OBS}"
echo "[best] obs=${best_global_obs}, rank=${best_global_rank}, seed=${best_global_seed}"
echo "[best] dir=${best_global_dir}"
echo "[done] summary=${SUMMARY}"
exit 2
