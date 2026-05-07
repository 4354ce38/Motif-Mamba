#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float32}
SEED=${SEED:-42}

RANK_START=${RANK_START:-1}
RANK_END=${RANK_END:-3}
MOTIF_START=${MOTIF_START:-1}
MOTIF_END=${MOTIF_END:-13}

# Fixed-step timing experiment
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-200}
TRAIN_ONLY_LAYER=${TRAIN_ONLY_LAYER:-0}
MOTIF_LOSS_ONLY_LAYER=${MOTIF_LOSS_ONLY_LAYER:-0}

# Use the same params as rank1to4_motif1to13_uniform_20260428_182114 by default
PQ_K_INIT=${PQ_K_INIT:-1e-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_LR=${WARMUP_LR:-3e-3}
WARMUP_COEF=${WARMUP_COEF:-2e6}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-3e4}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}
MOTIF_LOSS_FORM=${MOTIF_LOSS_FORM:-mse}
MOTIF_LOG_EPS=${MOTIF_LOG_EPS:-1e-12}

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/time_impact_motif1to13_rank1to3_${TS}}
mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/time_summary.csv"

echo "rank,motif,status,wall_sec,steps_run,max_steps,reached,best_motif,init_motif,target_motif,train_elapsed_sec,run_dir" > "${SUMMARY_CSV}"

cd "${ROOT}/mamba-main"

echo "[run] out_root=${OUT_ROOT}"
echo "[run] ranks=${RANK_START}..${RANK_END}, motifs=${MOTIF_START}..${MOTIF_END}"
echo "[run] fixed warmup steps=${WARMUP_MAX_STEPS}, warmup_no_early_stop=1"

for ((rank=RANK_START; rank<=RANK_END; rank++)); do
  for ((motif=MOTIF_START; motif<=MOTIF_END; motif++)); do
    RUN_DIR="${OUT_ROOT}/rank_${rank}/motif${motif}"
    mkdir -p "${RUN_DIR}"

    echo "[run] rank=${rank}, motif=${motif}"

    cmd=(
      "${PY}" -u motif/train_kpq_motif_pile_lm.py
      --pretrained-dir "${PRETRAINED_DIR}"
      --pile-root "${PILE_ROOT}"
      --out-dir "${RUN_DIR}"
      --device "${DEVICE}"
      --dtype "${DTYPE}"
      --seed "${SEED}"
      --motif-class "${motif}"
      --pq-rank "${rank}"
      --no-pq-per-dim
      --train-only-layer "${TRAIN_ONLY_LAYER}"
      --motif-loss-only-layer "${MOTIF_LOSS_ONLY_LAYER}"
      --train-pq-k
      --pq-k-init "${PQ_K_INIT}"
      --warmup-only
      --warmup-no-early-stop
      --no-warmup-strict
      --warmup-max-steps "${WARMUP_MAX_STEPS}"
      --warmup-ratio "${WARMUP_RATIO}"
      --warmup-lr "${WARMUP_LR}"
      --warmup-coef "${WARMUP_COEF}"
      --motif-amplitude "${MOTIF_AMPLITUDE}"
      --motif-bias "${MOTIF_BIAS}"
      --motif-loss-form "${MOTIF_LOSS_FORM}"
      --motif-log-eps "${MOTIF_LOG_EPS}"
      --motif-per-dim-mode sampled_channel
      --motif-channel-samples 16
      --max-train-tokens 0
      --min-train-tokens 0
      --batch-size 1
      --seq-len 2048
    )

    t0=$(date +%s)
    if ! env -u RANK -u WORLD_SIZE -u LOCAL_RANK \
      PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" MPLCONFIGDIR=/tmp/mpl \
      "${cmd[@]}" > "${RUN_DIR}/run.log" 2>&1; then
      t1=$(date +%s)
      wall=$((t1 - t0))
      echo "${rank},${motif},run_failed,${wall},NaN,${WARMUP_MAX_STEPS},false,NaN,NaN,NaN,NaN,${RUN_DIR}" >> "${SUMMARY_CSV}"
      continue
    fi
    t1=$(date +%s)
    wall=$((t1 - t0))

    SUMM="${RUN_DIR}/warmup_summary.json"
    LOGCSV="${RUN_DIR}/train_log.csv"
    if [[ ! -f "${SUMM}" || ! -f "${LOGCSV}" ]]; then
      echo "${rank},${motif},missing_outputs,${wall},NaN,${WARMUP_MAX_STEPS},false,NaN,NaN,NaN,NaN,${RUN_DIR}" >> "${SUMMARY_CSV}"
      continue
    fi

    read -r steps_run reached best init target elapsed_sec <<EOF2
$(${PY} - "${SUMM}" "${LOGCSV}" <<'PY'
import json,sys
import pandas as pd
summ=json.load(open(sys.argv[1]))
df=pd.read_csv(sys.argv[2])
elapsed=float(df['elapsed_sec'].iloc[-1]) if len(df)>0 else float('nan')
print(
    int(summ.get('steps_run',0)),
    str(bool(summ.get('reached',False))).lower(),
    float(summ.get('best_motif',float('nan'))),
    float(summ.get('init_motif',float('nan'))),
    float(summ.get('target_motif',float('nan'))),
    elapsed,
)
PY
)
EOF2

    echo "${rank},${motif},ok,${wall},${steps_run},${WARMUP_MAX_STEPS},${reached},${best},${init},${target},${elapsed_sec},${RUN_DIR}" >> "${SUMMARY_CSV}"
  done
done

echo "[done] summary=${SUMMARY_CSV}"
