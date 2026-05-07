#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
SEED=${SEED:-42}

START_RANK=${START_RANK:-1}
END_RANK=${END_RANK:-16}
MOTIF_START=${MOTIF_START:-1}
MOTIF_END=${MOTIF_END:-13}

# Target criterion: motif_loss <= init_motif * WARMUP_RATIO
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-3000}

# Small-experiment defaults: train single-layer PQ, shared PQ, warmup-only
TRAIN_ONLY_LAYER=${TRAIN_ONLY_LAYER:-0}
MOTIF_LOSS_ONLY_LAYER=${MOTIF_LOSS_ONLY_LAYER:-0}
PQ_PER_DIM=${PQ_PER_DIM:-0}
TRAIN_PQ_K=${TRAIN_PQ_K:-1}
WARMUP_ONLY=${WARMUP_ONLY:-1}

PQ_K_INIT=${PQ_K_INIT:-1e-4}
WARMUP_LR=${WARMUP_LR:-1e-3}
WARMUP_COEF=${WARMUP_COEF:-1e6}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-1e5}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}
MOTIF_LOSS_FORM=${MOTIF_LOSS_FORM:-mse}
MOTIF_LOG_EPS=${MOTIF_LOG_EPS:-1e-12}
MOTIF_LOSS_WEIGHTS=${MOTIF_LOSS_WEIGHTS:-}
MOTIF_PER_DIM_MODE=${MOTIF_PER_DIM_MODE:-sampled_channel}
MOTIF_CHANNEL_SAMPLES=${MOTIF_CHANNEL_SAMPLES:-16}

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/pq_rank_search_motif1to13_${TS}}
mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/rank_search_summary.csv"

cd "${ROOT}/mamba-main"

cat > "${SUMMARY_CSV}" <<CSV
rank,motif,status,reached,ratio,best_motif,init_motif,target_motif,out_dir
CSV
RANK_SUMMARY_CSV="${OUT_ROOT}/rank_overview.csv"
cat > "${RANK_SUMMARY_CSV}" <<CSV
rank,all_ok,n_ok,n_fail,n_total,rank_dir
CSV

echo "[run] out_root=${OUT_ROOT}"
echo "[run] rank_range=${START_RANK}..${END_RANK}, motif_range=${MOTIF_START}..${MOTIF_END}"
echo "[run] criterion: motif_loss <= init * ${WARMUP_RATIO}"
echo "[run] setup: train_only_layer=${TRAIN_ONLY_LAYER}, motif_loss_only_layer=${MOTIF_LOSS_ONLY_LAYER}, pq_per_dim=${PQ_PER_DIM}, warmup_only=${WARMUP_ONLY}"

found_rank=""

for ((rank=START_RANK; rank<=END_RANK; rank++)); do
  RANK_DIR="${OUT_ROOT}/rank_${rank}"
  mkdir -p "${RANK_DIR}"
  echo "[rank=${rank}] start"

  rank_all_ok=1
  n_ok=0
  n_fail=0
  n_total=0

  for ((motif=MOTIF_START; motif<=MOTIF_END; motif++)); do
    n_total=$((n_total + 1))
    OUT_DIR="${RANK_DIR}/motif${motif}"
    mkdir -p "${OUT_DIR}"

    echo "[rank=${rank}] motif=${motif}"

    cmd=(
      "${PY}" -u motif/train_kpq_motif_pile_lm.py
      --pretrained-dir "${PRETRAINED_DIR}"
      --pile-root "${PILE_ROOT}"
      --out-dir "${OUT_DIR}"
      --device "${DEVICE}"
      --dtype "${DTYPE}"
      --seed "${SEED}"
      --motif-class "${motif}"
      --pq-rank "${rank}"
      --train-only-layer "${TRAIN_ONLY_LAYER}"
      --motif-loss-only-layer "${MOTIF_LOSS_ONLY_LAYER}"
      --warmup-ratio "${WARMUP_RATIO}"
      --warmup-max-steps "${WARMUP_MAX_STEPS}"
      --pq-k-init "${PQ_K_INIT}"
      --warmup-lr "${WARMUP_LR}"
      --warmup-coef "${WARMUP_COEF}"
      --motif-amplitude "${MOTIF_AMPLITUDE}"
      --motif-bias "${MOTIF_BIAS}"
      --motif-loss-form "${MOTIF_LOSS_FORM}"
      --motif-log-eps "${MOTIF_LOG_EPS}"
      --motif-per-dim-mode "${MOTIF_PER_DIM_MODE}"
      --motif-channel-samples "${MOTIF_CHANNEL_SAMPLES}"
      --max-train-tokens 0
      --min-train-tokens 0
      --batch-size 1
      --seq-len 2048
    )

    if [[ "${PQ_PER_DIM}" == "1" ]]; then
      cmd+=(--pq-per-dim)
    else
      cmd+=(--no-pq-per-dim)
    fi

    if [[ "${TRAIN_PQ_K}" == "1" ]]; then
      cmd+=(--train-pq-k)
    else
      cmd+=(--no-train-pq-k)
    fi

    if [[ "${WARMUP_ONLY}" == "1" ]]; then
      cmd+=(--warmup-only --no-warmup-strict)
    fi
    if [[ -n "${MOTIF_LOSS_WEIGHTS}" ]]; then
      cmd+=(--motif-loss-weights "${MOTIF_LOSS_WEIGHTS}")
    fi

    LOG_PATH="${OUT_DIR}/run.log"
    if ! PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"; then
      echo "${rank},${motif},run_failed,false,NaN,NaN,NaN,NaN,${OUT_DIR}" >> "${SUMMARY_CSV}"
      echo "[rank=${rank}] motif=${motif} failed"
      rank_all_ok=0
      n_fail=$((n_fail + 1))
      continue
    fi

    WARMUP_JSON="${OUT_DIR}/warmup_summary.json"
    if [[ ! -f "${WARMUP_JSON}" ]]; then
      echo "${rank},${motif},missing_warmup_summary,false,NaN,NaN,NaN,NaN,${OUT_DIR}" >> "${SUMMARY_CSV}"
      echo "[rank=${rank}] motif=${motif} missing warmup_summary"
      rank_all_ok=0
      n_fail=$((n_fail + 1))
      continue
    fi

    read -r reached ratio best init target <<EOF2
$(${PY} - "${WARMUP_JSON}" <<'PY'
import json
import math
import sys

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
init = float(d.get("init_motif", float("nan")))
best = float(d.get("best_motif", float("nan")))
target = float(d.get("target_motif", float("nan")))
reached = bool(d.get("reached", False))
ratio = (best / init) if (math.isfinite(init) and init != 0.0 and math.isfinite(best)) else float("nan")
print(str(reached).lower(), ratio, best, init, target)
PY
)
EOF2

    echo "${rank},${motif},ok,${reached},${ratio},${best},${init},${target},${OUT_DIR}" >> "${SUMMARY_CSV}"
    echo "[rank=${rank}] motif=${motif}, reached=${reached}, ratio=${ratio}"

    if [[ "${reached}" != "true" ]]; then
      rank_all_ok=0
      n_fail=$((n_fail + 1))
      echo "[rank=${rank}] motif=${motif} not reached"
    else
      n_ok=$((n_ok + 1))
    fi
  done

  echo "${rank},${rank_all_ok},${n_ok},${n_fail},${n_total},${RANK_DIR}" >> "${RANK_SUMMARY_CSV}"
  echo "[rank=${rank}] done: ok=${n_ok}, fail=${n_fail}, total=${n_total}"

  if [[ "${rank_all_ok}" == "1" ]]; then
    found_rank="${rank}"
    echo "[result] all motifs reached at rank=${rank}"
    break
  fi

done

if [[ -n "${found_rank}" ]]; then
  echo "[result] minimum_rank=${found_rank}"
  echo "[result] summary_csv=${SUMMARY_CSV}"
  echo "[result] rank_summary_csv=${RANK_SUMMARY_CSV}"
  echo "[result] out_root=${OUT_ROOT}"
  exit 0
fi

echo "[result] no rank in [${START_RANK}, ${END_RANK}] can satisfy all motifs ${MOTIF_START}..${MOTIF_END}"
echo "[result] summary_csv=${SUMMARY_CSV}"
echo "[result] rank_summary_csv=${RANK_SUMMARY_CSV}"
echo "[result] out_root=${OUT_ROOT}"
exit 2
