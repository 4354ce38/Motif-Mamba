#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
SCRIPT_DIR="${ROOT}/mamba-main/motif"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_kpq_motif_pile_lm.py"

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}

# Required: 13 comma-separated values, e.g. "0.07,0.08,...(13 values)"
MOTIF_TARGET_VALUES=${MOTIF_TARGET_VALUES:-}
if [[ -z "${MOTIF_TARGET_VALUES}" ]]; then
  echo "[error] MOTIF_TARGET_VALUES is empty. Please provide 13 comma-separated motif target values." >&2
  exit 1
fi

START_RANK=${START_RANK:-1}
END_RANK=${END_RANK:-32}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-20000}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
SEED=${SEED:-42}

# Keep same training env knobs for fair rank comparison
PQ_PER_DIM=${PQ_PER_DIM:-1}
PQ_K_INIT=${PQ_K_INIT:-1e-4}
TRAIN_PQ_K=${TRAIN_PQ_K:-1}
WARMUP_LR=${WARMUP_LR:-1e-3}
WARMUP_COEF=${WARMUP_COEF:-1e6}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-1e5}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}
MOTIF_PER_DIM_MODE=${MOTIF_PER_DIM_MODE:-sampled_channel}
MOTIF_CHANNEL_SAMPLES=${MOTIF_CHANNEL_SAMPLES:-16}

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/pq_rank_search_13motif_${TS}}
mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/rank_search_summary.csv"

cd "${ROOT}/mamba-main"

cat > "${SUMMARY_CSV}" <<CSV
rank,status,reached,ratio,best_motif,init_motif,target_motif,out_dir
CSV

echo "[run] out_root=${OUT_ROOT}"
echo "[run] rank_range=${START_RANK}..${END_RANK}"
echo "[run] threshold=motif_loss <= init * ${WARMUP_RATIO}"

auto_hit_rank=""

for ((rank=START_RANK; rank<=END_RANK; rank++)); do
  RUN_DIR="${OUT_ROOT}/rank_${rank}"
  mkdir -p "${RUN_DIR}"

  echo "[run] rank=${rank}, out_dir=${RUN_DIR}"

  cmd=(
    "${PY}" -u "${TRAIN_SCRIPT}"
    --pretrained-dir "${PRETRAINED_DIR}"
    --pile-root "${PILE_ROOT}"
    --out-dir "${RUN_DIR}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
    --pq-rank "${rank}"
    --motif-target-values "${MOTIF_TARGET_VALUES}"
    --warmup-ratio "${WARMUP_RATIO}"
    --warmup-max-steps "${WARMUP_MAX_STEPS}"
    --warmup-only
    --no-warmup-strict
    --warmup-lr "${WARMUP_LR}"
    --warmup-coef "${WARMUP_COEF}"
    --motif-amplitude "${MOTIF_AMPLITUDE}"
    --motif-bias "${MOTIF_BIAS}"
    --motif-per-dim-mode "${MOTIF_PER_DIM_MODE}"
    --motif-channel-samples "${MOTIF_CHANNEL_SAMPLES}"
  )

  if [[ "${PQ_PER_DIM}" == "0" ]]; then
    cmd+=(--no-pq-per-dim)
  else
    cmd+=(--pq-per-dim)
  fi

  if [[ "${TRAIN_PQ_K}" == "0" ]]; then
    cmd+=(--no-train-pq-k)
  else
    cmd+=(--train-pq-k)
  fi

  cmd+=(--pq-k-init "${PQ_K_INIT}")

  LOG_PATH="${RUN_DIR}/run.log"
  if ! PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"; then
    echo "${rank},run_failed,false,NaN,NaN,NaN,NaN,${RUN_DIR}" >> "${SUMMARY_CSV}"
    echo "[warn] rank=${rank} failed, continue to next rank"
    continue
  fi

  WARMUP_JSON="${RUN_DIR}/warmup_summary.json"
  if [[ ! -f "${WARMUP_JSON}" ]]; then
    echo "${rank},missing_warmup_summary,false,NaN,NaN,NaN,NaN,${RUN_DIR}" >> "${SUMMARY_CSV}"
    echo "[warn] rank=${rank} missing warmup_summary.json, continue"
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

  echo "${rank},ok,${reached},${ratio},${best},${init},${target},${RUN_DIR}" >> "${SUMMARY_CSV}"
  echo "[rank=${rank}] reached=${reached}, ratio=${ratio}, best=${best}, init=${init}, target=${target}"

  if [[ "${reached}" == "true" ]]; then
    auto_hit_rank="${rank}"
    break
  fi
done

if [[ -n "${auto_hit_rank}" ]]; then
  echo "[result] minimum_rank=${auto_hit_rank}"
  echo "[result] summary_csv=${SUMMARY_CSV}"
  echo "[result] out_root=${OUT_ROOT}"
  exit 0
fi

echo "[result] no rank in [${START_RANK}, ${END_RANK}] reached motif target ratio ${WARMUP_RATIO}"
echo "[result] summary_csv=${SUMMARY_CSV}"
echo "[result] out_root=${OUT_ROOT}"
exit 2
