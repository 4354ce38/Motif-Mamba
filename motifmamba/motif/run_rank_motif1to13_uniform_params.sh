#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float32}
SEED=${SEED:-42}

RANK=${RANK:-2}
MOTIF_START=${MOTIF_START:-1}
MOTIF_END=${MOTIF_END:-13}
TRAIN_ONLY_LAYER=${TRAIN_ONLY_LAYER:-0}
MOTIF_LOSS_ONLY_LAYER=${MOTIF_LOSS_ONLY_LAYER:-0}

# Uniform training params (matched to rank1to4_motif1to13_uniform_20260428_182114)
PQ_K_INIT=${PQ_K_INIT:-1e-3}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-3000}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_LR=${WARMUP_LR:-3e-3}
WARMUP_COEF=${WARMUP_COEF:-2e6}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-3e4}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}
MOTIF_LOSS_FORM=${MOTIF_LOSS_FORM:-mse}
MOTIF_LOG_EPS=${MOTIF_LOG_EPS:-1e-12}

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/rank${RANK}_motif1to13_uniform_${TS}}
mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
echo "motif,status,reached,ratio,best_motif,init_motif,target_motif,run_dir" > "${SUMMARY_CSV}"

cd "${ROOT}/mamba-main"

ok_count=0
fail_count=0

for motif in $(seq "${MOTIF_START}" "${MOTIF_END}"); do
  OUT_DIR="${OUT_ROOT}/motif${motif}"
  mkdir -p "${OUT_DIR}"
  echo "[run] rank=${RANK}, motif=${motif}"

  cmd=(
    "${PY}" -u motif/train_kpq_motif_pile_lm.py
    --pretrained-dir "${PRETRAINED_DIR}"
    --pile-root "${PILE_ROOT}"
    --out-dir "${OUT_DIR}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
    --motif-class "${motif}"
    --pq-rank "${RANK}"
    --no-pq-per-dim
    --train-only-layer "${TRAIN_ONLY_LAYER}"
    --motif-loss-only-layer "${MOTIF_LOSS_ONLY_LAYER}"
    --train-pq-k
    --pq-k-init "${PQ_K_INIT}"
    --warmup-only
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

  if ! env -u RANK -u WORLD_SIZE -u LOCAL_RANK \
      PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" MPLCONFIGDIR=/tmp/mpl \
      "${cmd[@]}" > "${OUT_DIR}/run.log" 2>&1; then
    echo "${motif},run_failed,false,NaN,NaN,NaN,NaN,${OUT_DIR}" >> "${SUMMARY_CSV}"
    fail_count=$((fail_count+1))
    continue
  fi

  SUMM="${OUT_DIR}/warmup_summary.json"
  if [[ ! -f "${SUMM}" ]]; then
    echo "${motif},missing_summary,false,NaN,NaN,NaN,NaN,${OUT_DIR}" >> "${SUMMARY_CSV}"
    fail_count=$((fail_count+1))
    continue
  fi

  read -r reached ratio best init target <<EOF2
$(${PY} - "${SUMM}" <<'PY'
import json,sys,math
p=sys.argv[1]
d=json.load(open(p))
init=float(d.get('init_motif', float('nan')))
best=float(d.get('best_motif', float('nan')))
target=float(d.get('target_motif', float('nan')))
reached=bool(d.get('reached',False))
ratio=(best/init) if (math.isfinite(init) and init!=0 and math.isfinite(best)) else float('nan')
print(str(reached).lower(), ratio, best, init, target)
PY
)
EOF2

  echo "${motif},ok,${reached},${ratio},${best},${init},${target},${OUT_DIR}" >> "${SUMMARY_CSV}"
  if [[ "${reached}" == "true" ]]; then
    ok_count=$((ok_count+1))
  else
    fail_count=$((fail_count+1))
  fi
done

echo "[done] rank=${RANK}, ok=${ok_count}, fail=${fail_count}, total=$((ok_count+fail_count))"
echo "[done] summary=${SUMMARY_CSV}"
if [[ "${fail_count}" == "0" ]]; then
  echo "[result] SUCCESS: motif${MOTIF_START}..${MOTIF_END} all reached"
  exit 0
fi

echo "[result] PARTIAL: ${ok_count}/$((ok_count+fail_count)) reached"
exit 2
