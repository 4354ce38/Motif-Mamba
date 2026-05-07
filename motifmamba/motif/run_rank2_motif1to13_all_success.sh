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
TRAIN_ONLY_LAYER=${TRAIN_ONLY_LAYER:-0}
MOTIF_LOSS_ONLY_LAYER=${MOTIF_LOSS_ONLY_LAYER:-0}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-3000}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/rank2_motif1to13_all_success_${TS}}
mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
echo "motif,status,reached,ratio,best_motif,init_motif,target_motif,trial_tag,run_dir" > "${SUMMARY_CSV}"

cd "${ROOT}/mamba-main"

run_one() {
  local motif="$1"
  local trial_tag="$2"
  local extra_args="$3"
  local out_dir="$4"

  mkdir -p "${out_dir}"

  local cmd=(
    "${PY}" -u motif/train_kpq_motif_pile_lm.py
    --pretrained-dir "${PRETRAINED_DIR}"
    --pile-root "${PILE_ROOT}"
    --out-dir "${out_dir}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
    --motif-class "${motif}"
    --pq-rank "${RANK}"
    --no-pq-per-dim
    --train-only-layer "${TRAIN_ONLY_LAYER}"
    --motif-loss-only-layer "${MOTIF_LOSS_ONLY_LAYER}"
    --train-pq-k
    --warmup-only
    --no-warmup-strict
    --warmup-max-steps "${WARMUP_MAX_STEPS}"
    --warmup-ratio "${WARMUP_RATIO}"
    --max-train-tokens 0
    --min-train-tokens 0
    --batch-size 1
    --seq-len 2048
  )

  # shellcheck disable=SC2206
  local extras=( ${extra_args} )
  cmd+=("${extras[@]}")

  if ! env -u RANK -u WORLD_SIZE -u LOCAL_RANK \
      PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" MPLCONFIGDIR=/tmp/mpl \
      "${cmd[@]}" > "${out_dir}/run.log" 2>&1; then
    echo "${motif},run_failed,false,NaN,NaN,NaN,NaN,${trial_tag},${out_dir}" >> "${SUMMARY_CSV}"
    return 1
  fi

  local summ="${out_dir}/warmup_summary.json"
  if [[ ! -f "${summ}" ]]; then
    echo "${motif},missing_summary,false,NaN,NaN,NaN,NaN,${trial_tag},${out_dir}" >> "${SUMMARY_CSV}"
    return 1
  fi

  read -r reached ratio best init target <<EOF2
$(${PY} - "${summ}" <<'PY'
import json,sys,math
p=sys.argv[1]
d=json.load(open(p))
init=float(d.get('init_motif', float('nan')))
best=float(d.get('best_motif', float('nan')))
target=float(d.get('target_motif', float('nan')))
reached=bool(d.get('reached',False))
ratio=(best/init) if (math.isfinite(init) and init!=0.0 and math.isfinite(best)) else float('nan')
print(str(reached).lower(), ratio, best, init, target)
PY
)
EOF2

  echo "${motif},ok,${reached},${ratio},${best},${init},${target},${trial_tag},${out_dir}" >> "${SUMMARY_CSV}"
  [[ "${reached}" == "true" ]]
}

# motifs 1..12: stable baseline
BASE_ARGS="--pq-k-init 1e-4 --warmup-lr 1e-3 --warmup-coef 1e6 --motif-amplitude 1e5 --motif-bias 5e-5 --motif-loss-form mse"

ok_count=0
fail_count=0

for motif in $(seq 1 12); do
  out_dir="${OUT_ROOT}/motif${motif}_baseline"
  echo "[run] motif=${motif} baseline"
  if run_one "${motif}" "baseline" "${BASE_ARGS}" "${out_dir}"; then
    ok_count=$((ok_count+1))
  else
    fail_count=$((fail_count+1))
  fi
done

# motif13: aggressive trials (stop at first success)
# format: tag|args
TRIALS=(
  "m13_t1|--pq-k-init 1.0 --warmup-lr 3e-2 --warmup-coef 1e7 --motif-amplitude 300 --motif-bias 0.0 --motif-loss-form log_mse --motif-log-eps 1e-18 --motif-loss-weights 1,1,1,1,1,1,1,1,1,1,1,1,300"
  "m13_t2|--pq-k-init 1.0 --warmup-lr 5e-2 --warmup-coef 1e7 --motif-amplitude 300 --motif-bias -1e-6 --motif-loss-form log_mse --motif-log-eps 1e-18 --motif-loss-weights 1,1,1,1,1,1,1,1,1,1,1,1,400"
  "m13_t3|--pq-k-init 3.0 --warmup-lr 3e-2 --warmup-coef 1e7 --motif-amplitude 100 --motif-bias 0.0 --motif-loss-form log_mse --motif-log-eps 1e-18 --motif-loss-weights 1,1,1,1,1,1,1,1,1,1,1,1,400"
  "m13_t4|--pq-k-init 1.0 --warmup-lr 1e-1 --warmup-coef 5e6 --motif-amplitude 300 --motif-bias 0.0 --motif-loss-form log_mse --motif-log-eps 1e-18 --motif-loss-weights 1,1,1,1,1,1,1,1,1,1,1,1,500"
)

m13_ok=0
for t in "${TRIALS[@]}"; do
  tag="${t%%|*}"
  args="${t#*|}"
  out_dir="${OUT_ROOT}/motif13_${tag}"
  echo "[run] motif=13 trial=${tag}"
  if run_one "13" "${tag}" "${args}" "${out_dir}"; then
    m13_ok=1
    ok_count=$((ok_count+1))
    break
  fi
done
if [[ "${m13_ok}" == "0" ]]; then
  fail_count=$((fail_count+1))
fi

echo "[done] ok=${ok_count}, fail=${fail_count}, total=13"
echo "[done] summary=${SUMMARY_CSV}"
if [[ "${ok_count}" == "13" ]]; then
  echo "[result] SUCCESS: rank=${RANK}, motif1..13 all reached"
  exit 0
fi

echo "[result] NOT FULLY SUCCESS: rank=${RANK}, ok=${ok_count}/13"
exit 2
