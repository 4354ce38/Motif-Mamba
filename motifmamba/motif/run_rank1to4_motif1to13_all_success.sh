#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
BASE_SCRIPT=${BASE_SCRIPT:-${ROOT}/mamba-main/motif/run_rank2_motif1to13_all_success.sh}

RANK_START=${RANK_START:-1}
RANK_END=${RANK_END:-4}
TS=$(date +%Y%m%d_%H%M%S)
MASTER_OUT=${MASTER_OUT:-${ROOT}/motif/logs/rank1to4_motif1to13_all_success_${TS}}
mkdir -p "${MASTER_OUT}"
MASTER_SUMMARY="${MASTER_OUT}/master_summary.csv"

echo "rank,exit_code,ok_count,total_count,summary_csv,run_root" > "${MASTER_SUMMARY}"

echo "[run] base_script=${BASE_SCRIPT}"
echo "[run] ranks=${RANK_START}..${RANK_END}"
echo "[run] master_out=${MASTER_OUT}"

for ((rank=RANK_START; rank<=RANK_END; rank++)); do
  RUN_ROOT="${MASTER_OUT}/rank_${rank}"
  mkdir -p "${RUN_ROOT}"

  echo "[run] rank=${rank}"

  set +e
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK \
    RANK="${rank}" OUT_ROOT="${RUN_ROOT}" bash "${BASE_SCRIPT}"
  rc=$?
  set -e

  SUMMARY_CSV=""
  if ls -1d "${RUN_ROOT}"/rank2_motif1to13_all_success_* >/dev/null 2>&1; then
    # fallback if base script ignored OUT_ROOT (should not happen)
    latest=$(ls -1dt "${RUN_ROOT}"/rank2_motif1to13_all_success_* | head -n1)
    SUMMARY_CSV="${latest}/summary.csv"
  elif [[ -f "${RUN_ROOT}/summary.csv" ]]; then
    SUMMARY_CSV="${RUN_ROOT}/summary.csv"
  else
    # base script writes directly to OUT_ROOT/summary.csv
    SUMMARY_CSV="${RUN_ROOT}/summary.csv"
  fi

  read -r ok_count total_count <<EOF2
$(python - "${SUMMARY_CSV}" <<'PY'
import csv,sys,os
p=sys.argv[1]
if not os.path.exists(p):
    print('0 13')
    raise SystemExit
ok=0
tot=0
with open(p,newline='') as f:
    r=csv.DictReader(f)
    for row in r:
        tot += 1
        if row.get('status') == 'ok' and row.get('reached') == 'true':
            ok += 1
print(ok, tot if tot>0 else 13)
PY
)
EOF2

  echo "${rank},${rc},${ok_count},${total_count},${SUMMARY_CSV},${RUN_ROOT}" >> "${MASTER_SUMMARY}"
  echo "[done] rank=${rank}, rc=${rc}, ok=${ok_count}/${total_count}"
done

echo "[done] master_summary=${MASTER_SUMMARY}"
