#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
BASE_SCRIPT=${BASE_SCRIPT:-${ROOT}/mamba-main/motif/run_rank_motif1to13_uniform_params.sh}
RANK_START=${RANK_START:-1}
RANK_END=${RANK_END:-4}

TS=$(date +%Y%m%d_%H%M%S)
MASTER_OUT=${MASTER_OUT:-${ROOT}/motif/logs/rank1to4_motif1to13_uniform_${TS}}
mkdir -p "${MASTER_OUT}"
MASTER_SUMMARY="${MASTER_OUT}/master_summary.csv"
echo "rank,exit_code,ok_count,total_count,summary_csv,run_root" > "${MASTER_SUMMARY}"

for ((rank=RANK_START; rank<=RANK_END; rank++)); do
  RUN_ROOT="${MASTER_OUT}/rank_${rank}"
  mkdir -p "${RUN_ROOT}"

  set +e
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK \
    RANK="${rank}" OUT_ROOT="${RUN_ROOT}" bash "${BASE_SCRIPT}"
  rc=$?
  set -e

  SUMMARY_CSV="${RUN_ROOT}/summary.csv"
  read -r ok_count total_count <<EOF2
$(python - "${SUMMARY_CSV}" <<'PY'
import csv,sys,os
p=sys.argv[1]
if not os.path.exists(p):
    print('0 13'); raise SystemExit
ok=0; tot=0
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
done

echo "[done] master_summary=${MASTER_SUMMARY}"
