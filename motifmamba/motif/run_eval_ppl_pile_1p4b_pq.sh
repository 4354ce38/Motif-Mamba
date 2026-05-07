#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
ROOT=${ROOT:-/workspace/mamba-main/motif}
DEVICE=${DEVICE:-cuda:0}

PRETRAINED_DIR=${PRETRAINED_DIR:-/workspace/mamba-1.4b}
PILE_ROOT=${PILE_ROOT:-/workspace/datasets/pile_standard_pythia}
ADAPTER_PATH=${ADAPTER_PATH:-/workspace/logs/pq_motif_pile/mamba1.4b_motif2_pq2/pq_adapter_best.pt}
OUT_CSV=${OUT_CSV:-/workspace/logs/pq_motif_pile/ppl_eval.csv}

cd "${ROOT}"
"${PY}" eval_ppl_pile_with_pq.py \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --adapter-path "${ADAPTER_PATH}" \
  --pile-root "${PILE_ROOT}" \
  --pile-shards 1 \
  --device "${DEVICE}" \
  --dtype float16 \
  --batch-size 1 \
  --seq-len 1024 \
  --eval-seqs 128 \
  --out-csv "${OUT_CSV}" \
  "$@"
