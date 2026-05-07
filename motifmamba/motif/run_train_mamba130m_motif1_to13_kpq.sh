#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}

PQ_RANK=${PQ_RANK:-2}
PQ_PER_DIM=${PQ_PER_DIM:-0}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_TOKENS=${MAX_TOKENS:-1000000}
MIN_TOKENS=${MIN_TOKENS:-0}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-3000}
SEED=${SEED:-42}
TRAIN_ONLY_LAYER=${TRAIN_ONLY_LAYER:-0}
MOTIF_LOSS_ONLY_LAYER=${MOTIF_LOSS_ONLY_LAYER:-0}
REPLICATE_ALL_LAYERS=${REPLICATE_ALL_LAYERS:-1}
SOURCE_CHANNEL=${SOURCE_CHANNEL:--1}
WARMUP_ONLY=${WARMUP_ONLY:-0}

MOTIF_START=${MOTIF_START:-1}
MOTIF_END=${MOTIF_END:-13}

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/logs/pile_mamba130m_motif1to13_rank${PQ_RANK}_${TS}}
mkdir -p "${OUT_ROOT}"

cd "${ROOT}/mamba-main"

echo "[run] device=${DEVICE}, pq_rank=${PQ_RANK}, motif=${MOTIF_START}..${MOTIF_END}"
echo "[run] out_root=${OUT_ROOT}"
echo "[run] train_only_layer=${TRAIN_ONLY_LAYER}, motif_loss_only_layer=${MOTIF_LOSS_ONLY_LAYER}"
if [[ "${PQ_PER_DIM}" == "1" ]]; then
  echo "[run] pq_per_dim=1 (many trainable params per selected layer)"
else
  echo "[run] pq_per_dim=0 (shared PQ; expected trainable per selected layer ~= 32*rank + 1)"
fi

for motif in $(seq "${MOTIF_START}" "${MOTIF_END}"); do
  OUT_DIR="${OUT_ROOT}/motif${motif}"
  mkdir -p "${OUT_DIR}"

  echo "[run] motif=${motif}, out_dir=${OUT_DIR}"

  cmd=(
    "${PY}" -u motif/train_kpq_motif_pile_lm.py
    --pretrained-dir "${PRETRAINED_DIR}" \
    --pile-root "${PILE_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --motif-class "${motif}" \
    --pq-rank "${PQ_RANK}" \
    --train-only-layer "${TRAIN_ONLY_LAYER}" \
    --motif-loss-only-layer "${MOTIF_LOSS_ONLY_LAYER}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-tokens "${MAX_TOKENS}" \
    --min-train-tokens "${MIN_TOKENS}" \
    --warmup-max-steps "${WARMUP_MAX_STEPS}"
  )

  if [[ "${WARMUP_ONLY}" == "1" ]]; then
    cmd+=(--warmup-only --no-warmup-strict)
  fi
  if [[ "${PQ_PER_DIM}" == "1" ]]; then
    cmd+=(--pq-per-dim)
  else
    cmd+=(--no-pq-per-dim)
  fi

  PYTHONPATH="${ROOT}/mamba-main:${PYTHONPATH:-}" \
  "${cmd[@]}"

  if [[ "${REPLICATE_ALL_LAYERS}" == "1" ]]; then
    IN_ADAPTER="${OUT_DIR}/pq_adapter_latest.pt"
    OUT_ADAPTER="${OUT_DIR}/pq_adapter_replicated.pt"
    if [[ -f "${IN_ADAPTER}" ]]; then
      "${PY}" -u motif/replicate_pq_adapter.py \
        --in-adapter "${IN_ADAPTER}" \
        --out-adapter "${OUT_ADAPTER}" \
        --source-layer "${TRAIN_ONLY_LAYER}" \
        --source-channel "${SOURCE_CHANNEL}"
      echo "[done] motif=${motif}, replicated_adapter=${OUT_ADAPTER}"
    else
      echo "[warn] motif=${motif}, missing ${IN_ADAPTER}, skip replication"
    fi
  fi

done

echo "[done] all motifs finished: ${MOTIF_START}..${MOTIF_END}"
echo "[done] out_root=${OUT_ROOT}"
