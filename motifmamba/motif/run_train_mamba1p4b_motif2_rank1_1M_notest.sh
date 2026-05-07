#!/usr/bin/env bash
set -euo pipefail

# Train motifmamba on Pile (10M shard path) with:
# - base model: mamba-1.4b
# - motif class: 2
# - pq rank: 1
# - token budget: 1M
# - no mid-training eval

PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
ROOT=${ROOT:-/workspace}
SCRIPT=${SCRIPT:-${ROOT}/mamba-main/motif/train_kpq_motif_pile_lm.py}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-1.4b}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia_10M}
OUT_DIR=${OUT_DIR:-${ROOT}/logs/pq_motif_pile/mamba1p4b_motif2_rank1_1M_notest}

DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
SEED=${SEED:-42}

# Distributed launch options.
# Single GPU: NPROC_PER_NODE=1 (default)
# Multi GPU (single node): set NPROC_PER_NODE>1, e.g. 4
# Multi node: also set NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
DIST_STRATEGY=${DIST_STRATEGY:-ddp}
FSDP_SHARDING_STRATEGY=${FSDP_SHARDING_STRATEGY:-shard_grad_op}
FSDP_OFFLOAD_PARAMS=${FSDP_OFFLOAD_PARAMS:-0}

PQ_RANK=${PQ_RANK:-1}
PQ_K_INIT=${PQ_K_INIT:-1e-4}

MOTIF_CLASS=${MOTIF_CLASS:-2}
MOTIF_COEF=${MOTIF_COEF:-0.1}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-1e5}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}

WARMUP_LR=${WARMUP_LR:-5e-4}
WARMUP_COEF=${WARMUP_COEF:-1e6}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-5000}
WARMUP_LOG_EVERY=${WARMUP_LOG_EVERY:-20}

TRAIN_LR=${TRAIN_LR:-1e-4}
TRAIN_STEPS=${TRAIN_STEPS:-0}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1.0}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000}
MIN_TRAIN_TOKENS=${MIN_TRAIN_TOKENS:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-1024}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-0}
EVAL_TOKENS=${EVAL_TOKENS:-200000}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
GRAD_CLIP=${GRAD_CLIP:-0.5}
LOG_EVERY=${LOG_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-200}

mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1

echo "[run] script=${SCRIPT}"
echo "[run] model=${PRETRAINED_DIR}"
echo "[run] data=${PILE_ROOT}"
echo "[run] out_dir=${OUT_DIR}"
echo "[run] device=${DEVICE}, dtype=${DTYPE}"
echo "[run] motif=${MOTIF_CLASS}, pq_rank=${PQ_RANK}, max_train_tokens=${MAX_TRAIN_TOKENS}"
echo "[run] eval_every_steps=${EVAL_EVERY_STEPS} (0 means disabled)"
echo "[run] dist: nproc_per_node=${NPROC_PER_NODE}, nnodes=${NNODES}, node_rank=${NODE_RANK}, strategy=${DIST_STRATEGY}"

DIST_ARGS=(
  --dist-strategy "${DIST_STRATEGY}"
  --fsdp-sharding-strategy "${FSDP_SHARDING_STRATEGY}"
)
if [[ "${FSDP_OFFLOAD_PARAMS}" == "1" ]]; then
  DIST_ARGS+=(--fsdp-offload-params)
fi

if [[ "${NPROC_PER_NODE}" -gt 1 || "${NNODES}" -gt 1 ]]; then
  LAUNCHER=(
    "${PY}" -m torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
  )
  DEVICE_ARG="cuda"
else
  LAUNCHER=("${PY}" -u)
  DEVICE_ARG="${DEVICE}"
fi

"${LAUNCHER[@]}" "${SCRIPT}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE_ARG}" \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  "${DIST_ARGS[@]}" \
  --pq-rank "${PQ_RANK}" \
  --pq-k-init "${PQ_K_INIT}" \
  --train-pq-k \
  --no-train-base-model \
  --motif-class "${MOTIF_CLASS}" \
  --motif-coef "${MOTIF_COEF}" \
  --motif-amplitude "${MOTIF_AMPLITUDE}" \
  --motif-bias "${MOTIF_BIAS}" \
  --warmup-lr "${WARMUP_LR}" \
  --warmup-coef "${WARMUP_COEF}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --warmup-max-steps "${WARMUP_MAX_STEPS}" \
  --warmup-log-every "${WARMUP_LOG_EVERY}" \
  --no-warmup-strict \
  --train-lr "${TRAIN_LR}" \
  --train-steps "${TRAIN_STEPS}" \
  --train-epochs "${TRAIN_EPOCHS}" \
  --max-train-tokens "${MAX_TRAIN_TOKENS}" \
  --min-train-tokens "${MIN_TRAIN_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --eval-tokens "${EVAL_TOKENS}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
  --disable-early-stop \
  --grad-clip "${GRAD_CLIP}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo "[done] log=${OUT_DIR}/train.log"
echo "[done] csv=${OUT_DIR}/train_log.csv"
echo "[done] adapter=${OUT_DIR}/pq_adapter_latest.pt"
