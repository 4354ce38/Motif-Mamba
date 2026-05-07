#!/usr/bin/env bash
set -euo pipefail

# Mamba130M + motif12 + pq_rank=2
# - Shared k/P/Q inside each block (pq_per_dim=0)
# - Different k/P/Q across blocks (naturally one set per block)
# - Train base Mamba params together with k/P/Q
# - Pile training budget: 1B tokens

PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}
SCRIPT=${SCRIPT:-/workspace/mamba-main/motif/train_kpq_motif_pile_lm.py}

PRETRAINED_DIR=${PRETRAINED_DIR:-/workspace/mamba-130m}
PILE_ROOT=${PILE_ROOT:-/workspace/datasets/pile_standard_pythia}
OUT_DIR=${OUT_DIR:-/workspace/motif/logs/pile_mamba130m_motif12_rank2_fulltrain_1B}

DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
SEED=${SEED:-42}

PQ_RANK=${PQ_RANK:-2}
PQ_K_INIT=${PQ_K_INIT:-1e-4}

MOTIF_CLASS=${MOTIF_CLASS:-12}
MOTIF_COEF=${MOTIF_COEF:-0.1}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-1e5}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}

WARMUP_LR=${WARMUP_LR:-1e-3}
WARMUP_COEF=${WARMUP_COEF:-1e6}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-20000}
WARMUP_LOG_EVERY=${WARMUP_LOG_EVERY:-20}

TRAIN_LR=${TRAIN_LR:-2e-5}
TRAIN_STEPS=${TRAIN_STEPS:-0}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1.0}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000000}
MIN_TRAIN_TOKENS=${MIN_TRAIN_TOKENS:-100000000}
BATCH_SIZE=${BATCH_SIZE:-2}
SEQ_LEN=${SEQ_LEN:-512}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-0}
EVAL_TOKENS=${EVAL_TOKENS:-2000000}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
GRAD_CLIP=${GRAD_CLIP:-0.5}
LOG_EVERY=${LOG_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-200}

RUN_EVAL_AFTER_TRAIN=${RUN_EVAL_AFTER_TRAIN:-1}
ROOT=${ROOT:-$(cd "$(dirname "${PRETRAINED_DIR}")" && pwd)}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}
EVAL_DEVICE=${EVAL_DEVICE:-${DEVICE}}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-2048}
EVAL_DTYPE=${EVAL_DTYPE:-float16}
EVAL_LIMIT=${EVAL_LIMIT:-0}
EVAL_OFFLINE=${EVAL_OFFLINE:-0}
EVAL_TASKS=${EVAL_TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,swde,squadv2,fda,triviaqa,nq_open,drop}
EVAL_TAG=${EVAL_TAG:-mamba130m_motif12_rank2_fulltrain_1B}

mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1

echo "[run] out_dir=${OUT_DIR}"
echo "[run] model=${PRETRAINED_DIR}"
echo "[run] device=${DEVICE}, dtype=${DTYPE}, pq_rank=${PQ_RANK}, pq_per_dim=0(shared-in-block), train_base_model=1"
echo "[run] motif=${MOTIF_CLASS}, target_tokens=${MAX_TRAIN_TOKENS}"

"${PY}" -u "${SCRIPT}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  --pq-rank "${PQ_RANK}" \
  --no-pq-per-dim \
  --pq-k-init "${PQ_K_INIT}" \
  --train-pq-k \
  --train-base-model \
  --save-full-model \
  --motif-class "${MOTIF_CLASS}" \
  --motif-coef "${MOTIF_COEF}" \
  --motif-amplitude "${MOTIF_AMPLITUDE}" \
  --motif-bias "${MOTIF_BIAS}" \
  --warmup-lr "${WARMUP_LR}" \
  --warmup-coef "${WARMUP_COEF}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --warmup-max-steps "${WARMUP_MAX_STEPS}" \
  --warmup-log-every "${WARMUP_LOG_EVERY}" \
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
  --grad-clip "${GRAD_CLIP}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo "[done] log=${OUT_DIR}/train.log"
echo "[done] csv=${OUT_DIR}/train_log.csv"

if [[ "${RUN_EVAL_AFTER_TRAIN}" == "1" ]]; then
  ADAPTER_PATH="${OUT_DIR}/pq_adapter_latest.pt"
  if [[ ! -f "${ADAPTER_PATH}" ]]; then
    echo "[warn] skip eval: adapter not found: ${ADAPTER_PATH}"
    exit 0
  fi
  if [[ ! -x "${EVAL_SCRIPT}" ]]; then
    echo "[warn] eval script not executable: ${EVAL_SCRIPT}"
    exit 0
  fi
  echo "[eval] running lm-eval on trained adapter..."
  ADAPTER="${ADAPTER_PATH}" \
  ROOT="${ROOT}" \
  BASE_MODEL="${PRETRAINED_DIR}" \
  DEVICE="${EVAL_DEVICE}" \
  BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  MAX_LENGTH="${EVAL_MAX_LENGTH}" \
  DTYPE="${EVAL_DTYPE}" \
  LIMIT="${EVAL_LIMIT}" \
  OFFLINE="${EVAL_OFFLINE}" \
  TASKS="${EVAL_TASKS}" \
  TAG="${EVAL_TAG}" \
  bash "${EVAL_SCRIPT}"
fi
