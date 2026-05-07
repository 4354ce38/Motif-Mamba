#!/usr/bin/env bash
set -euo pipefail

# Mamba-130M + motifFRP constraint + full-model training on Pile.
# Target: 1M tokens, seq_len=2048, then run lm-eval automatically.

PY=${PY:-/home/user/miniconda3/envs/motifmamba/bin/python}

ROOT=${ROOT:-/workspace}
SCRIPT=${SCRIPT:-${ROOT}/mamba-main/motif/train_kpq_motif_pile_lm.py}
EVAL_SCRIPT=${EVAL_SCRIPT:-${ROOT}/mamba-main/motif/run_eval_kpq_full_suite.sh}

PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia}
OUT_DIR=${OUT_DIR:-${ROOT}/motif/logs/pile_mamba130m_motifFRP_rank2_fulltrain_1M_seq2048_cuda2}

DEVICE=${DEVICE:-cuda:2}
DTYPE=${DTYPE:-float16}
SEED=${SEED:-42}

PQ_RANK=${PQ_RANK:-2}
PQ_K_INIT=${PQ_K_INIT:-1e-4}

MOTIF_CLASS=${MOTIF_CLASS:-motif_FRP}
MOTIF_COEF=${MOTIF_COEF:-0.1}
MOTIF_AMPLITUDE=${MOTIF_AMPLITUDE:-1e5}
MOTIF_BIAS=${MOTIF_BIAS:-5e-5}

WARMUP_LR=${WARMUP_LR:-5e-4}
WARMUP_COEF=${WARMUP_COEF:-1e6}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-5000}
WARMUP_LOG_EVERY=${WARMUP_LOG_EVERY:-20}

TRAIN_LR=${TRAIN_LR:-1e-5}
TRAIN_STEPS=${TRAIN_STEPS:-0}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1.0}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000}
MIN_TRAIN_TOKENS=${MIN_TRAIN_TOKENS:-200000}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-2048}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-0}
EVAL_TOKENS=${EVAL_TOKENS:-200000}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.001}
GRAD_CLIP=${GRAD_CLIP:-0.5}
LOG_EVERY=${LOG_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-20}

# Build extensions only once if requested.
AUTO_BUILD=${AUTO_BUILD:-0}
BUILD_STAMP=${BUILD_STAMP:-${ROOT}/mamba-main/.cuda_ext_built.ok}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
MAX_JOBS=${MAX_JOBS:-1}
CC=${CC:-/usr/bin/gcc-11}
CXX=${CXX:-/usr/bin/g++-11}

# Auto lm-eval after training
RUN_EVAL_AFTER_TRAIN=${RUN_EVAL_AFTER_TRAIN:-1}
TOKENIZER_PATH=${TOKENIZER_PATH:-EleutherAI/gpt-neox-20b}
LM_TASKS=${LM_TASKS:-lambada_openai,lambada_standard,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,swde,squadv2,fda,triviaqa,nq_open,drop}
LM_BATCH_SIZE=${LM_BATCH_SIZE:-8}
LM_MAX_LENGTH=${LM_MAX_LENGTH:-2048}
LM_DTYPE=${LM_DTYPE:-float16}
LM_LIMIT=${LM_LIMIT:-0}
LM_CACHE_REQUESTS=${LM_CACHE_REQUESTS:-true}
LM_USE_CACHE=${LM_USE_CACHE:-${ROOT}/.cache/lm_eval/core14_cache.sqlite}
LM_CACHE_ONLY=${LM_CACHE_ONLY:-1}
LM_TAG=${LM_TAG:-mamba130m_motifFRP_rank2_fulltrain_1M_seq2048_cuda2}

mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1

if [[ "${AUTO_BUILD}" == "1" ]]; then
  if [[ -f "${BUILD_STAMP}" ]]; then
    echo "[build] skip (stamp exists): ${BUILD_STAMP}"
  else
    echo "[build] first-time build extensions..."
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export MAX_JOBS
    export CC
    export CXX
    export CUDAHOSTCXX="${CXX}"
    (
      cd "${ROOT}/mamba-main"
      "${PY}" setup.py build_ext --inplace
    )
    (
      cd "${ROOT}/causal-conv1d"
      "${PY}" setup.py build_ext --inplace
    )
    touch "${BUILD_STAMP}"
    echo "[build] done and stamped: ${BUILD_STAMP}"
  fi
fi

echo "[run] out_dir=${OUT_DIR}"
echo "[run] model=${PRETRAINED_DIR}"
echo "[run] device=${DEVICE}, dtype=${DTYPE}, seq_len=${SEQ_LEN}, max_tokens=${MAX_TRAIN_TOKENS}"
echo "[run] motif=${MOTIF_CLASS}, pq_rank=${PQ_RANK}, full_train=1"

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
  if [[ -f "${ADAPTER_PATH}" && -x "${EVAL_SCRIPT}" ]]; then
    if [[ "${LM_CACHE_ONLY}" == "1" && ! -f "${LM_USE_CACHE}" ]]; then
      echo "[warn] cache-only mode: lm-eval cache not found: ${LM_USE_CACHE}"
      echo "[warn] skip eval to avoid rebuilding context"
      exit 0
    fi
    echo "[eval] running lm-eval..."
    echo "[eval] cache_requests=${LM_CACHE_REQUESTS}, use_cache=${LM_USE_CACHE}"
    ADAPTER="${ADAPTER_PATH}" \
    ROOT="${ROOT}" \
    BASE_MODEL="${PRETRAINED_DIR}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    DEVICE="${DEVICE}" \
    BATCH_SIZE="${LM_BATCH_SIZE}" \
    MAX_LENGTH="${LM_MAX_LENGTH}" \
    DTYPE="${LM_DTYPE}" \
    LIMIT="${LM_LIMIT}" \
    CACHE_REQUESTS="${LM_CACHE_REQUESTS}" \
    USE_CACHE="${LM_USE_CACHE}" \
    TASKS="${LM_TASKS}" \
    TAG="${LM_TAG}" \
    bash "${EVAL_SCRIPT}"
  else
    echo "[warn] skip eval: adapter or eval script not found"
  fi
fi
