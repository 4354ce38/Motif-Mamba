#!/bin/bash
#SBATCH -J m130m_m2_r1_1M_1g
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -o /data/run01/user/hch/logs/pq_motif_pile/%x-%j.out
#SBATCH --open-mode=append

set -eEuo pipefail
set -x
trap 'rc=$?; echo "[ERR] rc=${rc} line=${LINENO} time=$(date "+%F %T")"; exit ${rc}' ERR
trap 'rc=$?; echo "[EXIT] rc=${rc} time=$(date "+%F %T")"' EXIT

ROOT=${ROOT:-/home/user/run/hch}
SCRIPT=${SCRIPT:-${ROOT}/mamba-main/motif/train_kpq_motif_pile_lm.py}
PRETRAINED_DIR=${PRETRAINED_DIR:-${ROOT}/mamba-130m}
PILE_ROOT=${PILE_ROOT:-${ROOT}/datasets/pile_standard_pythia_10M}
PY_BIN=${PY_BIN:-/home/user/run/hch/miniconda3/envs/motifmamba/bin/python}

OUT_ROOT=${OUT_ROOT:-/data/run01/user/hch}
OUT_DIR=${OUT_DIR:-${OUT_ROOT}/logs/pq_motif_pile/mamba130m_motif2_rank1_seq256_1M_kpq_1gpu}
RUN_LOG=${RUN_LOG:-${OUT_ROOT}/logs/pq_motif_pile/train_${SLURM_JOB_ID}.log}
DEBUG_LOG=${DEBUG_LOG:-${OUT_ROOT}/logs/pq_motif_pile/debug_${SLURM_JOB_ID}.log}

DTYPE=${DTYPE:-float16}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-256}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000}

TOKENS_PER_STEP=$((BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS))
SAVE_EVERY=$((MAX_TRAIN_TOKENS / TOKENS_PER_STEP))
if [[ ${SAVE_EVERY} -lt 1 ]]; then SAVE_EVERY=1; fi

mkdir -p "${OUT_ROOT}/logs/pq_motif_pile" "${OUT_DIR}"
cd "${ROOT}"

exec > >(stdbuf -oL -eL tee -a "${DEBUG_LOG}") 2>&1

echo "[start] job=${SLURM_JOB_ID} host=$(hostname) time=$(date '+%F %T')"
echo "[cfg] batch=${BATCH_SIZE}, seq=${SEQ_LEN}, tokens_per_step=${TOKENS_PER_STEP}"
echo "[cfg] max_tokens=${MAX_TRAIN_TOKENS}, save_every_steps=${SAVE_EVERY}"

"${PY_BIN}" -c "import sys, numpy, torch; print('env ok:', sys.executable)"

touch "${RUN_LOG}" "${DEBUG_LOG}" "${OUT_DIR}/.write_test" || { echo "[error] output path not writable"; exit 1; }
rm -f "${OUT_DIR}/.write_test"

CUDA_VISIBLE_DEVICES=0 "${PY_BIN}" -u "${SCRIPT}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --pile-root "${PILE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --device cuda:0 \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  --pq-rank 1 \
  --no-pq-per-dim \
  --pq-k-init 1e-4 \
  --train-pq-k \
  --no-train-base-model \
  --motif-class 2 \
  --motif-coef 0.1 \
  --motif-amplitude 1e5 \
  --motif-bias 5e-5 \
  --warmup-lr 5e-4 \
  --warmup-coef 1e6 \
  --warmup-ratio 0.1 \
  --warmup-max-steps 5000 \
  --warmup-log-every 20 \
  --no-warmup-strict \
  --train-lr 1e-4 \
  --train-steps 0 \
  --train-epochs 1.0 \
  --max-train-tokens "${MAX_TRAIN_TOKENS}" \
  --min-train-tokens 0 \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --eval-every-steps 0 \
  --eval-tokens 200000 \
  --disable-early-stop \
  --grad-clip 0.5 \
  --log-every 10 \
  --save-every "${SAVE_EVERY}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "[done] out_dir=${OUT_DIR}"
echo "[done] run_log=${RUN_LOG}"
echo "[done] debug_log=${DEBUG_LOG}"
