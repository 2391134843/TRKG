#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../common.sh"
cd "${TRKG_ROOT}"

TASK_ID="main_primekg_stage1"
TASK="PrimeKG"
DATA_DIR="${PRIMEKG_DATA_DIR}"
OUT_DIR="${CKPT_ROOT}/PrimeKG_main_stage1_$(date +%F-%H%M.%S)"
LOG_FILE="${LOG_ROOT}/${TASK_ID}.log"
MODEL_PTR="${STATE_DIR}/${TASK_ID}.model_dir"

mark_state "${TASK_ID}" "running"
trap 'mark_state "${TASK_ID}" "failed"' ERR

log_banner "Main Stage 1 | PrimeKG | Qwen2.5-0.5B" | tee -a "${LOG_FILE}"
check_dataset "${DATA_DIR}" 2>&1 | tee -a "${LOG_FILE}"
check_python_env 2>&1 | tee -a "${LOG_FILE}"
mkdir -p "${OUT_DIR}"
echo "Output dir: ${OUT_DIR}" | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u main.py \
  --model-dir "${OUT_DIR}" \
  --pretrained-model "${BASE_MODEL}" \
  --pooling last \
  --lr 1.3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.03 \
  --warmup 1500 \
  --use-link-graph \
  --use-self-negative \
  --finetune-t \
  --t 0.035 \
  --pre-batch 6 \
  --pre-batch-weight 0.7 \
  --additive-margin 0.0 \
  --neighbor-weight 0.15 \
  --train-path "${DATA_DIR}/train.txt.json" \
  --valid-path "${DATA_DIR}/valid.txt.json" \
  --test-path  "${DATA_DIR}/test.txt.json" \
  --task ${TASK} \
  --batch-size 256 \
  --epochs 22 \
  --print-freq 50 \
  --use-amp \
  --workers 4 \
  --max-to-keep 3 \
  --proj-dim 512 \
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.02 \
  --max-num-tokens 512 \
  --lr-scheduler cosine \
  --gradient-accumulation-steps 1 \
  --eval-every-n-step 500 \
  --seed 7 \
  2>&1 | tee -a "${LOG_FILE}"

echo "${OUT_DIR}" > "${MODEL_PTR}"
mark_state "${TASK_ID}" "done"
echo "========== DONE ${TASK_ID} | ckpt at ${OUT_DIR} ==========" | tee -a "${LOG_FILE}"
