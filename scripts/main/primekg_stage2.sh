#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../common.sh"
cd "${TRKG_ROOT}"

TASK_ID="main_primekg_stage2"
TASK="PrimeKG"
DATA_DIR="${PRIMEKG_DATA_DIR}"
CAND_DIR="${CAND_ROOT}/PrimeKG_main"
CAND_LOG="${LOG_ROOT}/${TASK_ID}_candidate_gen.log"
GRPO_LOG="${LOG_ROOT}/${TASK_ID}_grpo.log"
LOG_FILE="${LOG_ROOT}/${TASK_ID}.log"

wait_for_state "main_primekg_stage1" 2>&1 | tee -a "${LOG_FILE}"

mark_state "${TASK_ID}" "running"
trap 'mark_state "${TASK_ID}" "failed"' ERR

STAGE1_DIR=$(cat "${STATE_DIR}/main_primekg_stage1.model_dir")
BEST_CKT="${STAGE1_DIR}/model_best.mdl"
if [[ ! -f "${BEST_CKT}" ]]; then
  [[ -f "${STAGE1_DIR}/model_last.mdl" ]] && BEST_CKT="${STAGE1_DIR}/model_last.mdl" \
    || { echo "No checkpoint in ${STAGE1_DIR}" >&2; exit 1; }
fi

GRPO_DIR="${CKPT_ROOT}/PrimeKG_main_grpo_$(date +%F-%H%M.%S)"
mkdir -p "${CAND_DIR}" "${GRPO_DIR}"

log_banner "Main Stage 2 | PrimeKG | Qwen2.5-0.5B(-Instruct) | ckpt=${BEST_CKT}" | tee -a "${LOG_FILE}"

echo "[Step 1] Generating candidates -> ${CAND_DIR}" | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u generate_candidates.py \
    --task ${TASK} \
    --pretrained-model "${BASE_MODEL}" \
    --eval-model-path "${BEST_CKT}" \
    --train-path "${DATA_DIR}/train.txt.json" \
    --valid-path "${DATA_DIR}/valid.txt.json" \
    --test-path  "${DATA_DIR}/test.txt.json" \
    --is-test \
    --use-link-graph \
    --use-amp \
    --batch-size 512 \
    --pooling mean \
    --proj-dim 256 \
    --lora-r 16 --lora-alpha 32 \
    --max-num-tokens 192 \
    --top-k 20 \
    --output-dir "${CAND_DIR}" \
    --splits "train,valid,test" \
    2>&1 | tee "${CAND_LOG}"

echo "[Step 2] GRPO training -> ${GRPO_DIR}" | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u run_grpo_rerank.py \
  --chat-model "${CHAT_MODEL}" \
  --candidates-dir "${CAND_DIR}" \
  --task ${TASK} \
  --grpo-model-dir "${GRPO_DIR}" \
  --grpo-epochs 5 \
  --grpo-lr 5e-5 \
  --grpo-batch-size 8 \
  --grpo-num-samples 8 \
  --grpo-beta 0.2 \
  --grpo-lora-r 16 --grpo-lora-alpha 32 \
  --grpo-print-freq 20 \
  --max-candidates 10 \
  --mode train_and_eval \
  --seed 42 \
  2>&1 | tee "${GRPO_LOG}"

mark_state "${TASK_ID}" "done"
echo "========== DONE ${TASK_ID} | GRPO ckpt at ${GRPO_DIR} ==========" | tee -a "${LOG_FILE}"
