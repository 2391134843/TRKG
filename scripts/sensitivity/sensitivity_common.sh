#!/usr/bin/env bash

run_beta_sens() {
  local DATASET="$1" TASK_ID="$2" BETA="$3" MAX_CANDS="$4" EPOCHS="$5"

  local LOG_FILE="${LOG_ROOT}/${TASK_ID}.log"
  local CAND_DIR="${CAND_ROOT}/${DATASET}_main"
  local CKPT_DIR="${CKPT_ROOT}/${DATASET}_${TASK_ID}_$(date +%F-%H%M.%S)"
  local TASK
  case "${DATASET}" in
    TCMKG)   TASK="TCM_KG" ;;
    PrimeKG) TASK="PrimeKG" ;;
  esac

  wait_for_state "main_${DATASET,,}_stage2" 2>&1 | tee -a "${LOG_FILE}"

  mark_state "${TASK_ID}" "running"
  trap 'mark_state "${TASK_ID}" "failed"' ERR

  mkdir -p "${CKPT_DIR}"
  log_banner "Sensitivity β=${BETA} | ${DATASET} | reuse main candidates" | tee -a "${LOG_FILE}"
  echo "Candidates: ${CAND_DIR}"   | tee -a "${LOG_FILE}"
  echo "GRPO ckpt:  ${CKPT_DIR}"   | tee -a "${LOG_FILE}"
  echo "Params:     beta=${BETA}  max_cand=${MAX_CANDS}  epochs=${EPOCHS}" | tee -a "${LOG_FILE}"

  "${PYTHON_BIN}" -u run_grpo_rerank.py \
    --chat-model "${CHAT_MODEL}" \
    --candidates-dir "${CAND_DIR}" \
    --task ${TASK} \
    --grpo-model-dir "${CKPT_DIR}" \
    --grpo-epochs ${EPOCHS} \
    --grpo-lr 5e-5 \
    --grpo-batch-size 8 \
    --grpo-num-samples 8 \
    --grpo-beta ${BETA} \
    --grpo-lora-r 16 --grpo-lora-alpha 32 \
    --grpo-print-freq 20 \
    --max-candidates ${MAX_CANDS} \
    --mode train_and_eval \
    --seed 42 \
    2>&1 | tee -a "${LOG_FILE}"

  mark_state "${TASK_ID}" "done"
  echo "========== DONE ${TASK_ID} ==========" | tee -a "${LOG_FILE}"
}

run_topk_sens() {
  local DATASET="$1" TASK_ID="$2" TOP_K="$3" DEFAULT_MAX="$4" BETA="$5" EPOCHS="$6" MAX_TOKENS="$7"

  local LOG_FILE="${LOG_ROOT}/${TASK_ID}.log"
  local CAND_LOG="${LOG_ROOT}/${TASK_ID}_candidate_gen.log"
  local CAND_DIR="${CAND_ROOT}/${DATASET}_topk${TOP_K}"
  local CKPT_DIR="${CKPT_ROOT}/${DATASET}_${TASK_ID}_$(date +%F-%H%M.%S)"
  local STAGE1_STATE="main_${DATASET,,}_stage1"
  local TASK DATA_DIR
  case "${DATASET}" in
    TCMKG)   TASK="TCM_KG";  DATA_DIR="${TCMKG_DATA_DIR}"   ;;
    PrimeKG) TASK="PrimeKG"; DATA_DIR="${PRIMEKG_DATA_DIR}" ;;
  esac

  wait_for_state "${STAGE1_STATE}" 2>&1 | tee -a "${LOG_FILE}"

  mark_state "${TASK_ID}" "running"
  trap 'mark_state "${TASK_ID}" "failed"' ERR

  local STAGE1_DIR STAGE1_CKT
  STAGE1_DIR=$(cat "${STATE_DIR}/${STAGE1_STATE}.model_dir")
  STAGE1_CKT="${STAGE1_DIR}/model_best.mdl"
  [[ -f "${STAGE1_CKT}" ]] || STAGE1_CKT="${STAGE1_DIR}/model_last.mdl"

  local MAX_CANDS=${DEFAULT_MAX}
  (( TOP_K < DEFAULT_MAX )) && MAX_CANDS=${TOP_K}

  mkdir -p "${CAND_DIR}" "${CKPT_DIR}"
  log_banner "Sensitivity top-K=${TOP_K} | ${DATASET}" | tee -a "${LOG_FILE}"
  echo "Stage1 ckpt:   ${STAGE1_CKT}"   | tee -a "${LOG_FILE}"
  echo "Candidates:    ${CAND_DIR}"     | tee -a "${LOG_FILE}"
  echo "GRPO ckpt:     ${CKPT_DIR}"     | tee -a "${LOG_FILE}"
  echo "Params: topK=${TOP_K}  maxCand=${MAX_CANDS}  beta=${BETA}  epochs=${EPOCHS}  maxTok=${MAX_TOKENS}" | tee -a "${LOG_FILE}"

  "${PYTHON_BIN}" -u generate_candidates.py \
      --task ${TASK} \
      --pretrained-model "${BASE_MODEL}" \
      --eval-model-path "${STAGE1_CKT}" \
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
      --max-num-tokens ${MAX_TOKENS} \
      --top-k ${TOP_K} \
      --output-dir "${CAND_DIR}" \
      --splits "train,valid,test" \
      2>&1 | tee "${CAND_LOG}"

  "${PYTHON_BIN}" -u run_grpo_rerank.py \
    --chat-model "${CHAT_MODEL}" \
    --candidates-dir "${CAND_DIR}" \
    --task ${TASK} \
    --grpo-model-dir "${CKPT_DIR}" \
    --grpo-epochs ${EPOCHS} \
    --grpo-lr 5e-5 \
    --grpo-batch-size 8 \
    --grpo-num-samples 8 \
    --grpo-beta ${BETA} \
    --grpo-lora-r 16 --grpo-lora-alpha 32 \
    --grpo-print-freq 20 \
    --max-candidates ${MAX_CANDS} \
    --mode train_and_eval \
    --seed 42 \
    2>&1 | tee -a "${LOG_FILE}"

  mark_state "${TASK_ID}" "done"
  echo "========== DONE ${TASK_ID} ==========" | tee -a "${LOG_FILE}"
}
