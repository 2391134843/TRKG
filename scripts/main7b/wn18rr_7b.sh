#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common_7b.sh"
cd "${TRKG_ROOT}"

JOB="wn18rr_7b"
TASK="WN18RR"
DATA_DIR="${TRKG_ROOT}/data/${TASK}"

DS_GPUS="${MAIN7B_GPUS:-0,1,2,3}"
FIRST_GPU="${DS_GPUS%%,*}"
DS_PORT=$(( 29700 + FIRST_GPU ))

S1_TASK_ID="main_${JOB}_stage1"
S2_TASK_ID="main_${JOB}_stage2"
TS=$(date +%F-%H%M.%S)
S1_OUT="${CKPT_ROOT}/WN18RR_${JOB}_stage1_${TS}"
S2_GRPO_OUT="${CKPT_ROOT}/WN18RR_${JOB}_grpo_${TS}"
CAND_DIR="${CAND_ROOT}/WN18RR_${JOB}"

TAG="${MAIN7B_TAG:-main}"
LOG_OUT_DIR="${TRKG_ROOT}/logs/log-new/wn18rr/${TAG}"
mkdir -p "${LOG_OUT_DIR}"
S1_LOG="${LOG_OUT_DIR}/stage1.log"
CAND_LOG="${LOG_OUT_DIR}/candidate_gen.log"
GRPO_LOG="${LOG_OUT_DIR}/grpo.log"
JOB_LOG="${LOG_OUT_DIR}/full.log"

wait_for_7b_models 2>&1 | tee -a "${JOB_LOG}"
mkdir -p "${S1_OUT}" "${S2_GRPO_OUT}" "${CAND_DIR}"

mark_state "${S1_TASK_ID}" "running"
trap 'mark_state "${S1_TASK_ID}" "failed"' ERR

log_banner "Stage1 7B | WN18RR | quad-GPU [${DS_GPUS}] | port ${DS_PORT}" \
  | tee -a "${JOB_LOG}" "${S1_LOG}"
echo "Output: ${S1_OUT}" | tee -a "${JOB_LOG}"

deepspeed --include "localhost:${DS_GPUS}" --master_port "${DS_PORT}" main.py \
  --deepspeed "${DS_CFG_STAGE1_7B}" \
  --model-dir "${S1_OUT}" \
  --pretrained-model "${BASE_MODEL_7B}" \
  --pooling mean \
  --lr 6e-5 \
  --weight-decay 5e-5 \
  --warmup 1500 \
  --t 0.05 \
  --use-link-graph \
  --use-self-negative \
  --finetune-t \
  --pre-batch 4 \
  --pre-batch-weight 0.5 \
  --additive-margin 0.02 \
  --neighbor-weight 0.10 \
  --train-path "${DATA_DIR}/train.txt.json" \
  --valid-path "${DATA_DIR}/valid.txt.json" \
  --test-path  "${DATA_DIR}/test.txt.json" \
  --task ${TASK} \
  --batch-size 16 \
  --epochs 5 \
  --print-freq 50 \
  --use-amp \
  --workers 6 \
  --max-to-keep 3 \
  --proj-dim 256 \
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 \
  --max-num-tokens 64 \
  --lr-scheduler cosine \
  --gradient-accumulation-steps 1 \
  --eval-every-n-step 2000 \
  --seed 42 \
  2>&1 | tee -a "${S1_LOG}" "${JOB_LOG}"

echo "${S1_OUT}" > "${STATE_DIR}/${S1_TASK_ID}.model_dir"
mark_state "${S1_TASK_ID}" "done"
echo "==== stage1 done ====" | tee -a "${JOB_LOG}"

mark_state "${S2_TASK_ID}" "running"
trap 'mark_state "${S2_TASK_ID}" "failed"' ERR

BEST_CKT="${S1_OUT}/model_best.mdl"
[[ ! -f "${BEST_CKT}" ]] && BEST_CKT="${S1_OUT}/model_last.mdl"
[[ ! -f "${BEST_CKT}" ]] && { echo "no checkpoint produced by Stage 1" >&2 | tee -a "${JOB_LOG}"; exit 1; }

log_banner "Stage2 7B | WN18RR-4gpu | candidate-gen | GPU ${FIRST_GPU}" \
  | tee -a "${JOB_LOG}" "${CAND_LOG}"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" "${PYTHON_BIN}" -u generate_candidates.py \
    --task ${TASK} \
    --pretrained-model "${BASE_MODEL_7B}" \
    --eval-model-path "${BEST_CKT}" \
    --train-path "${DATA_DIR}/train.txt.json" \
    --valid-path "${DATA_DIR}/valid.txt.json" \
    --test-path  "${DATA_DIR}/test.txt.json" \
    --is-test \
    --use-link-graph \
    --use-amp \
    --batch-size 64 \
    --pooling mean \
    --proj-dim 256 \
    --lora-r 64 --lora-alpha 128 \
    --max-num-tokens 64 \
    --top-k 20 \
    --output-dir "${CAND_DIR}" \
    --splits "train,valid,test" \
    2>&1 | tee -a "${CAND_LOG}" "${JOB_LOG}"

log_banner "Stage2 7B | WN18RR-4gpu | GRPO | GPUs [${DS_GPUS}]" \
  | tee -a "${JOB_LOG}" "${GRPO_LOG}"
deepspeed --include "localhost:${DS_GPUS}" --master_port "${DS_PORT}" run_grpo_rerank.py \
  --deepspeed_config "${DS_CFG_STAGE2_7B}" \
  --chat-model "${CHAT_MODEL_7B}" \
  --candidates-dir "${CAND_DIR}" \
  --task ${TASK} \
  --grpo-model-dir "${S2_GRPO_OUT}" \
  --grpo-epochs 4 \
  --grpo-lr 5e-5 \
  --grpo-batch-size 4 \
  --grpo-num-samples 8 \
  --grpo-beta 0.1 \
  --grpo-lora-r 32 --grpo-lora-alpha 64 \
  --grpo-print-freq 50 \
  --grpo-max-length 512 \
  --max-candidates 10 \
  --mode train_and_eval \
  --ref-model-quantize 4bit \
  --seed 42 \
  2>&1 | tee -a "${GRPO_LOG}" "${JOB_LOG}"

mark_state "${S2_TASK_ID}" "done"
echo "==== ${JOB} all done | s1=${S1_OUT} | grpo=${S2_GRPO_OUT} ====" \
  | tee -a "${JOB_LOG}"
