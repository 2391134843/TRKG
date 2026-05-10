#!/usr/bin/env bash
set -x
set -e
set -o pipefail

TASK="WN18RR"
DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"

LOG_DIR="${DIR}/logs"
DATA_DIR="${DIR}/data/${TASK}"
CANDIDATES_DIR="${DIR}/candidates/${TASK}_7B"
DS_CONFIG="${DIR}/ds_config_stage2_7b.json"
CHAT_MODEL="${CHAT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

LOG_TAG="${LOG_PREFIX:+${LOG_PREFIX}_}"

mkdir -p "${LOG_DIR}" "${CANDIDATES_DIR}"

if [[ ! -f "${LOG_DIR}/${LOG_TAG}stage1_7b_model_dir.txt" ]]; then
  echo "Missing ${LOG_DIR}/stage1_7b_model_dir.txt, Stage 1 may not have finished successfully." >&2
  exit 1
fi

MODEL_DIR=$(cat "${LOG_DIR}/${LOG_TAG}stage1_7b_model_dir.txt")
BEST_CKT="${MODEL_DIR}/model_best.mdl"
if [[ ! -f "${BEST_CKT}" ]]; then
  if [[ -f "${MODEL_DIR}/model_last.mdl" ]]; then
    BEST_CKT="${MODEL_DIR}/model_last.mdl"
    echo "model_best.mdl not found, fallback to ${BEST_CKT}"
  else
    echo "Neither model_best.mdl nor model_last.mdl exists in ${MODEL_DIR}" >&2
    echo "Stage 1 likely crashed before checkpointing; inspect ${LOG_DIR}/stage1_train_7B_${TASK}.log." >&2
    exit 1
  fi
fi
echo "========== Generating candidates with 7B model: ${BEST_CKT} =========="

CUDA_VISIBLE_DEVICES=0 python3 generate_candidates.py \
    --task ${TASK} \
    --pretrained-model "${CHAT_MODEL}" \
    --eval-model-path "${BEST_CKT}" \
    --train-path "${DATA_DIR}/train.txt.json" \
    --valid-path "${DATA_DIR}/valid.txt.json" \
    --test-path "${DATA_DIR}/test.txt.json" \
    --batch-size 64 \
    --pooling mean \
    --proj-dim 256 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-num-tokens 64 \
    --top-k 10 \
    --output-dir "${CANDIDATES_DIR}" \
    --splits "valid,test,train" \
    2>&1 | tee "${LOG_DIR}/${LOG_TAG}candidate_gen_7B_${TASK}.log"

echo "========== Candidate generation done! Output: ${CANDIDATES_DIR} =========="

GRPO_MODEL_DIR="${DIR}/checkpoint/grpo_${TASK}_7B_deepspeed_$(date +%F-%H%M.%S)"
mkdir -p "${GRPO_MODEL_DIR}"

echo "========== Stage 2: DeepSpeed GRPO (7B) for ${TASK} =========="

deepspeed --num_gpus=8 run_grpo_rerank.py \
  --deepspeed_config "${DS_CONFIG}" \
  --chat-model "${CHAT_MODEL}" \
  --candidates-dir "${CANDIDATES_DIR}" \
  --task ${TASK} \
  --grpo-model-dir "${GRPO_MODEL_DIR}" \
  --grpo-epochs 3 \
  --grpo-lr 3e-5 \
  --grpo-batch-size 4 \
  --grpo-num-samples 8 \
  --grpo-beta 0.1 \
  --grpo-lora-r 16 \
  --grpo-lora-alpha 32 \
  --grpo-print-freq 20 \
  --max-candidates 10 \
  --ref-model-quantize 4bit \
  --mode train_and_eval \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/${LOG_TAG}stage2_grpo_7B_${TASK}.log"

echo "========== Stage 2 (7B) done! Model: ${GRPO_MODEL_DIR} =========="
