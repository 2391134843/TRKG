#!/usr/bin/env bash
set -x
set -e
set -o pipefail

TASK="WN18RR"
DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"

LOG_DIR="${DIR}/logs"
DATA_DIR="${DIR}/data/${TASK}"
CANDIDATES_DIR="${DIR}/candidates/${TASK}_05B"
CHAT_MODEL="${CHAT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

LOG_TAG="${LOG_PREFIX:+${LOG_PREFIX}_}"

mkdir -p "${LOG_DIR}" "${CANDIDATES_DIR}"

MODEL_DIR=$(cat "${LOG_DIR}/${LOG_TAG}stage1_05b_model_dir.txt")
BEST_CKT="${MODEL_DIR}/model_best.mdl"
echo "========== Generating candidates with 0.5B model: ${BEST_CKT} =========="

CUDA_VISIBLE_DEVICES=0 python3 -u generate_candidates.py \
    --task ${TASK} \
    --pretrained-model "${BASE_MODEL:-Qwen/Qwen2.5-0.5B}" \
    --eval-model-path "${BEST_CKT}" \
    --train-path "${DATA_DIR}/train.txt.json" \
    --valid-path "${DATA_DIR}/valid.txt.json" \
    --test-path "${DATA_DIR}/test.txt.json" \
    --is-test \
    --use-link-graph \
    --use-amp \
    --batch-size 512 \
    --pooling mean \
    --proj-dim 256 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-num-tokens 64 \
    --top-k 10 \
    --output-dir "${CANDIDATES_DIR}" \
    --splits "valid,test,train" \
    2>&1 | tee "${LOG_DIR}/${LOG_TAG}candidate_gen_05B_${TASK}.log"

echo "========== Candidate generation done! Output: ${CANDIDATES_DIR} =========="

GRPO_MODEL_DIR="${DIR}/checkpoint/grpo_${TASK}_05B_$(date +%F-%H%M.%S)"
mkdir -p "${GRPO_MODEL_DIR}"

echo "========== Stage 2: GRPO (0.5B) for ${TASK} =========="

CUDA_VISIBLE_DEVICES=0 python3 -u run_grpo_rerank.py \
  --chat-model "${CHAT_MODEL}" \
  --candidates-dir "${CANDIDATES_DIR}" \
  --task ${TASK} \
  --grpo-model-dir "${GRPO_MODEL_DIR}" \
  --grpo-epochs 3 \
  --grpo-lr 5e-5 \
  --grpo-batch-size 8 \
  --grpo-num-samples 8 \
  --grpo-beta 0.1 \
  --grpo-lora-r 16 \
  --grpo-lora-alpha 32 \
  --grpo-print-freq 20 \
  --max-candidates 10 \
  --mode train_and_eval \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/${LOG_TAG}stage2_grpo_05B_${TASK}.log"

echo "========== Stage 2 (0.5B) done! Model: ${GRPO_MODEL_DIR} =========="
