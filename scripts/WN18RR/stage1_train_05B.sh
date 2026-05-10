#!/usr/bin/env bash
set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
echo "working directory: ${DIR}"

OUTPUT_DIR="${DIR}/checkpoint/${TASK}_05B_$(date +%F-%H%M.%S)"
DATA_DIR="${DIR}/data/${TASK}"
LOG_DIR="${DIR}/logs"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"

LOG_TAG="${LOG_PREFIX:+${LOG_PREFIX}_}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

python3 -u main.py \
  --model-dir "${OUTPUT_DIR}" \
  --pretrained-model "${MODEL_NAME}" \
  --pooling mean \
  --lr 2e-4 \
  --use-link-graph \
  --train-path "${DATA_DIR}/train.txt.json" \
  --valid-path "${DATA_DIR}/valid.txt.json" \
  --test-path "${DATA_DIR}/test.txt.json" \
  --task ${TASK} \
  --batch-size 512 \
  --print-freq 20 \
  --additive-margin 0.02 \
  --use-amp \
  --use-self-negative \
  --finetune-t \
  --pre-batch 2 \
  --epochs 10 \
  --workers 4 \
  --max-to-keep 3 \
  --proj-dim 256 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-num-tokens 64 \
  --lr-scheduler cosine \
  --gradient-accumulation-steps 1 \
  --eval-every-n-step 500 \
  --seed 42 "$@" \
  2>&1 | tee "${LOG_DIR}/${LOG_TAG}stage1_train_05B_${TASK}.log"

echo "========== Stage 1 (0.5B) training done! Model: ${OUTPUT_DIR} =========="
echo "${OUTPUT_DIR}" > "${LOG_DIR}/${LOG_TAG}stage1_05b_model_dir.txt"
