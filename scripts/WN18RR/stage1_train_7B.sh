#!/usr/bin/env bash
set -x
set -e
set -o pipefail

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
echo "working directory: ${DIR}"

OUTPUT_DIR="${DIR}/checkpoint/${TASK}_7B_deepspeed_$(date +%F-%H%M.%S)"
DATA_DIR="${DIR}/data/${TASK}"
LOG_DIR="${DIR}/logs"
DS_CONFIG="${DIR}/ds_config_stage1_7b.json"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

LOG_TAG="${LOG_PREFIX:+${LOG_PREFIX}_}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

deepspeed --num_gpus=8 main.py \
  --deepspeed "${DS_CONFIG}" \
  --model-dir "${OUTPUT_DIR}" \
  --pretrained-model "${MODEL_NAME}" \
  --pooling mean \
  --lr 1e-4 \
  --use-link-graph \
  --train-path "${DATA_DIR}/train.txt.json" \
  --valid-path "${DATA_DIR}/valid.txt.json" \
  --test-path "${DATA_DIR}/test.txt.json" \
  --task ${TASK} \
  --batch-size 16 \
  --print-freq 20 \
  --additive-margin 0.02 \
  --use-self-negative \
  --finetune-t \
  --pre-batch 0 \
  --epochs 10 \
  --workers 4 \
  --max-to-keep 3 \
  --proj-dim 256 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-num-tokens 64 \
  --lr-scheduler cosine \
  --gradient-accumulation-steps 2 \
  --eval-every-n-step 500 \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/${LOG_TAG}stage1_train_7B_${TASK}.log"

if [[ ! -f "${OUTPUT_DIR}/model_best.mdl" && ! -f "${OUTPUT_DIR}/model_last.mdl" ]]; then
  echo "Stage 1 finished without exporting checkpoint in ${OUTPUT_DIR}" >&2
  echo "Please check ${LOG_DIR}/stage1_train_7B_${TASK}.log for the first CUDA/DeepSpeed error." >&2
  exit 1
fi

echo "========== Stage 1 (7B) training done! Model: ${OUTPUT_DIR} =========="
echo "${OUTPUT_DIR}" > "${LOG_DIR}/${LOG_TAG}stage1_7b_model_dir.txt"
