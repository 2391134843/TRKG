#!/usr/bin/env bash
set -x
set -e
set -o pipefail

TASK="${1:?Missing TASK arg}"
CKPT_DIR="${2:?Missing CKPT_DIR arg}"
CKPT="${CKPT_DIR}/model_best.mdl"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
cd "${DIR}"

LOG_DIR="${DIR}/logs"
mkdir -p "${LOG_DIR}"

case "${TASK}" in
  WN18RR)   DATA_DIR="${DIR}/data/${TASK}" ;;
  FB15k237) DATA_DIR="${DIR}/data/${TASK}" ;;
  TCM_KG)   DATA_DIR="${DIR}/data/TCMKG/TCM_KG" ;;
  PrimeKG)  DATA_DIR="${DIR}/data/${TASK}" ;;
  *)        echo "Unknown task: ${TASK}" >&2; exit 1 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_FILE="${LOG_DIR}/eval_per_example_${TASK}.log"

echo "==================================================" | tee "${LOG_FILE}"
echo "  Per-example eval on TEST set: ${TASK}"     | tee -a "${LOG_FILE}"
echo "  Checkpoint: ${CKPT}"                       | tee -a "${LOG_FILE}"
echo "  Data dir: ${DATA_DIR}"                     | tee -a "${LOG_FILE}"
echo "  Start: $(date)"                            | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" -u evaluate.py \
    --task "${TASK}" \
    --is-test \
    --pretrained-model "${BASE_MODEL:-Qwen/Qwen2.5-0.5B}" \
    --eval-model-path "${CKPT}" \
    --train-path "${DATA_DIR}/train.txt.json" \
    --valid-path "${DATA_DIR}/test.txt.json" \
    --use-link-graph \
    --batch-size 512 \
    --pooling mean \
    --proj-dim 256 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-num-tokens 64 \
    2>&1 | tee -a "${LOG_FILE}"

echo "==================================================" | tee -a "${LOG_FILE}"
echo "  Done: ${TASK} at $(date)"                   | tee -a "${LOG_FILE}"
echo "  Output JSONs at: ${CKPT_DIR}/eval_test.txt.json_{forward,backward}_model_best.mdl.json" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"
