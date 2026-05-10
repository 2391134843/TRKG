#!/usr/bin/env bash
set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
LOG_DIR="${DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " Full WN18RR 0.5B Experiment Pipeline"
echo " Start: $(date)"
echo "=============================================="

echo ">>> Stage 1: TRKG Training (0.5B) <<<"
bash "${DIR}/scripts/WN18RR/stage1_train_05B.sh"

echo ">>> Stage 2: Candidate Generation + GRPO Reranking (0.5B) <<<"
bash "${DIR}/scripts/WN18RR/stage2_grpo_05B.sh"

echo "=============================================="
echo " Full WN18RR 0.5B Experiment Pipeline Complete"
echo " End: $(date)"
echo " Logs: ${LOG_DIR}"
echo "=============================================="
