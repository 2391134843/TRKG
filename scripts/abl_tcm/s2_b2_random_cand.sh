#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

EXP_ID="s2_b2_random_cand"
DESC="Stage2 B2 | random-sampled candidates (stage-1 retrieval bypassed, baseline GRPO knobs)"
init_exp_log "${EXP_ID}" "${DESC}"

RAND_CAND_DIR="${CAND_ROOT}/TCMKG_random_cand_${EXP_ID}"
mkdir -p "${RAND_CAND_DIR}"

echo "[B2] generating uniform-random candidates -> ${RAND_CAND_DIR}" \
  | tee -a "${EXP_FULL_LOG}"
"${PYTHON_BIN}" -u "${TRKG_ROOT}/generate_random_candidates.py" \
  --src-dir       "${TCM_BASELINE_CAND_DIR}" \
  --entities-file "${TCM_DATA_DIR}/entities.json" \
  --output-dir    "${RAND_CAND_DIR}" \
  --num-candidates 20 \
  --seed 42 \
  2>&1 | tee -a "${EXP_FULL_LOG}"

S2_OUT="${CKPT_ROOT}/abl_${EXP_ID}_grpo_$(date +%F-%H%M.%S)"
mkdir -p "${S2_OUT}"
run_grpo "${S2_OUT}" "${RAND_CAND_DIR}"
mark_exp_done
