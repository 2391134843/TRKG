#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

EXP_ID="primekg_s2_b2_random_cand"
DESC="PrimeKG Stage2 B2 | random-sampled candidates (stage-1 retrieval bypassed); epochs=3"
init_exp_log "${EXP_ID}" "${DESC}"

[[ -d "${PK_BASELINE_CAND_DIR}" ]] || {
  echo "[FATAL] missing baseline candidates dir ${PK_BASELINE_CAND_DIR}" \
    | tee -a "${EXP_FULL_LOG}" >&2
  exit 1
}

RAND_CAND_DIR="${CAND_ROOT}/PrimeKG_random_cand_${EXP_ID}"
mkdir -p "${RAND_CAND_DIR}"

echo "[B2] generating uniform-random candidates -> ${RAND_CAND_DIR}" \
  | tee -a "${EXP_FULL_LOG}"
"${PYTHON_BIN}" -u "${TRKG_ROOT}/generate_random_candidates.py" \
  --src-dir       "${PK_BASELINE_CAND_DIR}" \
  --entities-file "${PK_DATA_DIR}/entities.json" \
  --output-dir    "${RAND_CAND_DIR}" \
  --num-candidates 20 \
  --seed 42 \
  2>&1 | tee -a "${EXP_FULL_LOG}"

S2_OUT="${CKPT_ROOT}/abl_${EXP_ID}_grpo_$(date +%F-%H%M.%S)"
mkdir -p "${S2_OUT}"
run_grpo "${S2_OUT}" "${RAND_CAND_DIR}"
mark_exp_done
