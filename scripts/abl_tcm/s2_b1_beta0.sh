#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export S2_GRPO_BETA=0

source "${SCRIPT_DIR}/_common.sh"

EXP_ID="s2_b1_beta0"
DESC="Stage2 B1 | GRPO --grpo-beta 0 (no KL penalty); reuse pC candidates; else group-C"
init_exp_log "${EXP_ID}" "${DESC}"

[[ -d "${TCM_BASELINE_CAND_DIR}" ]] || {
  echo "[FATAL] missing baseline candidates dir ${TCM_BASELINE_CAND_DIR}" \
    | tee -a "${EXP_FULL_LOG}" >&2
  exit 1
}

S2_OUT="${CKPT_ROOT}/abl_${EXP_ID}_grpo_$(date +%F-%H%M.%S)"
mkdir -p "${S2_OUT}"
run_grpo "${S2_OUT}" "${TCM_BASELINE_CAND_DIR}"
mark_exp_done
