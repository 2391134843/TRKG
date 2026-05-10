#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export S1_PRE_BATCH=0
export S1_USE_SELF_NEG=0

source "${SCRIPT_DIR}/_common.sh"

EXP_ID="s1_a1_no_negsamp"
DESC="Stage1 A1 | drop pre-batch + self-negative (pre-batch=0, no --use-self-negative); else group-C"
init_exp_log "${EXP_ID}" "${DESC}"

S1_OUT="${CKPT_ROOT}/abl_${EXP_ID}_$(date +%F-%H%M.%S)"
mkdir -p "${S1_OUT}"
run_stage1 "${S1_OUT}"
mark_exp_done
