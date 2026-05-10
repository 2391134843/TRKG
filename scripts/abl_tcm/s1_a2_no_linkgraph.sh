#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export S1_USE_LINK_GRAPH=0

source "${SCRIPT_DIR}/_common.sh"

EXP_ID="s1_a2_no_linkgraph"
DESC="Stage1 A2 | drop link-graph neighbor context (no --use-link-graph); else group-C"
init_exp_log "${EXP_ID}" "${DESC}"

S1_OUT="${CKPT_ROOT}/abl_${EXP_ID}_$(date +%F-%H%M.%S)"
mkdir -p "${S1_OUT}"
run_stage1 "${S1_OUT}"
mark_exp_done
