#!/usr/bin/env bash

__abl_self="${BASH_SOURCE[0]}"
__abl_dir="$( cd "$( dirname "${__abl_self}" )" && pwd )"
source "${__abl_dir}/../main7b/_common_7b.sh"
cd "${TRKG_ROOT}"

export EXP_LOG_ROOT="${TRKG_ROOT}/exp_log"
mkdir -p "${EXP_LOG_ROOT}"

export PK_TASK="PrimeKG"
export PK_DATA_DIR="${PRIMEKG_DATA_DIR}"
export PK_BASELINE_CAND_DIR="${CAND_ROOT}/PrimeKG_primekg_7b"

: "${S1_POOLING:=last}"
: "${S1_LR:=6e-5}"
: "${S1_WD:=5e-5}"
: "${S1_WARMUP:=1500}"
: "${S1_T_INIT:=0.04}"
: "${S1_PRE_BATCH:=4}"
: "${S1_PRE_BATCH_W:=0.6}"
: "${S1_ADD_MARGIN:=0.0}"
: "${S1_NEIGH_WT:=0.15}"
: "${S1_BATCH:=16}"           # per-GPU; 4 GPUs -> effective 64 (baseline)
: "${S1_GRAD_ACCUM:=1}"
: "${S1_EPOCHS:=3}"
: "${S1_PROJ_DIM:=384}"
: "${S1_LORA_R:=64}"
: "${S1_LORA_A:=128}"
: "${S1_LORA_DROPOUT:=0.03}"
: "${S1_MAX_TOK:=384}"
: "${S1_EVAL_EVERY:=4000}"    # mid-epoch eval is expensive; baseline already loose
: "${S1_PRINT_FREQ:=50}"
: "${S1_WORKERS:=8}"
: "${S1_SEED:=42}"
: "${S1_USE_LINK_GRAPH:=1}"
: "${S1_USE_SELF_NEG:=1}"
: "${S1_FINETUNE_T:=1}"

: "${S2_GRPO_EPOCHS:=3}"
: "${S2_GRPO_LR:=5e-5}"
: "${S2_GRPO_BATCH:=4}"
: "${S2_GRPO_NUM_SAMPLES:=8}"
: "${S2_GRPO_BETA:=0.05}"
: "${S2_GRPO_LORA_R:=32}"
: "${S2_GRPO_LORA_A:=64}"
: "${S2_GRPO_MAX_LEN:=1024}"
: "${S2_MAX_CAND:=10}"
: "${S2_GRPO_PRINT_FREQ:=50}"
: "${S2_REF_QUANTIZE:=4bit}"
: "${S2_SEED:=42}"

init_exp_log() {
  export EXP_ID="$1"
  local desc="$2"
  export EXP_DIR="${EXP_LOG_ROOT}/${EXP_ID}"
  mkdir -p "${EXP_DIR}"
  export EXP_FULL_LOG="${EXP_DIR}/full.log"
  export EXP_CONFIG="${EXP_DIR}/config.txt"
  export EXP_STATE="${EXP_DIR}/STATE"

  echo "running  $(date '+%F %T')  pid=$$  gpus=${MAIN7B_GPUS:-?}" > "${EXP_STATE}"

  {
    echo "============================================================"
    echo "EXP_ID:    ${EXP_ID}"
    echo "DESC:      ${desc}"
    echo "GPUS:      ${MAIN7B_GPUS:-?}"
    echo "STARTED:   $(date '+%F %T')"
    echo "BASELINE:  PrimeKG 7B (scripts/main7b/primekg_7b.sh)"
    echo "HOST:      $(hostname)"
    echo "============================================================"
  } > "${EXP_CONFIG}"

  trap '__on_exp_fail' ERR
}

__on_exp_fail() {
  echo "failed  $(date '+%F %T')  pid=$$" > "${EXP_STATE}"
  echo "[FAIL] ${EXP_ID} at $(date '+%F %T')" | tee -a "${EXP_FULL_LOG}" >&2
}

mark_exp_done() {
  echo "done    $(date '+%F %T')  pid=$$" > "${EXP_STATE}"
  echo "==== ${EXP_ID} DONE at $(date '+%F %T') ====" | tee -a "${EXP_FULL_LOG}"
}

__pk_ds_port() {
  local first_gpu="${MAIN7B_GPUS%%,*}"
  echo $(( 29700 + first_gpu ))
}

run_stage1() {
  local out_dir="$1"
  local DS_GPUS="${MAIN7B_GPUS:?MAIN7B_GPUS must be set, e.g. \"4,5,6,7\"}"
  local DS_PORT
  DS_PORT=$(__pk_ds_port)

  local args=(
    --deepspeed "${DS_CFG_STAGE1_7B}"
    --model-dir "${out_dir}"
    --pretrained-model "${BASE_MODEL_7B}"
    --pooling "${S1_POOLING}"
    --lr "${S1_LR}"
    --weight-decay "${S1_WD}"
    --warmup "${S1_WARMUP}"
    --t "${S1_T_INIT}"
    --pre-batch "${S1_PRE_BATCH}"
    --pre-batch-weight "${S1_PRE_BATCH_W}"
    --additive-margin "${S1_ADD_MARGIN}"
    --neighbor-weight "${S1_NEIGH_WT}"
    --train-path "${PK_DATA_DIR}/train.txt.json"
    --valid-path "${PK_DATA_DIR}/valid.txt.json"
    --test-path  "${PK_DATA_DIR}/test.txt.json"
    --task "${PK_TASK}"
    --batch-size "${S1_BATCH}"
    --epochs "${S1_EPOCHS}"
    --print-freq "${S1_PRINT_FREQ}"
    --use-amp
    --workers "${S1_WORKERS}"
    --max-to-keep 2
    --proj-dim "${S1_PROJ_DIM}"
    --lora-r "${S1_LORA_R}" --lora-alpha "${S1_LORA_A}" --lora-dropout "${S1_LORA_DROPOUT}"
    --max-num-tokens "${S1_MAX_TOK}"
    --lr-scheduler cosine
    --gradient-accumulation-steps "${S1_GRAD_ACCUM}"
    --eval-every-n-step "${S1_EVAL_EVERY}"
    --seed "${S1_SEED}"
  )
  [[ "${S1_USE_LINK_GRAPH}" == "1" ]] && args+=( --use-link-graph )
  [[ "${S1_USE_SELF_NEG}"   == "1" ]] && args+=( --use-self-negative )
  [[ "${S1_FINETUNE_T}"     == "1" ]] && args+=( --finetune-t )

  {
    echo ""
    echo "=== STAGE 1 RESOLVED CMD (GPUs=${DS_GPUS}, master_port=${DS_PORT}) ==="
    echo "deepspeed --include localhost:${DS_GPUS} --master_port ${DS_PORT} main.py \\"
    printf '  %s \\\n' "${args[@]}"
    echo ""
    echo "=== ENV OVERRIDES ==="
    for v in S1_POOLING S1_LR S1_WD S1_WARMUP S1_T_INIT \
             S1_PRE_BATCH S1_PRE_BATCH_W S1_ADD_MARGIN S1_NEIGH_WT \
             S1_USE_LINK_GRAPH S1_USE_SELF_NEG S1_FINETUNE_T \
             S1_BATCH S1_EPOCHS S1_PROJ_DIM \
             S1_LORA_R S1_LORA_A S1_LORA_DROPOUT \
             S1_MAX_TOK S1_GRAD_ACCUM S1_EVAL_EVERY S1_WORKERS S1_SEED; do
      eval "echo \"  ${v} = \${${v}}\""
    done
  } >> "${EXP_CONFIG}"

  log_banner "ABL ${EXP_ID} | Stage 1 | GPUs=${DS_GPUS} | out=${out_dir}" \
    | tee -a "${EXP_FULL_LOG}"

  deepspeed --include "localhost:${DS_GPUS}" --master_port "${DS_PORT}" main.py \
    "${args[@]}" 2>&1 | tee -a "${EXP_FULL_LOG}"

  echo "${out_dir}" > "${EXP_DIR}/stage1_ckpt_dir.txt"
}

run_grpo() {
  local out_dir="$1"
  local cand_dir="$2"
  local DS_GPUS="${MAIN7B_GPUS:?MAIN7B_GPUS must be set, e.g. \"4,5,6,7\"}"
  local DS_PORT
  DS_PORT=$(__pk_ds_port)

  local args=(
    --deepspeed_config "${DS_CFG_STAGE2_7B}"
    --chat-model "${CHAT_MODEL_7B}"
    --candidates-dir "${cand_dir}"
    --task "${PK_TASK}"
    --grpo-model-dir "${out_dir}"
    --grpo-epochs "${S2_GRPO_EPOCHS}"
    --grpo-lr "${S2_GRPO_LR}"
    --grpo-batch-size "${S2_GRPO_BATCH}"
    --grpo-num-samples "${S2_GRPO_NUM_SAMPLES}"
    --grpo-beta "${S2_GRPO_BETA}"
    --grpo-lora-r "${S2_GRPO_LORA_R}" --grpo-lora-alpha "${S2_GRPO_LORA_A}"
    --grpo-print-freq "${S2_GRPO_PRINT_FREQ}"
    --grpo-max-length "${S2_GRPO_MAX_LEN}"
    --max-candidates "${S2_MAX_CAND}"
    --mode train_and_eval
    --seed "${S2_SEED}"
  )
  [[ -n "${S2_REF_QUANTIZE}" ]] && args+=( --ref-model-quantize "${S2_REF_QUANTIZE}" )

  {
    echo ""
    echo "=== STAGE 2 GRPO RESOLVED CMD (GPUs=${DS_GPUS}, master_port=${DS_PORT}) ==="
    echo "deepspeed --include localhost:${DS_GPUS} --master_port ${DS_PORT} run_grpo_rerank.py \\"
    printf '  %s \\\n' "${args[@]}"
    echo ""
    echo "=== ENV OVERRIDES ==="
    for v in S2_GRPO_EPOCHS S2_GRPO_LR S2_GRPO_BATCH S2_GRPO_NUM_SAMPLES \
             S2_GRPO_BETA S2_GRPO_LORA_R S2_GRPO_LORA_A S2_GRPO_MAX_LEN \
             S2_MAX_CAND S2_REF_QUANTIZE S2_SEED; do
      eval "echo \"  ${v} = \${${v}}\""
    done
  } >> "${EXP_CONFIG}"

  log_banner "ABL ${EXP_ID} | Stage 2 GRPO | GPUs=${DS_GPUS} | cand=${cand_dir}" \
    | tee -a "${EXP_FULL_LOG}"

  deepspeed --include "localhost:${DS_GPUS}" --master_port "${DS_PORT}" run_grpo_rerank.py \
    "${args[@]}" 2>&1 | tee -a "${EXP_FULL_LOG}"

  echo "${out_dir}" > "${EXP_DIR}/stage2_ckpt_dir.txt"
}
