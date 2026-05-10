#!/usr/bin/env bash

TRKG_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../.. && pwd )"
export TRKG_ROOT

source "${TRKG_ROOT}/scripts/common.sh"

export BASE_MODEL_7B="${BASE_MODEL_7B:-Qwen/Qwen2.5-7B}"
export CHAT_MODEL_7B="${CHAT_MODEL_7B:-Qwen/Qwen2.5-7B-Instruct}"

export RUN7B_LOG_DIR="${LOG_ROOT}/runs7b"
mkdir -p "${RUN7B_LOG_DIR}"

export DS_CFG_STAGE1_7B="${TRKG_ROOT}/ds_config_stage1_7b.json"
export DS_CFG_STAGE2_7B="${TRKG_ROOT}/ds_config_stage2_7b.json"

wait_for_7b_models() {
  BASE_MODEL_7B="${BASE_MODEL_7B}" CHAT_MODEL_7B="${CHAT_MODEL_7B}" python3 - <<'PY'
import json, os, sys, time
TARGETS = [
    p for p in (os.environ.get("BASE_MODEL_7B", ""),
                os.environ.get("CHAT_MODEL_7B", ""))
    if p and os.path.isabs(p)  # only check local-path targets
]
if not TARGETS:
    print("[wait] no local 7B model paths to wait for; skipping.")
    sys.exit(0)
def ready(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx) or not os.path.exists(os.path.join(d, "tokenizer_config.json")):
        return False
    with open(idx) as f:
        meta = json.load(f)
    shards = sorted({v for v in meta.get("weight_map", {}).values()})
    for s in shards:
        p = os.path.join(d, s)
        if not os.path.exists(p):
            return False
        if os.path.getsize(p) < 1_000_000:
            return False
    return True
deadline = time.time() + 24 * 3600
while True:
    miss = [d for d in TARGETS if not ready(d)]
    if not miss:
        print("[wait] both 7B models ready.")
        sys.exit(0)
    print(f"[wait] still downloading: {miss}", flush=True)
    if time.time() > deadline:
        sys.exit("[fatal] 7B models did not finish downloading in 24h")
    time.sleep(60)
PY
}

ds_master_port_for_gpu() {
  local gpu="$1"   # e.g. "3"
  echo $(( 29500 + gpu ))
}
