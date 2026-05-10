#!/usr/bin/env bash

TRKG_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
export TRKG_ROOT

export TCMKG_DATA_DIR="${TRKG_ROOT}/data/TCMKG"
export PRIMEKG_DATA_DIR="${TRKG_ROOT}/data/PrimeKG"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
export CHAT_MODEL="${CHAT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

export CKPT_ROOT="${TRKG_ROOT}/checkpoint"
export CAND_ROOT="${TRKG_ROOT}/candidates"
export LOG_ROOT="${TRKG_ROOT}/logs"
export STATE_DIR="${LOG_ROOT}/state"

mkdir -p "${CKPT_ROOT}" "${CAND_ROOT}" "${LOG_ROOT}" "${STATE_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  CONDA_SH="${CONDA_SH:-/usr/local/miniconda3/etc/profile.d/conda.sh}"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "torch20" ]] && [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate torch20 >/dev/null 2>&1 || true
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi
export PYTHON_BIN

export TOKENIZERS_PARALLELISM=false

log_banner() {
  local msg="$1"
  echo ""
  echo "============================================================"
  echo " ${msg}"
  echo " Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo " Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-all}"
  echo "============================================================"
}

check_dataset() {
  local data_dir="$1"
  if [[ ! -d "${data_dir}" ]]; then
    echo "[FATAL] Dataset dir not found: ${data_dir}" >&2
    return 1
  fi
  for f in train.txt.json valid.txt.json test.txt.json entities.json; do
    if [[ ! -f "${data_dir}/${f}" ]]; then
      echo "[FATAL] Missing required file: ${data_dir}/${f}" >&2
      return 1
    fi
  done
  return 0
}

check_python_env() {
  if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
    echo "[FATAL] torch not available in ${PYTHON_BIN}" >&2
    return 1
  fi
  "${PYTHON_BIN}" - <<'PY'
import re, sys
req = {"torch": (2, 0, 0), "transformers": (4, 37, 0), "peft": (0, 7, 0)}
def v(s): return tuple((list(map(int, re.findall(r"\d+", s))) + [0, 0, 0])[:3])
for p, m in req.items():
    try:
        mod = __import__(p)
    except Exception:
        sys.exit(f"missing {p}")
    if v(getattr(mod, "__version__", "0")) < m:
        sys.exit(f"{p} too old: {getattr(mod, '__version__', '?')} < {m}")
print("[env] torch/transformers/peft OK")
PY
}

mark_state() {
  local task_id="$1"
  local state="$2"   # "running" | "done" | "failed"
  echo "${state}  $(date '+%Y-%m-%d %H:%M:%S')  pid=$$" > "${STATE_DIR}/${task_id}.state"
}

wait_for_state() {
  local task_id="$1"
  local state_file="${STATE_DIR}/${task_id}.state"
  local waited=0
  while true; do
    if [[ -f "${state_file}" ]] && grep -q "^done" "${state_file}"; then
      return 0
    fi
    if [[ -f "${state_file}" ]] && grep -q "^failed" "${state_file}"; then
      echo "[FATAL] Dependency ${task_id} failed, aborting $$" >&2
      return 1
    fi
    sleep 15
    waited=$((waited + 15))
    if (( waited % 300 == 0 )); then
      echo "[wait] $$ still waiting for ${task_id} ... (${waited}s)"
    fi
  done
}
