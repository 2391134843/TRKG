#!/usr/bin/env bash
set -e; set -o pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/sensitivity_common.sh"
cd "${TRKG_ROOT}"
run_beta_sens TCMKG "sens_tcmkg_beta_05" 0.5 5 12
