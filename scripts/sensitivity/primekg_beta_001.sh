#!/usr/bin/env bash
set -e; set -o pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/sensitivity_common.sh"
cd "${TRKG_ROOT}"
run_beta_sens PrimeKG "sens_primekg_beta_001" 0.01 10 5
