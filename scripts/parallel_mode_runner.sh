#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly UNIFIED_RUNNER="${PROJECT_ROOT}/scripts/run_unified_d1_d6.py"
readonly RUN_ROOT="${PROJECT_ROOT}/outputs/runs/$(date '+%Y%m%d_%H%M%S')_formal"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly DRY_RUN="${DRY_RUN:-0}"
readonly MAX_JOBS="${MAX_JOBS:-10}"

if (($# > 0)); then
    printf 'Usage: DRY_RUN=1 bash scripts/parallel_mode_runner.sh\n' >&2
    exit 2
fi
if [[ "$(pwd -P)" != "${PROJECT_ROOT}" ]]; then
    printf 'ERROR: run this script from the project root: %s\n' "${PROJECT_ROOT}" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    printf 'ERROR: DRY_RUN must be 0 or 1\n' >&2
    exit 2
fi

printf '[LEGACY WRAPPER] MAX_JOBS=%s is ignored; unified runner owns scheduling.\n' "${MAX_JOBS}"
if [[ "${DRY_RUN}" == "1" ]]; then
    exec "${PYTHON_BIN}" "${UNIFIED_RUNNER}" --output-dir "${RUN_ROOT}" --dry-run
fi
exec "${PYTHON_BIN}" "${UNIFIED_RUNNER}" --output-dir "${RUN_ROOT}"
