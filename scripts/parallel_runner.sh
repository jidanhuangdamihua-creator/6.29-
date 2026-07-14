#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly UNIFIED_RUNNER="${PROJECT_ROOT}/scripts/run_unified_d1_d6.py"
readonly RUN_ROOT="${PROJECT_ROOT}/outputs/runs/$(date '+%Y%m%d_%H%M%S')_formal"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$(pwd -P)" != "${PROJECT_ROOT}" ]]; then
    printf 'ERROR: run this script from the project root: %s\n' "${PROJECT_ROOT}" >&2
    exit 2
fi
if [[ "${1:-}" == "--dry-run" && $# == 1 ]]; then
    exec "${PYTHON_BIN}" "${UNIFIED_RUNNER}" --output-dir "${RUN_ROOT}" --dry-run
fi
if (($# > 0)); then
    printf 'Usage: bash scripts/parallel_runner.sh [--dry-run]\n' >&2
    exit 2
fi

printf '[LEGACY WRAPPER] unified runner owns all 300 cells and acceptance.\n'
exec "${PYTHON_BIN}" "${UNIFIED_RUNNER}" --output-dir "${RUN_ROOT}"
