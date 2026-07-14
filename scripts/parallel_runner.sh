#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
readonly UNIFIED_RUNNER="${PROJECT_ROOT}/scripts/run_unified_d1_d6.py"
readonly RUN_ROOT="${PROJECT_ROOT}/outputs/runs/$(date '+%Y%m%d_%H%M%S')_formal"

resolve_python_bin() {
    local candidate
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if candidate="$(command -v "${PYTHON_BIN}" 2>/dev/null)"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        printf 'ERROR: PYTHON_BIN is not executable or on PATH: %s\n' "${PYTHON_BIN}" >&2
        return 127
    fi

    candidate="${PROJECT_ROOT}/.venv/bin/python"
    if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    if candidate="$(command -v python3 2>/dev/null)"; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    if candidate="$(command -v python 2>/dev/null)"; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    printf 'ERROR: no Python interpreter found; set PYTHON_BIN or create %s\n' \
        "${PROJECT_ROOT}/.venv/bin/python" >&2
    return 127
}

PYTHON_BIN="$(resolve_python_bin)"
readonly PYTHON_BIN

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
