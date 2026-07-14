#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1-}" == "--dry-run" && $# == 1 ]]; then
    export DRY_RUN=1
    shift
fi
exec "${SCRIPT_DIR}/parallel_mode_runner.sh" "$@"
