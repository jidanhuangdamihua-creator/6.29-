#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPECTED_BRANCH="codex/改"
CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
CURRENT_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG_DIR="/tmp/d5_knn_contract_fix_acceptance_${STAMP}"
SEALED_ROOT="$ROOT/数据集/固化数据/d1_d6_sealed_v1"

mkdir -p "$LOG_DIR"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  {
    printf '%s\n' '# FINAL_REPORT' '' 'BLOCKED_BY_BRANCH_MISMATCH' '' "branch=${CURRENT_BRANCH}" "expected_branch=${EXPECTED_BRANCH}"
  } > "$LOG_DIR/FINAL_REPORT.md"
  printf 'Acceptance blocked; report: %s\n' "$LOG_DIR/FINAL_REPORT.md" >&2
  exit 1
fi

COMMAND=(
  python tools/protection/codex_timeout.py --timeout 600 python
  tools/operations/gate1x_unified_acceptance.py
  --repository-root "$ROOT"
  --expected-branch "$EXPECTED_BRANCH"
  --expected-head "$CURRENT_HEAD"
  --sealed-root "$SEALED_ROOT"
  --output-dir "$LOG_DIR/gate1x"
  --run-full-tests
)

set +e
(cd "$ROOT" && "${COMMAND[@]}" > "$LOG_DIR/gate1x.stdout.log" 2> "$LOG_DIR/gate1x.stderr.log")
EXIT_CODE=$?
set -e

if [[ "$EXIT_CODE" -eq 0 ]]; then
  {
    printf '%s\n' '# FINAL_REPORT' '' 'FIXED' '' '# IDENTITY' '' "branch=${CURRENT_BRANCH}" "head=${CURRENT_HEAD}" '' '# ACCEPTANCE' '' 'gate1x_unified_acceptance=ACCEPTED' 'formal_training_started=false' 'formal_results_created=false' 'publication_performed=false' '' '# LOG_DIR' '' "$LOG_DIR"
  } > "$LOG_DIR/FINAL_REPORT.md"
  printf 'ACCEPTED\nLOG_DIR=%s\nFINAL_REPORT=%s\n' "$LOG_DIR" "$LOG_DIR/FINAL_REPORT.md"
  exit 0
fi

{
  printf '%s\n' '# FINAL_REPORT' '' 'BLOCKED_BY_ACCEPTANCE_FAILURE' '' '# IDENTITY' '' "branch=${CURRENT_BRANCH}" "head=${CURRENT_HEAD}" '' '# ACCEPTANCE' '' "gate1x_unified_acceptance_exit=${EXIT_CODE}" '' '# LOG_DIR' '' "$LOG_DIR" '' '# NEXT' '' 'Read gate1x.stdout.log and gate1x.stderr.log for the exact fail-closed stage and error code.'
} > "$LOG_DIR/FINAL_REPORT.md"
printf 'REJECTED\nLOG_DIR=%s\nFINAL_REPORT=%s\n' "$LOG_DIR" "$LOG_DIR/FINAL_REPORT.md" >&2
exit "$EXIT_CODE"
