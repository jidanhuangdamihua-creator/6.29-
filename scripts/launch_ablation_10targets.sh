#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results
set +e
python -u scripts/run_ablation_10targets.py --datasets D5,D6,D4
rc=$?
python -u scripts/analyze_ablation_10targets.py
exit "$rc"
