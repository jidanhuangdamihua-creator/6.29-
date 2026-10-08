#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
exec python scripts/run_baselines_revin_40target.py --datasets D5,D6,D4
