#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results
# D4 prediction cells are already on disk. D5/D6 train Stage A first, then SITA.
# TensorFlow is recycled via execv every 15 newly trained/adapted items.
python -u scripts/run_sita_sales_only_ablation.py --phase stage-a --datasets D5,D6
python -u scripts/run_sita_sales_only_ablation.py --phase sita --datasets D5,D6
python -u scripts/run_sita_sales_only_ablation.py --phase sita --datasets D4
python -u scripts/analyze_sita_sales_only_ablation.py
