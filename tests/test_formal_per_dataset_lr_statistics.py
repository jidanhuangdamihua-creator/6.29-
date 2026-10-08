"""Offline statistical audit for the completed formal per-dataset-LR matrix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "sita_ablation_formal_per_dataset_lr_ae_20260921_185137"
SCRIPT = PROJECT_ROOT / "scripts" / "analyze_formal_per_dataset_lr_ae.py"


def _module():
    spec = importlib.util.spec_from_file_location("formal_lr_stats", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_completed_formal_matrix_integrity_and_statistical_shape():
    module = _module()
    rows = module.load_formal_rows(RUN_ROOT)
    summary = module.integrity_audit_formal_rows(rows, RUN_ROOT)
    assert summary["n_rows"] == 300
    assert summary["n_missing"] == 0
    assert summary["n_duplicates"] == 0
    assert summary["n_failed"] == 0
    assert summary["n_nan"] == 0
    assert summary["runtime_lr"] == {
        "D1": "3e-5",
        "D4": "3e-5",
        "D5": "3e-4",
        "D6": "3e-5",
    }
    assert summary["d5_reused_rows"] == 75
    assert summary["d5_trained_rows"] == 0
    assert summary["non_d5_trained_rows"] == 225

    means, pairwise, statistics = module.compute_formal_statistics(rows)
    assert len(means) == 60
    assert len(pairwise) == 96
    assert len(statistics) == 8
    ca_rmse = next(row for row in statistics if row["comparison"] == "C-A" and row["metric"] == "RMSE")
    assert ca_rmse["n_pairs"] == 12
    assert ca_rmse["bootstrap_n"] == 100000
    assert ca_rmse["holm_family"] == "RMSE_4"
