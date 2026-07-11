from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_d1_d6_protocol_inputs import DATASET_CONFIG, validate_protocol_frames


def _rows(domain, item, *, group_col=None, group_value=None, periods=35):
    try:
        sales_value = float(item)
    except (TypeError, ValueError):
        sales_value = float(len(str(item)))
    payload = {
        "entity_id": str(domain),
        "item_id": str(item),
        "date": pd.date_range("2020-01-01", periods=periods, freq="D"),
        "sales": np.full(periods, sales_value, dtype=float),
    }
    if group_col:
        payload[group_col] = group_value
    return pd.DataFrame(payload)


class ProtocolPreflightTest(unittest.TestCase):
    def test_d1_d3_use_physical_domain_columns_not_composite_entity_id(self) -> None:
        self.assertEqual(DATASET_CONFIG[1]["group_cols"], ("store_id", "item_id"))
        self.assertEqual(DATASET_CONFIG[2]["group_cols"], ("brand_id", "item_id"))
        self.assertEqual(DATASET_CONFIG[3]["group_cols"], ("store_id",))

    def test_cli_help_can_import_project_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / "validate_d1_d6_protocol_inputs.py"), "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_incomplete_d1_with_pool_reports_missing_keys(self) -> None:
        source = pd.concat([_rows(1, item, periods=30) for item in range(1, 10)])
        target = _rows(1, 10)
        report = validate_protocol_frames(
            source,
            target,
            dataset_id="D1",
            scenario="with",
            group_cols=("entity_id", "item_id"),
            observed_start="2020-01-01",
            k=3,
        )
        self.assertEqual(report["status"], "failed")
        self.assertIn("missing required candidate keys", report["error"])

    def test_complete_d1_and_cross_store_d5_pass(self) -> None:
        d1_source = pd.concat(
            [_rows(store, item, periods=30) for store in range(1, 4) for item in range(1, 10)]
        )
        d1 = validate_protocol_frames(
            d1_source,
            _rows(1, 10),
            dataset_id="D1",
            scenario="with",
            group_cols=("entity_id", "item_id"),
            observed_start="2020-01-01",
            k=3,
        )
        self.assertEqual(d1["status"], "passed")
        self.assertEqual(d1["candidate_count"], 27)
        self.assertEqual(len(d1["candidate_pool_digest"]), 64)

        d5_source = pd.concat(
            [
                _rows("S1", "I2", group_col="family", group_value="F1", periods=30),
                _rows("S2", "I2", group_col="family", group_value="F1", periods=30),
            ]
        )
        d5 = validate_protocol_frames(
            d5_source,
            _rows("S1", "I1", group_col="family", group_value="F1"),
            dataset_id="D5",
            scenario="with",
            group_cols=("entity_id", "item_id"),
            grouping_col="family",
            observed_start="2020-01-01",
            k=2,
        )
        self.assertEqual(d5["status"], "passed")
        self.assertEqual(d5["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
