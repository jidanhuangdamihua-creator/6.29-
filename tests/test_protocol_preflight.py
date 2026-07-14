from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_d1_d6_protocol_inputs import (
    DATASET_CONFIG,
    D1D2_PROTOCOL_PARQUET_DIR,
    PARQUET_DIR,
    build_preflight_reports,
    resolve_parquet_dir,
    validate_protocol_frames,
)


def _rows(domain, item, *, group_col=None, group_value=None, periods=35):
    try:
        sales_value = float(item)
    except (TypeError, ValueError):
        sales_value = float(len(str(item)))
    payload = {
        "entity_id": str(domain),
        "store_id": str(domain),
        "item_id": str(item),
        "date": pd.date_range("2020-01-01", periods=periods, freq="D"),
        "sales": np.full(periods, sales_value, dtype=float),
    }
    if group_col:
        payload[group_col] = group_value
    return pd.DataFrame(payload)


class ProtocolPreflightTest(unittest.TestCase):
    def test_multi_target_preflight_prepares_source_pool_once(self) -> None:
        source = pd.concat(
            [
                _rows("S1", "I2", group_col="family", group_value="F1", periods=30),
                _rows("S2", "I2", group_col="family", group_value="F1", periods=30),
            ],
            ignore_index=True,
        )
        target = pd.concat(
            [
                _rows("S1", "I1", group_col="family", group_value="F1"),
                _rows("S1", "I3", group_col="family", group_value="F1"),
            ],
            ignore_index=True,
        )
        calls = []

        from src.protocols.candidate_pool import prepare_daily_sequence_pool

        def counting_factory(*args, **kwargs):
            calls.append(1)
            return prepare_daily_sequence_pool(*args, **kwargs)

        reports = build_preflight_reports(
            source,
            target,
            dataset_id=5,
            scenario="with",
            group_cols=("entity_id", "item_id"),
            grouping_col="family",
            observed_start="2020-01-01",
            k=1,
            pool_factory=counting_factory,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(reports), 2)
        self.assertEqual({report["status"] for report in reports}, {"passed"})

    def test_preflight_exclusions_are_bounded_with_complete_counts(self) -> None:
        frames = [_rows("S0", "VALID", group_col="family", group_value="F1", periods=30)]
        for index in range(25):
            incomplete = _rows(
                f"S{index + 1}",
                f"I{index}",
                group_col="family",
                group_value="F1",
                periods=30,
            )
            frames.append(incomplete.iloc[1:].copy())
        reports = build_preflight_reports(
            pd.concat(frames, ignore_index=True),
            _rows("T", "TARGET", group_col="family", group_value="F1"),
            dataset_id=5,
            scenario="with",
            group_cols=("entity_id", "item_id"),
            grouping_col="family",
            observed_start="2020-01-01",
            k=1,
            exclusion_sample_limit=20,
        )
        report = reports[0]
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["candidate_exclusion_count"], 25)
        self.assertEqual(report["candidate_exclusion_reason_counts"], {"missing_observed_dates": 25})
        self.assertEqual(len(report["candidate_exclusion_samples"]), 20)
        self.assertTrue(report["candidate_exclusions_truncated"])
        self.assertNotIn("candidate_exclusions", report)

        zero_sample_report = build_preflight_reports(
            pd.concat(frames, ignore_index=True),
            _rows("T", "TARGET", group_col="family", group_value="F1"),
            dataset_id=5,
            scenario="with",
            group_cols=("entity_id", "item_id"),
            grouping_col="family",
            observed_start="2020-01-01",
            k=1,
            exclusion_sample_limit=0,
        )[0]
        self.assertEqual(
            zero_sample_report["candidate_pool_digest"],
            report["candidate_pool_digest"],
        )
        self.assertEqual(
            zero_sample_report["selection_result_digest"],
            report["selection_result_digest"],
        )
        self.assertEqual(zero_sample_report["candidate_exclusion_samples"], [])

    def test_d1_d3_use_physical_domain_columns_not_composite_entity_id(self) -> None:
        self.assertEqual(DATASET_CONFIG[1]["group_cols"], ("store_id", "item_id"))
        self.assertEqual(DATASET_CONFIG[2]["group_cols"], ("brand_id", "item_id"))
        self.assertEqual(DATASET_CONFIG[3]["group_cols"], ("store_id",))

    def test_default_parquet_dir_routes_d1_d2_to_protocol_derived_data(self) -> None:
        self.assertEqual(resolve_parquet_dir(1), D1D2_PROTOCOL_PARQUET_DIR)
        self.assertEqual(resolve_parquet_dir(2), D1D2_PROTOCOL_PARQUET_DIR)

        for dataset_id in (3, 4, 5, 6):
            self.assertEqual(resolve_parquet_dir(dataset_id), PARQUET_DIR)

        explicit = Path("/tmp/custom-protocol-parquets")
        self.assertEqual(resolve_parquet_dir(1, explicit), explicit)
        self.assertEqual(resolve_parquet_dir(6, explicit), explicit)

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
            group_cols=("store_id", "item_id"),
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
            group_cols=("store_id", "item_id"),
            observed_start="2020-01-01",
            k=3,
        )
        self.assertEqual(d1["status"], "passed")
        self.assertEqual(d1["candidate_count"], 27)
        self.assertEqual(len(d1["candidate_pool_digest"]), 64)
        self.assertEqual(d1["candidate_count_valid"], 27)
        self.assertEqual(d1["candidate_exclusion_count"], 0)
        self.assertEqual(d1["candidate_exclusion_samples"], [])
        self.assertEqual(len(d1["ordered_top_k"]), 3)
        digest_summary = d1["candidate_pool_digest_input_summary"]
        self.assertEqual(digest_summary["candidate_keys_count"], 27)
        self.assertEqual(len(digest_summary["candidate_keys_sample"]), 20)
        self.assertTrue(digest_summary["candidate_keys_truncated"])
        self.assertTrue(d1["cnn_provenance_validated"])

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
