"""Horizon contract and full-run resume checks. These tests do not train."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from additional_target_robustness.full_run import (
    EXPECTED_RECORDS,
    _cell_dir,
    _cell_key,
    _digest,
    _inspect_cell,
    _mark_hold,
    _write_json,
)
from additional_target_robustness.horizon import verify_direct_horizon
from additional_target_robustness.manifest import ManifestRecord, load_manifest
from additional_target_robustness.protocol import BLOCKED_HORIZON_MISMATCH, HOLD_HASH_MISMATCH, ProtocolHold
from additional_target_robustness.source_selection import SelectedSource, SelectionOutcome


class HorizonContractTests(unittest.TestCase):
    def test_direct_h_label_is_origin_plus_horizon(self) -> None:
        contract = verify_direct_horizon()
        self.assertEqual(contract["mode"], "direct_h")
        self.assertEqual(contract["patchtst_target_window"], 1)
        self.assertEqual(contract["timesnet_pred_len"], 1)
        self.assertFalse(contract["recursive"])

    def test_h3_and_h5_are_not_the_one_step_label(self) -> None:
        from additional_target_robustness.paths import ensure_baseline_path

        ensure_baseline_path()
        from bl4_lstm import _direct_h_arrays

        values = np.arange(40, dtype=np.float64)
        one_step, one_label = _direct_h_arrays(values, horizon=1, lookback=10, name="series")
        for horizon in (3, 5):
            inputs, labels = _direct_h_arrays(values, horizon=horizon, lookback=10, name="series")
            self.assertFalse(np.array_equal(labels, one_label[: labels.shape[0]]))
            origin = 10 - 1
            self.assertEqual(float(labels[0, 0]), float(values[origin + horizon]))
            self.assertNotEqual(float(labels[0, 0]), float(values[origin + 1]))
            self.assertEqual(inputs.shape[1], one_step.shape[1])


class FullRunResumeTests(unittest.TestCase):
    def test_grid_is_2400_frozen_manifest_records(self) -> None:
        records = load_manifest()
        self.assertEqual(len(records), 40)
        self.assertEqual(len(records) * 3 * 5 * 4, EXPECTED_RECORDS)
        self.assertEqual(EXPECTED_RECORDS, 2400)

    def test_complete_cell_is_skipped_and_mismatch_is_held(self) -> None:
        record = ManifestRecord(
            dataset="D1",
            target_identity="2/6",
            target_sampling_seed=20261004,
            eligibility_status="eligible",
            target_train_start="2017-06-01",
            target_train_end="2017-06-15",
            target_val_start="2017-06-16",
            target_val_end="2017-06-30",
            target_test_start="2017-07-01",
            target_test_end="2017-12-27",
            target_total_days=210,
            source_candidate_pool_id="c57f402fbbc701b7",
            source_candidate_pool_hash="abc",
            selection_pool_size=18,
            selected_rank=1,
        )
        selection = SelectionOutcome(
            status="COMPLETED",
            dataset="D1",
            target_identity="2/6",
            selected=(SelectedSource("1/8", ("1", "8"), 0.5, 1.0, 0),),
            legal_count=3,
            comparison_start="2017-06-01",
            comparison_end="2017-06-30",
            source_universe_id="c57f402fbbc701b7",
        )
        dates = tuple(day.strftime("%Y-%m-%d") for day in pd.date_range("2017-07-01", periods=180, freq="D"))
        y_true = np.arange(180, dtype=np.float64)
        key = _cell_key(
            record,
            selection,
            horizon=1,
            seed=42,
            method="SITA",
            source_hashes={"D1:source": "abc"},
            protocol_sha="proto",
            target_sha="target",
            dates=dates,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = _cell_dir(root, "D1", "2/6", 1, 42, "SITA")
            directory.mkdir(parents=True)
            frame = pd.DataFrame({"target_date": list(dates), "y_true": y_true, "y_pred": y_true + 1.0})
            frame.to_parquet(directory / "predictions.parquet", index=False)
            before = (directory / "predictions.parquet").read_bytes()
            _write_json(directory / "key.json", key)
            _write_json(
                directory / "DONE.json",
                {"status": "COMPLETED", "key_sha256": _digest(key), "n_test": 180},
            )
            self.assertEqual(_inspect_cell(root, directory, key, dates, y_true), "COMPLETE")
            changed = dict(key)
            changed["protocol_sha256"] = "other-protocol"
            self.assertEqual(_inspect_cell(root, directory, changed, dates, y_true), "HOLD")
            _mark_hold(directory, changed)
            self.assertEqual((directory / "predictions.parquet").read_bytes(), before)
            hold = json.loads((directory / "HOLD.json").read_text(encoding="utf-8"))
            self.assertEqual(hold["status"], HOLD_HASH_MISMATCH)
            self.assertTrue((directory / "DONE.json").is_file())
            self.assertTrue((directory / "key.json").is_file())

    def test_runner_rejects_an_unknown_command(self) -> None:
        from additional_target_robustness.runner import main

        with self.assertRaises(SystemExit):
            main(["all-targets"])


class HorizonHoldTests(unittest.TestCase):
    def test_token_is_available_for_a_failed_contract(self) -> None:
        self.assertEqual(BLOCKED_HORIZON_MISMATCH, "BLOCKED_HORIZON_MISMATCH")
        with self.assertRaises(ProtocolHold) as caught:
            raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
        self.assertEqual(caught.exception.token, "BLOCKED_HORIZON_MISMATCH")


if __name__ == "__main__":
    unittest.main()
