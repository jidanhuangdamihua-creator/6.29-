from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.runner_adapter import configure_protocol_frames, source_key_mask


def _daily_rows(store, item, sales, *, group_col=None, group_value=None, periods=30):
    payload = {
        "entity_id": store,
        "item_id": item,
        "date": pd.date_range("2020-01-01", periods=periods, freq="D"),
        "sales": np.full(periods, sales, dtype=float),
    }
    if group_col is not None:
        payload[group_col] = group_value
    return pd.DataFrame(payload)


class RunnerProtocolIntegrationTest(unittest.TestCase):
    def test_normalized_selected_key_indexes_numeric_source_rows(self) -> None:
        frame = pd.DataFrame(
            {"entity_id": [1, 1, 2], "item_id": [3, 4, 3], "sales": [1.0, 2.0, 3.0]}
        )
        mask = source_key_mask(frame, ("entity_id", "item_id"), ("1", "3"))
        self.assertEqual(frame.loc[mask, "sales"].tolist(), [1.0])

    def test_d1_with_pool_is_exact_and_old_incomplete_pool_fails(self) -> None:
        target = _daily_rows(1, 10, 0.0, periods=35)
        complete = pd.concat(
            [
                _daily_rows(store, item, float(item))
                for store in range(1, 4)
                for item in range(1, 10)
            ],
            ignore_index=True,
        )
        source, configured_target = configure_protocol_frames(
            complete,
            target,
            dataset_id="D1",
            scenario="with",
            group_cols=("entity_id", "item_id"),
            observed_start="2020-01-01",
        )
        self.assertEqual(len(source.attrs["protocol_candidate_keys"]), 27)
        self.assertEqual(configured_target.attrs["protocol_target_key"], ("1", "10"))

        incomplete = complete[complete["entity_id"] == 1]
        with self.assertRaisesRegex(ProtocolViolation, "missing required candidate keys"):
            configure_protocol_frames(
                incomplete,
                target,
                dataset_id="D1",
                scenario="with",
                group_cols=("entity_id", "item_id"),
                observed_start="2020-01-01",
            )

    def test_d5_without_and_with_follow_same_family_store_semantics(self) -> None:
        target = _daily_rows(
            "S1", "I1", 0.0, group_col="family", group_value="F1", periods=35
        )
        source = pd.concat(
            [
                _daily_rows("S1", "I2", 1.0, group_col="family", group_value="F1"),
                _daily_rows("S2", "I2", 2.0, group_col="family", group_value="F1"),
                _daily_rows("S2", "I3", 3.0, group_col="family", group_value="F2"),
            ],
            ignore_index=True,
        )
        without, _ = configure_protocol_frames(
            source,
            target,
            dataset_id="D5",
            scenario="without",
            group_cols=("entity_id", "item_id"),
            grouping_col="family",
            observed_start="2020-01-01",
        )
        with_sharing, _ = configure_protocol_frames(
            source,
            target,
            dataset_id="D5",
            scenario="with",
            group_cols=("entity_id", "item_id"),
            grouping_col="family",
            observed_start="2020-01-01",
        )
        self.assertEqual(without.attrs["protocol_candidate_keys"], (("S1", "I2"),))
        self.assertEqual(
            with_sharing.attrs["protocol_candidate_keys"],
            (("S1", "I2"), ("S2", "I2")),
        )


if __name__ == "__main__":
    unittest.main()
