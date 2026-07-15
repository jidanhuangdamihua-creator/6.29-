from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.protocols.candidate_pool import (
    rank_source_distances,
    select_daily_sequence_sources,
)
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol


DATES = pd.date_range("2020-01-01", periods=30, freq="D")
FUTURE_DATES = pd.date_range("2020-01-31", periods=5, freq="D")


def _target(values: np.ndarray, *, include_future: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame({"date": DATES, "sales": values.astype(float)})
    if include_future:
        frame = pd.concat(
            [
                frame,
                pd.DataFrame({"date": FUTURE_DATES, "sales": np.arange(5) + 1000.0}),
            ],
            ignore_index=True,
        )
    return frame


def _source(key: tuple[str, str], values: np.ndarray, future: float = 2000.0) -> pd.DataFrame:
    observed = pd.DataFrame(
        {
            "store": key[0],
            "item": key[1],
            "date": DATES,
            "sales": values.astype(float),
        }
    )
    after = pd.DataFrame(
        {
            "store": key[0],
            "item": key[1],
            "date": FUTURE_DATES,
            "sales": np.full(len(FUTURE_DATES), future),
        }
    )
    return pd.concat([observed, after], ignore_index=True)


def _select(target: pd.DataFrame, source: pd.DataFrame, *, k: int = 2):
    return select_daily_sequence_sources(
        target_df=target,
        source_df=source,
        protocol=get_experiment_protocol("D4"),
        scenario="with",
        target_key=("T", "I0"),
        candidate_keys=(("S1", "I1"), ("S2", "I2"), ("S3", "I3")),
        group_cols=("store", "item"),
        observed_start="2020-01-01",
        feature_cols=("sales",),
        k=k,
    )


class DailyKnnProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = _target(np.zeros(30))
        self.source = pd.concat(
            [
                _source(("S1", "I1"), np.ones(30)),
                _source(("S2", "I2"), np.full(30, 3.0)),
                _source(("S3", "I3"), np.full(30, 10.0)),
            ],
            ignore_index=True,
        )

    def test_future_target_and_source_perturbation_cannot_change_selection(self) -> None:
        before = _select(self.target, self.source)
        target_after = self.target.copy()
        target_after.loc[target_after["date"] > DATES[-1], "sales"] = -1e12
        source_after = self.source.copy()
        source_after.loc[source_after["date"] > DATES[-1], "sales"] = 1e12
        after = _select(target_after, source_after)

        self.assertEqual(before.ordered_source_keys, after.ordered_source_keys)
        np.testing.assert_array_equal(before.distances, after.distances)
        np.testing.assert_array_equal(before.weights, after.weights)
        self.assertEqual(before.candidate_pool_digest, after.candidate_pool_digest)
        self.assertEqual(before.selection_result_digest, after.selection_result_digest)

    def test_observed_extreme_deterministically_flips_top1(self) -> None:
        before = _select(self.target, self.source)
        original_margin = before.entries[1].distance - before.entries[0].distance
        self.assertGreater(original_margin, 0.5)
        changed = self.source.copy()
        mask = (
            (changed["store"] == "S3")
            & (changed["item"] == "I3")
            & changed["date"].isin(DATES)
        )
        changed.loc[mask, "sales"] = 0.0

        after = _select(self.target, changed)
        self.assertEqual(after.ordered_source_keys[0], ("S3", "I3"))

    def test_ties_use_anchored_groups_and_lexical_key_order(self) -> None:
        ranked = rank_source_distances(
            (("S2", "I2"), ("S1", "I1"), ("S3", "I3")),
            np.asarray([1.0 + 0.75e-12, 1.0, 1.0 + 1.5e-12], dtype=np.float64),
            tie_tolerance=1e-12,
        )

        self.assertEqual([entry.source_key for entry in ranked], [("S1", "I1"), ("S2", "I2"), ("S3", "I3")])
        self.assertEqual([entry.tie_group for entry in ranked], [1, 1, 2])

    def test_target_missing_date_fails(self) -> None:
        with self.assertRaisesRegex(ProtocolViolation, "target.*missing observed dates"):
            _select(self.target.iloc[1:].copy(), self.source)

    def test_duplicate_target_date_fails(self) -> None:
        duplicated = pd.concat([self.target, self.target.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ProtocolViolation, "target.*duplicate observed dates"):
            _select(duplicated, self.source)

    def test_missing_candidate_dates_cannot_silently_shrink_k(self) -> None:
        incomplete = self.source[
            ~((self.source["store"] == "S2") & (self.source["date"] == DATES[0]))
        ]
        selection = _select(self.target, incomplete, k=3)
        self.assertEqual(len(selection.entries), 3)
        repaired = next(
            entry for entry in selection.entries if entry.source_key == ("S2", "I2")
        )
        self.assertTrue(repaired.source_repair_digest)

    def test_duplicate_source_date_fails(self) -> None:
        duplicated = pd.concat([self.source, self.source.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ProtocolViolation, "valid candidates.*required K=3"):
            _select(self.target, duplicated, k=3)

    def test_target_key_in_candidate_pool_fails(self) -> None:
        with self.assertRaisesRegex(ProtocolViolation, "target key.*candidate pool"):
            select_daily_sequence_sources(
                target_df=self.target,
                source_df=self.source,
                protocol=get_experiment_protocol("D4"),
                scenario="with",
                target_key=("S1", "I1"),
                candidate_keys=(("S1", "I1"),),
                group_cols=("store", "item"),
                observed_start="2020-01-01",
                feature_cols=("sales",),
                k=1,
            )

    def test_non_sales_features_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolViolation, "must start with sales"):
            select_daily_sequence_sources(
                target_df=self.target,
                source_df=self.source,
                protocol=get_experiment_protocol("D4"),
                scenario="with",
                target_key=("T", "I0"),
                candidate_keys=(("S1", "I1"),),
                group_cols=("store", "item"),
                observed_start="2020-01-01",
                feature_cols=("store", "sales"),
                k=1,
            )


if __name__ == "__main__":
    unittest.main()
