from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd

from src.protocols.candidate_pool import (
    prepare_daily_sequence_pool,
    rank_source_distances,
    select_daily_sequence_sources,
)
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol


DATES = pd.date_range("2020-01-01", periods=30, freq="D")
PRETRAIN_DATES = pd.date_range(DATES[0] - pd.Timedelta(days=150), periods=180, freq="D")
FUTURE = pd.date_range("2020-01-31", periods=4, freq="D")
CANDIDATES = (("S1", "I1"), ("S2", "I2"), ("S3", "I3"))


def _target(values: np.ndarray) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({"date": DATES, "sales": values.astype(float)}),
            pd.DataFrame({"date": FUTURE, "sales": 9999.0}),
        ],
        ignore_index=True,
    )


def _source(key: tuple[str, str], value: float) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "store": key[0],
                    "item": key[1],
                    "family": "F1",
                    "date": PRETRAIN_DATES,
                    "sales": np.full(180, value, dtype=float),
                }
            ),
            pd.DataFrame(
                {
                    "store": key[0],
                    "item": key[1],
                    "family": "F1",
                    "date": FUTURE,
                    "sales": 5000.0,
                }
            ),
        ],
        ignore_index=True,
    )


def _select(target: pd.DataFrame, source: pd.DataFrame, *, pool=None, k: int = 2):
    return select_daily_sequence_sources(
        target_df=target,
        source_df=source,
        prepared_pool=pool,
        protocol=get_experiment_protocol("D4"),
        scenario="with",
        target_key=("T", "I0"),
        candidate_keys=CANDIDATES,
        group_cols=("store", "item"),
        observed_start="2020-01-01",
        feature_cols=("sales",),
        k=k,
    )


class PreparedDailySequencePoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = _target(np.zeros(30))
        self.source = pd.concat(
            [_source(("S1", "I1"), 1.0), _source(("S2", "I2"), 3.0), _source(("S3", "I3"), 10.0)],
            ignore_index=True,
        )

    def _pool(self, source: pd.DataFrame | None = None):
        return prepare_daily_sequence_pool(
            self.source if source is None else source,
            group_cols=("store", "item"),
            observed_start="2020-01-01",
            pretrain_start=PRETRAIN_DATES[0],
            pretrain_end=PRETRAIN_DATES[-1],
            metadata_cols=("family",),
        )

    def test_prepared_selection_matches_frozen_reference_bit_for_bit(self) -> None:
        result = _select(self.target, self.source.iloc[0:0], pool=self._pool())

        self.assertEqual(
            result.candidate_pool_digest,
            "3c3a564e37fff144ace4155f7ecc31a9950c61e04326eecd156f196514ee321c",
        )
        self.assertEqual(
            result.candidate_pool_digest_input,
            {
                "protocol_version": "d1_d6_protocol_v1",
                "dataset_id": "D4",
                "scenario": "with",
                "target_key": ["T", "I0"],
                "group_cols": ["store", "item"],
                "candidate_keys": [["S1", "I1"], ["S2", "I2"], ["S3", "I3"]],
                "observed_start": "2020-01-01",
                "observed_end": "2020-01-30",
                "feature_cols": ["sales"],
            },
        )
        self.assertEqual(
            result.selection_result_digest,
            "69b5eda4261a5ec1cc648476af87cb303c8ab884199f90ba1a07a91b5c21a009",
        )
        self.assertEqual(result.ordered_source_keys, (("S1", "I1"), ("S2", "I2")))
        np.testing.assert_array_equal(
            result.distances,
            np.asarray([0.54772255750516619, 1.6431676725154982], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            result.weights,
            np.asarray([0.74999999771782266, 0.25000000228217739], dtype=np.float64),
        )
        self.assertEqual([entry.tie_group for entry in result.entries], [1, 2])
        self.assertEqual((result.scaler_min, result.scaler_max), (0.0, 10.0))
        self.assertEqual(result.excluded_candidates, ())

    def test_prepared_ties_keep_anchored_tolerance_and_lexical_order(self) -> None:
        tied_source = pd.concat(
            [_source(("S2", "I2"), 1.0), _source(("S1", "I1"), 1.0), _source(("S3", "I3"), 10.0)],
            ignore_index=True,
        )
        result = _select(
            self.target,
            tied_source.iloc[0:0],
            pool=self._pool(tied_source),
        )
        self.assertEqual(result.ordered_source_keys, (("S1", "I1"), ("S2", "I2")))
        self.assertEqual([entry.tie_group for entry in result.entries], [1, 1])
        np.testing.assert_array_equal(result.weights, np.asarray([0.5, 0.5]))

    def test_pool_preparation_does_not_use_dataframe_apply_axis_1(self) -> None:
        original_apply = pd.DataFrame.apply

        def guarded_apply(frame, func=None, axis=0, *args, **kwargs):
            if axis in (1, "columns"):
                raise AssertionError("axis=1 is forbidden for prepared source keys")
            return original_apply(frame, func=func, axis=axis, *args, **kwargs)

        pd.DataFrame.apply = guarded_apply
        try:
            pool = self._pool()
        finally:
            pd.DataFrame.apply = original_apply
        self.assertEqual(pool.sales_matrix.shape, (3, 30))
        self.assertEqual(pool.date_presence_matrix.shape, (3, 30))
        self.assertEqual(pool.date_presence_matrix.dtype, np.dtype(bool))

    def test_ranker_has_no_remaining_list_rescan_loop(self) -> None:
        source = inspect.getsource(rank_source_distances)
        self.assertNotIn("while remaining", source)
        self.assertNotIn("remaining = later", source)

    def test_mixed_raw_key_types_preserve_protocol_normalization(self) -> None:
        numeric = pd.DataFrame(
            {"store": 1, "item": 2, "date": PRETRAIN_DATES, "sales": 1.0}
        )
        textual = pd.DataFrame(
            {"store": "1", "item": "2", "date": PRETRAIN_DATES, "sales": 2.0}
        )
        mixed = pd.concat([numeric, textual], ignore_index=True)
        pool = prepare_daily_sequence_pool(
            mixed,
            group_cols=("store", "item"),
            observed_start="2020-01-01",
            pretrain_start=PRETRAIN_DATES[0],
            pretrain_end=PRETRAIN_DATES[-1],
        )
        self.assertEqual(pool.source_keys, (("1", "2"),))
        self.assertEqual(pool.duplicate_date_keys, frozenset({("1", "2")}))

    def test_future_is_invariant_and_observed_change_flips_top1(self) -> None:
        before = _select(self.target, self.source.iloc[0:0], pool=self._pool())
        future_changed = self.source.copy()
        future_changed.loc[future_changed["date"].isin(FUTURE), "sales"] = -1e12
        after_future = _select(
            self.target.assign(sales=lambda frame: np.where(frame["date"].isin(FUTURE), 1e12, frame["sales"])),
            future_changed.iloc[0:0],
            pool=self._pool(future_changed),
        )
        self.assertEqual(before.selection_result_digest, after_future.selection_result_digest)
        np.testing.assert_array_equal(before.distances, after_future.distances)

        observed_changed = self.source.copy()
        mask = (
            (observed_changed["store"] == "S3")
            & (observed_changed["item"] == "I3")
            & observed_changed["date"].isin(DATES)
        )
        observed_changed.loc[mask, "sales"] = 0.0
        after_observed = _select(
            self.target,
            observed_changed.iloc[0:0],
            pool=self._pool(observed_changed),
        )
        self.assertEqual(after_observed.ordered_source_keys[0], ("S3", "I3"))

    def test_missing_duplicate_nonfinite_and_insufficient_k_behave_as_before(self) -> None:
        missing = self.source[
            ~((self.source["store"] == "S2") & (self.source["date"] == DATES[0]))
        ]
        result = _select(self.target, missing.iloc[0:0], pool=self._pool(missing), k=2)
        self.assertEqual(
            result.excluded_candidates,
            (
                {
                    "source_key": ("S2", "I2"),
                    "reason": "missing_observed_dates",
                    "reasons": ("missing_observed_dates",),
                    "missing_dates": ("2020-01-01",),
                },
            ),
        )
        with self.assertRaisesRegex(ProtocolViolation, "valid candidates=2.*required K=3"):
            _select(self.target, missing.iloc[0:0], pool=self._pool(missing), k=3)

        duplicate_row = self.source[
            (self.source["store"] == "S1") & (self.source["date"] == DATES[0])
        ].iloc[[0]]
        duplicate = pd.concat([self.source, duplicate_row], ignore_index=True)
        duplicate_result = _select(
            self.target, duplicate.iloc[0:0], pool=self._pool(duplicate)
        )
        self.assertEqual(duplicate_result.excluded_candidates[0]["reason"], "duplicate_source_date")
        with self.assertRaisesRegex(ProtocolViolation, "valid candidates=2.*required K=3"):
            _select(self.target, duplicate.iloc[0:0], pool=self._pool(duplicate), k=3)

        nonfinite = self.source.copy()
        nonfinite.loc[
            (nonfinite["store"] == "S1") & (nonfinite["date"] == DATES[0]), "sales"
        ] = np.inf
        nonfinite_result = _select(
            self.target, nonfinite.iloc[0:0], pool=self._pool(nonfinite)
        )
        self.assertEqual(nonfinite_result.excluded_candidates[0]["reason"], "source_sales_infinity")
        with self.assertRaisesRegex(ProtocolViolation, "valid candidates=2.*required K=3"):
            _select(self.target, nonfinite.iloc[0:0], pool=self._pool(nonfinite), k=3)


if __name__ == "__main__":
    unittest.main()
