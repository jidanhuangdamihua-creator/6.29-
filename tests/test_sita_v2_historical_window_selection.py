from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.constants import SOURCE_HISTORY_COMPLETENESS_POLICY, SOURCE_HISTORY_DAYS
from src.protocols.candidate_pool import (
    SOURCE_SELECTION_MODE_ALIGNED_30D,
    SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D,
    InsufficientCandidatePoolError,
    build_selection_result_digest,
    select_daily_sequence_sources,
)
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol
from src.protocols.provenance import extract_selected_source_slices
from src.source_selection.source_selector import (
    K3SelectionEvidence,
    TargetK3SelectionContext,
)


OBSERVED_END = pd.Timestamp("2020-01-30")
HISTORY_DATES = pd.date_range(end=OBSERVED_END, periods=SOURCE_HISTORY_DAYS, freq="D")
OBSERVED_DATES = HISTORY_DATES[-30:]
FUTURE_DATES = pd.date_range(OBSERVED_END + pd.Timedelta(days=1), periods=5, freq="D")


def _history_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    frame.attrs["source_history_days"] = SOURCE_HISTORY_DAYS
    frame.attrs["source_history_start"] = HISTORY_DATES[0]
    frame.attrs["source_history_end"] = HISTORY_DATES[-1]
    frame.attrs["source_history_completeness_policy"] = SOURCE_HISTORY_COMPLETENESS_POLICY
    frame.attrs["source_history_frame_digest"] = "synthetic-source-history"
    return frame


def _series(length: int, value: float, *, early: float | None = None) -> np.ndarray:
    values = np.full(length, value, dtype=float)
    if early is not None:
        values[:30] = early
    return values


def _product(key: tuple[str, str], history: np.ndarray, *, future: float = 1e9) -> pd.DataFrame:
    observed = pd.DataFrame(
        {
            "store": key[0],
            "item": key[1],
            "date": HISTORY_DATES,
            "sales": history.astype(float),
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


def _target(values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"date": OBSERVED_DATES, "sales": values.astype(float)})
    future = pd.DataFrame({"date": FUTURE_DATES, "sales": np.full(len(FUTURE_DATES), 1e9)})
    return pd.concat([frame, future], ignore_index=True)


def _select(target: pd.DataFrame, source: pd.DataFrame, *, mode: str, k: int = 3, keys=None):
    return select_daily_sequence_sources(
        target_df=target,
        source_df=source,
        protocol=get_experiment_protocol("D4"),
        scenario="with",
        target_key=("T", "I0"),
        candidate_keys=keys or (("A", "1"), ("B", "1"), ("C", "1")),
        group_cols=("store", "item"),
        observed_start=OBSERVED_DATES[0].strftime("%Y-%m-%d"),
        feature_cols=("sales",),
        k=k,
        source_selection_mode=mode,
        source_history_frame=source,
    )


class SITAV2HistoricalWindowSelectionTest(unittest.TestCase):
    def test_complete_history_has_151_windows_and_earliest_tie(self) -> None:
        history = np.full(SOURCE_HISTORY_DAYS, 5.0)
        history[:30] = 0.0
        history[10:40] = 0.0
        source = _history_attrs(
            pd.concat(
                [
                    _product(("A", "1"), history),
                    _product(("B", "1"), _series(SOURCE_HISTORY_DAYS, 3.0)),
                    _product(("C", "1"), _series(SOURCE_HISTORY_DAYS, 8.0)),
                ],
                ignore_index=True,
            )
        )
        result = _select(_target(np.zeros(30)), source, mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D)
        chosen = result.entries[0]
        self.assertEqual(chosen.source_key, ("A", "1"))
        self.assertEqual(chosen.valid_window_count, 151)
        self.assertEqual(chosen.match_window_start, HISTORY_DATES[0].strftime("%Y-%m-%d"))
        self.assertEqual(chosen.match_window_end, HISTORY_DATES[29].strftime("%Y-%m-%d"))
        self.assertEqual(len({entry.source_key for entry in result.entries}), 3)
        self.assertAlmostEqual(float(np.sum(result.weights)), 1.0)
        self.assertEqual(result.source_history_end, OBSERVED_END.strftime("%Y-%m-%d"))

    def test_product_level_top3_and_key_tie_break(self) -> None:
        source = _history_attrs(
            pd.concat(
                [
                    _product(("B", "1"), _series(SOURCE_HISTORY_DAYS, 1.0)),
                    _product(("A", "1"), _series(SOURCE_HISTORY_DAYS, 1.0)),
                    _product(("C", "1"), _series(SOURCE_HISTORY_DAYS, 4.0)),
                ],
                ignore_index=True,
            )
        )
        result = _select(_target(np.ones(30)), source, mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D)
        self.assertEqual(
            [entry.source_key for entry in result.entries[:2]],
            [("A", "1"), ("B", "1")],
        )
        self.assertEqual(len({entry.source_key for entry in result.entries}), 3)

    def test_scaler_uses_aligned_values_and_does_not_clip(self) -> None:
        aligned = np.linspace(0.0, 1.0, 30)
        history = np.concatenate([np.full(150, 10.0), aligned])
        source = _history_attrs(
            pd.concat(
                [
                    _product(("A", "1"), history),
                    _product(("B", "1"), np.concatenate([np.full(150, 0.2), aligned])),
                    _product(("C", "1"), np.concatenate([np.full(150, 0.4), aligned])),
                ],
                ignore_index=True,
            )
        )
        target = _target(aligned)
        historical = _select(target, source, mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D)
        aligned_result = _select(target, source, mode=SOURCE_SELECTION_MODE_ALIGNED_30D)
        self.assertEqual(historical.scaler_min, aligned_result.scaler_min)
        self.assertEqual(historical.scaler_max, aligned_result.scaler_max)
        self.assertLessEqual(historical.scaler_max, 1.0)
        self.assertTrue(historical.scaled_outside_unit_interval)
        self.assertTrue(any(entry.scaled_outside_unit_interval for entry in historical.entries))
        scaled_spike = (10.0 - historical.scaler_min) / (historical.scaler_max - historical.scaler_min)
        self.assertGreater(scaled_spike, 1.0)
        self.assertFalse(historical.scaler_range_is_zero)

    def test_future_values_do_not_change_distance_or_scaler(self) -> None:
        source = _history_attrs(
            pd.concat(
                [
                    _product(("A", "1"), _series(SOURCE_HISTORY_DAYS, 0.0, early=0.0)),
                    _product(("B", "1"), _series(SOURCE_HISTORY_DAYS, 2.0)),
                    _product(("C", "1"), _series(SOURCE_HISTORY_DAYS, 4.0)),
                ],
                ignore_index=True,
            )
        )
        before = _select(_target(np.zeros(30)), source, mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D)
        changed = source.copy()
        changed.attrs = dict(source.attrs)
        changed.loc[changed["date"] > OBSERVED_END, "sales"] = -1e12
        after = _select(_target(np.zeros(30)), changed, mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D)
        self.assertEqual(before.selection_result_digest, after.selection_result_digest)
        self.assertEqual(before.scaler_min, after.scaler_min)
        self.assertEqual(before.scaler_max, after.scaler_max)
        self.assertTrue(
            all(
                pd.Timestamp(entry.match_window_end) <= OBSERVED_END
                for entry in before.entries
            )
        )

    def test_aligned_digest_stays_compatible(self) -> None:
        source = pd.concat(
            [
                _product(("A", "1"), _series(SOURCE_HISTORY_DAYS, 1.0)),
                _product(("B", "1"), _series(SOURCE_HISTORY_DAYS, 2.0)),
                _product(("C", "1"), _series(SOURCE_HISTORY_DAYS, 3.0)),
            ],
            ignore_index=True,
        )
        target = _target(np.zeros(30))
        implicit = select_daily_sequence_sources(
            target_df=target,
            source_df=source,
            protocol=get_experiment_protocol("D4"),
            scenario="with",
            target_key=("T", "I0"),
            candidate_keys=(("A", "1"), ("B", "1"), ("C", "1")),
            group_cols=("store", "item"),
            observed_start=OBSERVED_DATES[0].strftime("%Y-%m-%d"),
            feature_cols=("sales",),
            k=3,
        )
        explicit = _select(target, source, mode=SOURCE_SELECTION_MODE_ALIGNED_30D)
        manual = build_selection_result_digest(
            protocol_version=implicit.protocol_version,
            candidate_pool_digest=implicit.candidate_pool_digest,
            k=3,
            weight_mode=get_experiment_protocol("D4").weight_mode,
            weight_epsilon=get_experiment_protocol("D4").weight_epsilon,
            entries=implicit.entries,
        )
        self.assertEqual(implicit.selection_result_digest, explicit.selection_result_digest)
        self.assertEqual(implicit.selection_result_digest, manual)
        self.assertIsNone(implicit.entries[0].match_window_start)
        historical = _select(
            target,
            _history_attrs(source.copy()),
            mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D,
        )
        self.assertNotEqual(historical.selection_result_digest, implicit.selection_result_digest)
        self.assertIn("historical_window_30d", historical.source_selection_mode)

    def test_fewer_than_three_complete_histories_fail_closed(self) -> None:
        incomplete = _product(("C", "1"), _series(SOURCE_HISTORY_DAYS, 1.0))
        incomplete = incomplete.loc[incomplete["date"] != HISTORY_DATES[0]]
        source = _history_attrs(
            pd.concat(
                [
                    _product(("A", "1"), _series(SOURCE_HISTORY_DAYS, 0.0)),
                    _product(("B", "1"), _series(SOURCE_HISTORY_DAYS, 1.0)),
                    incomplete,
                ],
                ignore_index=True,
            )
        )
        with self.assertRaises(InsufficientCandidatePoolError):
            _select(_target(np.zeros(30)), source, mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D)

    def test_duplicate_history_dates_raise(self) -> None:
        source = _history_attrs(
            pd.concat(
                [
                    _product(("A", "1"), _series(SOURCE_HISTORY_DAYS, 0.0)),
                    _product(("B", "1"), _series(SOURCE_HISTORY_DAYS, 1.0)),
                    _product(("C", "1"), _series(SOURCE_HISTORY_DAYS, 2.0)),
                ],
                ignore_index=True,
            )
        )
        duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        duplicated.attrs = dict(source.attrs)
        with self.assertRaisesRegex(ProtocolViolation, "duplicate dates"):
            _select(_target(np.zeros(30)), duplicated, mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D)

    def test_d1_without_frozen_history_fails_only_for_v2(self) -> None:
        dates = pd.date_range("2017-06-01", periods=30, freq="D")
        target = pd.DataFrame({"date": dates, "sales": np.zeros(30)})
        rows = []
        for key in (("2", "1"), ("3", "1"), ("4", "1")):
            rows.append(
                pd.DataFrame(
                    {
                        "store_id": key[0],
                        "item_id": key[1],
                        "date": dates,
                        "sales": np.arange(30, dtype=float),
                    }
                )
            )
        source = pd.concat(rows, ignore_index=True)
        aligned = select_daily_sequence_sources(
            target_df=target,
            source_df=source,
            protocol=get_experiment_protocol("D1"),
            scenario="without",
            target_key=("1", "10"),
            candidate_keys=(("2", "1"), ("3", "1"), ("4", "1")),
            group_cols=("store_id", "item_id"),
            observed_start="2017-06-01",
            feature_cols=("sales",),
            k=3,
        )
        self.assertEqual(len(aligned.entries), 3)
        with self.assertRaisesRegex(ProtocolViolation, "source_history_days == 180"):
            select_daily_sequence_sources(
                target_df=target,
                source_df=source,
                protocol=get_experiment_protocol("D1"),
                scenario="without",
                target_key=("1", "10"),
                candidate_keys=(("2", "1"), ("3", "1"), ("4", "1")),
                group_cols=("store_id", "item_id"),
                observed_start="2017-06-01",
                feature_cols=("sales",),
                k=3,
                source_selection_mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D,
                source_history_frame=source,
            )

    def test_provenance_keeps_cnn_slice_and_checks_match_window(self) -> None:
        history = np.arange(SOURCE_HISTORY_DAYS, dtype=float)
        source = _history_attrs(
            pd.concat(
                [
                    _product(("A", "1"), history, future=0.0),
                    _product(("B", "1"), history + 20.0, future=0.0),
                    _product(("C", "1"), history + 40.0, future=0.0),
                ],
                ignore_index=True,
            )
        )
        result = _select(
            _target(history[:30]),
            source,
            mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D,
            k=1,
        )
        slices = extract_selected_source_slices(
            result,
            source,
            training_start=OBSERVED_DATES[0],
            training_end=OBSERVED_DATES[-1],
            model_feature_cols=("sales",),
            match_history_frame=source,
        )
        self.assertEqual(slices[0].date_start, OBSERVED_DATES[0].strftime("%Y-%m-%d"))
        self.assertEqual(slices[0].date_end, OBSERVED_DATES[-1].strftime("%Y-%m-%d"))
        self.assertNotEqual(result.entries[0].match_window_start, slices[0].date_start)
        matched = source[
            (source["store"] == "A")
            & (source["item"] == "1")
            & source["date"].between(
                result.entries[0].match_window_start,
                result.entries[0].match_window_end,
            )
        ].sort_values("date")
        np.testing.assert_array_equal(
            np.asarray(result.entries[0].match_raw_vector),
            matched["sales"].to_numpy(dtype=float),
        )

    def test_window_vector_is_feature_major(self) -> None:
        sales = np.arange(SOURCE_HISTORY_DAYS, dtype=float)
        promo = sales + 1000.0
        oil = sales + 2000.0
        frames = []
        for index, shift in enumerate((0.0, 50.0, 80.0)):
            frames.append(
                pd.DataFrame(
                    {
                        "store_nbr": "9",
                        "item_nbr": str(index),
                        "date": HISTORY_DATES,
                        "sales": sales + shift,
                        "onpromotion": promo + shift,
                        "oil_price": oil + shift,
                    }
                )
            )
        source = _history_attrs(pd.concat(frames, ignore_index=True))
        target = pd.DataFrame(
            {
                "date": OBSERVED_DATES,
                "sales": sales[:30],
                "onpromotion": promo[:30],
                "oil_price": oil[:30],
            }
        )
        result = select_daily_sequence_sources(
            target_df=target,
            source_df=source,
            protocol=get_experiment_protocol("D5"),
            scenario="with",
            target_key=("48", "364606"),
            candidate_keys=(("9", "0"), ("9", "1"), ("9", "2")),
            group_cols=("store_nbr", "item_nbr"),
            observed_start=OBSERVED_DATES[0].strftime("%Y-%m-%d"),
            feature_cols=("sales", "onpromotion", "oil_price"),
            k=3,
            source_selection_mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D,
            source_history_frame=source,
        )
        chosen = result.entries[0]
        raw = np.asarray(chosen.match_raw_vector)
        self.assertEqual(chosen.source_key, ("9", "0"))
        self.assertEqual(chosen.match_window_start, HISTORY_DATES[0].strftime("%Y-%m-%d"))
        np.testing.assert_array_equal(raw[:30], sales[:30])
        np.testing.assert_array_equal(raw[30:60], promo[:30])
        np.testing.assert_array_equal(raw[60:], oil[:30])

    def test_k3_evidence_rejects_cached_mode_mismatch(self) -> None:
        evidence = K3SelectionEvidence(
            lifecycle_identity=("D4", "without", 1, 42, ("166", "258")),
            dataset_id="D4",
            scenario="without_information_sharing",
            target_key=("166", "258"),
            candidate_keys=(("166", "1"),),
            group_cols=("store_id", "product_id"),
            model_feature_cols=("sales",),
            knn_feature_cols=("sales",),
            target_frame_digest="target",
            source_frame_digest="source",
            candidate_pool_digest="pool",
            selection_result_digest="selection",
            prepared_pool_id=1,
            _selection_payload=b"",
            source_selection_mode=SOURCE_SELECTION_MODE_ALIGNED_30D,
        )
        context = TargetK3SelectionContext(lifecycle_identity=evidence.lifecycle_identity)
        context._evidence = evidence
        with self.assertRaisesRegex(ProtocolViolation, "source_selection_mode mismatch"):
            context.selection_for_method(
                target_df=pd.DataFrame(),
                source_df=pd.DataFrame(),
                feature_cols=("sales",),
                group_cols=("store_id", "product_id"),
                k=3,
                weight_mode="inverse_distance",
                source_selection_mode=SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D,
            )


if __name__ == "__main__":
    unittest.main()
