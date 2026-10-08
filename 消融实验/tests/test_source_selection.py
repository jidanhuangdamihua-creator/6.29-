"""Synthetic checks for source selection. Expected distances are closed forms."""

from __future__ import annotations

import hashlib
import math
import unittest

import numpy as np
import pandas as pd

from sita_ablation.data import (
    ConditionFrames,
    ContractError,
    DateSpan,
    load_dataset_contract,
    similarity_windows,
)
from sita_ablation.scaling import transform_selection
from sita_ablation.source_selection import (
    anchored_rank,
    derived_random3_seed,
    rank_candidates,
    source_key_overlap,
)


def _fill(value, periods=30):
    return np.full(periods, value, dtype=np.float64)


def make_frames(dataset, specs, extras=()):
    contract = load_dataset_contract(dataset)
    dates = pd.date_range(contract.target_train.start, contract.target_val.end, freq="D")
    history = pd.date_range(contract.source_history.start, contract.source_history.end, freq="D")
    if len(dates) != 30 or len(history) != 180:
        raise AssertionError(f"{dataset} frozen windows are {len(dates)} and {len(history)} days")
    comparison = {pd.Timestamp(day).normalize() for day in dates}
    target_records = []
    source_records = []
    for spec in specs:
        dropped = {pd.Timestamp(day).normalize() for day in spec.get("drop_dates", ())}
        bucket = target_records if spec["role"] == "target" else source_records
        if contract.grouping_field is not None and spec.get("group") is None:
            raise AssertionError(f"{dataset} spec {spec['key']!r} needs a group value")
        for offset, day in enumerate(dates):
            if pd.Timestamp(day).normalize() in dropped:
                continue
            bucket.append(_row(contract, spec["key"], day, spec["values"], offset, spec.get("group")))
        if spec["role"] == "source" and spec.get("history", "full") != "comparison_only":
            prefix_values = {name: _prefix_value(series) for name, series in spec["values"].items()}
            for day in history:
                normalized = pd.Timestamp(day).normalize()
                if normalized in comparison or normalized in dropped:
                    continue
                bucket.append(_row(contract, spec["key"], normalized, prefix_values, None, spec.get("group")))
    for extra in extras:
        bucket = target_records if extra["role"] == "target" else source_records
        bucket.append(
            _row(contract, extra["key"], pd.Timestamp(extra["date"]), extra["values"], None, extra.get("group"))
        )
    return _frames_from_records(contract, target_records, source_records)


def _row(contract, key, day, values, offset, group):
    row = {"date": pd.Timestamp(day).normalize()}
    for field, part in zip(contract.key_fields, key):
        row[field] = part
    for name, series in values.items():
        row[name] = series if offset is None else series[offset]
    if contract.grouping_field is not None:
        row[contract.grouping_field] = group
    if contract.dataset == "D5":
        row["entity_id"] = f"{key[0]}_{key[1]}"
    return row


def _prefix_value(series):
    if isinstance(series, (list, tuple, np.ndarray)):
        return series[0]
    return series


def _frames_from_records(contract, target_records, source_records):
    return ConditionFrames(
        dataset=contract.dataset,
        source_rows=pd.DataFrame(source_records),
        target_rows=pd.DataFrame(target_records),
        key_fields=contract.key_fields,
        target_key=contract.target_key,
        model_features=contract.model_features,
        similarity_variables=contract.similarity_variables,
        source_history=contract.source_history,
        target_train=contract.target_train,
        target_val=contract.target_val,
        target_test=contract.target_test,
        comparison_30=DateSpan(contract.target_train.start, contract.target_val.end),
        comparison_15=DateSpan(contract.target_train.start, contract.target_train.end),
        data_sha256={},
        grouping_field=contract.grouping_field,
    )


def _keys(result):
    return tuple(item.source_key for item in result.selected)


class SourceSelectionTests(unittest.TestCase):
    def test_per_variable_scaling_distinguishes_a_global_scalar_and_keeps_d5_order(self):
        contract = load_dataset_contract("D5")
        self.assertEqual(contract.similarity_variables, ("sales", "onpromotion", "oil_price"))
        self.assertEqual(contract.target_key, ("48", "1159414"))
        # Three scales: sales 0..10, onpromotion 0..1, oil 0..1000.
        # Per variable, each candidate is exactly one scaled unit from the origin, so the
        # distances tie and the frozen key order selects ("48","1"). A global scalar
        # min/max of 1000 makes the onpromotion candidate ("48","9") nearest.
        frames = make_frames(
            "D5",
            [
                _d5("target", ("48", "1159414"), 0.0, 0.0, 0.0),
                _d5("source", ("48", "1"), 0.0, 0.0, 1000.0),
                _d5("source", ("48", "5"), 10.0, 0.0, 0.0),
                _d5("source", ("48", "9"), 0.0, 1.0, 0.0),
                _d5("source", ("49", "1"), 1.0e6, 0.0, 0.0, group="GROCERY"),
                _d5("source", ("48", "2"), 0.0, 0.0, 1.0e6, group="OTHER"),
            ],
            extras=[
                {
                    "role": "target",
                    "key": ("48", "1159414"),
                    "date": "2017-02-16",
                    "group": "GROCERY",
                    "values": {"sales": 1.0e12, "onpromotion": 0.0, "oil_price": 1.0e9},
                }
            ],
        )
        result = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=42)
        repeated = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=46)
        global_distance = {
            ("48", "1"): math.sqrt(30.0) * (1000.0 / 1000.0),
            ("48", "5"): math.sqrt(30.0) * (10.0 / 1000.0),
            ("48", "9"): math.sqrt(30.0) * (1.0 / 1000.0),
        }
        global_top = min(global_distance, key=global_distance.get)

        self.assertEqual(global_top, ("48", "9"))
        self.assertEqual(_keys(result), (("48", "1"), ("48", "5"), ("48", "9")))
        self.assertEqual(_keys(repeated), _keys(result))
        self.assertNotEqual(result.selected[0].source_key, global_top)
        self.assertEqual({item.tie_group for item in result.selected}, {1})
        np.testing.assert_allclose(result.distances, np.full(3, math.sqrt(30.0)), rtol=0.0, atol=1e-12)
        self.assertEqual(result.selection_scaler.variables, ("sales", "onpromotion", "oil_price"))
        np.testing.assert_array_equal(result.selection_scaler.minimum, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(result.selection_scaler.maximum, np.array([10.0, 1.0, 1000.0]))
        oil_candidate = np.vstack([_fill(0.0), _fill(0.0), _fill(1000.0)])
        scaled = transform_selection(result.selection_scaler, oil_candidate)
        self.assertEqual(scaled.shape, (90,))
        np.testing.assert_array_equal(scaled[:30], np.zeros(30))
        np.testing.assert_array_equal(scaled[30:60], np.zeros(30))
        np.testing.assert_array_equal(scaled[60:], np.ones(30))
        self.assertEqual(result.comparison_dates[0], "2017-01-17")
        self.assertEqual(result.comparison_dates[-1], "2017-02-15")
        self.assertNotIn("2017-02-16", result.comparison_dates)
        self.assertNotIn(("49", "1"), result.legal_keys)
        self.assertNotIn(("48", "2"), result.legal_keys)

        short = rank_candidates(frames.dataset, frames, comparison_days=15, rule="top", k=1, seed=42)
        self.assertEqual(short.comparison_dates[0], "2017-01-17")
        self.assertEqual(short.comparison_dates[-1], "2017-01-31")
        self.assertEqual(len(short.comparison_dates), 15)
        self.assertNotIn("2017-02-16", short.comparison_dates)
        self.assertEqual(short.selected[0].source_key, ("48", "1"))
        short_scaled = transform_selection(short.selection_scaler, oil_candidate[:, :15])
        self.assertEqual(short_scaled.shape, (45,))
        np.testing.assert_array_equal(short_scaled[30:], np.ones(15))

    def test_zero_range_variable_is_zero_for_the_target_and_every_candidate(self):
        frames = make_frames(
            "D5",
            [
                _d5("target", ("48", "1159414"), 0.0, 0.0, 5.0),
                _d5("source", ("48", "1"), 0.0, 0.0, 5.0),
                _d5("source", ("48", "2"), 10.0, 1.0, 5.0),
            ],
            extras=[
                {
                    "role": "target",
                    "key": ("48", "1159414"),
                    "date": "2017-02-16",
                    "group": "GROCERY",
                    "values": {"sales": 1.0e12, "onpromotion": 0.0, "oil_price": 999.0},
                }
            ],
        )
        result = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=1, seed=42)
        oil = result.selection_scaler.variables.index("oil_price")
        self.assertEqual(oil, 2)
        self.assertEqual(result.selection_scaler.range[oil], 0.0)
        self.assertEqual(result.selection_scaler.minimum[oil], 5.0)
        self.assertEqual(result.selection_scaler.maximum[oil], 5.0)
        self.assertEqual(result.selection_scaler.maximum[0], 10.0)
        target = np.vstack([_fill(0.0), _fill(0.0), _fill(5.0)])
        other = np.vstack([_fill(10.0), _fill(1.0), _fill(5.0)])
        scaled_target = transform_selection(result.selection_scaler, target)
        scaled_other = transform_selection(result.selection_scaler, other)
        np.testing.assert_array_equal(scaled_target[60:], np.zeros(30))
        np.testing.assert_array_equal(scaled_other[60:], np.zeros(30))
        np.testing.assert_array_equal(scaled_other[:30], np.ones(30))
        np.testing.assert_array_equal(scaled_other[30:60], np.ones(30))
        global_oil = (5.0 - 0.0) / (10.0 - 0.0)
        self.assertEqual(global_oil, 0.5)
        self.assertNotEqual(float(scaled_target[60]), global_oil)
        self.assertNotEqual(float(scaled_other[60]), global_oil)

    def test_euclidean_top_bottom_inverse_uniform_and_single_source_weights(self):
        frames = make_frames(
            "D1",
            [
                {"role": "target", "key": ("1", "10"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("1", "1"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("1", "2"), "values": {"sales": _fill(3.0)}},
                {"role": "source", "key": ("1", "3"), "values": {"sales": _fill(6.0)}},
                {"role": "source", "key": ("1", "4"), "values": {"sales": _fill(9.0)}},
                {"role": "source", "key": ("1", "11"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("2", "1"), "values": {"sales": _fill(0.0)}},
            ],
        )
        expected = np.array([0.0, 3.0, 6.0, 9.0], dtype=np.float64) / 9.0 * math.sqrt(30.0)
        top = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=42)
        top_other_seed = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=45)
        bottom = rank_candidates(frames.dataset, frames, comparison_days=30, rule="bottom", k=2, seed=42)
        single = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=1, seed=44)
        _target, _candidates, window_keys = similarity_windows(frames, 30)

        self.assertEqual(top.legal_keys, window_keys)
        self.assertEqual(_keys(top), (("1", "1"), ("1", "2"), ("1", "3")))
        self.assertEqual(_keys(top_other_seed), _keys(top))
        self.assertNotIn(("1", "11"), top.legal_keys)
        self.assertNotIn(("2", "1"), top.legal_keys)
        np.testing.assert_allclose(top.distances, expected[:3], rtol=0.0, atol=1e-12)
        self.assertEqual(_keys(bottom), (("1", "3"), ("1", "4")))
        np.testing.assert_allclose(bottom.distances, expected[2:], rtol=0.0, atol=1e-12)
        self.assertEqual(tuple(item.rank for item in bottom.selected), (0, 1))
        pair = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=2, seed=42)
        scores = 1.0 / (expected[:2] + 1e-8)
        np.testing.assert_allclose(pair.inverse_distance_weights, scores / scores.sum(), rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(pair.inverse_distance_weights.sum()), 1.0)
        uniform = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=42)
        np.testing.assert_array_equal(uniform.uniform_weights, np.full(3, 1.0 / 3.0))
        self.assertAlmostEqual(float(uniform.uniform_weights.sum()), 1.0)
        np.testing.assert_array_equal(single.inverse_distance_weights, np.array([1.0]))
        np.testing.assert_array_equal(single.uniform_weights, np.array([1.0]))
        self.assertEqual(single.selected[0].source_key, ("1", "1"))
        wide = rank_candidates(frames.dataset, frames, comparison_days=30, rule="bottom", k=3, seed=42)
        self.assertEqual(source_key_overlap(top, wide), 2)

    def test_tied_distances_follow_the_anchored_rule_and_repeat(self):
        inside = anchored_rank(
            [("1", "2"), ("1", "1")],
            np.array([0.0, 1e-12], dtype=np.float64),
        )
        outside_gap = float(np.nextafter(np.float64(1e-12), np.float64(1.0)))
        outside = anchored_rank(
            [("1", "2"), ("1", "1")],
            np.array([0.0, outside_gap], dtype=np.float64),
        )
        unchained = anchored_rank(
            [("1", "9"), ("1", "8"), ("1", "1")],
            np.array([0.0, 1e-12, 2e-12], dtype=np.float64),
        )
        self.assertEqual(_keys_of(inside), (("1", "1"), ("1", "2")))
        self.assertEqual(inside[0].tie_group, inside[1].tie_group)
        self.assertEqual(_keys_of(outside), (("1", "2"), ("1", "1")))
        self.assertNotEqual(outside[0].tie_group, outside[1].tie_group)
        self.assertEqual(_keys_of(unchained), (("1", "8"), ("1", "9"), ("1", "1")))
        self.assertEqual([item.tie_group for item in unchained], [1, 1, 2])
        self.assertEqual(anchored_rank([("1", "9"), ("1", "8"), ("1", "1")], np.array([0.0, 1e-12, 2e-12])), unchained)

        frames = make_frames(
            "D1",
            [
                {"role": "target", "key": ("1", "10"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("1", "1"), "values": {"sales": _fill(2e-13)}},
                {"role": "source", "key": ("1", "2"), "values": {"sales": _fill(1e-13)}},
                {"role": "source", "key": ("1", "3"), "values": {"sales": _fill(1.0)}},
            ],
        )
        first = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=42)
        second = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=42)
        self.assertEqual(_keys(first), (("1", "1"), ("1", "2"), ("1", "3")))
        self.assertEqual(_keys(second), _keys(first))
        np.testing.assert_array_equal(first.distances, second.distances)
        self.assertGreater(first.selected[0].distance, first.selected[1].distance)
        self.assertLessEqual(first.selected[0].distance - first.selected[1].distance, 1e-12)
        self.assertEqual(first.selected[0].tie_group, first.selected[1].tie_group)
        self.assertNotEqual(first.selected[0].tie_group, first.selected[2].tie_group)

    def test_random3_repeats_for_one_cell_seed_and_uses_a_separate_stream(self):
        specs = [{"role": "target", "key": ("1", "10"), "values": {"sales": _fill(0.0)}}]
        for item, sales in enumerate((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), start=1):
            specs.append({"role": "source", "key": ("1", str(item)), "values": {"sales": _fill(sales)}})
        frames = make_frames("D1", specs)
        legal = tuple(("1", str(item)) for item in range(1, 7))
        self.assertEqual(derived_random3_seed(42), _sha256_seed(42))
        self.assertEqual(derived_random3_seed(43), _sha256_seed(43))
        self.assertNotEqual(derived_random3_seed(42), derived_random3_seed(43))
        expected = {}
        for cell_seed in (42, 43):
            generator = np.random.default_rng(_sha256_seed(cell_seed))
            order = generator.permutation(len(legal))
            expected[cell_seed] = tuple(legal[int(index)] for index in order[:3])

        np.random.seed(123)
        before = np.random.rand(3).copy()
        first = rank_candidates(frames.dataset, frames, comparison_days=30, rule="random3", k=3, seed=42)
        other = rank_candidates(frames.dataset, frames, comparison_days=30, rule="random3", k=3, seed=43)
        second = rank_candidates(frames.dataset, frames, comparison_days=30, rule="random3", k=3, seed=42)
        np.random.seed(123)
        after = np.random.rand(3)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(_keys(first), expected[42])
        self.assertEqual(_keys(second), expected[42])
        self.assertEqual(_keys(other), expected[43])
        np.testing.assert_array_equal(first.distances, second.distances)
        np.testing.assert_array_equal(first.inverse_distance_weights, second.inverse_distance_weights)
        drawn_sales = np.array([float(key[1]) for key in _keys(first)], dtype=np.float64)
        np.testing.assert_allclose(first.distances, drawn_sales / 6.0 * math.sqrt(30.0), rtol=0.0, atol=1e-12)
        self.assertEqual(tuple(item.rank for item in first.selected), (0, 1, 2))
        with self.assertRaises(ContractError):
            rank_candidates(frames.dataset, frames, comparison_days=30, rule="random3", k=1, seed=42)
        with self.assertRaises(ContractError):
            rank_candidates(frames.dataset, frames, comparison_days=45, rule="top", k=1, seed=42)

    def test_missing_days_non_finite_values_and_too_few_candidates_skip(self):
        broken = _fill(4.0)
        broken[0] = np.nan
        frames = make_frames(
            "D1",
            [
                {"role": "target", "key": ("1", "10"), "values": {"sales": _fill(1.0)}},
                {"role": "source", "key": ("1", "1"), "values": {"sales": _fill(1.0)}},
                {"role": "source", "key": ("1", "2"), "values": {"sales": _fill(2.0)}},
                {
                    "role": "source",
                    "key": ("1", "3"),
                    "values": {"sales": _fill(3.0)},
                    "drop_dates": ("2017-06-02",),
                },
                {"role": "source", "key": ("1", "4"), "values": {"sales": broken}},
            ],
        )
        result = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=3, seed=42)
        again = rank_candidates(frames.dataset, frames, comparison_days=30, rule="bottom", k=3, seed=46)
        reasons = {record.source_key: record for record in result.exclusions}
        self.assertEqual(result.status, "SKIPPED_INSUFFICIENT_CANDIDATES")
        self.assertEqual(result.selected, ())
        self.assertEqual(result.n_legal, 2)
        self.assertEqual(result.legal_keys, (("1", "1"), ("1", "2")))
        self.assertEqual(result.distances.shape, (0,))
        self.assertEqual(result.inverse_distance_weights.shape, (0,))
        self.assertEqual(result.uniform_weights.shape, (0,))
        self.assertIsNone(result.selection_scaler)
        self.assertEqual(reasons[("1", "3")].reason, "missing_observed_dates")
        self.assertIn("2017-06-02", reasons[("1", "3")].missing_dates)
        self.assertEqual(reasons[("1", "4")].reason, "non_finite_similarity")
        self.assertEqual(reasons[("1", "4")].variable, "sales")
        self.assertEqual(again.status, result.status)
        self.assertEqual(again.exclusions, result.exclusions)
        self.assertEqual(len(result.selected), 0)

    def test_fifteen_and_thirty_day_boundaries_exclude_the_test_date(self):
        train_peak = _fill(0.0)
        train_peak[0] = 10.0
        validation_peak = _fill(0.0)
        validation_peak[15] = 777.0
        leaked = _fill(888.0)
        leaked[9] = np.nan
        frames = make_frames(
            "D1",
            [
                {"role": "target", "key": ("1", "10"), "values": {"sales": train_peak + validation_peak}},
                {"role": "source", "key": ("1", "1"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("1", "2"), "values": {"sales": _fill(1.0)}},
                {
                    "role": "source",
                    "key": ("1", "3"),
                    "values": {"sales": _fill(0.0)},
                    "drop_dates": ("2017-06-16",),
                },
                {"role": "source", "key": ("1", "4"), "values": {"sales": leaked}},
            ],
            extras=[
                {
                    "role": "target",
                    "key": ("1", "10"),
                    "date": "2017-07-01",
                    "values": {"sales": 1.0e12},
                }
            ],
        )
        short = rank_candidates(frames.dataset, frames, comparison_days=15, rule="top", k=2, seed=42)
        long = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=2, seed=42)
        self.assertEqual(short.comparison_dates[0], "2017-06-01")
        self.assertEqual(short.comparison_dates[-1], "2017-06-15")
        self.assertEqual(len(short.comparison_dates), 15)
        self.assertEqual(long.comparison_dates[0], "2017-06-01")
        self.assertEqual(long.comparison_dates[-1], "2017-06-30")
        self.assertEqual(len(long.comparison_dates), 30)
        self.assertTrue(all(day < "2017-07-01" for day in short.comparison_dates))
        self.assertTrue(all(day < "2017-07-01" for day in long.comparison_dates))
        self.assertNotIn("2017-07-01", long.comparison_dates)
        self.assertNotIn(("1", "3"), short.legal_keys)
        self.assertNotIn(("1", "3"), long.legal_keys)
        short_missing = {record.source_key: record for record in short.exclusions}
        self.assertEqual(short_missing[("1", "3")].reason, "incomplete_source_history")
        self.assertEqual(short_missing[("1", "3")].gate, "source_history")
        self.assertIn("2017-06-16", short_missing[("1", "3")].missing_dates)
        long_missing = {record.source_key: record for record in long.exclusions}
        self.assertEqual(long_missing[("1", "3")].reason, "missing_observed_dates")
        self.assertEqual(long_missing[("1", "3")].missing_dates, ("2017-06-16",))
        self.assertEqual(long_missing[("1", "4")].reason, "non_finite_similarity")
        self.assertEqual(short.selection_scaler.maximum[0], 10.0)
        self.assertEqual(long.selection_scaler.maximum[0], 777.0)
        self.assertLess(long.selection_scaler.maximum[0], 888.0)
        self.assertLess(short.selection_scaler.maximum[0], 1.0e12)

    def test_d4_and_d6_candidate_rules_drop_other_stores_and_groups(self):
        d4 = make_frames(
            "D4",
            [
                {"role": "target", "key": ("166", "258"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("166", "1"), "values": {"sales": _fill(4.0)}},
                {"role": "source", "key": ("167", "1"), "values": {"sales": _fill(0.0)}},
            ],
        )
        d4_result = rank_candidates(d4.dataset, d4, comparison_days=30, rule="top", k=1, seed=42)
        self.assertEqual(d4_result.selected[0].source_key, ("166", "1"))
        self.assertEqual(d4_result.legal_keys, (("166", "1"),))

        d6 = make_frames(
            "D6",
            [
                _d6("target", ("CA_1", "FOODS_3_080"), "FOODS_3", 0.0),
                _d6("source", ("CA_1", "FOODS_3_090"), "FOODS_3", 4.0),
                _d6("source", ("CA_1", "HOUSEHOLD_1_001"), "HOUSEHOLD", 0.0),
                _d6("source", ("CA_2", "FOODS_3_090"), "FOODS_3", 0.0),
            ],
        )
        d6_result = rank_candidates(d6.dataset, d6, comparison_days=30, rule="top", k=1, seed=42)
        self.assertEqual(d6_result.target_key, ("CA_1", "FOODS_3_080"))
        self.assertEqual(d6_result.selected[0].source_key, ("CA_1", "FOODS_3_090"))
        self.assertNotIn(("CA_1", "HOUSEHOLD_1_001"), d6_result.legal_keys)
        self.assertNotIn(("CA_2", "FOODS_3_090"), d6_result.legal_keys)


def _d5(role, key, sales, onpromotion, oil, group="GROCERY"):
    return {
        "role": role,
        "key": key,
        "group": group,
        "values": {
            "sales": _fill(sales),
            "onpromotion": _fill(onpromotion),
            "oil_price": _fill(oil),
        },
    }


def _d6(role, key, department, sales):
    return {"role": role, "key": key, "group": department, "values": {"sales": _fill(sales)}}


def _keys_of(ranked):
    return tuple(item.source_key for item in ranked)


def _sha256_seed(cell_seed):
    digest = hashlib.sha256(f"{cell_seed}|random3|v1".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


if __name__ == "__main__":
    unittest.main()
