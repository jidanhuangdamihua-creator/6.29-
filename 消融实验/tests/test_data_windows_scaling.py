"""Contract tests for sealed dates, windows, and the three scalers."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

from sita_ablation.data import (
    ContractError,
    load_condition_frames,
    observed_through_test,
    partition_rows,
    split_source_frame,
    clip_source_history,
    similarity_windows,
)
from sita_ablation.scaling import (
    fit_selection_scaler,
    fit_source_model_scaler,
    fit_target_model_scaler,
    inverse_sales,
    transform_model_frame,
    transform_selection,
)
from sita_ablation.windows import build_target_partition_windows, build_test_windows

DATASETS = ("D1", "D4", "D5", "D6")
HORIZONS = (1, 3, 5)
_FRAMES = {}


def frames(dataset):
    if dataset not in _FRAMES:
        _FRAMES[dataset] = load_condition_frames(dataset)
    return _FRAMES[dataset]


def _day(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


class FrozenCalendarTests(unittest.TestCase):
    def test_d5_drops_2017_08_15_before_the_split(self):
        raw_target = pd.read_parquet(
            _REPO_ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1" / "dataset5" / "target.parquet",
            columns=["date"],
        )
        raw_dates = pd.to_datetime(raw_target["date"]).dt.normalize()
        self.assertTrue((raw_dates == pd.Timestamp("2017-08-15")).any())
        loaded = frames("D5")
        self.assertFalse((loaded.target_rows["date"] > pd.Timestamp("2017-08-14")).any())
        self.assertFalse((loaded.source_rows["date"] > pd.Timestamp("2017-08-14")).any())
        self.assertEqual(_day(loaded.target_test.end), "2017-08-14")
        test_rows = partition_rows(loaded, "test")
        self.assertEqual(_day(test_rows["date"].max()), "2017-08-14")
        self.assertEqual(len(test_rows), 180)

    def test_partition_dates_match_the_frozen_file(self):
        expected_days = {"train": 15, "validation": 15, "test": 180}
        for dataset in DATASETS:
            loaded = frames(dataset)
            for name, span in (
                ("train", loaded.target_train),
                ("validation", loaded.target_val),
                ("test", loaded.target_test),
            ):
                rows = partition_rows(loaded, name)
                expected = pd.date_range(span.start, span.end, freq="D")
                self.assertEqual(len(expected), expected_days[name], dataset)
                self.assertTrue(rows["date"].reset_index(drop=True).equals(pd.Series(expected)))
                self.assertFalse(rows["date"].duplicated().any())
            self.assertLess(loaded.target_train.end, loaded.target_val.start)
            self.assertLess(loaded.target_val.end, loaded.target_test.start)
            self.assertEqual(loaded.comparison_30.start, loaded.target_train.start)
            self.assertEqual(loaded.comparison_30.end, loaded.target_val.end)
            self.assertEqual(loaded.comparison_15.iso(), loaded.target_train.iso())


class WindowTests(unittest.TestCase):
    def test_train_and_validation_stay_inside_their_slices(self):
        for dataset in DATASETS:
            loaded = frames(dataset)
            scaler = fit_target_model_scaler(partition_rows(loaded, "train"), loaded.model_features)
            for partition in ("train", "validation"):
                raw = partition_rows(loaded, partition)
                scaled = transform_model_frame(scaler, raw)
                span_start = raw["date"].min()
                span_end = raw["date"].max()
                for horizon in HORIZONS:
                    batch = build_target_partition_windows(
                        scaled,
                        key_fields=loaded.key_fields,
                        key=loaded.target_key,
                        feature_columns=loaded.model_features,
                        horizon=horizon,
                        partition=partition,
                    )
                    self.assertEqual(len(batch.label_date), 6 - horizon)
                    self.assertEqual(batch.partition, partition)
                    self.assertTrue((batch.input_start >= np.datetime64(span_start, "D")).all())
                    self.assertTrue((batch.label_date <= np.datetime64(span_end, "D")).all())
                    self.assertTrue((batch.input_end < batch.label_date).all())
                    self._assert_tensor_contract(batch, loaded.model_features, 6 - horizon)

    def test_test_labels_cover_every_frozen_day(self):
        for dataset in DATASETS:
            loaded = frames(dataset)
            scaler = fit_target_model_scaler(partition_rows(loaded, "train"), loaded.model_features)
            observed = transform_model_frame(scaler, observed_through_test(loaded))
            expected_labels = pd.date_range(loaded.target_test.start, loaded.target_test.end, freq="D")
            self.assertEqual(len(expected_labels), 180)
            for horizon in HORIZONS:
                batch = build_test_windows(
                    observed,
                    test_start=_day(loaded.target_test.start),
                    test_end=_day(loaded.target_test.end),
                    feature_columns=loaded.model_features,
                    horizon=horizon,
                )
                self.assertEqual(list(batch.label_date), list(expected_labels.to_numpy(dtype="datetime64[D]")))
                self.assertEqual(len(set(batch.label_date)), 180)
                self.assertEqual(batch.unscored_test_dates, ())
                self.assertTrue((batch.input_end < batch.label_date).all())
                self.assertTrue((batch.input_start < batch.label_date).all())
                input_end = loaded.target_test.start - pd.Timedelta(days=horizon)
                input_start = input_end - pd.Timedelta(days=9)
                self.assertEqual(batch.input_start[0], np.datetime64(input_start, "D"))
                self.assertEqual(batch.input_end[0], np.datetime64(input_end, "D"))
                self.assertLess(batch.input_end[0], np.datetime64(loaded.target_test.start, "D"))
                if horizon == 1:
                    preceding = pd.date_range(
                        loaded.target_test.start - pd.Timedelta(days=10),
                        loaded.target_test.start - pd.Timedelta(days=1),
                        freq="D",
                    )
                    self.assertEqual(batch.input_start[0], np.datetime64(preceding[0], "D"))
                    self.assertEqual(batch.input_end[0], np.datetime64(preceding[-1], "D"))
                    self.assertEqual(len(preceding), 10)
                for start, end in zip(batch.input_start, batch.input_end):
                    self.assertEqual(int((end - start) / np.timedelta64(1, "D")), 9)
                self._assert_tensor_contract(batch, loaded.model_features, 180)

    def _assert_tensor_contract(self, batch, columns, count):
        self.assertEqual(batch.x.shape, (count, 10, len(columns)))
        self.assertEqual(batch.x.dtype, np.float32)
        self.assertEqual(batch.x_sales.shape, (count, 10, 1))
        self.assertEqual(batch.x_sales.dtype, np.float32)
        self.assertTrue(np.array_equal(batch.x_sales[:, :, 0], batch.x[:, :, 0]))
        self.assertEqual(batch.y_scaled.shape, (count,))
        self.assertEqual(batch.y_scaled.dtype, np.float32)
        self.assertEqual(batch.y_original.shape, (count,))
        self.assertEqual(batch.y_original.dtype, np.float64)
        self.assertTrue(np.isfinite(batch.x).all())
        self.assertTrue(np.isfinite(batch.y_scaled).all())
        self.assertTrue(np.isfinite(batch.y_original).all())


class ScalerTests(unittest.TestCase):
    def test_target_scaler_uses_only_target_train_rows(self):
        loaded = frames("D1")
        train = partition_rows(loaded, "train")
        validation = partition_rows(loaded, "validation")
        test = partition_rows(loaded, "test")
        scaler = fit_target_model_scaler(train, loaded.model_features)
        self.assertEqual(scaler.role, "target_model")
        self.assertEqual(scaler.fit_dates, tuple(_day(day) for day in train["date"]))
        sales = train["sales"].to_numpy(dtype=np.float64)
        sales_index = loaded.model_features.index("sales")
        self.assertEqual(scaler.data_min[sales_index], np.min(sales))
        self.assertEqual(scaler.data_max[sales_index], np.max(sales))
        self.assertTrue(set(scaler.fit_dates).isdisjoint(_day(day) for day in validation["date"]))
        self.assertTrue(set(scaler.fit_dates).isdisjoint(_day(day) for day in test["date"]))
        restored = inverse_sales(scaler, transform_model_frame(scaler, train)["sales"])
        np.testing.assert_allclose(restored, sales)

    def test_source_scaler_uses_only_that_source_train_rows(self):
        loaded = frames("D1")
        _target, _candidates, keys = similarity_windows(loaded, 30)
        source = clip_source_history(loaded, keys[0])
        split = split_source_frame(source, loaded.model_features)
        scaler = fit_source_model_scaler(split.train, loaded.model_features)
        self.assertEqual(scaler.role, "source_model")
        self.assertEqual(scaler.fit_dates, tuple(_day(day) for day in split.train_dates))
        self.assertTrue(set(scaler.fit_dates).isdisjoint(_day(day) for day in split.validation_dates))
        self.assertTrue(set(scaler.fit_dates).isdisjoint(_day(day) for day in split.test_dates))
        sales_index = loaded.model_features.index("sales")
        train_sales = split.train["sales"].to_numpy(dtype=np.float64)
        self.assertEqual(scaler.data_min[sales_index], np.min(train_sales))
        self.assertEqual(scaler.data_max[sales_index], np.max(train_sales))

    def test_selection_scaler_uses_the_comparison_window_only(self):
        loaded = frames("D1")
        target_window, candidate_windows, keys = similarity_windows(loaded, 30)
        self.assertEqual(keys, tuple(("1", str(item)) for item in range(1, 10)))
        self.assertEqual(target_window.shape, (1, 30))
        self.assertEqual(candidate_windows.shape, (9, 1, 30))
        scaler = fit_selection_scaler(target_window, candidate_windows, loaded.similarity_variables)
        population = np.concatenate([target_window[0], candidate_windows[:, 0, :].reshape(-1)])
        self.assertEqual(scaler.minimum[0], np.min(population))
        self.assertEqual(scaler.maximum[0], np.max(population))
        self.assertEqual(scaler.role, "source_selection")
        short_target, short_candidates, short_keys = similarity_windows(loaded, 15)
        self.assertEqual(short_keys, keys)
        self.assertEqual(short_target.shape, (1, 15))
        self.assertEqual(short_candidates.shape, (9, 1, 15))
        train_sales = partition_rows(loaded, "train")["sales"].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(short_target[0], train_sales)

    def test_incomplete_candidate_is_excluded_from_selection(self):
        loaded = frames("D1")
        source = loaded.source_rows.copy()
        drop = (source["item_id"].astype(str) == "9") & (source["date"] == loaded.comparison_30.start)
        source = source.loc[~drop].copy()
        replaced = type(loaded)(**{**loaded.__dict__, "source_rows": source})
        _target, _candidates, keys = similarity_windows(replaced, 30)
        self.assertNotIn(("1", "9"), keys)
        self.assertIn(("1", "1"), keys)

    def test_scalers_do_not_share_parameters_and_zero_range_is_explicit(self):
        loaded = frames("D1")
        target_scaler = fit_target_model_scaler(partition_rows(loaded, "train"), loaded.model_features)
        source = clip_source_history(loaded, ("1", "1"))
        source_scaler = fit_source_model_scaler(
            split_source_frame(source, loaded.model_features).train,
            loaded.model_features,
        )
        target_window, candidate_windows, _keys = similarity_windows(loaded, 30)
        selection_scaler = fit_selection_scaler(target_window, candidate_windows, loaded.similarity_variables)
        self.assertIsNot(target_scaler.data_min, source_scaler.data_min)
        self.assertIsNot(target_scaler.scale, selection_scaler.minimum)
        original_target_min = target_scaler.data_min.copy()
        source_scaler.data_min[0] = 12345.0
        selection_scaler.minimum[0] = -12345.0
        np.testing.assert_array_equal(target_scaler.data_min, original_target_min)

        constant = _synthetic_model_frame([4.0, 4.0, 4.0, 4.0], [1.0, 2.0, 3.0, 5.0])
        constant_scaler = fit_target_model_scaler(constant, ("sales", "other"))
        transformed = transform_model_frame(constant_scaler, constant)
        self.assertEqual(constant_scaler.scale[0], 1.0)
        np.testing.assert_array_equal(transformed["sales"].to_numpy(), np.zeros(4))
        self.assertEqual(inverse_sales(constant_scaler, np.zeros(4))[0], 4.0)

        zero_target = np.array([[1.0, 3.0], [8.0, 8.0]], dtype=np.float64)
        zero_candidates = np.array([[[5.0, 7.0], [8.0, 8.0]]], dtype=np.float64)
        zero_scaler = fit_selection_scaler(zero_target, zero_candidates, ("moving", "flat"))
        scaled = transform_selection(zero_scaler, zero_target)
        self.assertEqual(zero_scaler.range[1], 0.0)
        np.testing.assert_array_equal(scaled[2:], np.zeros(2))
        self.assertGreater(zero_scaler.range[0], 0.0)

    def test_source_missing_values_are_filled_before_the_split(self):
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        sales = np.arange(30, dtype=float)
        sales[2] = np.nan
        sales[28] = np.nan
        sales[24] = 10000.0
        frame = pd.DataFrame(
            {
                "date": dates,
                "sales": sales,
                "family": pd.Series([None] * 30, dtype="object"),
                "flag": pd.Series([True] * 29 + [pd.NA], dtype="boolean"),
            }
        )
        split = split_source_frame(frame, ("sales", "family", "flag"))
        self.assertEqual(int(split.train["sales"].isna().sum()), 0)
        self.assertEqual(float(split.train.loc[split.train["date"] == dates[2], "sales"].iloc[0]), 0.0)
        self.assertEqual(float(split.test.loc[split.test["date"] == dates[28], "sales"].iloc[0]), 0.0)
        self.assertTrue(split.train["family"].isna().all())
        self.assertTrue(pd.isna(split.test["flag"].iloc[-1]))
        scaler = fit_source_model_scaler(split.train, ("sales",))
        self.assertLess(scaler.data_max[0], 10000.0)
        self.assertEqual(scaler.data_max[0], 23.0)

    def test_target_non_finite_model_values_are_not_filled(self):
        frame = _synthetic_model_frame([1.0, np.nan, 3.0], [0.0, 1.0, 2.0])
        with self.assertRaises(ContractError):
            fit_target_model_scaler(frame, ("sales", "other"))


def _synthetic_model_frame(sales, other):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(sales), freq="D"),
            "sales": sales,
            "other": other,
        }
    )


if __name__ == "__main__":
    unittest.main()
