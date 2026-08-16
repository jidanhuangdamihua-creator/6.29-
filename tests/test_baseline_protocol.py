from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.baselines.baseline_data_loader import _build_entity_slice, load_baseline_data
from scripts.baselines.bl4_lstm import _direct_h_arrays, _scale_partitions
from scripts.baselines.run_baselines_multiseed import (
    _validate_matured_record,
    evaluate_entity_protocol,
)
from src.evaluation.metrics import compute_original_scale_metrics
from src.protocols.experiment_protocol import ProtocolViolation, formal_target_entity_keys
from src.protocols.gate1_transformation import dataset_contract
from src.protocols.rolling_origin import aggregate_protocol_results


class BaselineProtocolTest(unittest.TestCase):
    def test_baseline_loader_returns_shared_manifest(self) -> None:
        window = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=45, freq="D"),
                "sales": np.arange(45, dtype=float),
                "store_id": "1",
                "item_id": "10",
            }
        )
        data = _build_entity_slice(window, "d1", "1_10")
        manifest = data["sample_manifest"]

        self.assertEqual(manifest.horizons, (1, 2, 3, 4, 5))
        self.assertEqual(manifest.for_horizon(1)[0].label_date, "2020-02-10")
        self.assertEqual(len(manifest.for_horizon(1)[0].input_dates), 10)
        self.assertEqual(len(data["train_sales"]), 15)
        self.assertEqual(len(data["val_sales"]), 15)
        self.assertEqual(data["lookback"], 10)
        self.assertEqual(data["sample_manifest_digest"], manifest.digest)
        self.assertEqual(data["protocol_version"], "d1_d6_protocol_v1")

    def test_original_scale_metrics_are_primary(self) -> None:
        result = compute_original_scale_metrics(
            np.asarray([1.0, 3.0]),
            np.asarray([2.0, 1.0]),
        )
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["accuracy"], 1.0 / (np.sqrt(2.5) + 1e-8))
        self.assertEqual(result["primary_metric_space"], "original_sales")

    def test_every_baseline_uses_every_manifest_sample_seed_and_horizon(self) -> None:
        window = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=45, freq="D"),
                "sales": np.arange(45, dtype=float),
                "store_id": "1",
                "item_id": "10",
            }
        )
        data = _build_entity_slice(window, "d1", "1_10")

        def predictor(method, record, seed):
            del method, seed
            return float(record.input_sales[-1])

        rows = evaluate_entity_protocol(data, predictor=predictor)
        self.assertEqual(len(rows), 5 * 5 * 5)
        self.assertEqual(set(rows["seed"]), set(range(42, 47)))
        self.assertEqual(set(rows["horizon"]), set(range(1, 6)))
        self.assertEqual(set(rows["primary_metric_space"]), {"original_sales"})
        self.assertEqual(set(rows["rmse_metric_space"]), {"original_sales_space"})
        self.assertEqual(set(rows["smape_metric_space"]), {"original_sales_space"})
        self.assertEqual(rows["sample_manifest_digest"].nunique(), 1)
        expected_counts = {
            horizon: len(data["sample_manifest"].for_horizon(horizon))
            for horizon in range(1, 6)
        }
        for horizon, count in expected_counts.items():
            self.assertEqual(
                set(rows.loc[rows["horizon"] == horizon, "sample_count"]),
                {count},
            )

    def test_bl4_direct_h_arrays_use_direct_labels(self) -> None:
        scaled = np.arange(15, dtype=float)
        h1_x, h1_y = _direct_h_arrays(scaled, horizon=1, lookback=10, name="train")
        h5_x, h5_y = _direct_h_arrays(scaled, horizon=5, lookback=10, name="train")

        self.assertEqual(h1_x.shape, (5, 10, 1))
        np.testing.assert_array_equal(h1_y.reshape(-1), np.arange(10, 15, dtype=float))
        self.assertEqual(h5_x.shape, (1, 10, 1))
        np.testing.assert_array_equal(h5_x[0, :, 0], np.arange(10, dtype=float))
        self.assertEqual(float(h5_y[0, 0]), 14.0)

    def test_bl4_scaler_fits_train_only(self) -> None:
        train = np.arange(15, dtype=float)
        validation = np.arange(100, 115, dtype=float)
        train_scaled, validation_scaled, scaler = _scale_partitions(train, validation)

        self.assertEqual(int(scaler.n_samples_seen_), 15)
        self.assertEqual(float(scaler.data_min_[0]), 0.0)
        self.assertEqual(float(scaler.data_max_[0]), 14.0)
        self.assertGreater(float(validation_scaled.min()), 1.0)
        self.assertEqual(train_scaled.shape, (15,))

    def test_bl4_direct_h_fails_closed_one_day_short(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 15"):
            _direct_h_arrays(
                np.arange(14, dtype=float),
                horizon=5,
                lookback=10,
                name="train",
            )

    def test_matured_truth_guard_rejects_future_input_and_wrong_label(self) -> None:
        valid = SimpleNamespace(
            input_dates=tuple(pd.date_range("2020-01-01", periods=10).strftime("%Y-%m-%d")),
            input_sales=tuple(range(10)),
            forecast_origin="2020-01-10",
            horizon=5,
            label_date="2020-01-15",
        )
        _validate_matured_record(valid, lookback=10)

        future = SimpleNamespace(**vars(valid))
        future.input_dates = (*valid.input_dates[:-1], "2020-01-11")
        with self.assertRaisesRegex(ProtocolViolation, "future target truth"):
            _validate_matured_record(future, lookback=10)

        wrong_label = SimpleNamespace(**vars(valid))
        wrong_label.label_date = "2020-01-14"
        with self.assertRaisesRegex(ProtocolViolation, "label date"):
            _validate_matured_record(wrong_label, lookback=10)

    def test_formal_loader_matches_sealed_authority(self) -> None:
        for dataset_id in tuple(f"d{number}" for number in range(1, 7)):
            with self.subTest(dataset_id=dataset_id):
                spec = dataset_contract(dataset_id)
                slices = load_baseline_data(dataset_id)
                self.assertEqual(
                    tuple(item["entity_key"] for item in slices),
                    formal_target_entity_keys(dataset_id),
                )
                for item in slices:
                    window = item["target_window"]
                    self.assertEqual(window["date"].min().date(), spec.target_train_start)
                    self.assertEqual(window["date"].max().date(), spec.blind_end)
                    self.assertEqual(len(item["train_sales"]), 15)
                    self.assertEqual(len(item["val_sales"]), 15)
                    self.assertEqual(item["train_dates"][0], spec.target_train_start.isoformat())
                    self.assertEqual(item["train_dates"][-1], spec.target_train_end.isoformat())
                    self.assertEqual(item["validation_dates"][0], spec.validation_start.isoformat())
                    self.assertEqual(item["validation_dates"][-1], spec.validation_end.isoformat())

    def test_bl4_fits_once_per_horizon_seed_and_never_on_rolling_truth(self) -> None:
        window = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=45, freq="D"),
                "sales": np.arange(45, dtype=float),
                "store_id": "1",
                "item_id": "10",
            }
        )
        data = _build_entity_slice(window, "d1", "1_10", entity_values=("1", "10"))
        fitted_calls = []
        prediction_calls = []

        def fake_fit(train, validation, *, horizon, lookback, seed):
            fitted_calls.append(
                (tuple(train), tuple(validation), int(horizon), int(lookback), int(seed))
            )
            return (int(horizon), int(seed))

        def fake_predict(fitted, input_sales):
            prediction_calls.append((fitted, tuple(input_sales)))
            return float(input_sales[-1])

        with patch(
            "scripts.baselines.run_baselines_multiseed.fit_bl4",
            side_effect=fake_fit,
        ), patch(
            "scripts.baselines.run_baselines_multiseed.predict_bl4",
            side_effect=fake_predict,
        ):
            rows = evaluate_entity_protocol(data, methods=("BL4_LSTM",))

        self.assertEqual(len(rows), 25)
        self.assertEqual(len(fitted_calls), 25)
        self.assertEqual(
            {(call[2], call[4]) for call in fitted_calls},
            {(horizon, seed) for horizon in range(1, 6) for seed in range(42, 47)},
        )
        self.assertTrue(all(len(call[0]) == 15 and len(call[1]) == 15 for call in fitted_calls))
        self.assertTrue(all(call[0] == fitted_calls[0][0] for call in fitted_calls))
        self.assertTrue(all(call[1] == fitted_calls[0][1] for call in fitted_calls))
        expected_predictions = sum(
            len(data["sample_manifest"].for_horizon(horizon)) * 5
            for horizon in range(1, 6)
        )
        self.assertEqual(len(prediction_calls), expected_predictions)

    def test_bl3_fits_once_per_horizon_seed_without_rolling_refit(self) -> None:
        window = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=45, freq="D"),
                "sales": np.arange(45, dtype=float),
                "store_id": "1",
                "item_id": "10",
            }
        )
        data = _build_entity_slice(window, "d1", "1_10", entity_values=("1", "10"))
        calls = []

        def fake_predict(train_features, test_features, random_state):
            calls.append((train_features.copy(), test_features.copy(), int(random_state)))
            return np.zeros(len(test_features), dtype=float)

        with patch(
            "scripts.baselines.run_baselines_multiseed.predict_bl3",
            side_effect=fake_predict,
        ):
            rows = evaluate_entity_protocol(data, methods=("BL3_LightGBM",))

        self.assertEqual(len(rows), 25)
        self.assertEqual(len(calls), 25)
        self.assertEqual({seed for _, _, seed in calls}, set(range(42, 47)))
        self.assertTrue(all("sales" in train.columns for train, _, _ in calls))
        self.assertTrue(all("sales" not in test.columns for _, test, _ in calls))
        for index, (_, test, _) in enumerate(calls):
            horizon = index // 5 + 1
            self.assertEqual(
                len(test),
                len(data["sample_manifest"].for_horizon(horizon)),
            )

    def test_aggregation_requires_five_horizons_and_five_seeds(self) -> None:
        rows = []
        for horizon in range(1, 6):
            for seed in range(42, 47):
                rows.append(
                    {
                        "dataset_id": "D1",
                        "target_entity_key": "Store1/Item10",
                        "scenario": "without",
                        "method": "BL1",
                        "horizon": horizon,
                        "seed": seed,
                        "rmse": float(horizon + seed - 42),
                        "mae": float(horizon),
                        "smape": float(horizon * 2),
                        "accuracy": 1.0 / float(horizon + seed - 42),
                        "primary_metric_space": "original_sales",
                        "sample_manifest_digest": "digest-all",
                    }
                )
        aggregates = aggregate_protocol_results(pd.DataFrame(rows))
        self.assertEqual(set(aggregates["aggregate_scope"]), {"horizon", "horizons_1_5"})
        self.assertEqual(len(aggregates), 6)
        horizon_one = aggregates[
            (aggregates["aggregate_scope"] == "horizon")
            & (aggregates["horizon"] == 1)
        ].iloc[0]
        self.assertEqual(horizon_one["rmse_mean"], 3.0)

        with self.assertRaisesRegex(ProtocolViolation, "formal seed coverage"):
            aggregate_protocol_results(pd.DataFrame(rows[:-1]))


if __name__ == "__main__":
    unittest.main()
