from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.baselines.baseline_data_loader import _build_entity_slice
from scripts.baselines.run_baselines_multiseed import evaluate_entity_protocol
from src.evaluation.metrics import compute_original_scale_metrics
from src.protocols.experiment_protocol import ProtocolViolation
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
        self.assertEqual(len(rows), 4 * 5 * 5)
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
