from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.baselines.baseline_data_loader import _build_entity_slice
from scripts.baselines.bl2_moving_average import predict_bl2
from scripts.baselines.bl5_seasonal_naive import predict_bl5
from scripts.baselines.run_baselines_multiseed import (
    _default_protocol_predictor,
    evaluate_entity_protocol,
)


class SeasonalNaiveBaselineTest(unittest.TestCase):
    def test_basic_formula_is_calendar_lag_seven(self) -> None:
        dates = pd.date_range("2026-01-01", periods=7, freq="D")
        sales = np.asarray([10, 20, 30, 40, 50, 60, 70], dtype=float)

        prediction = predict_bl5(
            dates,
            sales,
            forecast_origin="2026-01-07",
            horizon=1,
            dataset_id="d1",
            target_identity=("1", "10"),
        )

        self.assertEqual(prediction, 10.0)
        self.assertNotEqual(prediction, float(sales.mean()))

    def test_h1_to_h5_map_to_t_minus_6_through_t_minus_2(self) -> None:
        origin = pd.Timestamp("2026-01-10")
        dates = pd.date_range(origin - pd.Timedelta(days=9), origin, freq="D")
        sales = np.arange(100.0, 110.0)
        by_date = {date: float(value) for date, value in zip(dates, sales)}

        for horizon in range(1, 6):
            with self.subTest(horizon=horizon):
                prediction = predict_bl5(
                    dates,
                    sales,
                    forecast_origin=origin,
                    horizon=horizon,
                    dataset_id="d1",
                    target_identity=("1", "10"),
                )
                required_date = origin + pd.Timedelta(days=horizon - 7)
                self.assertEqual(prediction, by_date[required_date])

    def test_future_target_values_do_not_change_same_origin_prediction(self) -> None:
        dates = pd.date_range("2020-01-01", periods=45, freq="D")
        base = pd.DataFrame(
            {
                "date": dates,
                "sales": np.arange(45, dtype=float),
                "store_id": "1",
                "item_id": "10",
            }
        )
        original = _build_entity_slice(base, "d1", "1_10", entity_values=("1", "10"))
        original_sample = original["sample_manifest"].for_horizon(1)[0]
        origin = pd.Timestamp(original_sample.forecast_origin)

        mutated = base.copy()
        mutated.loc[mutated["date"] > origin, "sales"] = 999999.0
        mutated_data = _build_entity_slice(
            mutated,
            "d1",
            "1_10",
            entity_values=("1", "10"),
        )
        mutated_sample = mutated_data["sample_manifest"].for_horizon(1)[0]

        before = _default_protocol_predictor("BL5_SeasonalNaive", original_sample, 42)
        after = _default_protocol_predictor("BL5_SeasonalNaive", mutated_sample, 42)
        self.assertEqual(before, after)
        self.assertEqual(original_sample.input_sales, mutated_sample.input_sales)

    def test_bl5_is_distinct_from_seven_day_moving_average(self) -> None:
        origin = pd.Timestamp("2026-01-10")
        dates = pd.date_range(origin - pd.Timedelta(days=9), origin, freq="D")
        sales = np.asarray([1, 2, 3, 4, 10, 20, 30, 40, 50, 60], dtype=float)

        seasonal = predict_bl5(
            dates,
            sales,
            forecast_origin=origin,
            horizon=1,
            dataset_id="d1",
            target_identity=("1", "10"),
        )
        moving_average = float(predict_bl2(sales, 1)[0])

        self.assertEqual(seasonal, 4.0)
        self.assertNotEqual(seasonal, moving_average)

    def test_missing_required_lag_fails_closed(self) -> None:
        origin = pd.Timestamp("2026-01-10")
        dates = pd.date_range(origin - pd.Timedelta(days=9), origin, freq="D")
        sales = np.arange(10, dtype=float)
        required = origin - pd.Timedelta(days=6)
        keep = dates != required

        with self.assertRaisesRegex(
            ValueError,
            "SEASONAL_NAIVE_LAG7_HISTORY_MISSING",
        ) as caught:
            predict_bl5(
                dates[keep],
                sales[keep],
                forecast_origin=origin,
                horizon=1,
                dataset_id="d1",
                target_identity=("1", "10"),
            )

        message = str(caught.exception)
        for token in (
            "dataset=D1",
            "target_identity=1/10",
            "forecast_origin=2026-01-10",
            "target_date=2026-01-11",
            "required_lag_date=2026-01-04",
            "horizon=1",
        ):
            self.assertIn(token, message)

    def test_horizon_requiring_post_origin_lag_is_rejected(self) -> None:
        origin = pd.Timestamp("2026-01-10")
        dates = pd.date_range(origin - pd.Timedelta(days=9), origin, freq="D")
        sales = np.arange(10, dtype=float)

        with self.assertRaisesRegex(
            ValueError,
            "SEASONAL_NAIVE_LAG7_NOT_OBSERVED_AT_ORIGIN",
        ):
            predict_bl5(
                dates,
                sales,
                forecast_origin=origin,
                horizon=8,
                dataset_id="d1",
                target_identity=("1", "10"),
            )

    def test_runner_bl5_smoke_is_complete_deterministic_and_cached(self) -> None:
        window = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=45, freq="D"),
                "sales": np.arange(45, dtype=float),
                "store_id": "1",
                "item_id": "10",
            }
        )
        data = _build_entity_slice(
            window,
            "d1",
            "1_10",
            entity_values=("1", "10"),
        )
        manifest = data["sample_manifest"]
        expected_prediction_calls = sum(
            len(manifest.for_horizon(horizon)) for horizon in range(1, 6)
        )

        import scripts.baselines.run_baselines_multiseed as runner

        with mock.patch.object(runner, "predict_bl5", wraps=runner.predict_bl5) as wrapped:
            rows = evaluate_entity_protocol(
                data,
                methods=("BL5_SeasonalNaive",),
            )

        self.assertEqual(wrapped.call_count, expected_prediction_calls)
        self.assertEqual(len(rows), 25)
        self.assertEqual(set(rows["method"]), {"BL5_SeasonalNaive"})
        self.assertEqual(set(rows["horizon"]), {1, 2, 3, 4, 5})
        self.assertEqual(set(rows["seed"]), {42, 43, 44, 45, 46})
        self.assertEqual(rows["sample_manifest_digest"].nunique(), 1)
        self.assertTrue(
            np.isfinite(rows[["rmse", "mae", "smape", "accuracy"]].to_numpy(dtype=float)).all()
        )
        for _, group in rows.groupby("horizon"):
            for metric in ("rmse", "mae", "smape", "accuracy"):
                self.assertEqual(group[metric].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
