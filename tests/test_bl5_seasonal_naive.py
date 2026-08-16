from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.baselines.baseline_data_loader import _build_entity_slice
from scripts.baselines.bl2_moving_average import predict_bl2
from scripts.baselines.bl5_seasonal_naive import predict_bl5
from scripts.baselines.run_baselines_multiseed import METHODS, evaluate_entity_protocol


class SeasonalNaiveBaselineTest(unittest.TestCase):
    @staticmethod
    def _seasonal_record(horizon: int = 1):
        window = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=45, freq="D"),
                "sales": np.arange(10.0, 460.0, 10.0),
                "store_id": "1",
                "item_id": "10",
            }
        )
        data = _build_entity_slice(window, "d1", "1_10", entity_values=("1", "10"))
        return data, data["sample_manifest"].for_horizon(horizon)[0]

    def test_true_calendar_lag_seven_not_moving_average(self) -> None:
        record = SimpleNamespace(
            dataset_id="D1",
            target_key=("1", "10"),
            forecast_origin="2026-01-07",
            label_date="2026-01-08",
            horizon=1,
            input_dates=tuple(
                pd.date_range("2026-01-01", periods=7, freq="D").strftime("%Y-%m-%d")
            ),
            input_sales=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
        )

        self.assertEqual(predict_bl5(record), 10.0)
        self.assertNotEqual(predict_bl5(record), float(predict_bl2(record.input_sales, 1)[0]))

    def test_h1_to_h5_use_t_minus_6_through_t_minus_2(self) -> None:
        for horizon in range(1, 6):
            _, record = self._seasonal_record(horizon=horizon)
            origin = pd.Timestamp(record.forecast_origin)
            required_date = pd.Timestamp(record.label_date) - pd.Timedelta(days=7)
            self.assertEqual(required_date, origin + pd.Timedelta(days=horizon - 7))
            expected = dict(zip(map(pd.Timestamp, record.input_dates), record.input_sales))[
                required_date
            ]
            self.assertEqual(predict_bl5(record), expected)

    def test_future_target_values_do_not_change_predictions(self) -> None:
        data, record = self._seasonal_record(horizon=5)
        changed_window = data["target_window"].copy()
        origin = pd.Timestamp(record.forecast_origin)
        changed_window.loc[changed_window["date"] > origin, "sales"] = 999999.0
        changed = _build_entity_slice(
            changed_window,
            "d1",
            "1_10",
            entity_values=("1", "10"),
        )
        changed_record = changed["sample_manifest"].for_horizon(5)[0]

        self.assertEqual(record.forecast_origin, changed_record.forecast_origin)
        self.assertEqual(predict_bl5(record), predict_bl5(changed_record))

    def test_missing_lag_fails_closed_with_identity(self) -> None:
        _, record = self._seasonal_record(horizon=3)
        required_date = pd.Timestamp(record.label_date) - pd.Timedelta(days=7)
        kept = [
            (date, sales)
            for date, sales in zip(record.input_dates, record.input_sales)
            if pd.Timestamp(date) != required_date
        ]
        missing = replace(
            record,
            input_dates=tuple(date for date, _ in kept),
            input_sales=tuple(sales for _, sales in kept),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"SEASONAL_NAIVE_LAG7_HISTORY_MISSING .*dataset=D1 .*"
            r"target_identity=1/10 .*forecast_origin=.* target_date=.* "
            r"required_lag_date=.* horizon=3",
        ):
            predict_bl5(missing)

    def test_multiseed_is_deterministic_and_computed_once(self) -> None:
        data, _ = self._seasonal_record()
        calls = []

        def predictor(method, record, seed):
            calls.append((method, record.sample_key, seed))
            return predict_bl5(record)

        rows = evaluate_entity_protocol(
            data,
            predictor=predictor,
            methods=("BL5_SeasonalNaive",),
        )
        self.assertEqual(len(rows), 25)
        self.assertEqual(set(rows["seed"]), set(range(42, 47)))
        self.assertEqual(set(rows["horizon"]), set(range(1, 6)))
        self.assertEqual(
            len(calls),
            sum(
                len(data["sample_manifest"].for_horizon(horizon))
                for horizon in range(1, 6)
            ),
        )
        for _, group in rows.groupby("horizon"):
            for metric in ("rmse", "mae", "smape", "accuracy"):
                self.assertEqual(group[metric].nunique(), 1)
        self.assertTrue(
            np.isfinite(rows[["rmse", "mae", "smape", "accuracy"]].to_numpy()).all()
        )

    def test_registry_order_and_output_schema_match(self) -> None:
        self.assertEqual(
            METHODS,
            (
                "BL1_HistoricalMean",
                "BL2_MovingAverage",
                "BL3_LightGBM",
                "BL4_LSTM",
                "BL5_SeasonalNaive",
            ),
        )
        data, _ = self._seasonal_record()

        def predictor(method, record, seed):
            del method, seed
            return float(record.input_sales[-1])

        reference = evaluate_entity_protocol(
            data,
            predictor=predictor,
            methods=("BL1_HistoricalMean",),
        )
        seasonal = evaluate_entity_protocol(
            data,
            predictor=predictor,
            methods=("BL5_SeasonalNaive",),
        )
        self.assertEqual(list(reference.columns), list(seasonal.columns))
        self.assertEqual(set(seasonal["method"]), {"BL5_SeasonalNaive"})


if __name__ == "__main__":
    unittest.main()
