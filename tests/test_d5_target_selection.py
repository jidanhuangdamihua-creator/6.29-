import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import select_d5_target_domain as selector


def _favorita_rows(store, item, start="2015-06-01", periods=760, base=10.0, promo_every=0):
    rows = []
    for i, dt in enumerate(pd.date_range(start, periods=periods, freq="D")):
        rows.append(
            {
                "id": len(rows),
                "date": dt.strftime("%Y-%m-%d"),
                "store_nbr": store,
                "item_nbr": item,
                "unit_sales": float(base + (1.0 if i % 7 == 0 else 0.0)),
                "onpromotion": bool(promo_every and i % promo_every == 0),
            }
        )
    return rows


class D5TargetSelectionHelperTests(unittest.TestCase):
    def test_windows_match_favorita_global_dates(self):
        windows = selector.compute_target_windows("2017-08-15", selector.WindowConfig())
        source = selector.compute_source_window(windows["train_start"], 300)

        self.assertEqual(windows["train_start"], pd.Timestamp("2017-01-17"))
        self.assertEqual(windows["val_start"], pd.Timestamp("2017-02-01"))
        self.assertEqual(windows["test_start"], pd.Timestamp("2017-02-16"))
        self.assertEqual(source["source_start"], pd.Timestamp("2016-04-22"))

    def test_cleaning_clips_negative_sales_and_complete_series_fills_missing_days(self):
        df = pd.DataFrame(
            {
                "date": ["2017-01-01", "2017-01-03"],
                "unit_sales": [-2.0, 5.0],
                "onpromotion": [True, False],
            }
        )

        clean = selector.clean_sales_frame(df)
        full = selector.complete_daily_series(clean, "2017-01-01", "2017-01-03")

        self.assertEqual(full["unit_sales"].tolist(), [0.0, 0.0, 5.0])
        self.assertEqual(full["onpromotion"].tolist(), [True, False, False])

    def test_transaction_gap_uses_30_day_hard_gate_and_7_day_audit_flag(self):
        tx = pd.DataFrame(
            {
                "date": ["2017-01-17", "2017-01-25", "2017-03-05"],
                "store_nbr": [1, 1, 1],
                "transactions": [1, 1, 1],
            }
        )

        gap = selector.transaction_gap_summary(tx, 1, "2017-01-17", "2017-03-05")

        self.assertGreater(gap["max_transaction_gap_days"], 30)
        self.assertTrue(gap["has_gap_gt_7_days"])
        self.assertFalse(gap["passes_transaction_gap_gate"])

    def test_structural_shift_matches_d4_absolute_difference_semantics(self):
        target = [{"trend_slope": 3.0, "acf_lag7": 0.6, "cv": 0.4, "coverage_ratio": 0.9, "mean": 10.0}]
        source = [{"trend_slope": 1.0, "acf_lag7": -0.2, "cv": 0.1, "coverage_ratio": 0.5, "mean": 7.0}]

        shift, _, _ = selector.compute_structural_shift(target, source)

        self.assertAlmostEqual(shift["trend"], 2.0)
        self.assertAlmostEqual(shift["seasonality"], 0.8)
        self.assertAlmostEqual(shift["scale"], 3.0)

    def test_store_level_coverage_ratio_uses_global_calendar_days(self):
        series = pd.DataFrame(
            {
                "date": pd.date_range("2017-01-01", periods=10, freq="D"),
                "unit_sales": [1.0] * 10,
            }
        )

        summary = selector.summarize_sales_series(series, coverage_denominator=1688)

        self.assertAlmostEqual(summary["coverage_ratio"], 10 / 1688)


class D5TargetSelectionSelectionTests(unittest.TestCase):
    def test_family_fallback_mixes_families_and_source_follows_actual_target_families(self):
        metrics = pd.DataFrame(
            {
                "store_nbr": [1, 1, 1, 1],
                "item_nbr": [101, 102, 201, 202],
                "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "BEVERAGES"],
                "date_coverage_days": [520, 520, 520, 520],
                "coverage_ratio": [0.8, 0.8, 0.8, 0.8],
                "onpromotion_ratio": [0.0, 0.0, 0.0, 0.0],
                "cv": [0.2, 0.3, 0.1, 0.4],
                "spike_ratio": [2.0, 2.0, 2.0, 2.0],
            }
        )

        selected = selector.select_target_skus_with_fallback(metrics)

        self.assertEqual(selected["target_family"], "MIXED")
        self.assertEqual(set(selected["target_families"]), {"GROCERY I", "BEVERAGES"})

        items = pd.DataFrame(
            {
                "item_nbr": [101, 102, 201, 202, 301],
                "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "BEVERAGES", "DAIRY"],
            }
        )
        train = pd.DataFrame(
            {
                "store_nbr": [2, 2, 3, 3, 4],
                "item_nbr": [101, 201, 102, 202, 301],
                "date": pd.to_datetime(["2016-04-22"] * 5),
                "unit_sales": [1, 1, 1, 1, 1],
            }
        )

        source = selector.build_source_entities(
            train,
            items,
            target_store=1,
            target_families=selected["target_families"],
        )

        self.assertEqual({row["item_nbr"] for row in source}, {101, 102, 201, 202})

    def test_scaled_mmd_fits_one_shared_scaler(self):
        target = np.asarray([[1.0, 2.0], [2.0, 3.0]])
        source = np.asarray([[10.0, 20.0], [12.0, 24.0]])

        mmd, gamma, scaler = selector.scaled_mmd(target, source)

        np.testing.assert_allclose(scaler.mean_, np.vstack([target, source]).mean(axis=0))
        self.assertGreaterEqual(mmd, 0.0)
        self.assertGreater(gamma, 0.0)


class D5TargetSelectionOutputTests(unittest.TestCase):
    def test_run_target_selection_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "Dataset5"
            out = root / "target_selection"
            data.mkdir()
            rows = []
            for store, base in [(1, 10.0), (2, 12.0), (3, 14.0)]:
                for item in [101, 102, 201, 202, 301]:
                    rows.extend(_favorita_rows(store, item, base=base + item / 1000))
            pd.DataFrame(rows).to_csv(data / "train.csv", index=False)
            pd.DataFrame(
                {
                    "item_nbr": [101, 102, 201, 202, 301],
                    "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "BEVERAGES", "DAIRY"],
                    "class": [1, 1, 2, 2, 3],
                    "perishable": [0, 0, 0, 0, 0],
                }
            ).to_csv(data / "items.csv", index=False)
            pd.DataFrame(
                {
                    "store_nbr": [1, 2, 3],
                    "city": ["Quito", "Quito", "Quito"],
                    "state": ["Pichincha", "Pichincha", "Pichincha"],
                    "type": ["A", "A", "B"],
                    "cluster": [1, 1, 2],
                }
            ).to_csv(data / "stores.csv", index=False)
            tx_rows = [
                {"date": dt.strftime("%Y-%m-%d"), "store_nbr": store, "transactions": 100}
                for store in [1, 2, 3]
                for dt in pd.date_range("2017-01-17", "2017-08-15")
            ]
            pd.DataFrame(tx_rows).to_csv(data / "transactions.csv", index=False)

            result = selector.run_target_selection(
                data,
                out,
                permutations=3,
                store_sku_sample=3,
                random_state=7,
            )

            for name in [
                "target_selection_result.json",
                "store_candidate_profile.csv",
                "target_sku_metrics.csv",
                "target_selection_report.md",
            ]:
                self.assertTrue((out / name).exists(), name)

            payload = json.loads((out / "target_selection_result.json").read_text(encoding="utf-8"))
            self.assertIn("source_entities", payload)
            self.assertGreaterEqual(len(payload["target_skus"]), 3)
            self.assertEqual(result["target_store"], payload["target_store"])
            profile = pd.read_csv(out / "store_candidate_profile.csv")
            self.assertIn("max_transaction_gap_days", profile.columns)
            self.assertIn("has_gap_gt_7_days", profile.columns)


if __name__ == "__main__":
    unittest.main()
