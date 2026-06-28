import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import select_d4_target_domain as selector


def _daily_rows(store_id, product_id, category, start, periods, base, wobble=0.0):
    dates = pd.date_range(start, periods=periods, freq="D")
    rows = []
    for i, dt in enumerate(dates):
        sale = base + (wobble if i % 7 == 0 else 0.0)
        rows.append(
            {
                "city_id": 1,
                "store_id": store_id,
                "product_id": product_id,
                "dt": dt,
                "sale_amount": float(sale),
                "second_category_id": category,
            }
        )
    return rows


class D4TargetSelectionHelperTests(unittest.TestCase):
    def test_target_and_source_windows_are_inclusive_and_ordered(self):
        cfg = selector.WindowConfig()

        windows = selector.compute_target_windows("2024-12-31", cfg)
        source = selector.compute_source_window(windows["target_train_start"], 300)

        self.assertEqual(windows["test_start"], pd.Timestamp("2024-07-05"))
        self.assertEqual(windows["val_start"], pd.Timestamp("2024-06-20"))
        self.assertEqual(windows["target_train_start"], pd.Timestamp("2024-06-05"))
        self.assertEqual(source["source_history_start"], pd.Timestamp("2023-08-11"))
        self.assertEqual(source["source_history_end"], pd.Timestamp("2024-06-05"))

    def test_calendar_coverage_and_gap_count_missing_natural_days(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])})

        self.assertEqual(selector.date_coverage_days(df), 5)
        self.assertEqual(selector.max_calendar_gap_days(df), 2)

    def test_sku_metrics_use_passed_training_history(self):
        df = pd.DataFrame(
            {
                "dt": pd.date_range("2024-01-01", periods=5, freq="D"),
                "sale_amount": [10.0, 10.0, 0.0, 10.0, 50.0],
            }
        )

        metrics = selector.compute_sku_metrics(df)

        self.assertEqual(metrics["date_coverage_days"], 5)
        self.assertEqual(metrics["max_gap_days"], 0)
        self.assertAlmostEqual(metrics["nonzero_ratio"], 0.8)
        self.assertAlmostEqual(metrics["spike_ratio"], 5.0)
        self.assertGreater(metrics["cv"], 0)

    def test_scaled_mmd_fits_one_shared_scaler_on_target_and_source_features(self):
        target = np.asarray([[1.0, 2.0], [2.0, 3.0]])
        source = np.asarray([[10.0, 20.0], [12.0, 24.0], [14.0, 28.0]])

        mmd, gamma, scaler = selector.scaled_mmd(target, source)

        np.testing.assert_allclose(scaler.mean_, np.vstack([target, source]).mean(axis=0))
        self.assertGreaterEqual(mmd, 0.0)
        self.assertGreater(gamma, 0.0)

    def test_choose_target_skus_uses_largest_category_then_lowest_cv(self):
        metrics = pd.DataFrame(
            {
                "product_id": [101, 102, 103, 201, 202],
                "second_category_id": [10, 10, 10, 20, 20],
                "date_coverage_days": [510, 510, 510, 510, 510],
                "max_gap_days": [0, 0, 0, 0, 0],
                "nonzero_ratio": [0.9, 0.9, 0.9, 0.9, 0.9],
                "cv": [0.3, 0.1, 0.2, 0.01, 0.02],
                "spike_ratio": [2, 2, 2, 2, 2],
            }
        )

        eligible = selector.filter_eligible_skus(metrics)
        skus, category = selector.choose_target_skus(eligible)

        self.assertEqual(category, 10)
        self.assertEqual(skus, [102, 103, 101])

    def test_select_target_store_prefers_quality_inside_mmd_interquartile_range(self):
        scan = pd.DataFrame(
            {
                "store_id": [1, 2, 3, 4],
                "mmd": [0.10, 0.20, 0.30, 0.40],
                "eligible_sku_count": [50, 10, 30, 80],
                "source_sku_count": [100, 100, 100, 100],
                "store_max_gap_days": [0, 0, 0, 0],
            }
        )

        selected = selector.select_target_store(scan)

        self.assertEqual(selected["store_id"], 3)
        self.assertAlmostEqual(selected["mmd_q25"], 0.175)
        self.assertAlmostEqual(selected["mmd_q75"], 0.325)


class D4TargetSelectionOutputTests(unittest.TestCase):
    def test_run_target_selection_writes_required_artifacts_from_raw_data(self):
        rows = []
        for store_id, base in [(1, 10.0), (2, 14.0), (3, 22.0), (4, 30.0)]:
            for product_id in range(100, 105):
                rows.extend(_daily_rows(store_id, product_id, 10, "2023-01-01", 520, base + product_id / 1000))
            for product_id in range(200, 202):
                rows.extend(_daily_rows(store_id, product_id, 20, "2023-01-01", 520, base + 5.0))
        raw = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "train.parquet"
            profile_dir = root / "profile"
            output_dir = root / "target_selection"
            profile_dir.mkdir()
            raw.to_parquet(raw_path, index=False)
            pd.DataFrame(
                {
                    "entity_id": [f"store_id={store}|product_id={product}" for store in [1, 2, 3, 4] for product in range(100, 105)],
                    "total_calendar_days": [520] * 20,
                    "nonzero_sales_days": [520] * 20,
                    "coefficient_of_variation": [0.1] * 20,
                    "quality_score": [0.9] * 20,
                }
            ).to_csv(profile_dir / "source_target_candidate_report.csv", index=False)

            result = selector.run_target_selection(
                raw_path=raw_path,
                profile_dir=profile_dir,
                output_dir=output_dir,
                permutations=5,
                random_state=7,
            )

            for name in [
                "store_candidate_profile.csv",
                "warehouse_mmd_scan.csv",
                "target_sku_metrics.csv",
                "target_selection_result.json",
                "target_selection_report.md",
            ]:
                self.assertTrue((output_dir / name).exists(), name)

            with (output_dir / "target_selection_result.json").open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(result["target_store_id"], payload["target_store_id"])
            self.assertGreaterEqual(len(payload["target_skus"]), 3)
            self.assertLessEqual(len(payload["target_skus"]), 5)
            self.assertEqual(payload["target_categories"], [10])
            self.assertGreater(payload["source_store_count"], 0)
            self.assertGreater(payload["source_sku_count"], 0)
            self.assertIn("value", payload["mmd"])
            self.assertIn("p_value", payload["permutation_test"])
            self.assertIn("trend", payload["structural_shift"])


if __name__ == "__main__":
    unittest.main()
