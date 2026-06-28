import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import select_d6_target_domain as selector


def _make_wide_row(store_id, item_id, dept_id, cat_id, state_id, sales_array):
    row = {
        "id": f"{item_id}_{store_id}",
        "item_id": item_id,
        "dept_id": dept_id,
        "cat_id": cat_id,
        "store_id": store_id,
        "state_id": state_id,
    }
    for i, value in enumerate(sales_array, start=1):
        row[f"d_{i}"] = int(value)
    return row


def _make_calendar(n_days=1941, start="2011-01-29"):
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({"d": [f"d_{i}" for i in range(1, n_days + 1)], "date": dates})


class D6WindowTests(unittest.TestCase):
    def test_real_m5_windows_match_formula_not_obsolete_example_dates(self):
        cal = _make_calendar(1941)
        max_date = selector.resolve_max_date(cal, n_d_cols=1941)
        windows = selector.compute_target_windows(max_date, selector.WindowConfig())
        source = selector.compute_source_window(windows["train_start"], 300)

        self.assertEqual(max_date, pd.Timestamp("2016-05-22"))
        self.assertEqual(windows["train_start"], pd.Timestamp("2015-10-26"))
        self.assertEqual(windows["train_end"], pd.Timestamp("2015-11-09"))
        self.assertEqual(windows["val_start"], pd.Timestamp("2015-11-10"))
        self.assertEqual(windows["val_end"], pd.Timestamp("2015-11-24"))
        self.assertEqual(windows["test_start"], pd.Timestamp("2015-11-25"))
        self.assertEqual(windows["test_end"], pd.Timestamp("2016-05-22"))
        self.assertEqual(source["source_start"], pd.Timestamp("2014-12-30"))
        self.assertEqual(source["source_end"], pd.Timestamp("2015-10-25"))
        self.assertEqual((windows["test_end"] - windows["train_start"]).days + 1, 210)
        self.assertEqual((source["source_end"] - source["source_start"]).days + 1, 300)

    def test_screening_columns_end_at_train_end(self):
        cal = _make_calendar(1941)
        windows = selector.compute_target_windows(selector.resolve_max_date(cal, 1941))

        d_cols = selector.get_screening_d_cols(cal, windows["train_end"])

        self.assertEqual(d_cols[0], "d_1")
        self.assertEqual(d_cols[-1], "d_1746")
        self.assertEqual(len(d_cols), 1746)


class D6FeatureTests(unittest.TestCase):
    def test_constant_series_gives_zero_cv_and_zero_acf(self):
        feat = selector.extract_sku_summary(np.full(100, 5.0))

        self.assertEqual(feat["cv"], 0.0)
        self.assertEqual(feat["acf_lag7"], 0.0)

    def test_all_zero_series_gives_zero_cv_and_full_zero_ratio_for_mmd(self):
        feat = selector.extract_sku_summary(np.zeros(100))

        self.assertEqual(feat["cv"], 0.0)
        self.assertEqual(feat["zero_ratio"], 1.0)
        self.assertEqual(feat["acf_lag7"], 0.0)

    def test_trend_slope_uses_natural_day_index(self):
        feat = selector.extract_sku_summary(np.arange(50, dtype=float))

        self.assertAlmostEqual(feat["trend_slope"], 1.0, places=5)

    def test_screening_metrics_mean_zero_eliminates_sku(self):
        metrics = selector.compute_sku_screening_metrics(
            pd.Series({"item_id": "FOODS_3_001", "dept_id": "FOODS_3", "cat_id": "FOODS"}),
            np.zeros(30),
        )

        self.assertEqual(metrics["nonzero_ratio"], 0.0)
        self.assertTrue(np.isinf(metrics["cv"]))
        self.assertTrue(np.isinf(metrics["spike_ratio"]))


class D6MMDTests(unittest.TestCase):
    def test_shared_scaler_mean_matches_combined_data(self):
        target = np.asarray([[1.0, 2.0], [2.0, 3.0]])
        source = np.asarray([[10.0, 20.0], [12.0, 24.0]])

        mmd, gamma, scaler = selector.scaled_mmd(target, source)

        np.testing.assert_allclose(scaler.mean_, np.vstack([target, source]).mean(axis=0))
        self.assertGreaterEqual(mmd, 0.0)
        self.assertGreater(gamma, 0.0)

    def test_permutation_p_value_is_corrected(self):
        rng = np.random.RandomState(99)
        X = rng.randn(20, 7)
        Y = rng.randn(20, 7) + 5.0
        _, gamma, scaler = selector.scaled_mmd(X, Y)
        Xs = scaler.transform(X)
        Ys = scaler.transform(Y)

        p_value = selector.permutation_p_value(Xs, Ys, gamma, n_perm=19, random_state=99)

        self.assertGreaterEqual(p_value, 1 / 20)
        self.assertLessEqual(p_value, 1.0)


class D6FallbackTests(unittest.TestCase):
    def test_round1_foods3_selects_up_to_5_by_cv(self):
        metrics = pd.DataFrame(
            {
                "item_id": [f"FOODS_3_{i:03d}" for i in range(10)],
                "dept_id": ["FOODS_3"] * 10,
                "cat_id": ["FOODS"] * 10,
                "nonzero_ratio": [0.5] * 10,
                "cv": [0.1 * i for i in range(10)],
                "spike_ratio": [3.0] * 10,
            }
        )

        result = selector.select_target_skus_with_fallback(metrics)

        self.assertEqual(result["fallback_round"], 1)
        self.assertEqual(result["target_department"], "FOODS_3")
        self.assertEqual(len(result["target_skus"]), 5)
        self.assertEqual(result["target_skus"][0], "FOODS_3_000")

    def test_round3_picks_department_with_lowest_cv_median(self):
        rows = []
        for dept, base_cv in [("FOODS_1", 0.8), ("FOODS_2", 0.3), ("FOODS_3", 2.1)]:
            for i in range(4):
                rows.append(
                    {
                        "item_id": f"{dept}_{i:03d}",
                        "dept_id": dept,
                        "cat_id": "FOODS",
                        "nonzero_ratio": 0.5,
                        "cv": base_cv + 0.01 * i,
                        "spike_ratio": 3.0,
                    }
                )
        metrics = pd.DataFrame(rows)

        result = selector.select_target_skus_with_fallback(metrics)

        self.assertEqual(result["fallback_round"], 3)
        self.assertEqual(result["target_department"], "FOODS_2")
        self.assertEqual({sku.rsplit("_", 1)[0] for sku in result["target_skus"]}, {"FOODS_2"})

    def test_build_source_entities_excludes_target_store_and_matches_department(self):
        df = pd.DataFrame(
            [
                {"store_id": "CA_1", "item_id": "FOODS_3_001", "dept_id": "FOODS_3"},
                {"store_id": "CA_2", "item_id": "FOODS_3_001", "dept_id": "FOODS_3"},
                {"store_id": "CA_2", "item_id": "FOODS_2_001", "dept_id": "FOODS_2"},
            ]
        )

        source = selector.build_source_entities(df, target_store="CA_1", target_department="FOODS_3")

        self.assertEqual(source, [{"store_id": "CA_2", "item_id": "FOODS_3_001"}])

    def test_structural_shift_is_signed_target_minus_source(self):
        target = [{"mean": 5.0, "cv": 0.5, "zero_ratio": 0.1, "acf_lag7": 0.4, "trend_slope": 2.0}]
        source = [{"mean": 8.0, "cv": 0.2, "zero_ratio": 0.3, "acf_lag7": 0.1, "trend_slope": 5.0}]

        shift, _, _ = selector.compute_structural_shift(target, source)

        self.assertEqual(shift["scale"], -3.0)
        self.assertEqual(shift["volatility"], 0.3)
        self.assertAlmostEqual(shift["sparsity"], -0.2)
        self.assertEqual(shift["trend"], -3.0)


class D6OutputTests(unittest.TestCase):
    def test_run_produces_all_artifacts_and_both_mmd_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out = root / "output"
            cal = _make_calendar(620)
            cal.to_csv(root / "calendar.csv", index=False)
            pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_3_001"], "wm_yr_wk": [1], "sell_price": [1.0]}).to_csv(
                root / "sell_prices.csv", index=False
            )
            rows = []
            stores = [("CA_1", "CA"), ("CA_2", "CA"), ("TX_1", "TX"), ("WI_1", "WI")]
            depts = ["FOODS_3", "FOODS_2", "FOODS_1", "HOBBIES_1"]
            for store_index, (store, state) in enumerate(stores):
                for dept in depts:
                    for sku_index in range(6):
                        item = f"{dept}_{sku_index:03d}"
                        base = store_index + sku_index + 1
                        sales = np.asarray([(base + day % 5) if day % (sku_index + 2) != 0 else 0 for day in range(620)])
                        rows.append(_make_wide_row(store, item, dept, dept.split("_")[0], state, sales))
            pd.DataFrame(rows).to_csv(root / "sales_train_evaluation.csv", index=False)

            result = selector.run_target_selection(
                root,
                out,
                n_perm=3,
                store_sample_size=5,
                random_seed=42,
                source_history_days=300,
            )

            for name in ["target_selection_result.json", "store_candidate_profile.csv", "target_sku_metrics.csv", "target_selection_report.md"]:
                self.assertTrue((out / name).exists(), name)

            payload = json.loads((out / "target_selection_result.json").read_text(encoding="utf-8"))
            self.assertIn("store_mmd", payload)
            self.assertIn("final_mmd", payload)
            self.assertIn("source_entities", payload)
            self.assertIn("target_entities", payload)
            self.assertEqual(payload["structural_shift_semantics"], "signed_target_minus_source")
            self.assertGreaterEqual(len(payload["target_skus"]), 3)
            self.assertLessEqual(len(payload["target_skus"]), 5)
            self.assertEqual({entity["store_id"] for entity in payload["target_entities"]}, {payload["target_store"]})
            self.assertEqual(result["target_store"], payload["target_store"])


if __name__ == "__main__":
    unittest.main()
