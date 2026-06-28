import unittest

import pandas as pd

from msml_tl_rfe import build_joint_rfe_training_dataframe, run_rfe_feature_selection
from data_preprocessing import infer_modeling_feature_columns, normalize_features, build_tabular_sequence


class TestMsmlTlRfeLeakageFix(unittest.TestCase):
    def _toy_frame(self, offset: int = 0) -> pd.DataFrame:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
                    "entity_id": 1 + offset,
                    "item_id": 10 + offset,
                    "sales": float(100 + offset + idx),
                    "year": 2024,
                    "month": 1,
                    "week": (idx // 7) + 1,
                    "day": idx + 1,
                }
            )
        return pd.DataFrame(rows)

    def test_rfe_excludes_current_target_but_adds_sales_back_as_history_input(self):
        train_df = self._toy_frame()
        result = run_rfe_feature_selection(
            train_df=train_df,
            feature_cols=["sales", "year", "month", "week", "day"],
            target_col="sales",
            estimator_name="linear_regression",
            keep_ratio=0.5,
            random_state=42,
        )

        self.assertNotIn("sales", result["rfe_candidate_cols"])
        self.assertNotIn("sales", result["rfe_selected_features"])
        self.assertEqual(["sales"], result["final_selected_features"][:1])
        self.assertEqual(2, result["n_features_to_select"])
        self.assertEqual((12, 4), tuple(result["rfe_input_shape"]))
        self.assertFalse(result["contains_target_in_rfe_X"])
        self.assertTrue(result["target_removed_from_rfe"])
        self.assertTrue(result["sales_added_back_as_history_input"])

    def test_joint_rfe_dataframe_dedupes_sales_column(self):
        target_df = self._toy_frame(offset=0)
        source_df = self._toy_frame(offset=100)

        joint_df = build_joint_rfe_training_dataframe(
            target_train_df=target_df,
            selected_source_dfs=[source_df],
            feature_cols=["sales", "year", "month", "week", "day"],
            target_col="sales",
        )

        self.assertFalse(joint_df.columns.duplicated().any())
        self.assertEqual(1, list(joint_df.columns).count("sales"))
        self.assertEqual(0, joint_df.attrs["rfe_audit"]["duplicate_sales_after"])

    def test_modeling_feature_inference_excludes_identifier_and_code_columns(self):
        df = self._toy_frame()
        df["entity_id_code"] = 1
        df["brand_code"] = 2
        df["store_id"] = 3
        df["region_id"] = 4
        df["region_code"] = 5
        df["safe_feature"] = 6.0

        inferred = infer_modeling_feature_columns(df)

        self.assertEqual("sales", inferred[0])
        self.assertIn("safe_feature", inferred)
        for leaked_col in (
            "entity_id",
            "item_id",
            "entity_id_code",
            "brand_code",
            "store_id",
            "region_id",
            "region_code",
        ):
            self.assertNotIn(leaked_col, inferred)

    def test_normalize_and_sequence_use_safe_explicit_feature_columns(self):
        df = self._toy_frame()
        df["entity_id_code"] = 1
        df["brand_code"] = 2
        df["store_id"] = 3
        df["region_id"] = 4
        df["region_code"] = 5
        df["safe_feature"] = range(len(df))
        train_df = df.iloc[:8].copy()
        val_df = df.iloc[8:10].copy()
        test_df = df.iloc[10:].copy()

        safe_cols = infer_modeling_feature_columns(train_df)
        train_scaled, val_scaled, test_scaled, _, feature_columns = normalize_features(
            train_df,
            val_df,
            test_df,
            feature_cols=safe_cols,
        )
        X_train, _ = build_tabular_sequence(
            train_scaled,
            horizon=1,
            window_size=2,
            feature_cols=feature_columns,
        )

        self.assertEqual(safe_cols, feature_columns)
        self.assertEqual(len(feature_columns), X_train.shape[-1])
        self.assertIn("safe_feature", feature_columns)
        self.assertFalse(any(col.endswith("_id") or col.endswith("_code") for col in feature_columns))

    def test_rfe_excludes_identifier_and_code_candidates(self):
        train_df = self._toy_frame()
        train_df["entity_id_code"] = 1
        train_df["brand_code"] = 2
        train_df["store_id"] = 3
        train_df["region_id"] = 4
        train_df["region_code"] = 5
        train_df["safe_feature"] = [float(i) for i in range(len(train_df))]

        result = run_rfe_feature_selection(
            train_df=train_df,
            feature_cols=[
                "sales",
                "year",
                "month",
                "week",
                "day",
                "entity_id",
                "item_id",
                "entity_id_code",
                "brand_code",
                "store_id",
                "region_id",
                "region_code",
                "safe_feature",
            ],
            target_col="sales",
            estimator_name="linear_regression",
            keep_ratio=0.5,
            random_state=42,
        )

        unsafe = {
            "entity_id",
            "item_id",
            "entity_id_code",
            "brand_code",
            "store_id",
            "region_id",
            "region_code",
        }
        self.assertTrue(unsafe.isdisjoint(result["rfe_candidate_features"]))
        self.assertTrue(unsafe.isdisjoint(result["rfe_selected_features"]))
        self.assertTrue(unsafe.isdisjoint(result["final_selected_features"]))


if __name__ == "__main__":
    unittest.main()
