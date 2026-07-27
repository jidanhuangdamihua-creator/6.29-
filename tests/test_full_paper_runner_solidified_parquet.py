import unittest
import tempfile
from pathlib import Path

import pandas as pd

from scripts import run_full_paper_experiments
from src.data_processing.data_preprocessing import temporal_split_by_ratio_or_dates
from scripts.aggregate_d1_d6_results import (
    _assert_dataset3_result_target_is_store10,
    _normalize_row,
)
from scripts.run_full_paper_experiments import (
    _apply_information_sharing_filter,
    _assert_dataset3_target_is_store10,
    _attach_target_metadata,
    _dataset3_target_metadata,
    _load_config,
    _load_solidified_base_data,
    _materialize_result_dataframes,
    _project_modeling_frames,
    _resolve_dataset_feature_cols,
    _save_dataset_result_csvs,
)
from src.protocols.experiment_protocol import get_experiment_protocol
from src.utils.parquet_data_loader import attach_window_attrs


def _fixture_root() -> Path:
    checkout = Path(__file__).resolve().parents[1]
    if (checkout / "数据集" / "固化数据" / "dataset1-source.parquet").is_file():
        return checkout
    git_pointer = checkout / ".git"
    if git_pointer.is_file():
        git_dir = Path(git_pointer.read_text(encoding="utf-8").strip().split(":", 1)[1].strip())
        primary = git_dir.parents[2]
        if (primary / "数据集" / "固化数据" / "dataset1-source.parquet").is_file():
            return primary
    return checkout


FIXTURE_ROOT = _fixture_root()


class FullPaperRunnerSolidifiedParquetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_root = run_full_paper_experiments.ROOT
        run_full_paper_experiments.ROOT = FIXTURE_ROOT

    @classmethod
    def tearDownClass(cls):
        run_full_paper_experiments.ROOT = cls._original_root

    def test_d3_solidified_base_data_has_no_todo_region_and_normalized_columns(self):
        base = _load_solidified_base_data(
            dataset_name="Dataset3",
            cfg=_load_config(),
            strict_paper_mode=False,
            strict_paper_split=False,
        )

        source_df = base["source_df"]
        target_df = base["target_df"]

        self.assertIn("customers", source_df.columns)
        self.assertIn("open", source_df.columns)
        self.assertIn("promo", source_df.columns)
        self.assertIn("school_holiday", source_df.columns)
        self.assertNotIn("Customers", source_df.columns)
        self.assertNotIn("Open", source_df.columns)
        self.assertNotIn("Promo", source_df.columns)
        self.assertNotIn("SchoolHoliday", source_df.columns)

        combined_text = pd.concat([source_df, target_df], ignore_index=True).astype(str)
        self.assertFalse(
            combined_text.apply(
                lambda col: col.str.contains("TODO_REGION_UNAVAILABLE", regex=False).any()
            ).any()
        )

        self.assertEqual("Dataset3", source_df.attrs["dataset_name"])
        self.assertEqual("Dataset3", target_df.attrs["dataset_name"])
        self.assertEqual("source", source_df.attrs["split_role"])
        self.assertEqual("target", target_df.attrs["split_role"])
        self.assertEqual(
            int(target_df["date"].nunique()),
            target_df.attrs["target_window_unique_days"],
        )
        self.assertEqual(210, len(target_df))
        self.assertEqual(["10"], sorted(target_df["entity_id"].astype(str).unique().tolist()))
        self.assertEqual(["10"], sorted(target_df["store_id"].astype(str).unique().tolist()))
        self.assertEqual(
            {
                "target_entity_id": "10",
                "target_store_id": "10",
                "target_item_id": "1",
            },
            _dataset3_target_metadata(target_df),
        )

        with self.assertRaisesRegex(ValueError, "D3 target missing required columns"):
            _assert_dataset3_target_is_store10(pd.DataFrame({"entity_id": ["10"]}))
        with self.assertRaisesRegex(ValueError, "D3 target store mismatch"):
            _assert_dataset3_target_is_store10(
                pd.DataFrame({"entity_id": ["10"], "store_id": ["2"]})
            )
        with self.assertRaisesRegex(ValueError, "D3 target entity mismatch"):
            _assert_dataset3_target_is_store10(
                pd.DataFrame({"entity_id": ["2"], "store_id": ["10"]})
            )

        d3_result_path = Path("/tmp/dataset3_results.csv")
        _assert_dataset3_result_target_is_store10(
            pd.DataFrame(
                {
                    "target_entity_id": ["10", "10"],
                    "target_store_id": ["10", "10"],
                }
            ),
            d3_result_path,
        )
        with self.assertRaisesRegex(ValueError, "has no target identity columns"):
            _assert_dataset3_result_target_is_store10(
                pd.DataFrame({"method": ["No-TL"]}),
                d3_result_path,
            )
        with self.assertRaisesRegex(ValueError, "wrong/stale result target_store_id"):
            _assert_dataset3_result_target_is_store10(
                pd.DataFrame({"target_store_id": ["2"]}),
                d3_result_path,
            )
        with self.assertRaisesRegex(ValueError, "refusing to set target_entity_key=GLOBAL"):
            _normalize_row({}, 3, d3_result_path)
        self.assertEqual(
            "GLOBAL",
            _normalize_row({}, 2, Path("/tmp/dataset2_results.csv"))["target_entity_key"],
        )

    def test_aggregate_d3_source_csv_proves_target_store10(self):
        fixture = pd.DataFrame(
            {
                "target_entity_id": ["10", "10"],
                "target_store_id": ["10", "10"],
                "error": ["", "completed without import failures"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            d3_source = Path(tmpdir) / "dataset3_results.csv"
            fixture.to_csv(d3_source, index=False)
            df = pd.read_csv(d3_source, dtype=str, keep_default_na=False)

        _assert_dataset3_result_target_is_store10(df, d3_source)
        self.assertEqual({"10"}, set(df["target_entity_id"]))
        self.assertEqual({"10"}, set(df["target_store_id"]))
        self.assertFalse(
            df["error"].astype(str).str.contains("Traceback|ImportError", regex=True).any()
        )
        bad_df = df.copy()
        bad_df.loc[0, "error"] = "Traceback: ImportError while loading dependency"
        self.assertTrue(
            bad_df["error"].astype(str).str.contains("Traceback|ImportError", regex=True).any()
        )

    def test_d1_d2_d3_solidified_non_strict_targets_carry_paper_window_attrs(self):
        cfg = _load_config()
        for dataset_name in ("Dataset1", "Dataset2", "Dataset3"):
            with self.subTest(dataset_name=dataset_name):
                base = _load_solidified_base_data(
                    dataset_name=dataset_name,
                    cfg=cfg,
                    strict_paper_mode=False,
                    strict_paper_split=False,
                )
                target_df = base["target_df"]

                self.assertEqual(
                    "solidified_non_strict_train15_val15_test180",
                    target_df.attrs["paper_split_protocol"],
                )
                self.assertEqual(30, target_df.attrs["observed_days"])
                self.assertEqual(180, target_df.attrs["test_days"])
                self.assertEqual(15, target_df.attrs["train_days"])
                self.assertEqual(15, target_df.attrs["val_days"])
                self.assertEqual("paper_split_protocol", target_df.attrs["split_mode"])
                self.assertEqual(
                    {"train_days": 15, "val_days": 15, "test_days": 180},
                    target_df.attrs["split_config"],
                )
                self.assertEqual(210, target_df.attrs["target_window_expected_days"])
                self.assertEqual(210, target_df.attrs["target_window_unique_days"])

    def test_paper_split_protocol_maps_to_fixed_day_split(self):
        target_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=210, freq="D"),
                "entity_id": ["target"] * 210,
                "item_id": [1] * 210,
                "sales": range(210),
            }
        )
        target_df.attrs.update(
            {
                "split_role": "target",
                "split_mode": "paper_split_protocol",
                "split_config": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25},
            }
        )

        train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df)

        self.assertEqual([15, 15, 180], [part["date"].nunique() for part in (train_df, val_df, test_df)])

    def test_d4_d5_d6_target_window_attrs_use_paper_split_protocol(self):
        for dataset_id in (4, 5, 6):
            with self.subTest(dataset_id=dataset_id):
                target_df = pd.DataFrame(
                    {
                        "date": pd.date_range("2024-01-01", periods=210, freq="D"),
                        "entity_id": ["target"] * 210,
                        "item_id": [1] * 210,
                        "sales": range(210),
                    }
                )
                windows = {
                    "dataset_id": dataset_id,
                    "train_start": "2024-01-01",
                    "test_end": "2024-07-28",
                    "target_train_window": {"start": "2024-01-01", "end": "2024-01-30"},
                }

                attached = attach_window_attrs(target_df, windows, role="target")

                self.assertEqual("paper_split_protocol", attached.attrs["split_mode"])
                self.assertEqual(15, attached.attrs["train_days"])
                self.assertEqual(15, attached.attrs["val_days"])
                self.assertEqual(30, attached.attrs["observed_days"])
                self.assertEqual(180, attached.attrs["test_days"])
                self.assertEqual("days", attached.attrs["split_config"]["mode"])

    def test_d3_target_metadata_materializes_in_result_dataframe_and_csv(self):
        target_metadata = {
            "target_entity_id": "10",
            "target_store_id": "10",
            "target_item_id": "1",
        }
        success_row = _attach_target_metadata(
            dataset_name="Dataset3",
            row={
                "dataset": "Dataset3",
                "method": "No-TL",
                "information_sharing": "without_information_sharing",
                "experiment_track": "paper",
                "rmse": 1.0,
                "accuracy": 0.9,
                "smape": 2.0,
                "error": "",
            },
            target_metadata=target_metadata,
        )
        paper_df, extended_df = _materialize_result_dataframes([success_row], [])

        self.assertTrue(extended_df.empty)
        self.assertEqual("10", paper_df.loc[0, "target_entity_id"])
        self.assertEqual("10", paper_df.loc[0, "target_store_id"])
        self.assertEqual("1", paper_df.loc[0, "target_item_id"])
        self.assertLess(
            paper_df.columns.get_loc("target_item_id"),
            paper_df.columns.get_loc("method"),
        )

        with self.subTest("dataset CSV writer preserves target metadata"):
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = _save_dataset_result_csvs(paper_df, Path(tmpdir))
                d3_df = pd.read_csv(paths["Dataset3"], dtype=str, keep_default_na=False)

            self.assertEqual("10", d3_df.loc[0, "target_entity_id"])
            self.assertEqual("10", d3_df.loc[0, "target_store_id"])
            self.assertEqual("1", d3_df.loc[0, "target_item_id"])

    def test_d3_feature_cols_are_numeric_non_identifier_and_projected(self):
        cfg = _load_config()
        base = _load_solidified_base_data(
            dataset_name="Dataset3",
            cfg=cfg,
            strict_paper_mode=False,
            strict_paper_split=False,
        )
        source_df = base["source_df"].assign(StoreType="a", Assortment="basic", PromoInterval="Jan")
        target_df = base["target_df"].assign(StoreType="a", Assortment="basic", PromoInterval="Jan")

        feature_cols = _resolve_dataset_feature_cols("Dataset3", source_df, target_df, cfg)

        self.assertEqual("sales", feature_cols[0])
        self.assertNotIn("store_id", feature_cols)
        self.assertNotIn("StoreType", feature_cols)
        self.assertNotIn("Assortment", feature_cols)
        self.assertNotIn("PromoInterval", feature_cols)
        self.assertTrue(feature_cols)
        self.assertTrue(all(source_df[c].dtype.kind in ("i", "u", "f") for c in feature_cols))

        projected_source, projected_target = _project_modeling_frames(
            source_df,
            target_df,
            modeling_feature_cols=feature_cols,
        )

        for bad_col in ("StoreType", "Assortment", "PromoInterval"):
            self.assertNotIn(bad_col, projected_source.columns)
            self.assertNotIn(bad_col, projected_target.columns)
        self.assertTrue(set(feature_cols).issubset(projected_source.columns))
        self.assertTrue(set(feature_cols).issubset(projected_target.columns))

    def test_d2_projection_preserves_protocol_knn_promo_without_modeling_promo(self):
        source_df = pd.DataFrame(
            {
                "date": pd.date_range("2018-06-01", periods=2),
                "entity_id": ["source-1", "source-1"],
                "item_id": [1, 1],
                "brand_id": [1, 1],
                "sales": [10.0, 11.0],
                "promo": [0.0, 1.0],
                "year": [2018, 2018],
                "month": [6, 6],
                "week": [22, 22],
                "day": [1, 2],
            }
        )
        target_df = source_df.copy()
        target_df["entity_id"] = "target"
        source_df.attrs["projection_test"] = "source"
        target_df.attrs["projection_test"] = "target"

        modeling_feature_cols = _resolve_dataset_feature_cols(
            "Dataset2",
            source_df,
            target_df,
            _load_config(),
        )
        self.assertEqual(
            ["sales", "year", "month", "week", "day"],
            modeling_feature_cols,
        )
        knn_required_cols = get_experiment_protocol("Dataset2").knn_feature_columns

        projected_source, projected_target = _project_modeling_frames(
            source_df,
            target_df,
            modeling_feature_cols=modeling_feature_cols,
            required_passthrough_cols=knn_required_cols,
        )

        self.assertNotIn("promo", modeling_feature_cols)
        self.assertIn("promo", projected_source.columns)
        self.assertIn("promo", projected_target.columns)
        self.assertEqual(source_df["promo"].tolist(), projected_source["promo"].tolist())
        self.assertEqual(target_df["promo"].tolist(), projected_target["promo"].tolist())
        self.assertEqual(source_df.attrs, projected_source.attrs)
        self.assertEqual(target_df.attrs, projected_target.attrs)
        self.assertEqual(len(projected_source.columns), len(set(projected_source.columns)))
        self.assertEqual(len(projected_target.columns), len(set(projected_target.columns)))

    def test_d1_without_information_sharing_uses_store_domain_not_entity_overlap(self):
        source_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "entity_id": ["source-a", "source-b", "source-c", "source-d"],
                "item_id": [1, 2, 3, 4],
                "store_id": [1, 1, 2, 2],
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "entity_id": ["target-only", "target-only"],
                "item_id": [10, 10],
                "store_id": [1, 1],
                "sales": [10.0, 11.0],
            }
        )

        filtered = _apply_information_sharing_filter(
            dataset_name="Dataset1",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=False,
            strict_paper_mode=True,
            protocol={},
            cfg={},
        )

        self.assertFalse(filtered.empty)
        self.assertEqual({1}, set(filtered["store_id"].unique()))
        self.assertEqual(
            {"column": "store_id", "value": 1},
            filtered.attrs["domain_filter_used"],
        )
        self.assertEqual(
            "without_information_sharing_same_store",
            filtered.attrs["source_pool_scope_mode"],
        )

    def test_d2_without_information_sharing_uses_brand_domain_not_entity_overlap(self):
        source_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "entity_id": ["source-a", "source-b", "source-c", "source-d"],
                "item_id": [1, 2, 3, 4],
                "brand_id": [1, 1, 2, 2],
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "entity_id": ["target-only", "target-only"],
                "item_id": [10, 10],
                "brand_id": [1, 1],
                "sales": [10.0, 11.0],
            }
        )

        filtered = _apply_information_sharing_filter(
            dataset_name="Dataset2",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=False,
            strict_paper_mode=True,
            protocol={},
            cfg={},
        )

        self.assertFalse(filtered.empty)
        self.assertEqual({1}, set(filtered["brand_id"].unique()))
        self.assertEqual(
            {"column": "brand_id", "value": 1},
            filtered.attrs["domain_filter_used"],
        )
        self.assertEqual(
            "without_information_sharing_same_brand",
            filtered.attrs["source_pool_scope_mode"],
        )

    def test_d3_without_information_sharing_uses_knn_domain_filter(self):
        source_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "entity_id": ["source-a", "source-b", "source-c", "source-d"],
                "item_id": [1, 2, 3, 4],
                "region": [1, 1, 2, 2],
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "entity_id": ["target-only", "target-only"],
                "item_id": [10, 10],
                "region": [1, 1],
                "sales": [10.0, 11.0],
            }
        )

        filtered = _apply_information_sharing_filter(
            dataset_name="Dataset3",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=False,
            strict_paper_mode=False,
            protocol={},
            cfg={},
        )

        self.assertFalse(filtered.empty)
        self.assertEqual({1}, set(filtered["region"].unique()))
        self.assertEqual(
            {"column": "region", "value": 1},
            filtered.attrs["domain_filter_used"],
        )
        self.assertEqual(
            "without_information_sharing_domain_filter",
            filtered.attrs["source_pool_scope_mode"],
        )

    def test_strict_solidified_target_split_attrs_use_positive_days(self):
        cfg = _load_config()
        expected = {
            "Dataset1": {"train_days": 15, "val_days": 15, "test_days": 180},
            "Dataset2": {"train_days": 14, "val_days": 15, "test_days": 179},
            "Dataset3": {"train_days": 16, "val_days": 15, "test_days": 181},
        }

        for dataset_name, split_days in expected.items():
            with self.subTest(dataset_name=dataset_name):
                base = _load_solidified_base_data(
                    dataset_name=dataset_name,
                    cfg=cfg,
                    strict_paper_mode=True,
                    strict_paper_split=True,
                )

                target_df = base["target_df"]
                self.assertEqual("days", target_df.attrs["split_mode"])
                self.assertEqual(split_days, target_df.attrs["split_config"])
                self.assertTrue(all(v > 0 for v in target_df.attrs["split_config"].values()))


if __name__ == "__main__":
    unittest.main()
