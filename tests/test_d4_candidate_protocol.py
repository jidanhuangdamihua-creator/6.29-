from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.constants import RESULT_SCHEMA_COLUMNS
from src.protocols import experiment_protocol
from src.protocols.candidate_pool import (
    build_candidate_pool_digest,
    prepare_daily_sequence_pool,
)
from src.protocols.experiment_protocol import (
    ProtocolViolation,
    SourceIdentity,
    build_candidate_keys,
    get_experiment_protocol,
)
from src.protocols.runner_adapter import configure_protocol_frames
from src.utils.d4_d6_runtime import (
    apply_runtime_source_domain_policy,
    validate_runtime_target_domain,
)
from src.utils import entity_experiment
from src.utils.dataframe_attrs import get_protocol_frame_context
from src.utils.result_schema import (
    REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY,
    align_d4_d6_result_records,
)


DATES = pd.date_range("2020-01-01", periods=35, freq="D")
OBSERVED_DATES = DATES[:30]
TARGET_KEY = ("166", "258")
SAME_STORE_DIFFERENT_CATEGORY_KEY = ("166", "259")
SAME_STORE_SAME_CATEGORY_KEY = ("166", "261")
CROSS_STORE_DIFFERENT_CATEGORY_KEY = ("167", "260")
CROSS_STORE_SAME_PRODUCT_KEY = ("168", "258")


def _rows(
    store_id: int,
    product_id: int,
    second_category_id: int,
    dates: pd.DatetimeIndex,
    first_category_id: int = 15,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": store_id,
            "product_id": product_id,
            "first_category_id": first_category_id,
            "second_category_id": second_category_id,
            "date": dates,
            "sales": np.full(len(dates), float(product_id)),
            "onpromotion": np.zeros(len(dates), dtype=float),
            "oil_price": np.full(len(dates), 50.0, dtype=float),
        }
    )


def _d4_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    target = _rows(166, 258, 20, DATES)
    source = pd.concat(
        [
            _rows(166, 258, 20, OBSERVED_DATES),
            _rows(166, 259, 30, OBSERVED_DATES, first_category_id=99),
            _rows(166, 261, 20, OBSERVED_DATES),
            _rows(167, 260, 40, OBSERVED_DATES, first_category_id=99),
            _rows(168, 258, 50, OBSERVED_DATES, first_category_id=99),
        ],
        ignore_index=True,
    )
    return source, target


def _candidate_keys(
    *, scenario: str, prepared: bool = False
) -> tuple[tuple[str, str], ...]:
    source, target = _d4_frames()
    pool = (
        prepare_daily_sequence_pool(
            source,
            group_cols=("store_id", "product_id"),
            observed_start="2020-01-01",
            metadata_cols=("second_category_id",),
        )
        if prepared
        else None
    )
    configured_source, _ = configure_protocol_frames(
        source,
        target,
        dataset_id="D4",
        scenario=scenario,
        group_cols=("store_id", "product_id"),
        grouping_col="second_category_id",
        observed_start="2020-01-01",
        prepared_pool=pool,
    )
    return get_protocol_frame_context(configured_source).candidate_keys


class Dataset4CandidateProtocolTest(unittest.TestCase):
    def test_formal_d4_row_preserves_tracks_and_uses_only_registered_columns(self) -> None:
        source, target = _d4_frames()
        source = source.assign(
            entity_id=source["store_id"].astype(str)
            + "_"
            + source["product_id"].astype(str)
        )
        target = target.assign(entity_id="166_258")
        config = {
            "dataset_id": 4,
            "dataset_name": "Dataset4",
            "info_sharing": "without",
            "entity_col": "entity_id",
            "group_cols": ("store_id", "product_id"),
            "source_count": 3,
            "horizon": 1,
            "window_size": 1,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
            "smoke": False,
            "metric_protocol": {
                "current_metric_space": "normalized_minmax_space",
                "paper_metric_space": "original_sales_space",
                "strict_paper_metrics": False,
            },
        }
        knn_data = {
            "domain_filter": {"column": "first_category_id", "value": 15},
            "group_cols": ["store_id", "product_id"],
        }
        source = apply_runtime_source_domain_policy(source, knn_data, config)
        validate_runtime_target_domain(target, ["166_258"], knn_data, config)

        def fake_no_tl(**_kwargs):
            return {
                "rmse": 1.0,
                "accuracy": 0.5,
                "mae": 0.5,
                "mape": 1.0,
                "smape": 2.0,
                "rmse_metric_space": "original_sales_space",
                "smape_metric_space": "original_sales_space",
                "paper_metric_computed_valid": True,
                "paper_metric_status": "valid",
                "paper_metric_error": "",
                "paper_metric_aligned": True,
                "inverse_transform_applied": True,
                "inverse_transform_available": True,
                "error": "",
            }

        with patch.object(
            entity_experiment,
            "_method_runner",
            return_value=fake_no_tl,
        ):
            rows = entity_experiment.run_single_entity_experiment(
                entity_key="166_258",
                source_df=source,
                target_entity_df=target,
                feature_cols=["sales"],
                config=config,
                enabled_methods=["No-TL"],
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["protocol_track"], experiment_protocol.FORMAL_PROTOCOL_TRACK)
        self.assertEqual(rows[0]["source_pool_track"], experiment_protocol.EXTENDED_TRACK)

        aligned = align_d4_d6_result_records(rows)
        allowed = set(RESULT_SCHEMA_COLUMNS)
        allowed.update(
            REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY[
                aligned.loc[0, "schema_family"]
            ]
        )
        self.assertEqual(set(aligned.columns).difference(allowed), set())

    def test_d4_without_domain_filter_keeps_all_source_categories(self) -> None:
        source, _ = _d4_frames()
        config = {"dataset_id": 4, "info_sharing": "without"}

        after = apply_runtime_source_domain_policy(
            source,
            {
                "domain_filter": {"column": "first_category_id", "value": 15},
                "group_cols": ["store_id", "product_id"],
            },
            config,
        )

        self.assertEqual(len(after), len(source))
        self.assertEqual(
            config["source_pool_entities_after_filter"],
            config["source_pool_entities_before_filter"],
        )
        self.assertIn(99, after["first_category_id"].unique())
        self.assertFalse(config["source_domain_filter_applied"])
        self.assertFalse(config["domain_filter_applied_to_source"])
        self.assertEqual(config["domain_filter_scope"], "target_only")
        self.assertEqual(config["domain_filter_column"], "first_category_id")
        self.assertEqual(config["domain_filter_value"], 15)
        self.assertEqual(config["source_domain_filter_reason"], "domain_filter_target_only")
        self.assertEqual(
            config["source_pool_policy"], "without_information_sharing_same_store"
        )

    def test_d4_with_domain_filter_keeps_all_source_categories(self) -> None:
        source, _ = _d4_frames()
        config = {"dataset_id": 4, "info_sharing": "with"}

        after = apply_runtime_source_domain_policy(
            source,
            {
                "domain_filter": {"column": "first_category_id", "value": 15},
                "group_cols": ["store_id", "product_id"],
            },
            config,
        )

        self.assertEqual(len(after), len(source))
        self.assertEqual(
            config["source_pool_entities_after_filter"],
            config["source_pool_entities_before_filter"],
        )
        self.assertIn(99, after["first_category_id"].unique())
        self.assertFalse(config["source_domain_filter_applied"])
        self.assertFalse(config["domain_filter_applied_to_source"])
        self.assertEqual(config["domain_filter_scope"], "target_only")
        self.assertEqual(config["source_domain_filter_reason"], "domain_filter_target_only")
        self.assertEqual(
            config["source_pool_policy"], "with_information_sharing_cross_store"
        )

    def test_d4_target_domain_validation_rejects_nonmatching_json_target(self) -> None:
        _, target = _d4_frames()
        config = {"dataset_id": 4, "entity_col": "entity_id"}
        target = target.assign(entity_id="166_258")
        knn_data = {"domain_filter": {"column": "first_category_id", "value": 15}}

        validate_runtime_target_domain(target, ["166_258"], knn_data, config)
        self.assertTrue(config["target_domain_validation_passed"])

        invalid = target.assign(first_category_id=99)
        with self.assertRaisesRegex(ProtocolViolation, "target domain validation failed"):
            validate_runtime_target_domain(invalid, ["166_258"], knn_data, config)

    def test_d5_source_domain_filter_remains_enabled(self) -> None:
        source, _ = _d4_frames()
        config = {"dataset_id": 5, "info_sharing": "without"}

        after = apply_runtime_source_domain_policy(
            source,
            {
                "domain_filter": {"column": "first_category_id", "value": 15},
                "group_cols": ["store_id", "product_id"],
            },
            config,
        )

        self.assertLess(len(after), len(source))
        self.assertTrue(config["domain_filter_applied_to_source"])

    def test_shared_d4_protocol_excludes_only_exact_composite_target_key(
        self,
    ) -> None:
        identities = (
            SourceIdentity(TARGET_KEY, "20"),
            SourceIdentity(SAME_STORE_DIFFERENT_CATEGORY_KEY, "30"),
            SourceIdentity(SAME_STORE_SAME_CATEGORY_KEY, "20"),
            SourceIdentity(CROSS_STORE_DIFFERENT_CATEGORY_KEY, "40"),
            SourceIdentity(CROSS_STORE_SAME_PRODUCT_KEY, "50"),
        )
        protocol = get_experiment_protocol("D4")

        without_candidate_keys = build_candidate_keys(
            protocol, "without", TARGET_KEY, identities
        )
        with_candidate_keys = build_candidate_keys(protocol, "with", TARGET_KEY, identities)

        self.assertEqual(
            without_candidate_keys,
            (SAME_STORE_DIFFERENT_CATEGORY_KEY, SAME_STORE_SAME_CATEGORY_KEY),
        )
        self.assertEqual(
            with_candidate_keys,
            (
                SAME_STORE_DIFFERENT_CATEGORY_KEY,
                SAME_STORE_SAME_CATEGORY_KEY,
                CROSS_STORE_DIFFERENT_CATEGORY_KEY,
                CROSS_STORE_SAME_PRODUCT_KEY,
            ),
        )
        self.assertEqual(
            protocol.source_pool_rule.excluded_candidate_key_fields,
            (),
        )

    def test_d5_still_excludes_different_group_candidates(self) -> None:
        target = _rows(166, 258, 20, DATES).assign(family="F1")
        source = pd.concat(
            [
                _rows(166, 259, 30, OBSERVED_DATES).assign(family="F1"),
                _rows(167, 260, 40, OBSERVED_DATES).assign(family="F1"),
                _rows(166, 261, 20, OBSERVED_DATES).assign(family="F2"),
            ],
            ignore_index=True,
        )

        configured_source, _ = configure_protocol_frames(
            source,
            target,
            dataset_id="D5",
            scenario="with",
            group_cols=("store_id", "product_id"),
            grouping_col="family",
            observed_start="2020-01-01",
        )

        self.assertEqual(
            get_protocol_frame_context(configured_source).candidate_keys,
            (("166", "259"), ("167", "260")),
        )

    def test_without_allows_different_second_category_but_not_cross_store(
        self,
    ) -> None:
        candidate_keys = _candidate_keys(scenario="without")

        self.assertIn(SAME_STORE_DIFFERENT_CATEGORY_KEY, candidate_keys)
        self.assertNotIn(TARGET_KEY, candidate_keys)
        self.assertNotIn(CROSS_STORE_DIFFERENT_CATEGORY_KEY, candidate_keys)
        self.assertNotIn(CROSS_STORE_SAME_PRODUCT_KEY, candidate_keys)
        source, target = _d4_frames()
        candidate = source[(source["store_id"] == 166) & (source["product_id"] == 259)]
        self.assertNotEqual(
            candidate["first_category_id"].iloc[0], target["first_category_id"].iloc[0]
        )
        self.assertNotEqual(
            candidate["second_category_id"].iloc[0], target["second_category_id"].iloc[0]
        )

    def test_with_excludes_only_exact_key_and_retains_cross_store_same_product(
        self,
    ) -> None:
        candidate_keys = _candidate_keys(scenario="with")

        self.assertIn(CROSS_STORE_DIFFERENT_CATEGORY_KEY, candidate_keys)
        self.assertNotIn(TARGET_KEY, candidate_keys)
        self.assertIn(CROSS_STORE_SAME_PRODUCT_KEY, candidate_keys)
        source, target = _d4_frames()
        candidate = source[(source["store_id"] == 167) & (source["product_id"] == 260)]
        self.assertNotEqual(candidate["store_id"].iloc[0], target["store_id"].iloc[0])
        self.assertNotEqual(
            candidate["first_category_id"].iloc[0], target["first_category_id"].iloc[0]
        )
        self.assertNotEqual(
            candidate["second_category_id"].iloc[0], target["second_category_id"].iloc[0]
        )

    def test_prepared_pool_does_not_restore_second_category_restriction(
        self,
    ) -> None:
        without_candidate_keys = _candidate_keys(scenario="without", prepared=True)
        with_candidate_keys = _candidate_keys(scenario="with", prepared=True)

        self.assertIn(SAME_STORE_DIFFERENT_CATEGORY_KEY, without_candidate_keys)
        self.assertIn(CROSS_STORE_DIFFERENT_CATEGORY_KEY, with_candidate_keys)
        self.assertIn(CROSS_STORE_SAME_PRODUCT_KEY, with_candidate_keys)

    def test_candidate_digest_binds_store_and_product_components(self) -> None:
        payload = {
            "protocol_version": "d1_d6_protocol_v1",
            "dataset_id": "D4",
            "scenario": "with",
            "target_key": TARGET_KEY,
            "group_cols": ("store_id", "product_id"),
            "candidate_keys": (CROSS_STORE_SAME_PRODUCT_KEY,),
            "observed_start": "2020-01-01",
            "observed_end": "2020-01-30",
            "feature_cols": ("sales",),
        }
        baseline = build_candidate_pool_digest(**payload)
        changed = build_candidate_pool_digest(
            **{**payload, "candidate_keys": (("169", "258"),)}
        )
        self.assertNotEqual(baseline, changed)
