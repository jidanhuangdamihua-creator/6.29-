from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.regenerate_solidified_knn import (
    _build_regenerated_payload,
    _prepare_d4_runtime_source_pool,
    _select_d4_shared_protocol,
)
from src.protocols.candidate_pool import InsufficientCandidatePoolError
from src.protocols.experiment_protocol import PROTOCOL_VERSION, ProtocolViolation
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector
from src.utils.d4_d6_runtime import (
    apply_runtime_source_domain_policy,
    validate_runtime_target_domain,
)


OBSERVED_DATES = pd.date_range("2024-01-01", periods=30, freq="D")
SOURCE_DATES = pd.date_range(OBSERVED_DATES[0] - pd.Timedelta(days=150), periods=180, freq="D")
TARGET_DATES = pd.date_range("2024-01-01", periods=31, freq="D")


def _rows(
    store_id: int,
    product_id: int,
    second_category_id: int,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": f"{store_id}_{product_id}",
            "store_id": store_id,
            "product_id": product_id,
            "first_category_id": 15,
            "second_category_id": second_category_id,
            "date": dates,
            "sales": np.asarray(
                [(date - OBSERVED_DATES[0]).days for date in dates], dtype=float
            ) + float(product_id),
        }
    )


def _configured_d4_without_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.concat(
        [
            _rows(166, 259, 30, SOURCE_DATES),
            _rows(166, 260, 20, SOURCE_DATES),
            _rows(166, 261, 40, SOURCE_DATES),
            _rows(166, 262, 50, SOURCE_DATES.delete(-1)),
            _rows(167, 258, 60, SOURCE_DATES),
            _rows(167, 263, 70, SOURCE_DATES),
            _rows(168, 264, 80, SOURCE_DATES),
        ],
        ignore_index=True,
    )
    target = _rows(166, 258, 20, TARGET_DATES)
    return configure_protocol_frames(
        source,
        target,
        dataset_id="D4",
        scenario="without",
        group_cols=("store_id", "product_id"),
        observed_start="2024-01-01",
    )


def _d4_payload(scenario: str) -> dict[str, object]:
    return {
        "dataset_id": 4,
        "dataset": "D4",
        "info_sharing": scenario,
        "k": 3,
        "group_cols": ["store_id", "product_id"],
        "domain_filter": {"column": "second_category_id", "value": 20},
        "results": {"166_258": []},
    }


def _raw_d4_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source, _ = _configured_d4_without_frames()
    target = _rows(166, 258, 20, TARGET_DATES)
    return source, target


def _runtime_select(scenario: str, source: pd.DataFrame, target: pd.DataFrame) -> dict[str, object]:
    payload = _d4_payload(scenario)
    runtime_config = {"dataset_id": 4, "info_sharing": scenario, "entity_col": "entity_id"}
    runtime_source = apply_runtime_source_domain_policy(source, payload, runtime_config)
    validate_runtime_target_domain(target, ["166_258"], payload, runtime_config)
    configured_source, configured_target = configure_protocol_frames(
        runtime_source,
        target,
        dataset_id="D4",
        scenario=scenario,
        group_cols=("store_id", "product_id"),
        observed_start="2024-01-01",
    )
    return SourceSelector().select_top_k_sources(
        target_df=configured_target,
        source_df=configured_source,
        feature_cols=("sales",),
        k=3,
        group_cols=("store_id", "product_id"),
    )


def _regeneration_select(scenario: str, source: pd.DataFrame, target: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    policy = _prepare_d4_runtime_source_pool(
        source_df=source,
        target_df=target,
        target_entity_keys=["166_258"],
        scenario=scenario,
        old_payload=_d4_payload(scenario),
    )
    selection = _select_d4_shared_protocol(
        source_df=policy.frame,
        target_entity_df=target,
        scenario=scenario,
        feature_cols=("sales",),
        k=3,
        group_cols=("store_id", "product_id"),
    )
    return selection, policy.diagnostics


def _key_set(keys: object) -> set[tuple[str, str]]:
    return {tuple(str(part) for part in key) for key in keys}


def _source_keys(selection: dict[str, object]) -> list[tuple[str, str]]:
    return [
        tuple(str(part) for part in row["source_key"])
        for row in selection["sources"]
    ]


class Dataset4RegenerateRuntimeProtocolParityTest(unittest.TestCase):
    def test_d4_payload_accepts_shared_protocol_metadata(self) -> None:
        payload = _build_regenerated_payload(
            old_payload=_d4_payload("without"),
            feature_cols=("sales",),
            feature_info={"selected_features": ["sales"]},
            source_pool_size=210,
            source_domain_policy_diagnostics={
                "domain_filter_scope": "target_only",
                "domain_filter_applied_to_source": False,
                "source_pool_policy": "without_information_sharing_same_store",
                "source_pool_entity_count": 7,
            },
            results={"166_258": []},
            selection_metadata={
                "166_258": {
                    "selection_authority": "shared_protocol",
                    "selection_path": "shared_protocol",
                    "protocol_version": PROTOCOL_VERSION,
                    "eligible_candidate_count": 4,
                    "valid_30d_candidate_count": 3,
                    "selected_count": 3,
                    "observed_days": 30,
                    "require_same_group": False,
                    "excluded_candidate_key_fields": ["product_id"],
                }
            },
        )

        self.assertEqual(payload["domain_filter"], _d4_payload("without")["domain_filter"])
        self.assertEqual(payload["selection_metadata"]["166_258"]["selection_path"], "shared_protocol")
        self.assertEqual(payload["selection_metadata"]["166_258"]["valid_30d_candidate_count"], 3)
        self.assertFalse(payload["selection_metadata"]["166_258"]["require_same_group"])

    def test_shared_selection_reports_explicit_protocol_path_and_candidate_layers(self) -> None:
        source, target = _configured_d4_without_frames()

        result = SourceSelector().select_top_k_sources(
            target_df=target,
            source_df=source,
            feature_cols=("sales",),
            k=3,
            group_cols=("store_id", "product_id"),
        )

        self.assertEqual(result["meta"]["selection_path"], "shared_protocol")
        self.assertEqual(result["meta"]["eligible_candidate_count"], 4)
        self.assertEqual(result["meta"]["valid_30d_candidate_count"], 3)
        self.assertEqual(result["meta"]["selected_count"], 3)
        self.assertEqual(result["meta"]["observed_days"], 30)
        self.assertEqual(source.attrs["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(target.attrs["protocol_dataset_id"], "D4")
        self.assertEqual(target.attrs["protocol_scenario"], "without")
        self.assertEqual(target.attrs["protocol_observed_start"], "2024-01-01")
        self.assertEqual(target.attrs["protocol_observed_days"], 30)
        self.assertEqual(
            _key_set(target.attrs["protocol_candidate_keys"]),
            {("166", "259"), ("166", "260"), ("166", "261"), ("166", "262")},
        )

    def test_d4_without_runtime_and_regeneration_candidate_parity(self) -> None:
        source, target = _raw_d4_frames()
        runtime = _runtime_select("without", source, target)
        regenerated, diagnostics = _regeneration_select("without", source, target)

        self.assertEqual(runtime["meta"]["selection_path"], "shared_protocol")
        self.assertEqual(regenerated["meta"]["selection_path"], "shared_protocol")
        self.assertEqual(_key_set(runtime["meta"]["eligible_candidate_keys"]), _key_set(regenerated["meta"]["eligible_candidate_keys"]))
        self.assertEqual(_key_set(runtime["meta"]["valid_30d_candidate_keys"]), _key_set(regenerated["meta"]["valid_30d_candidate_keys"]))
        self.assertEqual(_source_keys(runtime), _source_keys(regenerated))
        np.testing.assert_allclose(
            [row["distance"] for row in runtime["sources"]],
            [row["distance"] for row in regenerated["sources"]],
        )
        np.testing.assert_allclose(
            [row["weight"] for row in runtime["sources"]],
            [row["weight"] for row in regenerated["sources"]],
        )
        self.assertEqual(diagnostics["domain_filter_scope"], "target_only")
        self.assertFalse(diagnostics["domain_filter_applied_to_source"])
        self.assertEqual(_key_set(regenerated["meta"]["eligible_candidate_keys"]), {("166", "259"), ("166", "260"), ("166", "261"), ("166", "262")})
        self.assertEqual(_key_set(regenerated["meta"]["valid_30d_candidate_keys"]), {("166", "259"), ("166", "260"), ("166", "261")})

    def test_d4_with_runtime_and_regeneration_candidate_parity(self) -> None:
        source, target = _raw_d4_frames()
        runtime = _runtime_select("with", source, target)
        regenerated, _ = _regeneration_select("with", source, target)

        self.assertEqual(_key_set(runtime["meta"]["eligible_candidate_keys"]), _key_set(regenerated["meta"]["eligible_candidate_keys"]))
        self.assertIn(("167", "263"), _key_set(regenerated["meta"]["eligible_candidate_keys"]))
        self.assertIn(("168", "264"), _key_set(regenerated["meta"]["eligible_candidate_keys"]))
        self.assertNotIn(("167", "258"), _key_set(regenerated["meta"]["eligible_candidate_keys"]))
        self.assertEqual(_source_keys(runtime), _source_keys(regenerated))
        self.assertEqual(
            runtime["meta"]["source_skip_diagnostics"],
            regenerated["meta"]["source_skip_diagnostics"],
        )
        np.testing.assert_allclose(
            [row["distance"] for row in runtime["sources"]],
            [row["distance"] for row in regenerated["sources"]],
        )
        np.testing.assert_allclose(
            [row["weight"] for row in runtime["sources"]],
            [row["weight"] for row in regenerated["sources"]],
        )

    def test_d4_domain_filter_is_target_only_and_invalid_target_fails_validation(self) -> None:
        source, target = _raw_d4_frames()
        before_entities = source[["store_id", "product_id"]].drop_duplicates().shape[0]
        policy = _prepare_d4_runtime_source_pool(
            source_df=source,
            target_df=target,
            target_entity_keys=["166_258"],
            scenario="without",
            old_payload=_d4_payload("without"),
        )

        self.assertFalse(policy.diagnostics["domain_filter_applied_to_source"])
        self.assertEqual(
            policy.frame[["store_id", "product_id"]].drop_duplicates().shape[0],
            before_entities,
        )
        with self.assertRaisesRegex(ProtocolViolation, "target domain validation failed"):
            _prepare_d4_runtime_source_pool(
                source_df=source,
                target_df=target.assign(second_category_id=99),
                target_entity_keys=["166_258"],
                scenario="without",
                old_payload=_d4_payload("without"),
            )

    def test_d4_k3_failure_is_identical_for_runtime_and_regeneration(self) -> None:
        source, target = _raw_d4_frames()
        two_valid = source[
            source["product_id"].isin([259, 260, 258])
        ].copy()
        with self.assertRaises(InsufficientCandidatePoolError) as runtime_error:
            _runtime_select("without", two_valid, target)
        with self.assertRaises(InsufficientCandidatePoolError) as regeneration_error:
            _regeneration_select("without", two_valid, target)

        self.assertEqual(type(runtime_error.exception), type(regeneration_error.exception))
        self.assertEqual(runtime_error.exception.required_k, regeneration_error.exception.required_k)
        self.assertEqual(runtime_error.exception.valid_count, regeneration_error.exception.valid_count)
        self.assertEqual(runtime_error.exception.eligible_count, regeneration_error.exception.eligible_count)
        self.assertEqual(runtime_error.exception.target_key, regeneration_error.exception.target_key)
        self.assertEqual(runtime_error.exception.scenario, regeneration_error.exception.scenario)
        self.assertEqual(runtime_error.exception.exclusions, regeneration_error.exception.exclusions)
        self.assertEqual(
            (
                runtime_error.exception.required_k,
                runtime_error.exception.eligible_count,
                runtime_error.exception.valid_count,
                runtime_error.exception.target_key,
                runtime_error.exception.scenario,
            ),
            (3, 2, 2, ("166", "258"), "without"),
        )

    def test_d5_and_d6_same_group_protocol_still_excludes_other_groups(self) -> None:
        for dataset_id, grouping_col in (("D5", "family"), ("D6", "dept_id")):
            source = pd.concat(
                [
                    _rows(166, 259, 20, OBSERVED_DATES).assign(**{grouping_col: "A"}),
                    _rows(167, 260, 20, OBSERVED_DATES).assign(**{grouping_col: "A"}),
                    _rows(168, 261, 20, OBSERVED_DATES).assign(**{grouping_col: "B"}),
                ],
                ignore_index=True,
            )
            target = _rows(166, 258, 20, TARGET_DATES).assign(**{grouping_col: "A"})
            _, configured_target = configure_protocol_frames(
                source,
                target,
                dataset_id=dataset_id,
                scenario="with",
                group_cols=("store_id", "product_id"),
                grouping_col=grouping_col,
                observed_start="2024-01-01",
            )
            self.assertEqual(
                configured_target.attrs["protocol_candidate_keys"],
                (("166", "259"), ("167", "260")),
            )
