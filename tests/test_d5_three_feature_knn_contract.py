from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector
from src.utils.knn_feature_loader import resolve_knn_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
D5_KNN_FEATURES = ["sales", "onpromotion", "oil_price"]
D5_MODEL_FEATURES = [
    "sales",
    "year",
    "month",
    "week",
    "day",
    "class",
    "perishable",
    "cluster",
    "transactions",
    "oil_price",
    "is_holiday",
]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    source_parts = []
    for index, (store, item, sales) in enumerate(
        (("48", "S1", 1.0), ("49", "S2", 2.0), ("50", "S3", 3.0))
    ):
        source_parts.append(
            pd.DataFrame(
                {
                    "store_nbr": store,
                    "item_nbr": item,
                    "family": "F1",
                    "date": dates,
                    "sales": np.full(len(dates), sales, dtype=float),
                    "onpromotion": np.full(len(dates), float(index), dtype=float),
                    "oil_price": np.linspace(10.0 + index, 20.0 + index, len(dates)),
                }
            )
        )
    source = pd.concat(source_parts, ignore_index=True)
    # The configured KNN pool is intentionally only the three protocol
    # fields, while CNN provenance still needs the pre-existing model fields.
    for column in D5_MODEL_FEATURES:
        if column not in source:
            source[column] = 0.0
    target = pd.DataFrame(
        {
            "store_nbr": "48",
            "item_nbr": "T1",
            "family": "F1",
            "date": dates,
            "sales": np.zeros(len(dates), dtype=float),
            "onpromotion": np.zeros(len(dates), dtype=float),
            "oil_price": np.linspace(10.0, 20.0, len(dates)),
        }
    )
    for column in D5_MODEL_FEATURES:
        if column not in target:
            target[column] = 0.0
    return source, target


def _select(
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=("store_nbr", "item_nbr"),
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=tuple(D5_KNN_FEATURES),
    )
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D5",
        scenario="with",
        group_cols=("store_nbr", "item_nbr"),
        grouping_col="family",
        observed_start="2020-01-01",
        prepared_pool=pool,
    )
    result = SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=tuple(D5_MODEL_FEATURES),
        k=2,
        group_cols=("store_nbr", "item_nbr"),
        consumer_source_df=source,
    )
    return result, configured_source, configured_target


def test_d5_protocol_order_is_immutable_and_other_protocols_do_not_drift() -> None:
    assert get_experiment_protocol(5).knn_feature_columns == tuple(D5_KNN_FEATURES)
    assert get_experiment_protocol(1).knn_feature_columns == ("sales",)
    assert get_experiment_protocol(2).knn_feature_columns == ("sales", "promo")
    assert get_experiment_protocol(3).knn_feature_columns == ("sales",)
    assert get_experiment_protocol(4).knn_feature_columns == ("sales",)
    assert get_experiment_protocol(6).knn_feature_columns == ("sales",)


def test_d5_three_fields_reach_frames_pool_and_distance() -> None:
    source, target = _frames()
    result, configured_source, configured_target = _select(source, target)
    expected = D5_KNN_FEATURES

    assert configured_source.attrs["knn_feature_columns"] == expected
    assert configured_target.attrs["knn_feature_columns"] == expected
    pool = configured_source.attrs["prepared_daily_sequence_pool"]
    assert list(pool.feature_matrices.keys()) == expected
    assert result["meta"]["knn_feature_columns"] == expected
    assert result["meta"]["feature_cols"] == expected
    assert result["meta"]["target_signature_dim"] == 90
    proof = result["meta"]["runtime_knn_proof"]
    assert proof["target_exact_vector_shape"] == [90]
    assert proof["source_matrix_shapes"] == {
        "sales": [3, 30],
        "onpromotion": [3, 30],
        "oil_price": [3, 30],
    }
    assert proof["distance_feature_columns"] == expected
    assert proof["nan_count"] == 0
    assert proof["inf_count"] == 0


@pytest.mark.parametrize("feature", ["onpromotion", "oil_price"])
def test_d5_missing_knn_feature_fails_closed(feature: str) -> None:
    source, target = _frames()
    source = source.drop(columns=[feature])
    target = target.drop(columns=[feature])
    with pytest.raises(ProtocolViolation, match=feature):
        configure_protocol_frames(
            source,
            target,
            dataset_id="D5",
            scenario="with",
            group_cols=("store_nbr", "item_nbr"),
            grouping_col="family",
            observed_start="2020-01-01",
        )


def test_d5_historical_feature_change_changes_distance_digest() -> None:
    source, target = _frames()
    baseline, _, _ = _select(source, target)
    changed = source.copy()
    changed.loc[
        (changed["store_nbr"] == "49") & (changed["item_nbr"] == "S2")
        & (changed["date"] <= pd.Timestamp("2020-01-30")),
        "onpromotion",
    ] += 100.0
    altered, _, _ = _select(changed, target)
    assert baseline["meta"]["selection_result_digest"] != altered["meta"]["selection_result_digest"]
    assert baseline["meta"]["selected_sources_runtime"] != altered["meta"]["selected_sources_runtime"]


def test_d5_future_feature_change_does_not_change_knn_selection_authority() -> None:
    source, target = _frames()
    baseline, _, _ = _select(source, target)
    changed = source.copy()
    future = changed["date"] > pd.Timestamp("2020-01-30")
    changed.loc[future, "onpromotion"] += 1000.0
    changed.loc[future, "oil_price"] += 1000.0
    altered, _, _ = _select(changed, target)
    assert baseline["meta"]["selection_result_digest"] == altered["meta"]["selection_result_digest"]
    assert baseline["meta"]["selected_sources_runtime"] == altered["meta"]["selected_sources_runtime"]


def test_d5_sales_only_authority_is_rejected_by_feature_resolver() -> None:
    path = (
        PROJECT_ROOT
        / "configs"
        / "solidified"
        / "knn"
        / "Dataset5"
        / "knn_with_info_sharing.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["knn_feature_columns"] = ["sales"]
    source = pd.DataFrame({column: [1.0] for column in D5_MODEL_FEATURES})
    source["onpromotion"] = 0.0
    target = source.copy()
    with pytest.raises(ValueError, match="expected features=.*sales.*onpromotion.*oil_price"):
        resolve_knn_feature_columns(
            dataset_id=5,
            information_sharing="with",
            knn_root=PROJECT_ROOT / "configs" / "solidified" / "knn",
            source_df=source,
            target_df=target,
            knn_payload=payload,
        )
