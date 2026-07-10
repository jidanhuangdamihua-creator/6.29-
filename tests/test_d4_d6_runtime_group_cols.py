"""Focused D4-D6 runtime source-grouping contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.constants import D4_D6_RUNTIME_KNN_PROTOCOL_VERSION, SOLIDIFIED_TARGET_WINDOWS
from src.source_selection.source_selector import SourceSelector
from src.utils import entity_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNN_ROOT = PROJECT_ROOT / "configs" / "solidified" / "knn"


def _payload(dataset_id: int, scenario: str) -> dict[str, Any]:
    path = KNN_ROOT / f"Dataset{dataset_id}" / f"knn_{scenario}_info_sharing.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_attrs(dataset_id: int, scenario: str, role: str) -> dict[str, Any]:
    observed_start = pd.Timestamp(SOLIDIFIED_TARGET_WINDOWS[dataset_id]["train_start"])
    observed_end = observed_start + pd.Timedelta(days=29)
    return {
        "dataset_name": f"Dataset{dataset_id}",
        "role": role,
        "selection_authority": "runtime",
        "protocol_version": D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
        "target_observed_start": observed_start,
        "target_observed_end": observed_end,
        "source_history_start": observed_end - pd.Timedelta(days=299),
        "source_history_end": observed_end,
        "target_test_excluded": True,
        "source_future_excluded": True,
        "source_alignment_mode": "exact_target_observed_dates",
        "representation": "mean_std_min_max_last",
        "scaling": "none",
        "scaler_fit_scope": "not_applicable",
        "information_sharing_scenario": f"{scenario}_information_sharing",
    }


def _selector_frames(
    dataset_id: int,
    scenario: str,
    group_cols: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(
        pd.Timestamp(SOLIDIFIED_TARGET_WINDOWS[dataset_id]["train_start"]), periods=30, freq="D"
    )
    source_rows: list[dict[str, object]] = []
    source_pairs = (("near-store", "near-item", 1.0), ("far-store", "far-item", 9.0))
    for store, item, sales in source_pairs:
        for date in dates:
            source_rows.append(
                {
                    "date": date,
                    # Deliberately identical legacy keys: only group_cols distinguish the sources.
                    "entity_id": "legacy-entity",
                    "item_id": "legacy-item",
                    group_cols[0]: store,
                    group_cols[1]: item,
                    "sales": sales,
                }
            )
    target_df = pd.DataFrame(
        [
            {
                "date": date,
                "entity_id": "target-entity",
                "item_id": "target-item",
                group_cols[0]: "target-store",
                group_cols[1]: "target-item",
                "sales": 1.0,
            }
            for date in dates
        ]
    )
    source_df = pd.DataFrame(source_rows)
    source_df.attrs.update(_runtime_attrs(dataset_id, scenario, "source"))
    target_df.attrs.update(_runtime_attrs(dataset_id, scenario, "target"))
    return source_df, target_df


@pytest.mark.parametrize("dataset_id", (4, 5, 6))
@pytest.mark.parametrize("scenario", ("without", "with"))
def test_runtime_selector_records_payload_group_cols_and_group_keys(
    dataset_id: int,
    scenario: str,
) -> None:
    payload = _payload(dataset_id, scenario)
    group_cols = tuple(payload["group_cols"])
    source_df, target_df = _selector_frames(dataset_id, scenario, group_cols)

    result = SourceSelector().select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=["sales"],
        k=2,
        group_cols=group_cols,
    )

    assert result["meta"]["group_cols"] == list(group_cols)
    assert result["meta"]["selected_sources_runtime"] == result["sources"]
    assert {tuple(row["source_key"]) for row in result["sources"]} == {
        ("near-store", "near-item"),
        ("far-store", "far-item"),
    }


@pytest.mark.parametrize("dataset_id", (4, 5, 6))
@pytest.mark.parametrize("scenario", ("without", "with"))
@pytest.mark.parametrize("method", ("SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"))
def test_entity_runtime_forwards_payload_group_cols_to_transfer_runner(
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: int,
    scenario: str,
    method: str,
) -> None:
    payload = _payload(dataset_id, scenario)
    group_cols = tuple(payload["group_cols"])
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    source_df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "entity_id": ["source-a"] * 3 + ["source-b"] * 3,
            "item_id": ["legacy-item"] * 6,
            group_cols[0]: ["store-a"] * 3 + ["store-b"] * 3,
            group_cols[1]: ["item-a"] * 3 + ["item-b"] * 3,
            "sales": [1.0] * 6,
        }
    )
    target_df = pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["target"] * 3,
            "item_id": ["target-item"] * 3,
            group_cols[0]: ["target-store"] * 3,
            group_cols[1]: ["target-item"] * 3,
            "sales": [1.0] * 3,
        }
    )

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        assert tuple(kwargs["group_cols"]) == group_cols
        assert set(group_cols).issubset(kwargs["source_df"].columns)
        assert set(group_cols).issubset(kwargs["target_df"].columns)
        return {"rmse": 0.0, "accuracy": 1.0, "mae": 0.0, "mape": 0.0, "smape": 0.0, "meta": {}}

    monkeypatch.setattr(entity_experiment, "_method_runner", lambda method: fake_runner)
    rows = entity_experiment.run_single_entity_experiment(
        entity_key="target",
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=["sales"],
        config={
            "dataset_id": dataset_id,
            "info_sharing": scenario,
            "horizon": 1,
            "window_size": 1,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
            "source_selection_group_cols": list(group_cols),
        },
        enabled_methods=[method],
    )

    assert len(rows) == 1
