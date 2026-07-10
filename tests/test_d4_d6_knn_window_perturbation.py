"""Focused perturbation tests for D4–D6 runtime KNN window leakage.

Verifies that Top-K source selection is invariant to values in the target test
window and in source rows after the observed cutoff, while remaining sensitive
to values inside the legal observed/history window (positive control).

KNN payloads are written only under tmp_path. Formal configs/solidified/knn
JSON files are never modified.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.regenerate_solidified_knn import (
    _build_regenerated_payload,
    _write_json,
    snapshot_knn_config_files,
    verify_knn_config_unchanged,
)
from src.constants import (
    D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
    SOLIDIFIED_KNN_ROOT,
    SOLIDIFIED_TARGET_WINDOWS,
    SOURCE_HISTORY_DAYS,
)
from src.source_selection.source_selector import SourceSelector

DATASET_IDS = (4, 5, 6)
INFO_SHARING_MODES = ("without", "with")

# Match solidified KNN group_cols (read-only contract; never written back).
GROUP_COLS_BY_DATASET: dict[int, tuple[str, str]] = {
    4: ("store_id", "product_id"),
    5: ("store_nbr", "item_nbr"),
    6: ("store_id", "item_id"),
}

FEATURE_COLS = ("sales", "promo_flag")
K = 2
EXTREME_VALUE = 1_000_000.0


def _train_start(dataset_id: int) -> pd.Timestamp:
    return pd.Timestamp(SOLIDIFIED_TARGET_WINDOWS[dataset_id]["train_start"]).normalize()


def _observed_bounds(dataset_id: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _train_start(dataset_id)
    end = start + pd.Timedelta(days=29)
    return start, end


def _history_bounds(dataset_id: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, observed_end = _observed_bounds(dataset_id)
    history_end = observed_end
    history_start = history_end - pd.Timedelta(days=SOURCE_HISTORY_DAYS - 1)
    return history_start, history_end


def _runtime_attrs(dataset_id: int, mode: str) -> dict[str, Any]:
    observed_start, observed_end = _observed_bounds(dataset_id)
    history_start, history_end = _history_bounds(dataset_id)
    scenario = (
        "with_information_sharing" if mode == "with" else "without_information_sharing"
    )
    return {
        "dataset_name": f"Dataset{dataset_id}",
        "selection_authority": "runtime",
        "protocol_version": D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
        "target_observed_start": observed_start,
        "target_observed_end": observed_end,
        "source_history_start": history_start,
        "source_history_end": history_end,
        "target_test_excluded": True,
        "source_future_excluded": True,
        "source_alignment_mode": "exact_target_observed_dates",
        "representation": "mean_std_min_max_last",
        "scaling": "none",
        "scaler_fit_scope": "not_applicable",
        "information_sharing_scenario": scenario,
    }


def _group_cols(dataset_id: int) -> tuple[str, str]:
    return GROUP_COLS_BY_DATASET[dataset_id]


def _entity_key(store: str, item: str) -> str:
    return f"{store}_{item}"


def _build_target_rows(
    *,
    dataset_id: int,
    store: str,
    item: str,
    observed_sales: float,
    test_sales: float,
    observed_promo: float = 0.0,
    test_promo: float = 0.0,
) -> list[dict[str, Any]]:
    observed_start, observed_end = _observed_bounds(dataset_id)
    group_a, group_b = _group_cols(dataset_id)
    entity_id = _entity_key(store, item)
    rows: list[dict[str, Any]] = []
    for date in pd.date_range(observed_start, observed_end, freq="D"):
        rows.append(
            {
                "entity_id": entity_id,
                group_a: store,
                group_b: item,
                "date": date,
                "sales": observed_sales,
                "promo_flag": observed_promo,
            }
        )
    for date in pd.date_range(observed_end + pd.Timedelta(days=1), periods=10, freq="D"):
        rows.append(
            {
                "entity_id": entity_id,
                group_a: store,
                group_b: item,
                "date": date,
                "sales": test_sales,
                "promo_flag": test_promo,
            }
        )
    return rows


def _build_source_rows(
    *,
    dataset_id: int,
    store: str,
    item: str,
    observed_sales: float,
    future_sales: float,
    observed_promo: float = 0.0,
    future_promo: float = 0.0,
) -> list[dict[str, Any]]:
    observed_start, observed_end = _observed_bounds(dataset_id)
    group_a, group_b = _group_cols(dataset_id)
    rows: list[dict[str, Any]] = []
    for date in pd.date_range(observed_start, observed_end, freq="D"):
        rows.append(
            {
                group_a: store,
                group_b: item,
                "date": date,
                "sales": observed_sales,
                "promo_flag": observed_promo,
            }
        )
    # One post-cutoff row per source: eligible for future-window perturbation only.
    rows.append(
        {
            group_a: store,
            group_b: item,
            "date": observed_end + pd.Timedelta(days=1),
            "sales": future_sales,
            "promo_flag": future_promo,
        }
    )
    return rows


def _attach_attrs(frame: pd.DataFrame, dataset_id: int, mode: str, role: str) -> pd.DataFrame:
    out = frame.copy()
    out.attrs.update(_runtime_attrs(dataset_id, mode))
    out.attrs["role"] = role
    return out


def _fixture_frames(
    dataset_id: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build two targets and three sources with a deterministic Top-K ranking.

    Baseline ranking for each target (k=2):
      1. source-near  (sales match target observed=1.0)
      2. source-mid   (sales=5.0)
    source-far (sales=20.0) remains unselected until observed values flip.
    """
    target_specs = (
        ("tstore1", "titem1", 1.0),
        ("tstore2", "titem2", 1.0),
    )
    source_specs = (
        ("snear", "item", 1.0),
        ("smid", "item", 5.0),
        ("sfar", "item", 20.0),
    )

    target_rows: list[dict[str, Any]] = []
    for store, item, sales in target_specs:
        target_rows.extend(
            _build_target_rows(
                dataset_id=dataset_id,
                store=store,
                item=item,
                observed_sales=sales,
                test_sales=sales,
            )
        )
    source_rows: list[dict[str, Any]] = []
    for store, item, sales in source_specs:
        source_rows.extend(
            _build_source_rows(
                dataset_id=dataset_id,
                store=store,
                item=item,
                observed_sales=sales,
                future_sales=sales,
            )
        )

    target_df = _attach_attrs(pd.DataFrame(target_rows), dataset_id, mode, "target")
    source_df = _attach_attrs(pd.DataFrame(source_rows), dataset_id, mode, "source")
    target_ids = [_entity_key(store, item) for store, item, _ in target_specs]
    return source_df, target_df, target_ids


def _source_key_tuple(source: dict[str, Any]) -> tuple[Any, ...]:
    key = source["source_key"]
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return (key,)


def _topk_by_target(selection: dict[str, list[dict[str, Any]]]) -> dict[str, list[tuple[Any, ...]]]:
    return {
        target_id: [_source_key_tuple(row) for row in rows]
        for target_id, rows in selection.items()
    }


def _format_topk(topk: dict[str, list[tuple[Any, ...]]]) -> str:
    return json.dumps(
        {target: [list(key) for key in keys] for target, keys in topk.items()},
        ensure_ascii=False,
        sort_keys=True,
    )


def _assert_topk_unchanged(
    baseline: dict[str, list[tuple[Any, ...]]],
    perturbed: dict[str, list[tuple[Any, ...]]],
    *,
    dataset_id: int,
    mode: str,
    case: str,
) -> None:
    if baseline == perturbed:
        return
    changed_targets = sorted(
        target_id
        for target_id in set(baseline) | set(perturbed)
        if baseline.get(target_id) != perturbed.get(target_id)
    )
    details = []
    for target_id in changed_targets:
        details.append(
            f"target={target_id!r} baseline={baseline.get(target_id)!r} "
            f"perturbed={perturbed.get(target_id)!r}"
        )
    raise AssertionError(
        f"Dataset{dataset_id} mode={mode} case={case}: Top-K changed after "
        f"forbidden-window perturbation. changed_targets={changed_targets}; "
        f"details=[{'; '.join(details)}]; "
        f"baseline_all={_format_topk(baseline)}; perturbed_all={_format_topk(perturbed)}"
    )


def _assert_topk_changed(
    baseline: dict[str, list[tuple[Any, ...]]],
    perturbed: dict[str, list[tuple[Any, ...]]],
    *,
    dataset_id: int,
    mode: str,
    case: str,
) -> None:
    if baseline != perturbed:
        return
    raise AssertionError(
        f"Dataset{dataset_id} mode={mode} case={case}: positive-control Top-K "
        f"did not change after legal observed-window perturbation "
        f"(possible false-positive leakage test). "
        f"topk={_format_topk(baseline)}"
    )


def _select_with_meta(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_ids: list[str],
    dataset_id: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    selector = SourceSelector()
    group_cols = _group_cols(dataset_id)
    selected: dict[str, list[dict[str, Any]]] = {}
    selection_metadata: dict[str, dict[str, Any]] = {}
    for target_id in target_ids:
        target_entity_df = target_df[target_df["entity_id"].astype(str) == str(target_id)].copy()
        assert not target_entity_df.empty, f"missing synthetic target entity: {target_id}"
        target_entity_df.attrs.update(target_df.attrs)
        result = selector.select_top_k_sources(
            target_df=target_entity_df,
            source_df=source_df,
            feature_cols=list(FEATURE_COLS),
            k=K,
            group_cols=group_cols,
        )
        sources = result["sources"]
        meta = result["meta"]
        assert isinstance(sources, list)
        assert isinstance(meta, dict)
        selected[str(target_id)] = copy.deepcopy(sources)
        selection_metadata[str(target_id)] = copy.deepcopy(meta)
    return selected, selection_metadata


def _generate_knn_payload(
    *,
    tmp_path: Path,
    dataset_id: int,
    mode: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_ids: list[str],
    label: str,
) -> tuple[dict[str, list[tuple[Any, ...]]], Path]:
    """Generate a diagnostic KNN payload under tmp_path only."""
    group_cols = _group_cols(dataset_id)
    selected, selection_metadata = _select_with_meta(
        source_df=source_df,
        target_df=target_df,
        target_ids=target_ids,
        dataset_id=dataset_id,
    )

    old_payload = {
        "dataset_id": dataset_id,
        "dataset": f"D{dataset_id}",
        "info_sharing": mode,
        "k": K,
        "window_size": 10,
        "horizon": 1,
        "target_train_window": {
            "start": str(_train_start(dataset_id).date()),
            "end": str((_train_start(dataset_id) + pd.Timedelta(days=14)).date()),
        },
        "domain_filter": {"column": "synthetic_domain", "value": "perturbation"},
        "group_cols": list(group_cols),
    }
    payload = _build_regenerated_payload(
        old_payload=old_payload,
        feature_cols=list(FEATURE_COLS),
        feature_info={"selected_features": list(FEATURE_COLS)},
        source_pool_size=int(source_df.groupby(list(group_cols)).ngroups),
        results={
            target_id: [
                {
                    "source_entity": "_".join(str(part) for part in _source_key_tuple(row)),
                    "distance": float(row["distance"]),
                    "weight": float(row["weight"]),
                    "source_key": list(_source_key_tuple(row)),
                }
                for row in rows
            ]
            for target_id, rows in selected.items()
        },
        selection_metadata=selection_metadata,
    )
    out_path = (
        tmp_path
        / "generated_knn"
        / label
        / f"Dataset{dataset_id}"
        / f"knn_{mode}_info_sharing.json"
    )
    _write_json(out_path, payload)
    assert out_path.exists()
    assert SOLIDIFIED_KNN_ROOT.resolve() not in out_path.resolve().parents
    return _topk_by_target(selected), out_path


def _perturb_target_test_window(target_df: pd.DataFrame, dataset_id: int) -> pd.DataFrame:
    _, observed_end = _observed_bounds(dataset_id)
    perturbed = target_df.copy()
    perturbed.attrs.update(target_df.attrs)
    mask = perturbed["date"] > observed_end
    perturbed.loc[mask, "sales"] = EXTREME_VALUE
    perturbed.loc[mask, "promo_flag"] = EXTREME_VALUE
    return perturbed


def _perturb_source_future_window(source_df: pd.DataFrame, dataset_id: int) -> pd.DataFrame:
    _, observed_end = _observed_bounds(dataset_id)
    perturbed = source_df.copy()
    perturbed.attrs.update(source_df.attrs)
    mask = perturbed["date"] > observed_end
    perturbed.loc[mask, "sales"] = EXTREME_VALUE
    perturbed.loc[mask, "promo_flag"] = EXTREME_VALUE
    return perturbed


def _perturb_observed_window_flip_ranking(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    dataset_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flip ranking by moving target observed sales toward source-far."""
    _, observed_end = _observed_bounds(dataset_id)
    group_a, _ = _group_cols(dataset_id)

    new_target = target_df.copy()
    new_target.attrs.update(target_df.attrs)
    observed = new_target["date"] <= observed_end
    new_target.loc[observed, "sales"] = 20.0

    new_source = source_df.copy()
    new_source.attrs.update(source_df.attrs)
    # Also nudge the near source's observed history away from the new target.
    near_mask = (new_source[group_a] == "snear") & (new_source["date"] <= observed_end)
    new_source.loc[near_mask, "sales"] = 0.0
    return new_source, new_target


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
@pytest.mark.parametrize("mode", INFO_SHARING_MODES)
def test_forbidden_windows_do_not_change_generated_topk(
    tmp_path: Path,
    dataset_id: int,
    mode: str,
) -> None:
    solidified_before = snapshot_knn_config_files(SOLIDIFIED_KNN_ROOT)
    source_df, target_df, target_ids = _fixture_frames(dataset_id, mode)

    baseline_topk, baseline_path = _generate_knn_payload(
        tmp_path=tmp_path,
        dataset_id=dataset_id,
        mode=mode,
        source_df=source_df,
        target_df=target_df,
        target_ids=target_ids,
        label="baseline",
    )
    assert baseline_topk, "baseline Top-K must be non-empty"
    for target_id, keys in baseline_topk.items():
        assert keys[0] == ("snear", "item"), (
            f"Dataset{dataset_id} mode={mode} target={target_id}: "
            f"expected baseline rank-1=('snear','item'), got={keys}"
        )

    # Case A: extreme values only in target test window.
    test_perturbed_target = _perturb_target_test_window(target_df, dataset_id)
    test_topk, _ = _generate_knn_payload(
        tmp_path=tmp_path,
        dataset_id=dataset_id,
        mode=mode,
        source_df=source_df,
        target_df=test_perturbed_target,
        target_ids=target_ids,
        label="target_test_perturbed",
    )
    _assert_topk_unchanged(
        baseline_topk,
        test_topk,
        dataset_id=dataset_id,
        mode=mode,
        case="target_test_window",
    )

    # Case B: extreme values only in source rows after cutoff.
    future_perturbed_source = _perturb_source_future_window(source_df, dataset_id)
    future_topk, _ = _generate_knn_payload(
        tmp_path=tmp_path,
        dataset_id=dataset_id,
        mode=mode,
        source_df=future_perturbed_source,
        target_df=target_df,
        target_ids=target_ids,
        label="source_future_perturbed",
    )
    _assert_topk_unchanged(
        baseline_topk,
        future_topk,
        dataset_id=dataset_id,
        mode=mode,
        case="source_future_window",
    )

    verify_knn_config_unchanged(SOLIDIFIED_KNN_ROOT, solidified_before)
    assert baseline_path.exists()
    assert "configs/solidified/knn" not in str(baseline_path)


@pytest.mark.parametrize("dataset_id", DATASET_IDS)
@pytest.mark.parametrize("mode", INFO_SHARING_MODES)
def test_observed_window_perturbation_can_change_generated_topk(
    tmp_path: Path,
    dataset_id: int,
    mode: str,
) -> None:
    """Positive control: legal observed-window edits must be able to change Top-K."""
    solidified_before = snapshot_knn_config_files(SOLIDIFIED_KNN_ROOT)
    source_df, target_df, target_ids = _fixture_frames(dataset_id, mode)

    baseline_topk, _ = _generate_knn_payload(
        tmp_path=tmp_path,
        dataset_id=dataset_id,
        mode=mode,
        source_df=source_df,
        target_df=target_df,
        target_ids=target_ids,
        label="baseline_positive_control",
    )

    flipped_source, flipped_target = _perturb_observed_window_flip_ranking(
        source_df, target_df, dataset_id
    )
    flipped_topk, _ = _generate_knn_payload(
        tmp_path=tmp_path,
        dataset_id=dataset_id,
        mode=mode,
        source_df=flipped_source,
        target_df=flipped_target,
        target_ids=target_ids,
        label="observed_perturbed",
    )

    _assert_topk_changed(
        baseline_topk,
        flipped_topk,
        dataset_id=dataset_id,
        mode=mode,
        case="observed_window_positive_control",
    )
    # Rank-1 should move toward the far source after the observed flip.
    for target_id, keys in flipped_topk.items():
        assert keys[0] == ("sfar", "item"), (
            f"Dataset{dataset_id} mode={mode} target={target_id}: "
            f"positive-control expected rank-1=('sfar','item'), got={keys}; "
            f"baseline={baseline_topk.get(target_id)}"
        )

    verify_knn_config_unchanged(SOLIDIFIED_KNN_ROOT, solidified_before)


def test_tmp_knn_generation_does_not_touch_solidified_tree(tmp_path: Path) -> None:
    """Smoke: one dataset write lands only under tmp_path."""
    solidified_before = snapshot_knn_config_files(SOLIDIFIED_KNN_ROOT)
    source_df, target_df, target_ids = _fixture_frames(4, "without")
    _, out_path = _generate_knn_payload(
        tmp_path=tmp_path,
        dataset_id=4,
        mode="without",
        source_df=source_df,
        target_df=target_df,
        target_ids=target_ids,
        label="smoke",
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["selection_authority"] == "runtime"
    assert payload["protocol_version"] == D4_D6_RUNTIME_KNN_PROTOCOL_VERSION
    assert set(payload["results"]) == set(target_ids)
    verify_knn_config_unchanged(SOLIDIFIED_KNN_ROOT, solidified_before)
    assert str(out_path.resolve()).startswith(str(tmp_path.resolve()))
