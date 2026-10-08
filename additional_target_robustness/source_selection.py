"""Target-specific Top-3 selection on a read-only source history.

The distance, scaling, ranking, and weights are the frozen ablation functions.
This module does not call ``rank_candidates`` or reconstruct a source calendar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from additional_target_robustness.paths import ensure_ablation_path
from additional_target_robustness.protocol import (
    D1_SOURCE_ITEMS,
    D1_SOURCE_STORES,
    TOP_K,
    Calendar,
    ProtocolError,
    calendar_for,
)

ensure_ablation_path()

from sita_ablation.data import _encode_similarity  # noqa: E402
from sita_ablation.scaling import fit_selection_scaler, transform_selection  # noqa: E402
from sita_ablation.source_selection import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_SKIPPED,
    anchored_rank,
    inverse_distance_weights,
)


@dataclass(frozen=True)
class SelectedSource:
    identity: str
    key: tuple[str, ...]
    distance: float
    weight: float
    rank: int


@dataclass(frozen=True)
class SelectionOutcome:
    status: str
    dataset: str
    target_identity: str
    selected: tuple[SelectedSource, ...]
    legal_count: int
    comparison_start: str
    comparison_end: str
    source_universe_id: str | None


def select_top3(
    dataset: str,
    target_identity: str,
    target_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    *,
    source_universe_id: str | None = None,
) -> SelectionOutcome:
    calendar = calendar_for(dataset)
    if calendar.comparison_end >= calendar.test_start:
        raise ProtocolError(f"{dataset} comparison window reaches the test period")
    target_key = _identity_key(target_identity)
    target_window = _comparison_matrix(dataset, target_frame, calendar, required=True)
    legal: list[tuple[tuple[str, ...], np.ndarray]] = []
    scoped = _scope_sources(dataset, target_key, target_frame, source_frame, calendar)
    for key, series in scoped:
        if key == target_key:
            continue
        if not _exact_history(series, calendar):
            continue
        window = _comparison_matrix(dataset, series, calendar, required=False)
        if window is None:
            continue
        legal.append((key, window))
    legal.sort(key=lambda item: item[0])
    outcome_common = dict(
        dataset=dataset,
        target_identity=target_identity,
        legal_count=len(legal),
        comparison_start=calendar.comparison_start,
        comparison_end=calendar.comparison_end,
        source_universe_id=source_universe_id if dataset == "D1" else source_universe_id,
    )
    if len(legal) < TOP_K:
        return SelectionOutcome(status=STATUS_SKIPPED, selected=(), **outcome_common)
    if target_window is None:
        raise ProtocolError(f"{dataset} {target_identity} comparison window is incomplete")
    keys = tuple(key for key, _window in legal)
    candidates = np.stack([window for _key, window in legal]).astype(np.float64, copy=False)
    scaler = fit_selection_scaler(target_window, candidates, calendar.similarity_variables)
    scaled_target = transform_selection(scaler, target_window)
    scaled_candidates = transform_selection(scaler, candidates)
    distances = np.linalg.norm(scaled_candidates - scaled_target, axis=1).astype(np.float64, copy=False)
    ranked = anchored_rank(keys, distances)
    chosen = ranked[:TOP_K]
    chosen_distances = np.array([item.distance for item in chosen], dtype=np.float64)
    weights = inverse_distance_weights(chosen_distances)
    selected = tuple(
        SelectedSource(
            identity="/".join(item.source_key),
            key=tuple(item.source_key),
            distance=float(item.distance),
            weight=float(weights[index]),
            rank=index,
        )
        for index, item in enumerate(chosen)
    )
    if any(item.identity == target_identity for item in selected):
        raise ProtocolError(f"{dataset} selected its own identity {target_identity}")
    return SelectionOutcome(status=STATUS_COMPLETED, selected=selected, **outcome_common)


def load_source_histories(dataset: str, *, target_identity: str, target_frame: pd.DataFrame) -> pd.DataFrame:
    """Read the frozen history slice. The caller has already checked the file hash."""
    from additional_target_robustness.target_data import _read_filtered, source_cache_path

    calendar = calendar_for(dataset)
    path = source_cache_path(dataset)
    if dataset == "D5" and path.name != "prepared_180day_source.parquet":
        raise ProtocolError("D5 source histories must be read from prepared_180day_source.parquet")
    columns = ["date", *calendar.key_fields, *calendar.model_features, *calendar.similarity_variables]
    if calendar.group_field is not None:
        columns.append(calendar.group_field)
    columns = list(dict.fromkeys(columns))
    key_fields, identity = _source_read_identity(dataset, target_identity, target_frame, calendar)
    if dataset == "D1":
        frame = _read_store_items(path, columns, calendar)
    else:
        frame = _read_filtered(
            path,
            key_fields=(key_fields[0],),
            identity=identity,
            date_min=calendar.history_start,
            date_max=calendar.history_end,
            columns=columns,
        )
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    start = pd.Timestamp(calendar.history_start)
    end = pd.Timestamp(calendar.history_end)
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)
    if (frame["date"] > end).any():
        raise ProtocolError(f"{dataset} source history includes a date after {calendar.history_end}")
    return frame


def _source_read_identity(
    dataset: str,
    target_identity: str,
    target_frame: pd.DataFrame,
    calendar: Calendar,
) -> tuple[tuple[str, ...], str]:
    if dataset == "D1":
        return calendar.key_fields, "1/1"
    target_key = _identity_key(target_identity)
    if dataset in {"D4", "D6"}:
        return calendar.key_fields, target_key[0]
    if dataset == "D5":
        return calendar.key_fields, target_key[0]
    del target_frame
    raise ProtocolError(f"unsupported dataset {dataset}")


def _read_store_items(path, columns: list[str], calendar: Calendar) -> pd.DataFrame:
    from additional_target_robustness.target_data import _read_filtered

    frames = []
    for item in D1_SOURCE_ITEMS:
        frames.append(
            _read_filtered(
                path,
                key_fields=calendar.key_fields,
                identity=f"{D1_SOURCE_STORES[0]}/{item}",
                date_min=calendar.history_start,
                date_max=calendar.history_end,
                columns=columns,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _scope_sources(
    dataset: str,
    target_key: tuple[str, ...],
    target_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    calendar: Calendar,
) -> list[tuple[tuple[str, ...], pd.DataFrame]]:
    source = source_frame.copy()
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    if dataset == "D1":
        mask = source["store_id"].map(_component).isin(set(D1_SOURCE_STORES))
        mask &= source["item_id"].map(_component).isin(set(D1_SOURCE_ITEMS))
    elif dataset == "D4":
        mask = source["store_id"].map(_component) == target_key[0]
    elif dataset == "D5":
        family = _group_value(target_frame, "family")
        mask = source["store_nbr"].map(_component) == target_key[0]
        mask &= source["family"].map(_component) == family
    elif dataset == "D6":
        department = _group_value(target_frame, "dept_id")
        mask = source["store_id"].map(_component) == target_key[0]
        mask &= source["dept_id"].map(_component) == department
    else:
        raise ProtocolError(f"unsupported dataset {dataset}")
    narrowed = source.loc[mask]
    groups: list[tuple[tuple[str, ...], pd.DataFrame]] = []
    if narrowed.empty:
        return groups
    history_start = pd.Timestamp(calendar.history_start)
    history_end = pd.Timestamp(calendar.history_end)
    for key_values, group in narrowed.groupby(list(calendar.key_fields), sort=False, dropna=False):
        key = _normalize(key_values, calendar.key_fields)
        ordered = group.sort_values("date", kind="mergesort")
        ordered = ordered.loc[(ordered["date"] >= history_start) & (ordered["date"] <= history_end)]
        ordered = ordered.reset_index(drop=True)
        if ordered.empty or ordered["date"].duplicated().any():
            continue
        groups.append((key, ordered))
    return groups


def _comparison_matrix(
    dataset: str,
    series: pd.DataFrame,
    calendar: Calendar,
    *,
    required: bool,
) -> np.ndarray | None:
    start = pd.Timestamp(calendar.comparison_start)
    end = pd.Timestamp(calendar.comparison_end)
    window = series.loc[(series["date"] >= start) & (series["date"] <= end)].sort_values("date", kind="mergesort")
    expected = pd.date_range(start, end, freq="D")
    dates = window["date"].reset_index(drop=True)
    if len(window) != len(expected) or window["date"].duplicated().any() or not dates.equals(pd.Series(expected)):
        if required:
            raise ProtocolError(f"{dataset} target comparison window {calendar.comparison_start}..{calendar.comparison_end} is incomplete")
        return None
    if dates.max() >= pd.Timestamp(calendar.test_start):
        raise ProtocolError(f"{dataset} comparison window includes a test date")
    columns = []
    for name in calendar.similarity_variables:
        if name not in window.columns:
            if required:
                raise ProtocolError(f"{dataset} comparison window is missing {name}")
            return None
        values = _encode_similarity(dataset, name, window[name])
        if values is None or not np.isfinite(values).all():
            if required:
                raise ProtocolError(f"{dataset} comparison window has a non-finite {name}")
            return None
        columns.append(np.asarray(values, dtype=np.float64))
    return np.stack(columns, axis=0)


def _exact_history(series: pd.DataFrame, calendar: Calendar) -> bool:
    expected = pd.date_range(calendar.history_start, calendar.history_end, freq="D")
    dates = series["date"].reset_index(drop=True)
    return len(dates) == len(expected) and not series["date"].duplicated().any() and dates.equals(pd.Series(expected))


def _group_value(frame: pd.DataFrame, field: str) -> str:
    if field not in frame.columns:
        raise ProtocolError(f"target frame is missing {field}")
    values = sorted({_component(value) for value in frame[field].tolist() if _component(value)})
    if len(values) != 1:
        raise ProtocolError(f"target {field} values are {values!r}")
    return values[0]


def _identity_key(identity: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in identity.split("/"))
    if len(parts) != 2 or any(part == "" for part in parts):
        raise ProtocolError(f"identity {identity!r} is not a two-part key")
    return parts


def _normalize(key_values: object, key_fields: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(key_values, tuple):
        key_values = (key_values,)
    if len(key_values) != len(key_fields):
        raise ProtocolError(f"key {key_values!r} does not match {key_fields!r}")
    return tuple(_component(value) for value in key_values)


def _component(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
