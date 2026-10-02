"""Load sealed frames and apply the frozen date and candidate rules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

DatasetId = Literal["D1", "D4", "D5", "D6"]
ComparisonDays = Literal[15, 30]
DATASETS: tuple[DatasetId, ...] = ("D1", "D4", "D5", "D6")
D5_LAST_DATE = "2017-08-14"

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_FEATURES_PATH = _PACKAGE_ROOT / "configs" / "features_frozen.json"
_SEALED_ROOT = _REPO_ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1"


class ContractError(RuntimeError):
    """The sealed data or a caller-supplied frame disagrees with the frozen contract."""


@dataclass(frozen=True)
class DateSpan:
    start: pd.Timestamp
    end: pd.Timestamp

    def iso(self) -> tuple[str, str]:
        return (_iso(self.start), _iso(self.end))

    def contains(self, day: pd.Timestamp) -> bool:
        return self.start <= day <= self.end


@dataclass(frozen=True)
class DatasetContract:
    dataset: DatasetId
    folder: str
    key_fields: tuple[str, ...]
    target_key: tuple[str, ...]
    model_features: tuple[str, ...]
    similarity_variables: tuple[str, ...]
    source_history: DateSpan
    target_train: DateSpan
    target_val: DateSpan
    target_test: DateSpan
    grouping_field: str | None


@dataclass
class ConditionFrames:
    dataset: DatasetId
    source_rows: pd.DataFrame
    target_rows: pd.DataFrame
    key_fields: tuple[str, ...]
    target_key: tuple[str, ...]
    model_features: tuple[str, ...]
    similarity_variables: tuple[str, ...]
    source_history: DateSpan
    target_train: DateSpan
    target_val: DateSpan
    target_test: DateSpan
    comparison_30: DateSpan
    comparison_15: DateSpan
    data_sha256: dict[str, str]
    grouping_field: str | None


@dataclass
class SourceSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]


def load_dataset_contract(dataset: DatasetId) -> DatasetContract:
    payload = json.loads(_FEATURES_PATH.read_text(encoding="utf-8"))
    block = payload["datasets"][dataset]
    key_fields, grouping_field = _KEY_RULES[dataset]
    return DatasetContract(
        dataset=dataset,
        folder=_FOLDERS[dataset],
        key_fields=key_fields,
        target_key=_parse_target_key(str(block["target_key"]["value"])),
        model_features=tuple(block["model_features"]["value"]),
        similarity_variables=tuple(block["similarity_variables"]["value"]),
        source_history=_span(block["source_history"]["value"]),
        target_train=_span(block["target_train"]["value"]),
        target_val=_span(block["target_val"]["value"]),
        target_test=_span(block["target_test"]["value"]),
        grouping_field=grouping_field,
    )


def load_condition_frames(dataset: DatasetId) -> ConditionFrames:
    contract = load_dataset_contract(dataset)
    source_path = _SEALED_ROOT / contract.folder / "source.parquet"
    target_path = _SEALED_ROOT / contract.folder / "target.parquet"
    columns = _read_columns(contract)
    source_rows = _read_frame(source_path, columns)
    target_rows = _read_frame(target_path, columns)
    if dataset == "D5":
        source_rows = _drop_after(source_rows, D5_LAST_DATE)
        target_rows = _drop_after(target_rows, D5_LAST_DATE)
    _require_columns(source_rows, columns, dataset=dataset, role="source")
    _require_columns(target_rows, columns, dataset=dataset, role="target")
    frames = ConditionFrames(
        dataset=dataset,
        source_rows=source_rows,
        target_rows=target_rows,
        key_fields=contract.key_fields,
        target_key=contract.target_key,
        model_features=contract.model_features,
        similarity_variables=contract.similarity_variables,
        source_history=contract.source_history,
        target_train=contract.target_train,
        target_val=contract.target_val,
        target_test=contract.target_test,
        comparison_30=DateSpan(contract.target_train.start, contract.target_val.end),
        comparison_15=DateSpan(contract.target_train.start, contract.target_train.end),
        data_sha256={
            "source.parquet": _sha256(source_path),
            "target.parquet": _sha256(target_path),
        },
        grouping_field=contract.grouping_field,
    )
    _assert_target_calendar(frames)
    return frames


def target_series(frames: ConditionFrames) -> pd.DataFrame:
    series = _filter_key(frames.target_rows, frames.key_fields, frames.target_key)
    if series.empty:
        raise ContractError(
            f"{frames.dataset} target {frames.target_key!r} is absent from target.parquet"
        )
    return _sort_series(series, frames.dataset)


def partition_rows(frames: ConditionFrames, partition: Literal["train", "validation", "test"]) -> pd.DataFrame:
    span = {
        "train": frames.target_train,
        "validation": frames.target_val,
        "test": frames.target_test,
    }[partition]
    series = target_series(frames)
    return _between(series, span).reset_index(drop=True)


def observed_through_test(frames: ConditionFrames) -> pd.DataFrame:
    """Validation rows plus test rows. Training dates are not required for these horizons."""
    series = target_series(frames)
    start = frames.target_val.start
    end = frames.target_test.end
    observed = series[(series["date"] >= start) & (series["date"] <= end)].copy()
    return observed.reset_index(drop=True)


def clip_source_history(frames: ConditionFrames, key: Sequence[str]) -> pd.DataFrame:
    series = _filter_key(frames.source_rows, frames.key_fields, tuple(key))
    if series.empty:
        raise ContractError(f"{frames.dataset} source {tuple(key)!r} is absent")
    clipped = _between(_sort_series(series, frames.dataset), frames.source_history)
    if clipped.empty:
        raise ContractError(f"{frames.dataset} source {tuple(key)!r} has no rows inside source history")
    return clipped.reset_index(drop=True)


def fill_source_numeric_na(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Fill numeric model columns with 0 before the source split. Booleans and categories stay untouched."""
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            raise ContractError(f"source frame is missing model column {column!r}")
        series = out[column]
        if _leave_unfilled(series):
            continue
        if not (pd.api.types.is_numeric_dtype(series) or _is_nullable_numeric(series)):
            raise ContractError(
                f"source model column {column!r} is {series.dtype}, not a numeric column; refusing to coerce it"
            )
        out[column] = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return out


def split_source_frame(frame: pd.DataFrame, columns: Sequence[str]) -> SourceSplit:
    filled = fill_source_numeric_na(frame, columns)
    dates = _unique_dates(filled)
    n_dates = len(dates)
    if n_dates < 3:
        raise ContractError(f"source split needs at least 3 dates, found {n_dates}")
    train_end = max(1, int(n_dates * 0.8))
    val_end = min(n_dates - 1, max(train_end + 1, int(n_dates * 0.9)))
    train_dates = tuple(dates[:train_end])
    validation_dates = tuple(dates[train_end:val_end])
    test_dates = tuple(dates[val_end:])
    return SourceSplit(
        train=_rows_on_dates(filled, train_dates),
        validation=_rows_on_dates(filled, validation_dates),
        test=_rows_on_dates(filled, test_dates),
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
    )


def legal_candidate_keys(frames: ConditionFrames, comparison_days: ComparisonDays = 30) -> tuple[tuple[str, ...], ...]:
    _target, candidates, keys = similarity_windows(frames, comparison_days)
    del _target, candidates
    return keys


def similarity_windows(
    frames: ConditionFrames,
    comparison_days: ComparisonDays = 30,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
    """Return target `(V, L)`, legal candidates `(N, V, L)`, and candidate keys.

    Only the comparison window is read. A candidate missing a day or a finite value is excluded.
    """
    span = _comparison_span(frames, comparison_days)
    if span.end >= frames.target_test.start:
        raise ContractError(
            f"{frames.dataset} comparison window {span.iso()} reaches test start {_iso(frames.target_test.start)}"
        )
    target_window = _similarity_window(frames, target_series(frames), span)
    if target_window is None:
        raise ContractError(f"{frames.dataset} target comparison window {span.iso()} is incomplete or non-finite")
    accepted: list[tuple[tuple[str, ...], np.ndarray]] = []
    for key, group in _candidate_groups(frames, span):
        window = _similarity_window(frames, group, span)
        if window is not None:
            accepted.append((key, window))
    if not accepted:
        raise ContractError(f"{frames.dataset} has no legal candidates on {span.iso()}")
    accepted.sort(key=lambda item: item[0])
    keys = tuple(key for key, _window in accepted)
    candidate_windows = np.stack([window for _key, window in accepted])
    return target_window, candidate_windows, keys


def _similarity_window(frames: ConditionFrames, series: pd.DataFrame, span: DateSpan) -> np.ndarray | None:
    window = _between(series, span)
    expected = pd.date_range(span.start, span.end, freq="D")
    if len(window) != len(expected) or window["date"].duplicated().any():
        return None
    if not window["date"].reset_index(drop=True).equals(pd.Series(expected)):
        return None
    columns = []
    for name in frames.similarity_variables:
        values = _encode_similarity(frames.dataset, name, window[name])
        if values is None or not np.isfinite(values).all():
            return None
        columns.append(values)
    return np.stack(columns, axis=0)


def _encode_similarity(dataset: DatasetId, name: str, series: pd.Series) -> np.ndarray | None:
    if dataset == "D5" and name == "onpromotion":
        try:
            return _encode_onpromotion(series)
        except ContractError:
            return None
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return numeric


def _encode_onpromotion(series: pd.Series) -> np.ndarray:
    """Map the sealed D5 strings with the formal encoding.

    `None` becomes 0, `True`/`1` becomes 1, and `False`/`0` becomes 0.
    Any other token is a contract error. See `src/protocols/gate1_transformation.py:431-448`.
    """
    encoded = np.empty(len(series), dtype=np.float64)
    for index, value in enumerate(series.tolist()):
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if value is None or value is pd.NA or missing:
            encoded[index] = 0.0
            continue
        if isinstance(value, (bool, np.bool_)):
            encoded[index] = 1.0 if bool(value) else 0.0
            continue
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in (0.0, 1.0):
            encoded[index] = float(value)
            continue
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            encoded[index] = 1.0 if value.strip().lower() in {"true", "1"} else 0.0
            continue
        raise ContractError(f"unknown D5 onpromotion value {value!r}")
    return encoded


def _candidate_groups(frames: ConditionFrames, span: DateSpan) -> list[tuple[tuple[str, ...], pd.DataFrame]]:
    source = frames.source_rows
    source = source[(source["date"] >= span.start) & (source["date"] <= span.end)]
    if frames.dataset == "D1":
        mask = source["store_id"].map(_component) == "1"
        mask &= source["item_id"].map(_component).isin({str(item) for item in range(1, 10)})
    elif frames.dataset == "D4":
        mask = source["store_id"].map(_component) == frames.target_key[0]
    elif frames.dataset == "D5":
        family = _target_group_value(frames)
        mask = source["store_nbr"].map(_component) == frames.target_key[0]
        mask &= source["family"].map(_component) == family
    elif frames.dataset == "D6":
        department = _target_group_value(frames)
        mask = source["store_id"].map(_component) == frames.target_key[0]
        mask &= source["dept_id"].map(_component) == department
    else:
        raise ContractError(f"unsupported dataset {frames.dataset}")
    narrowed = source.loc[mask]
    groups = []
    if narrowed.empty:
        return groups
    for key_values, group in narrowed.groupby(list(frames.key_fields), sort=False, dropna=False):
        key = _normalize_key(key_values, frames.key_fields)
        if key == frames.target_key:
            continue
        ordered = group.sort_values("date", kind="mergesort")
        if ordered["date"].duplicated().any():
            continue
        groups.append((key, ordered.reset_index(drop=True)))
    return groups


def _target_group_value(frames: ConditionFrames) -> str:
    if frames.grouping_field is None:
        raise ContractError(f"{frames.dataset} has no grouping field")
    series = target_series(frames)
    values = {_component(value) for value in series[frames.grouping_field].dropna().unique()}
    if len(values) != 1:
        raise ContractError(f"{frames.dataset} target {frames.grouping_field} values are {sorted(values)!r}")
    return next(iter(values))


def _assert_target_calendar(frames: ConditionFrames) -> None:
    series = target_series(frames)
    for name, span, expected_days in (
        ("train", frames.target_train, 15),
        ("validation", frames.target_val, 15),
        ("test", frames.target_test, 180),
    ):
        found = _between(series, span)
        expected = pd.date_range(span.start, span.end, freq="D")
        if len(expected) != expected_days or len(found) != expected_days or found["date"].duplicated().any():
            raise ContractError(
                f"{frames.dataset} {name} has {len(found)} rows for {span.iso()}, expected {expected_days} unique days"
            )
        if not found["date"].reset_index(drop=True).equals(pd.Series(expected)):
            missing = [str(day.date()) for day in expected if day not in set(found["date"])]
            raise ContractError(f"{frames.dataset} {name} does not match {span.iso()}; missing {missing[:5]}")
    if frames.dataset == "D5":
        last = pd.Timestamp(D5_LAST_DATE)
        for role, frame in (("source", frames.source_rows), ("target", frames.target_rows)):
            if (frame["date"] > last).any():
                raise ContractError(f"D5 {role} still contains a date after {D5_LAST_DATE}")


def _comparison_span(frames: ConditionFrames, comparison_days: ComparisonDays) -> DateSpan:
    if comparison_days == 30:
        return frames.comparison_30
    if comparison_days == 15:
        return frames.comparison_15
    raise ContractError(f"comparison window must be 15 or 30, got {comparison_days}")


def _read_columns(contract: DatasetContract) -> tuple[str, ...]:
    columns = {"date", *contract.key_fields, *contract.model_features, *contract.similarity_variables}
    if contract.grouping_field is not None:
        columns.add(contract.grouping_field)
    if contract.dataset == "D5":
        # Sealed columns required by the formal source-history calendarizer.
        # entity_id is the aggregation key and is not synthesized.
        columns.update({"entity_id", "item_id", "city", "state", "type"})
    return tuple(sorted(columns))


def _read_frame(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ContractError(f"sealed file is missing: {path}")
    frame = pd.read_parquet(path, columns=list(columns))
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.reset_index(drop=True)


def _drop_after(frame: pd.DataFrame, last_date: str) -> pd.DataFrame:
    last = pd.Timestamp(last_date)
    return frame.loc[frame["date"] <= last].reset_index(drop=True)


def _filter_key(frame: pd.DataFrame, key_fields: Sequence[str], key: Sequence[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for field, value in zip(key_fields, key):
        mask &= frame[field].map(_component) == value
    return frame.loc[mask].copy()


def _sort_series(frame: pd.DataFrame, dataset: DatasetId) -> pd.DataFrame:
    ordered = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if ordered["date"].duplicated().any():
        raise ContractError(f"{dataset} series has duplicate dates")
    return ordered


def _between(frame: pd.DataFrame, span: DateSpan) -> pd.DataFrame:
    selected = frame[(frame["date"] >= span.start) & (frame["date"] <= span.end)]
    return selected.sort_values("date", kind="mergesort").reset_index(drop=True)


def _rows_on_dates(frame: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    wanted = set(dates)
    selected = frame.loc[frame["date"].isin(wanted)]
    return selected.sort_values("date", kind="mergesort").reset_index(drop=True)


def _unique_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    return list(pd.to_datetime(frame["date"]).dt.normalize().drop_duplicates().sort_values())


def _normalize_key(key_values: object, key_fields: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(key_values, tuple):
        key_values = (key_values,)
    if len(key_values) != len(key_fields):
        raise ContractError(f"key {key_values!r} does not match fields {key_fields!r}")
    return tuple(_component(value) for value in key_values)


def _parse_target_key(text: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in text.split("/"))
    if len(parts) != 2 or any(part == "" for part in parts):
        raise ContractError(f"frozen target key {text!r} is not a two-part key")
    return parts


def _component(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _span(pair: Sequence[str]) -> DateSpan:
    if len(pair) != 2:
        raise ContractError(f"date span {pair!r} is not a closed pair")
    start = pd.Timestamp(pair[0])
    end = pd.Timestamp(pair[1])
    if end < start:
        raise ContractError(f"date span ends before it starts: {pair!r}")
    return DateSpan(start, end)


def _iso(day: pd.Timestamp) -> str:
    return pd.Timestamp(day).strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, dataset: str, role: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"{dataset} {role} is missing columns {missing}")


def _leave_unfilled(series: pd.Series) -> bool:
    """Boolean, categorical, and string columns are not numeric model inputs."""
    return bool(
        pd.api.types.is_bool_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )


def _is_nullable_numeric(series: pd.Series) -> bool:
    return str(series.dtype) in {"Int8", "Int16", "Int32", "Int64", "Float32", "Float64"}


_FOLDERS = {"D1": "dataset1", "D4": "dataset4", "D5": "dataset5", "D6": "dataset6"}
_KEY_RULES = {
    "D1": (("store_id", "item_id"), None),
    "D4": (("store_id", "product_id"), None),
    "D5": (("store_nbr", "item_nbr"), "family"),
    "D6": (("store_id", "item_id"), "dept_id"),
}
