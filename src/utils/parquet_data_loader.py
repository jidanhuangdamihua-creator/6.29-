from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.constants import (
    D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
    SOLIDIFIED_TARGET_WINDOWS,
    SOURCE_HISTORY_DAYS,
)
from src.protocols.gate1_transformation import dataset_contract
from src.protocols.experiment_protocol import get_experiment_protocol
from src.protocols.source_history import (
    build_exact_source_history_candidate_frame,
    source_history_frame_digest,
)
from src.utils.d5_calendar_reconstruction import (
    D5AuthorityBundle,
    D5ReconstructionReport,
    reconstruct_d5_target_calendar,
    reconstruct_d5_source_history_calendar,
)
from src.utils.d5_precomputed_source_history import (
    load_precomputed_d5_source_history,
)


LOGGER = logging.getLogger("experiment")


RUNTIME_KNN_WINDOW_ATTRS = (
    "selection_authority",
    "protocol_version",
    "target_observed_start",
    "target_observed_end",
    "source_history_start",
    "source_history_end",
    "source_history_days",
    "source_history_expected_date_count",
    "source_history_completeness_policy",
    "source_history_calendar",
    "source_history_inclusive_end",
    "source_history_frame_digest",
    "source_history_calendarization_rule",
    "source_history_synthetic_row_count",
    "target_test_excluded",
    "source_future_excluded",
    "source_alignment_mode",
    "representation",
    "scaling",
    "scaler_fit_scope",
)


@dataclass(frozen=True)
class ParquetSourceTargetLoad:
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    calendar_reconstruction: D5ReconstructionReport | None


def _dataset_name(dataset_id: int) -> str:
    return f"Dataset{int(dataset_id)}"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_target_selection_windows(dataset_id: int) -> Dict[str, Any]:
    """Read solidified D4-D6 target train/test bounds."""
    if int(dataset_id) not in {4, 5, 6}:
        return {}
    return dict(SOLIDIFIED_TARGET_WINDOWS[int(dataset_id)])


def derive_d4_d6_runtime_knn_windows(
    windows: Dict[str, Any],
    source_history_days: int,
) -> Dict[str, Any]:
    """Derive the explicit D4-D6 runtime KNN bounds using inclusive days."""
    dataset_id = int(windows.get("dataset_id", 0))
    if dataset_id not in {4, 5, 6}:
        raise ValueError(f"runtime KNN windows are only defined for D4-D6: dataset_id={dataset_id}")
    if "train_start" not in windows:
        raise ValueError("D4-D6 runtime KNN windows require train_start")
    if int(source_history_days) != SOURCE_HISTORY_DAYS:
        raise ValueError(
            "D4-D6 source_history_days must equal the frozen value "
            f"{SOURCE_HISTORY_DAYS}"
        )

    target_observed_start = pd.to_datetime(windows["train_start"], errors="coerce")
    if pd.isna(target_observed_start):
        raise ValueError(f"Invalid D4-D6 train_start: {windows['train_start']!r}")
    target_observed_start = pd.Timestamp(target_observed_start).normalize()
    target_observed_end = target_observed_start + pd.Timedelta(days=29)
    source_history_end = pd.Timestamp(dataset_contract(f"D{dataset_id}").origin).normalize()
    source_history_start = source_history_end - pd.Timedelta(days=int(source_history_days) - 1)

    return {
        "selection_authority": "runtime",
        "protocol_version": D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
        "target_observed_start": target_observed_start,
        "target_observed_end": target_observed_end,
        "source_history_start": source_history_start,
        "source_history_end": source_history_end,
        "target_test_excluded": True,
        "source_future_excluded": True,
        "source_alignment_mode": "exact_target_observed_dates",
        "representation": "mean_std_min_max_last",
        "scaling": "none",
        "scaler_fit_scope": "not_applicable",
    }


def expected_target_dates_from_windows(windows: Dict[str, Any]) -> pd.DatetimeIndex:
    """Materialize the target calendar only from the fixed window authority."""
    if "train_start" not in windows or "test_end" not in windows:
        raise ValueError("target window authority requires train_start and test_end")
    start = pd.to_datetime(windows["train_start"], errors="coerce")
    end = pd.to_datetime(windows["test_end"], errors="coerce")
    if pd.isna(start) or pd.isna(end) or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(
            f"invalid target window authority: train_start={windows.get('train_start')!r} "
            f"test_end={windows.get('test_end')!r}"
        )
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")


def _knn_json_path(knn_json_dir: str | Path, info_sharing: str) -> Path:
    scenario = str(info_sharing).strip().lower()
    if scenario not in {"with", "without"}:
        raise ValueError("info_sharing must be 'with' or 'without'")
    return Path(knn_json_dir) / f"knn_{scenario}_info_sharing.json"


def _payload_with_path(payload: Dict[str, Any], path: Path) -> Dict[str, Any]:
    loaded = dict(payload)
    if "_path" not in loaded:
        loaded["_path"] = str(path)
    return loaded


def load_knn_results(
    knn_json_dir: str | Path,
    info_sharing: str,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load precomputed KNN/source-selection JSON for a D4-D6 scenario."""
    path = _knn_json_path(knn_json_dir, info_sharing)
    if payload is not None:
        return _payload_with_path(payload, path)
    if not path.exists():
        raise FileNotFoundError(f"Missing KNN selection file: {path}")
    return _payload_with_path(_read_json(path), path)


def read_dataset_windows(
    dataset_id: int,
    knn_json_dir: str | Path,
    info_sharing: str | None = None,
    knn_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Read fixed target/source window metadata from existing KNN JSON files."""
    windows: Dict[str, Any] = {"dataset_id": int(dataset_id)}
    if info_sharing is None:
        scenarios = ("without", "with")
    else:
        scenario = str(info_sharing).strip().lower()
        if scenario not in {"with", "without"}:
            raise ValueError("info_sharing must be 'with' or 'without'")
        scenarios = (scenario,)

    for scenario in scenarios:
        if knn_payload is not None and info_sharing is not None:
            payload = _payload_with_path(knn_payload, _knn_json_path(knn_json_dir, scenario))
        else:
            payload = load_knn_results(knn_json_dir, scenario)
        windows[f"{scenario}_target_train_window"] = payload.get("target_train_window", {})
        windows[f"{scenario}_source_pool_size"] = payload.get("source_pool_size")
        windows[f"{scenario}_domain_filter"] = payload.get("domain_filter")
    first = windows.get("without_target_train_window") or windows.get("with_target_train_window") or {}
    windows["target_train_window"] = dict(first) if isinstance(first, dict) else {}
    windows.update(_read_target_selection_windows(int(dataset_id)))
    return windows


def _reindex_target_calendar(
    df: pd.DataFrame,
    date_col: str,
    entity_col: str,
    train_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    """Reindex every target entity to the complete daily evaluation calendar."""
    attrs = df.attrs.copy()
    bool_dtypes = {
        col: dtype
        for col, dtype in df.dtypes.items()
        if pd.api.types.is_bool_dtype(dtype)
    }
    numeric_cols = [
        col
        for col, dtype in df.dtypes.items()
        if col not in {date_col, entity_col, "sales"}
        and col not in bool_dtypes
        and pd.api.types.is_numeric_dtype(dtype)
    ]
    other_cols = [
        col
        for col in df.columns
        if col not in {date_col, entity_col, "sales"}
        and col not in bool_dtypes
        and col not in numeric_cols
    ]
    calendar = pd.date_range(train_start, test_end, freq="D")
    reindexed_entities = []

    for entity_key, entity_df in df.groupby(entity_col, sort=False, dropna=False):
        entity_df = entity_df.copy()
        entity_df[date_col] = pd.to_datetime(entity_df[date_col], errors="coerce")
        entity_df = entity_df.sort_values(date_col).set_index(date_col).reindex(calendar)
        entity_df[date_col] = calendar
        entity_df[entity_col] = entity_key
        if "sales" in entity_df.columns:
            entity_df["sales"] = entity_df["sales"].fillna(0)
        for col, dtype in bool_dtypes.items():
            entity_df[col] = entity_df[col].fillna(False).astype(dtype)
        for col in numeric_cols:
            entity_df[col] = entity_df[col].fillna(0)
        for col in other_cols:
            entity_df[col] = entity_df[col].ffill().bfill()
        reindexed_entities.append(entity_df.reset_index(drop=True))

    if not reindexed_entities:
        raise ValueError(
            f"No target entity groups remain after filtering from {train_start} through {test_end}"
        )
    result = pd.concat(reindexed_entities, ignore_index=True)
    result = result.loc[:, df.columns]
    result.attrs = attrs
    return result


def attach_window_attrs(
    df: pd.DataFrame,
    windows: Dict[str, Any],
    role: str,
    *,
    calendarize_target: bool = True,
) -> pd.DataFrame:
    """Attach split/window metadata and return the frame to use downstream."""
    dataset_id = int(windows.get("dataset_id", 0))
    df.attrs["dataset_name"] = _dataset_name(dataset_id) if dataset_id else "unknown"
    df.attrs["split_role"] = str(role)
    df.attrs["role"] = str(role)
    df.attrs["split_mode"] = "paper_split_protocol" if role == "target" else "ratio"
    for key in RUNTIME_KNN_WINDOW_ATTRS:
        if key in windows:
            df.attrs[key] = windows[key]
    if role == "target":
        date_col = "date" if "date" in df.columns else "dt" if "dt" in df.columns else None
        if date_col is None:
            raise AssertionError("target dataframe requires a date or dt column")
        entity_col = "entity_id"
        if entity_col not in df.columns:
            raise AssertionError("target dataframe requires entity_id as the entity column")

        if calendarize_target:
            train_start = pd.to_datetime(windows["train_start"])
            test_end = pd.to_datetime(windows["test_end"])
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[(df[date_col] >= train_start) & (df[date_col] <= test_end)].copy()
            df = _reindex_target_calendar(
                df,
                date_col=date_col,
                entity_col=entity_col,
                train_start=train_start,
                test_end=test_end,
            )

        train_days = 15
        val_days = 15
        n_unique = int(df[date_col].nunique())
        test_days = int(n_unique - train_days - val_days)
        if test_days <= 0:
            raise AssertionError(
                f"target split has non-positive test_days: "
                f"n_unique={n_unique} train_days={train_days} val_days={val_days}"
            )

        # The target frame retains evaluation dates. Explicit runtime attrs define
        # the separate 30-day train+validation window used by D4-D6 KNN selection.
        df.attrs["paper_split_protocol"] = "solidified_d4_d6_target_train_window"
        df.attrs["train_days"] = train_days
        df.attrs["val_days"] = val_days
        df.attrs["observed_days"] = int(train_days + val_days)
        df.attrs["test_days"] = test_days
        df.attrs["split_config"] = {
            "mode": "days",
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
        }
        window = windows.get("target_train_window", {})
        if isinstance(window, dict):
            df.attrs["target_train_window"] = dict(window)
    else:
        df.attrs["split_config"] = {
            "mode": "ratio",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
        }
    return df


def _coerce_bool_or_numeric_like(series: pd.Series, *, dataset_id: int, column: str, role: str) -> pd.Series | None:
    """Return a numeric version when conversion is deterministic; otherwise None."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_numeric_dtype(series):
        return series

    non_null = series.dropna()
    if non_null.empty:
        return pd.to_numeric(series, errors="coerce")

    text = non_null.astype("string").str.strip().str.lower()
    bool_map = {
        "true": 1,
        "false": 0,
        "t": 1,
        "f": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
    }
    if text.isin(bool_map).all():
        converted = series.astype("string").str.strip().str.lower().map(bool_map)
        return converted.astype("float64")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric[series.notna()].notna().all():
        return numeric

    LOGGER.warning(
        "[load_parquet_source_target] D%d %s column %s could not be safely converted to numeric; "
        "it will be excluded from model feature columns if still non-numeric.",
        int(dataset_id),
        role,
        column,
    )
    return None


def _coerce_known_model_candidate_columns(df: pd.DataFrame, *, dataset_id: int, role: str) -> pd.DataFrame:
    """Coerce only known D5/D6 model candidate columns in memory."""
    out = df.copy()
    if int(dataset_id) == 5 and "onpromotion" in out.columns:
        converted = _coerce_bool_or_numeric_like(out["onpromotion"], dataset_id=dataset_id, column="onpromotion", role=role)
        if converted is not None:
            out["onpromotion"] = converted

    if int(dataset_id) == 6:
        candidate_cols = [
            col
            for col in out.columns
            if col in {"sell_price", "is_event_1", "is_event_2", "snap"} or str(col).startswith("snap_")
        ]
        for col in candidate_cols:
            converted = _coerce_bool_or_numeric_like(out[col], dataset_id=dataset_id, column=str(col), role=role)
            if converted is not None:
                out[col] = converted
    return out


def load_parquet_source_target_with_diagnostics(
    dataset_id: int,
    source_path: str | Path,
    target_path: str | Path,
    windows: Dict[str, Any],
    source_history_days: int | None = None,
    *,
    expected_dates: pd.DatetimeIndex | None = None,
    d5_authorities: D5AuthorityBundle | None = None,
) -> ParquetSourceTargetLoad:
    """Load the explicit resolver-selected D4-D6 parquet files only."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(
            f"Missing solidified parquet paths: source={source_path} target={target_path}"
        )

    if source_history_days is None:
        raise ValueError("D4-D6 runtime KNN requires source_history_days")
    if int(source_history_days) != SOURCE_HISTORY_DAYS:
        raise ValueError(
            "D4-D6 runtime KNN rejects non-frozen source_history_days: "
            f"{source_history_days!r}"
        )
    runtime_windows = dict(windows)
    runtime_windows.update(
        derive_d4_d6_runtime_knn_windows(runtime_windows, int(source_history_days))
    )
    source_filters = [
        ("date", ">=", runtime_windows["source_history_start"]),
        ("date", "<=", runtime_windows["source_history_end"]),
    ]
    target_df = pd.read_parquet(target_path)
    target_df = _coerce_known_model_candidate_columns(target_df, dataset_id=dataset_id, role="target")

    key_fields = tuple(get_experiment_protocol(dataset_id).source_pool_rule.key_fields)
    source_history_reconstruction: D5ReconstructionReport | None = None
    source_history_precomputed = False
    precomputed: tuple[pd.DataFrame, Dict[str, Any]] | None = None
    if int(dataset_id) == 5:
        if d5_authorities is None:
            raise ValueError("D5 loader requires a preloaded D5AuthorityBundle for source history")
        precomputed = load_precomputed_d5_source_history(
            source_path=source_path,
            authorities=d5_authorities,
            source_history_start=runtime_windows["source_history_start"],
            source_history_end=runtime_windows["source_history_end"],
            source_history_days=int(source_history_days),
            key_fields=key_fields,
        )
    if precomputed is not None:
        source_df, _ = precomputed
        source_history_precomputed = True
    else:
        source_df = pd.read_parquet(source_path, filters=source_filters)
        source_df = _coerce_known_model_candidate_columns(
            source_df,
            dataset_id=dataset_id,
            role="source",
        )

    for role, frame in (("source", source_df), ("target", target_df)):
        if "date" not in frame.columns:
            raise ValueError(f"D4-D6 {role} dataframe requires date column")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValueError(f"D4-D6 {role} dataframe contains invalid date values")

    if not source_history_precomputed:
        source_df = source_df[
            source_df["date"].between(
                runtime_windows["source_history_start"],
                runtime_windows["source_history_end"],
                inclusive="both",
            )
        ].copy()

    source_history_eligibility: dict[str, object]
    source_history_calendarization_rule = "not_applicable"
    source_history_synthetic_row_count = 0
    source_history_original_row_count = len(source_df)
    if int(dataset_id) == 5:
        if precomputed is not None:
            source_df, precomputed_manifest = precomputed
            history = precomputed_manifest["source_history"]
            source_history_precomputed = True
            source_history_calendarization_rule = (
                "D5_APPROVED_SOURCE_HISTORY_CALENDARIZATION"
            )
            source_history_synthetic_row_count = int(history["synthetic_row_count"])
            source_history_original_row_count = max(
                0,
                int(len(source_df)) - source_history_synthetic_row_count,
            )
            source_history_frame_digest_value = str(
                history["source_history_frame_digest"]
            )
            static_keys = (
                source_df.loc[:, list(key_fields)]
                .drop_duplicates()
                .sort_values(list(key_fields), kind="mergesort")
            )
            source_history_eligibility = {
                "eligible_keys": [
                    [str(value) for value in raw_key]
                    for raw_key in static_keys.itertuples(index=False, name=None)
                ],
                "incomplete_keys": {},
                "duplicate_keys": [],
                "outside_window_row_count": 0,
            }
        else:
            source_history_dates = pd.date_range(
                runtime_windows["source_history_start"],
                runtime_windows["source_history_end"],
                freq="D",
            )
            source_df, source_history_reconstruction = reconstruct_d5_source_history_calendar(
                source_df,
                expected_dates=source_history_dates,
                authorities=d5_authorities,
            )
            eligibility = build_exact_source_history_candidate_frame(
                source_df,
                key_fields=key_fields,
                origin=runtime_windows["source_history_end"],
                source_history_days=int(source_history_days),
            )
            source_df = eligibility.candidate_frame
            source_history_calendarization_rule = (
                "D5_APPROVED_SOURCE_HISTORY_CALENDARIZATION"
            )
            source_history_synthetic_row_count = int(
                source_history_reconstruction.synthetic_row_count
            )
            source_history_original_row_count = int(
                source_history_reconstruction.original_row_count
            )
            source_history_frame_digest_value = source_history_frame_digest(
                source_df,
                key_fields=key_fields,
            )
            source_history_eligibility = {
                "eligible_keys": [list(key) for key in eligibility.eligible_keys],
                "incomplete_keys": {
                    "/".join(key): count
                    for key, count in eligibility.incomplete_keys.items()
                },
                "duplicate_keys": [list(key) for key in eligibility.duplicate_keys],
                "outside_window_row_count": eligibility.outside_window_row_count,
            }
    else:
        eligibility = build_exact_source_history_candidate_frame(
            source_df,
            key_fields=key_fields,
            origin=runtime_windows["source_history_end"],
            source_history_days=int(source_history_days),
        )
        source_df = eligibility.candidate_frame
        source_history_frame_digest_value = source_history_frame_digest(
            source_df,
            key_fields=key_fields,
        )
        source_history_eligibility = {
            "eligible_keys": [list(key) for key in eligibility.eligible_keys],
            "incomplete_keys": {
                "/".join(key): count
                for key, count in eligibility.incomplete_keys.items()
            },
            "duplicate_keys": [list(key) for key in eligibility.duplicate_keys],
            "outside_window_row_count": eligibility.outside_window_row_count,
        }

    runtime_windows.update(
        {
            "source_history_days": int(source_history_days),
            "source_history_expected_date_count": int(source_history_days),
            "source_history_completeness_policy": source_df.attrs[
                "source_history_completeness_policy"
            ]
            if "source_history_completeness_policy" in source_df.attrs
            else "exact_expected_date_set",
            "source_history_calendar": source_df.attrs.get(
                "source_history_calendar", "Gregorian daily"
            ),
            "source_history_inclusive_end": source_df.attrs.get(
                "source_history_inclusive_end", True
            ),
            "source_history_calendarization_rule": source_history_calendarization_rule,
            "source_history_synthetic_row_count": source_history_synthetic_row_count,
            "source_history_frame_digest": source_history_frame_digest_value,
        }
    )

    source_df = attach_window_attrs(source_df, runtime_windows, role="source")
    source_df.attrs.update(
        {
            "source_history_eligibility": source_history_eligibility,
            "source_history_key_fields": list(key_fields),
            "source_history_calendarization": {
                "rule": source_history_calendarization_rule,
                "synthetic_row_count": source_history_synthetic_row_count,
                "original_row_count": source_history_original_row_count,
            },
            "source_history_frame_digest": source_history_frame_digest_value,
        }
    )
    if source_history_precomputed:
        source_df.attrs.update(
            {
                "source_history_prevalidated_exact": True,
                "source_history_validation_path": "precomputed_static_file",
            }
        )
    reconstruction: D5ReconstructionReport | None = None
    if int(dataset_id) == 5:
        if expected_dates is None:
            raise ValueError("D5 loader requires expected_dates from the window authority")
        if d5_authorities is None:
            raise ValueError("D5 loader requires a preloaded D5AuthorityBundle")
        target_df = target_df[target_df["date"].isin(expected_dates)].copy()
        target_df, reconstruction = reconstruct_d5_target_calendar(
            target_df,
            date_col="date",
            entity_col="entity_id",
            expected_dates=expected_dates,
            authorities=d5_authorities,
        )
        target_df = attach_window_attrs(
            target_df,
            runtime_windows,
            role="target",
            calendarize_target=False,
        )
    else:
        target_df = attach_window_attrs(target_df, runtime_windows, role="target")
    source_df.attrs["solidified_parquet_path"] = str(source_path)
    target_df.attrs["solidified_parquet_path"] = str(target_path)
    return ParquetSourceTargetLoad(source_df, target_df, reconstruction)


def load_parquet_source_target(
    dataset_id: int,
    source_path: str | Path,
    target_path: str | Path,
    windows: Dict[str, Any],
    source_history_days: int | None = None,
    *,
    expected_dates: pd.DatetimeIndex | None = None,
    d5_authorities: D5AuthorityBundle | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility tuple wrapper around the diagnostics-returning loader."""
    loaded = load_parquet_source_target_with_diagnostics(
        dataset_id=dataset_id,
        source_path=source_path,
        target_path=target_path,
        windows=windows,
        source_history_days=source_history_days,
        expected_dates=expected_dates,
        d5_authorities=d5_authorities,
    )
    return loaded.source_df, loaded.target_df
