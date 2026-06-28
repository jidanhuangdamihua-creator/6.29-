from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


LOGGER = logging.getLogger("experiment")


def _dataset_name(dataset_id: int) -> str:
    return f"Dataset{int(dataset_id)}"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_target_selection_windows(dataset_id: int, knn_json_dir: str | Path) -> Dict[str, Any]:
    """Read D4-D6 target train/test bounds from target selection output."""
    if int(dataset_id) not in {4, 5, 6}:
        return {}

    knn_path = Path(knn_json_dir)
    outputs_root = knn_path.parent.parent
    path = (
        outputs_root
        / "domain_adaptation"
        / f"Dataset{int(dataset_id)}"
        / "target_selection"
        / "target_selection_result.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing target selection window file: {path}")

    payload = _read_json(path)
    if int(dataset_id) == 4:
        target_windows = payload["target_windows"]
        return {
            "train_start": target_windows["target_train_start"],
            "test_end": target_windows["test_end"],
        }
    if int(dataset_id) == 5:
        time_windows = payload["time_windows"]
        return {
            "train_start": time_windows["train_start"],
            "test_end": time_windows["test_end"],
        }
    if int(dataset_id) == 6:
        return {
            "train_start": payload["train_start"],
            "test_end": payload["test_end"],
        }
    return {}


def load_knn_results(knn_json_dir: str | Path, info_sharing: str) -> Dict[str, Any]:
    """Load precomputed KNN/source-selection JSON for a D4-D6 scenario."""
    scenario = str(info_sharing).strip().lower()
    if scenario not in {"with", "without"}:
        raise ValueError("info_sharing must be 'with' or 'without'")
    path = Path(knn_json_dir) / f"knn_{scenario}_info_sharing.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing KNN selection file: {path}")
    payload = _read_json(path)
    payload["_path"] = str(path)
    return payload


def read_dataset_windows(dataset_id: int, knn_json_dir: str | Path) -> Dict[str, Any]:
    """Read fixed target/source window metadata from existing KNN JSON files."""
    windows: Dict[str, Any] = {"dataset_id": int(dataset_id)}
    for scenario in ("without", "with"):
        payload = load_knn_results(knn_json_dir, scenario)
        windows[f"{scenario}_target_train_window"] = payload.get("target_train_window", {})
        windows[f"{scenario}_source_pool_size"] = payload.get("source_pool_size")
        windows[f"{scenario}_domain_filter"] = payload.get("domain_filter")
    first = windows.get("without_target_train_window") or windows.get("with_target_train_window") or {}
    windows["target_train_window"] = dict(first) if isinstance(first, dict) else {}
    windows.update(_read_target_selection_windows(int(dataset_id), knn_json_dir))
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


def attach_window_attrs(df: pd.DataFrame, windows: Dict[str, Any], role: str) -> pd.DataFrame:
    """Attach split/window metadata and return the frame to use downstream."""
    dataset_id = int(windows.get("dataset_id", 0))
    df.attrs["dataset_name"] = _dataset_name(dataset_id) if dataset_id else "unknown"
    df.attrs["split_role"] = str(role)
    df.attrs["role"] = str(role)
    df.attrs["split_mode"] = "paper_split_protocol" if role == "target" else "ratio"
    if role == "target":
        date_col = "date" if "date" in df.columns else "dt" if "dt" in df.columns else None
        if date_col is None:
            raise AssertionError("target dataframe requires a date or dt column")
        entity_col = "entity_id"
        if entity_col not in df.columns:
            raise AssertionError("target dataframe requires entity_id as the entity column")

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

        # D4-D6 KNN files use a 30-day observed target train+val window. The solidified
        # target parquet includes the later evaluation dates; preserving this
        # metadata lets SourceSelector exclude target test dates.
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


def load_parquet_source_target(
    dataset_id: int,
    parquet_dir: str | Path,
    windows: Dict[str, Any],
    source_history_days: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load D4-D6 fixed source/target parquet files only."""
    parquet_root = Path(parquet_dir)
    source_path = parquet_root / f"dataset{int(dataset_id)}-source.parquet"
    target_path = parquet_root / f"dataset{int(dataset_id)}-target.parquet"
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            f"Missing solidified parquet paths: source={source_path} target={target_path}"
        )

    source_df = pd.read_parquet(source_path)
    target_df = pd.read_parquet(target_path)
    source_df = _coerce_known_model_candidate_columns(source_df, dataset_id=dataset_id, role="source")
    target_df = _coerce_known_model_candidate_columns(target_df, dataset_id=dataset_id, role="target")
    if "date" in source_df.columns:
        source_df["date"] = pd.to_datetime(source_df["date"], errors="coerce")
    if "date" in target_df.columns:
        target_df["date"] = pd.to_datetime(target_df["date"], errors="coerce")

    if source_history_days and "date" in source_df.columns and "date" in target_df.columns and not target_df.empty:
        cutoff = pd.Timestamp(target_df["date"].min()) - pd.Timedelta(days=int(source_history_days))
        source_df = source_df[source_df["date"] >= cutoff].copy()

    source_df = attach_window_attrs(source_df, windows, role="source")
    target_df = attach_window_attrs(target_df, windows, role="target")
    source_df.attrs["solidified_parquet_path"] = str(source_path)
    target_df.attrs["solidified_parquet_path"] = str(target_path)
    return source_df, target_df
