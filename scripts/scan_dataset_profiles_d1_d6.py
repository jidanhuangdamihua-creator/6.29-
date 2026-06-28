#!/usr/bin/env python3
"""Auto-discover and profile Dataset1-Dataset6 under one data root."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import scan_dataset_profiles as base_scanner

DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
ARCHIVE_EXTENSIONS = {".zip"}
IGNORED_CODE_EXTENSIONS = {".ipynb"}
DATE_WORDS = ("date", "日期", "day", "time", "ds", "dt", "calendar", "d_")
SALES_WORDS = ("sales", "sale", "demand", "qty", "quantity", "unit_sales", "销量", "销售", "需求")
ENTITY_WORDS = (
    "store", "shop", "item", "product", "sku", "family", "dept", "cat", "id",
    "门店", "店铺", "商品", "品类",
)
PROMO_WORDS = ("promo", "promotion", "onpromotion", "促销")
HOLIDAY_WORDS = ("holiday", "event", "closed", "open", "节假日", "假日")
PRICE_WORDS = ("price", "cost", "sell_price", "价格")
WEATHER_WORDS = ("weather", "temp", "rain", "snow", "天气")
COLD_START = {"train_days": 15, "val_days": 15, "test_days": 180}
REQUIRED_DAYS = sum(COLD_START.values())


@dataclass
class DatasetPath:
    name: str
    path: Path
    kind: str
    priority: int
    alternatives: List[Path] = field(default_factory=list)


@dataclass
class DatasetDiscovery:
    data_root: Path
    datasets: Dict[str, DatasetPath]
    ignored_files: List[Path]


@dataclass
class AggregatedProfile:
    summary: Dict[str, Any]
    entity_time_span: pd.DataFrame
    entity_data_quality: pd.DataFrame
    sales_distribution: pd.DataFrame
    gap_report: pd.DataFrame
    feature_profile: pd.DataFrame
    monthly_missing: pd.DataFrame
    source_target_candidates: pd.DataFrame


def dataset_name_from_path(path: Path) -> Optional[str]:
    text = path.name
    match = re.search(r"dataset\s*([1-6])", text, flags=re.IGNORECASE)
    if not match:
        return None
    return f"Dataset{match.group(1)}"


def _candidate_kind(path: Path) -> Optional[Tuple[str, int]]:
    if path.is_dir():
        return "directory", 100
    suffix = path.suffix.lower()
    if suffix in DATA_EXTENSIONS:
        return "file", 80
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive", 40
    if suffix in IGNORED_CODE_EXTENSIONS:
        return "code", 5
    return None


def discover_dataset_paths(data_root: str | Path) -> DatasetDiscovery:
    root = Path(data_root).expanduser().resolve()
    datasets: Dict[str, DatasetPath] = {}
    ignored: List[Path] = []
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    for path in sorted(root.rglob("*"), key=lambda p: (len(p.parts), str(p))):
        if path.name.startswith("."):
            ignored.append(path)
            continue
        ds_name = dataset_name_from_path(path)
        kind_priority = _candidate_kind(path)
        if not ds_name or not kind_priority:
            if path.is_file():
                ignored.append(path)
            continue
        kind, priority = kind_priority
        if kind == "code":
            ignored.append(path)
            continue
        current = datasets.get(ds_name)
        if current is None or priority > current.priority:
            if current is not None:
                alternatives = [current.path] + current.alternatives
            else:
                alternatives = []
            datasets[ds_name] = DatasetPath(ds_name, path, kind, priority, alternatives)
        else:
            current.alternatives.append(path)

    return DatasetDiscovery(root, datasets, ignored)


def collect_dataset_files(path: str | Path) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in DATA_EXTENSIONS else []
    files: List[Path] = []
    for ext in DATA_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(
        [p for p in files if "sample_submission" not in p.name.lower()],
        key=lambda p: (str(p.parent), p.name.lower()),
    )


def _read_probe(path: Path, max_probe_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=max_probe_rows, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=max_probe_rows)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(path)
            batch = next(parquet_file.iter_batches(batch_size=max_probe_rows))
            return batch.to_pandas()
        except Exception:
            return pd.read_parquet(path).head(max_probe_rows)
    raise ValueError(f"Unsupported file type: {path}")


def _keyword_count(columns: Sequence[str], words: Sequence[str]) -> int:
    count = 0
    for col in columns:
        lower = str(col).lower()
        if any(word.lower() in lower for word in words):
            count += 1
    return count


def profile_candidate_file(path: Path, max_probe_rows: int = 1000) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "readable": False,
        "rows_probe": 0,
        "columns": [],
        "main_table_score": -1.0,
        "error": None,
    }
    try:
        df = _read_probe(path, max_probe_rows)
        columns = [str(c) for c in df.columns]
        date_hits = _keyword_count(columns, DATE_WORDS)
        sales_hits = _keyword_count(columns, SALES_WORDS)
        entity_hits = _keyword_count(columns, ENTITY_WORDS)
        row_factor = min(len(df) / max(max_probe_rows, 1), 1.0)
        train_bonus = 5 if path.name.lower().startswith(("train", "sales_train")) else 0
        aux_penalty = 8 if any(w in path.name.lower() for w in ("item", "store", "calendar", "price", "oil", "holiday", "transaction")) else 0
        score = date_hits * 25 + sales_hits * 30 + entity_hits * 10 + row_factor * 10 + train_bonus - aux_penalty
        profile.update({
            "readable": True,
            "rows_probe": int(len(df)),
            "columns": columns,
            "date_hits": date_hits,
            "sales_hits": sales_hits,
            "entity_hits": entity_hits,
            "main_table_score": float(score),
        })
    except Exception as exc:
        profile["error"] = str(exc)
    return profile


def choose_main_table(files: Sequence[Path], max_probe_rows: int = 1000) -> Tuple[Optional[Path], List[Path], Dict[str, Dict[str, Any]]]:
    profiles = {str(path): profile_candidate_file(path, max_probe_rows=max_probe_rows) for path in files}
    readable = [path for path in files if profiles[str(path)].get("readable")]
    if not readable:
        return None, list(files), profiles
    readable.sort(
        key=lambda p: (
            profiles[str(p)].get("main_table_score", -1),
            profiles[str(p)].get("size_bytes") or 0,
        ),
        reverse=True,
    )
    main = readable[0]
    auxiliaries = [p for p in files if p != main]
    return main, auxiliaries, profiles


def _extract_zip(zip_path: Path, extract_root: Path) -> Path:
    target = extract_root / zip_path.stem.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            suffix = Path(member.filename).suffix.lower()
            if suffix in DATA_EXTENSIONS and not member.is_dir():
                zf.extract(member, target)
    return target


def read_table(path: Path, max_rows: int, chunk_size: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    suffix = path.suffix.lower()
    meta = {
        "resolved_path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "read_mode": "full",
        "row_limit": None,
    }
    if suffix == ".csv":
        size = meta["size_bytes"] or 0
        if size > 500 * 1024 * 1024:
            chunks = []
            rows = 0
            for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                rows += len(chunk)
                if rows >= max_rows:
                    break
            meta["read_mode"] = "csv_chunked_capped"
            meta["row_limit"] = max_rows
            return pd.concat(chunks, ignore_index=True).head(max_rows), meta
        return pd.read_csv(path, low_memory=False), meta
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path), meta
    if suffix == ".parquet":
        size = meta["size_bytes"] or 0
        df = pd.read_parquet(path)
        if len(df) > max_rows or size > 500 * 1024 * 1024:
            meta["read_mode"] = "parquet_capped_after_read"
            meta["row_limit"] = max_rows
            df = df.head(max_rows)
        return df, meta
    raise ValueError(f"Unsupported file type: {path}")


def _m5_day_columns(columns: Sequence[str]) -> List[str]:
    return [str(c) for c in columns if re.match(r"^d_\d+$", str(c))]


def normalize_wide_day_sales(
    raw_df: pd.DataFrame,
    aux_files: Sequence[Path],
    max_entities: int = 500,
) -> Tuple[pd.DataFrame, Optional[str]]:
    day_cols = _m5_day_columns(raw_df.columns)
    if len(day_cols) < 30:
        return raw_df, None
    id_cols = [c for c in raw_df.columns if c not in day_cols]
    source = raw_df.head(max_entities).copy()
    long_df = source.melt(id_vars=id_cols, value_vars=day_cols, var_name="_m5_day", value_name="_m5_sales")
    calendar_file = next((p for p in aux_files if p.name.lower() == "calendar.csv"), None)
    if calendar_file and calendar_file.exists():
        try:
            cal = pd.read_csv(calendar_file, usecols=lambda c: c in {"d", "date"}, low_memory=False)
            if {"d", "date"}.issubset(cal.columns):
                long_df = long_df.merge(cal.rename(columns={"d": "_m5_day", "date": "_m5_date"}), on="_m5_day", how="left")
            else:
                long_df["_m5_date"] = long_df["_m5_day"]
        except Exception:
            long_df["_m5_date"] = long_df["_m5_day"]
    else:
        long_df["_m5_date"] = long_df["_m5_day"]
    note = f"M5/d_ wide table detected; melted first {len(source)} entities only to avoid oversized in-memory expansion."
    return long_df, note


def iter_table_chunks(path: Path, chunk_size: int, file_format: Optional[str] = None) -> Iterable[pd.DataFrame]:
    fmt = (file_format or path.suffix.lower().lstrip(".")).lower()
    if fmt == "csv":
        yield from pd.read_csv(path, chunksize=chunk_size, low_memory=False)
        return
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
            return
        except Exception:
            df = pd.read_parquet(path)
            for start in range(0, len(df), chunk_size):
                yield df.iloc[start:start + chunk_size].copy()
            return
    if fmt in {"xlsx", "xls"}:
        yield pd.read_excel(path)
        return
    raise ValueError(f"Unsupported chunked file format: {fmt}")


def _empty_entity_stat() -> Dict[str, Any]:
    return {
        "row_count": 0,
        "min_date": pd.NaT,
        "max_date": pd.NaT,
        "observed_days": 0,
        "valid_sales_days": 0,
        "nonzero_sales_days": 0,
        "sales_missing_count": 0,
        "zero_sales_days": 0,
        "sales_non_missing_count": 0,
        "sales_sum": 0.0,
        "sales_sumsq": 0.0,
        "sales_min": np.nan,
        "sales_max": np.nan,
    }


def _update_entity_stats(entity_stats: Dict[str, Dict[str, Any]], grouped: pd.DataFrame) -> None:
    for entity_id, row in grouped.iterrows():
        stat = entity_stats.setdefault(str(entity_id), _empty_entity_stat())
        stat["row_count"] += int(row.get("row_count", 0))
        min_date = row.get("min_date")
        max_date = row.get("max_date")
        if pd.notna(min_date) and (pd.isna(stat["min_date"]) or min_date < stat["min_date"]):
            stat["min_date"] = min_date
        if pd.notna(max_date) and (pd.isna(stat["max_date"]) or max_date > stat["max_date"]):
            stat["max_date"] = max_date
        for key in (
            "observed_days", "valid_sales_days", "nonzero_sales_days", "sales_missing_count",
            "zero_sales_days", "sales_non_missing_count",
        ):
            stat[key] += int(row.get(key, 0))
        stat["sales_sum"] += float(row.get("sales_sum", 0.0))
        stat["sales_sumsq"] += float(row.get("sales_sumsq", 0.0))
        sales_min = row.get("sales_min")
        sales_max = row.get("sales_max")
        if pd.notna(sales_min) and (pd.isna(stat["sales_min"]) or sales_min < stat["sales_min"]):
            stat["sales_min"] = float(sales_min)
        if pd.notna(sales_max) and (pd.isna(stat["sales_max"]) or sales_max > stat["sales_max"]):
            stat["sales_max"] = float(sales_max)


def _reduce_long_partials(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0)
    sum_cols = [
        "row_count", "observed_days", "valid_sales_days", "nonzero_sales_days",
        "sales_missing_count", "zero_sales_days", "sales_non_missing_count",
        "sales_sum", "sales_sumsq",
    ]
    agg_map: Dict[str, Any] = {col: "sum" for col in sum_cols if col in combined.columns}
    agg_map.update({
        "min_date": "min",
        "max_date": "max",
        "sales_min": "min",
        "sales_max": "max",
    })
    return combined.groupby(level=0, sort=False).agg(agg_map)


def _stats_dict_from_reduced(reduced: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    if reduced.empty:
        return stats
    for entity_id, row in reduced.iterrows():
        stat = _empty_entity_stat()
        for key in stat:
            if key in row:
                stat[key] = row[key]
        stats[str(entity_id)] = stat
    return stats


def _entity_frames_from_stats(entity_stats: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    span_rows = []
    quality_rows = []
    sales_rows = []
    gap_rows = []
    for entity_id, stat in entity_stats.items():
        min_date = stat["min_date"]
        max_date = stat["max_date"]
        total_days = int((max_date - min_date).days + 1) if pd.notna(min_date) and pd.notna(max_date) else 0
        sales_count = int(stat["sales_non_missing_count"])
        mean = stat["sales_sum"] / sales_count if sales_count else np.nan
        if sales_count > 1:
            variance = max((stat["sales_sumsq"] - (stat["sales_sum"] ** 2 / sales_count)) / (sales_count - 1), 0.0)
            std = math.sqrt(variance)
        else:
            std = np.nan
        span_rows.append({
            "entity_id": entity_id,
            "min_date": min_date,
            "max_date": max_date,
            "total_calendar_days": total_days,
            "observed_days": int(stat["observed_days"]),
            "valid_sales_days": int(stat["valid_sales_days"]),
            "nonzero_sales_days": int(stat["nonzero_sales_days"]),
            "meets_210_days": total_days >= REQUIRED_DAYS and int(stat["valid_sales_days"]) >= REQUIRED_DAYS,
        })
        quality_rows.append({
            "entity_id": entity_id,
            "row_count": int(stat["row_count"]),
            "missing_ratio": stat["sales_missing_count"] / max(int(stat["row_count"]), 1),
            "sales_missing_ratio": stat["sales_missing_count"] / max(int(stat["row_count"]), 1),
            "zero_sales_days": int(stat["zero_sales_days"]),
            "zero_sales_ratio": stat["zero_sales_days"] / max(sales_count, 1),
            "nonzero_sales_days": int(stat["nonzero_sales_days"]),
            "nonzero_sales_ratio": stat["nonzero_sales_days"] / max(sales_count, 1),
        })
        sales_rows.append({
            "entity_id": entity_id,
            "count": sales_count,
            "mean": mean,
            "std": std,
            "median": np.nan,
            "min": stat["sales_min"],
            "max": stat["sales_max"],
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "skewness": np.nan,
            "coefficient_of_variation": std / mean if mean and not pd.isna(std) else np.nan,
            "outlier_days_count": np.nan,
            "outlier_ratio": np.nan,
            "max_to_mean_ratio": stat["sales_max"] / mean if mean and pd.notna(stat["sales_max"]) else np.nan,
        })
        missing_calendar_days = max(total_days - int(stat["observed_days"]), 0)
        gap_rows.append({
            "entity_id": entity_id,
            "gap_count": np.nan,
            "max_gap_days": np.nan,
            "total_missing_calendar_days": missing_calendar_days,
            "longest_gap_start": None,
            "longest_gap_end": None,
        })
    ent_span = pd.DataFrame(span_rows)
    quality = pd.DataFrame(quality_rows)
    sales = pd.DataFrame(sales_rows)
    gaps = pd.DataFrame(gap_rows)
    if not ent_span.empty:
        ent_span = ent_span.sort_values(["meets_210_days", "valid_sales_days", "total_calendar_days"], ascending=[False, False, False])
    if not quality.empty:
        quality = quality.sort_values("sales_missing_ratio", ascending=False)
    if not sales.empty:
        sales = sales.sort_values("mean", ascending=False)
    return ent_span, quality, sales, gaps


def _source_target_candidates_from_entity_frames(
    ent_span: pd.DataFrame,
    sales_dist: pd.DataFrame,
) -> pd.DataFrame:
    if ent_span.empty:
        return pd.DataFrame()
    merged = ent_span.merge(
        sales_dist[["entity_id", "mean", "std", "coefficient_of_variation"]],
        on="entity_id",
        how="left",
    ) if not sales_dist.empty else ent_span.copy()
    merged["quality_score"] = (
        np.minimum(merged["valid_sales_days"] / REQUIRED_DAYS, 1.0) * 0.50
        + np.minimum(merged["total_calendar_days"] / REQUIRED_DAYS, 1.0) * 0.30
        + (1.0 - merged.get("coefficient_of_variation", pd.Series(1, index=merged.index)).fillna(1).clip(0, 5) / 5) * 0.20
    )
    merged = merged.sort_values("quality_score", ascending=False).copy()
    merged["candidate_role"] = "candidate_pool"
    if len(merged):
        merged.iloc[0, merged.columns.get_loc("candidate_role")] = "candidate_target"
    if len(merged) > 1:
        merged.iloc[1:min(len(merged), 21), merged.columns.get_loc("candidate_role")] = "candidate_source"
    merged["observed_train_days"] = COLD_START["train_days"]
    merged["observed_val_days"] = COLD_START["val_days"]
    merged["test_days"] = COLD_START["test_days"]
    merged["total_required_days"] = REQUIRED_DAYS
    merged["total_history_days"] = merged["total_calendar_days"]
    merged["cold_start_window_ratio"] = REQUIRED_DAYS / merged["total_history_days"].replace(0, np.nan)
    merged["observed_ratio"] = 30 / merged["total_history_days"].replace(0, np.nan)
    merged["test_ratio"] = 180 / merged["total_history_days"].replace(0, np.nan)
    return merged.head(50)


def _build_aggregated_summary(
    dataset_name: str,
    data_path: Path,
    main_file: Path,
    files: Sequence[Path],
    aux_files: Sequence[Path],
    notebooks: Sequence[Path],
    inference: Dict[str, Any],
    ent_span: pd.DataFrame,
    quality: pd.DataFrame,
    sales_dist: pd.DataFrame,
    features: pd.DataFrame,
    scan_coverage: str,
    read_mode: str,
    rows_seen: int,
    total_cells: int,
    missing_cells: int,
    entity_cols: Sequence[str],
    warnings: Sequence[str],
    zero_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    global_min = ent_span["min_date"].min() if not ent_span.empty else pd.NaT
    global_max = ent_span["max_date"].max() if not ent_span.empty else pd.NaT
    sales_nonmissing = int(sales_dist["count"].sum()) if not sales_dist.empty else 0
    zero_sales = int(quality["zero_sales_days"].sum()) if not quality.empty else 0
    span_dist = describe_span_distribution(ent_span, "total_calendar_days")
    valid_dist = describe_span_distribution(ent_span, "valid_sales_days")
    stores = []
    items = []
    for col in entity_cols:
        lower = str(col).lower()
        if "store" in lower or "shop" in lower:
            stores.append(col)
        if any(w in lower for w in ("item", "product", "sku")):
            items.append(col)
    return {
        "dataset": dataset_name,
        "data_path": str(data_path),
        "main_table_file": str(main_file),
        "file_count": len(files),
        "auxiliary_files": [str(p) for p in aux_files],
        "notebooks": [str(p) for p in notebooks],
        "main_table_rows_scanned": int(rows_seen),
        "main_table_cols": None,
        "read_mode": read_mode,
        "scan_coverage": scan_coverage,
        "date_col": inference.get("inferred_date_col"),
        "sales_col": inference.get("inferred_sales_col"),
        "entity_cols": list(entity_cols),
        "time_granularity": "daily",
        "global_min_date": str(global_min.date()) if pd.notna(global_min) else None,
        "global_max_date": str(global_max.date()) if pd.notna(global_max) else None,
        "global_total_days": int((global_max - global_min).days + 1) if pd.notna(global_min) and pd.notna(global_max) else 0,
        "entity_count": int(ent_span["entity_id"].nunique()) if not ent_span.empty else 0,
        "store_count": None,
        "item_count": None,
        "sku_count": None,
        "main_entity_grain": " + ".join(entity_cols or ["global_entity"]),
        "span_distribution": span_dist,
        "valid_sales_days_distribution": valid_dist,
        "entities_meeting_210_days": int(ent_span["meets_210_days"].sum()) if not ent_span.empty else 0,
        "entities_not_meeting_210_days": int((~ent_span["meets_210_days"]).sum()) if not ent_span.empty else 0,
        "missing_ratio": missing_cells / max(total_cells, 1),
        "zero_sales_ratio": zero_sales / max(sales_nonmissing, 1),
        "feature_count": int(len(features)),
        "rfe_candidate_count": int(features["recommended_for_rfe"].sum()) if not features.empty and "recommended_for_rfe" in features else 0,
        "source_target_method": "auto_candidate_by_full_aggregated_profile",
        "source_target_reason": "候选划分基于完整分块聚合后的实体历史长度、有效销售天数和销售分布稳定性。",
        "candidate_targets": ent_span.head(10)["entity_id"].tolist() if not ent_span.empty else [],
        "candidate_sources": ent_span.iloc[1:11]["entity_id"].tolist() if len(ent_span) > 1 else [],
        "warnings": list(warnings),
        "zero_context": zero_context or {"note": "分块聚合扫描未可靠区分真实零销售与缺货/关店，只统计 observed zero sales。"},
        "shortest_entities": ent_span.sort_values("total_calendar_days").head(10)["entity_id"].tolist() if not ent_span.empty else [],
        "longest_entities": ent_span.sort_values("total_calendar_days", ascending=False).head(10)["entity_id"].tolist() if not ent_span.empty else [],
    }


def aggregate_long_table_full_scan(
    path: Path,
    inference: Dict[str, Any],
    chunk_size: int,
    file_format: Optional[str] = None,
) -> AggregatedProfile:
    date_col = inference.get("inferred_date_col")
    sales_col = inference.get("inferred_sales_col")
    entity_cols = inference.get("inferred_entity_cols") or []
    if not date_col or not sales_col:
        raise ValueError("Long-table full scan requires inferred date and sales columns.")

    reduced_state = pd.DataFrame()
    partial_frames: List[pd.DataFrame] = []
    rows_seen = 0
    total_cells = 0
    missing_cells = 0
    feature_probe: Optional[pd.DataFrame] = None

    for chunk_idx, chunk in enumerate(iter_table_chunks(path, chunk_size=chunk_size, file_format=file_format)):
        if feature_probe is None:
            feature_probe = chunk.head(5000).copy()
        rows_seen += len(chunk)
        total_cells += int(chunk.shape[0] * chunk.shape[1])
        missing_cells += int(chunk.isna().sum().sum())
        work = pd.DataFrame(index=chunk.index)
        work["_date"] = pd.to_datetime(chunk[date_col], errors="coerce").dt.normalize()
        work["_sales"] = pd.to_numeric(chunk[sales_col], errors="coerce")
        work["_entity_id"] = base_scanner.build_entity_id_series(chunk, entity_cols)
        work["_date_present"] = work["_date"].notna().astype("int64")
        work["_valid_sales_day"] = (work["_date"].notna() & work["_sales"].notna()).astype("int64")
        work["_nonzero_sales_day"] = (work["_date"].notna() & work["_sales"].notna() & (work["_sales"] != 0)).astype("int64")
        work["_sales_missing"] = work["_sales"].isna().astype("int64")
        work["_zero_sales"] = (work["_sales"].notna() & (work["_sales"] == 0)).astype("int64")
        work["_sales_non_missing"] = work["_sales"].notna().astype("int64")
        work["_sales_sum"] = work["_sales"].fillna(0.0)
        work["_sales_sumsq"] = work["_sales"].fillna(0.0) ** 2
        grouped = work.groupby("_entity_id", sort=False).agg(
            row_count=("_sales", "size"),
            min_date=("_date", "min"),
            max_date=("_date", "max"),
            observed_days=("_date_present", "sum"),
            valid_sales_days=("_valid_sales_day", "sum"),
            nonzero_sales_days=("_nonzero_sales_day", "sum"),
            sales_missing_count=("_sales_missing", "sum"),
            zero_sales_days=("_zero_sales", "sum"),
            sales_non_missing_count=("_sales_non_missing", "sum"),
            sales_sum=("_sales_sum", "sum"),
            sales_sumsq=("_sales_sumsq", "sum"),
            sales_min=("_sales", "min"),
            sales_max=("_sales", "max"),
        )
        partial_frames.append(grouped)
        if len(partial_frames) >= 5:
            reduced = _reduce_long_partials(partial_frames)
            if reduced_state.empty:
                reduced_state = reduced
            else:
                reduced_state = _reduce_long_partials([reduced_state, reduced])
            partial_frames = []

    if partial_frames:
        reduced = _reduce_long_partials(partial_frames)
        reduced_state = reduced if reduced_state.empty else _reduce_long_partials([reduced_state, reduced])
    entity_stats = _stats_dict_from_reduced(reduced_state)
    ent_span, quality, sales_dist, gaps = _entity_frames_from_stats(entity_stats)
    if feature_probe is not None:
        probe_work = base_scanner.apply_inferred_schema(feature_probe, inference, base_scanner.ScanLogger())
        features = feature_profile(probe_work, inference)
    else:
        features = pd.DataFrame()
    candidates = _source_target_candidates_from_entity_frames(ent_span, sales_dist)
    summary = _build_aggregated_summary(
        dataset_name="",
        data_path=path,
        main_file=path,
        files=[path],
        aux_files=[],
        notebooks=[],
        inference=inference,
        ent_span=ent_span,
        quality=quality,
        sales_dist=sales_dist,
        features=features,
        scan_coverage="CHUNKED_FULL_SCAN",
        read_mode=f"{file_format or path.suffix.lower().lstrip('.')}_chunked_full",
        rows_seen=rows_seen,
        total_cells=total_cells,
        missing_cells=missing_cells,
        entity_cols=entity_cols,
        warnings=[],
    )
    return AggregatedProfile(summary, ent_span, quality, sales_dist, gaps, features, pd.DataFrame(), candidates)


def aggregate_m5_wide_full_scan(
    sales_path: Path,
    calendar_path: Optional[Path],
    chunk_size: int,
) -> AggregatedProfile:
    header = pd.read_csv(sales_path, nrows=0)
    day_cols = _m5_day_columns(header.columns)
    if not day_cols:
        raise ValueError(f"No d_ day columns found in {sales_path}")
    entity_cols = [c for c in ["store_id", "item_id"] if c in header.columns]
    if not entity_cols:
        entity_cols = ["id"] if "id" in header.columns else [c for c in header.columns if c not in day_cols][:1]
    metadata_cols = [c for c in header.columns if c not in day_cols]

    if calendar_path and calendar_path.exists():
        calendar = pd.read_csv(calendar_path, usecols=lambda c: c in {"d", "date"}, low_memory=False)
        calendar_map = dict(zip(calendar["d"].astype(str), pd.to_datetime(calendar["date"], errors="coerce")))
    else:
        calendar_map = {}
    day_dates = pd.Series([calendar_map.get(col, pd.NaT) for col in day_cols], index=day_cols)
    if day_dates.isna().any():
        fallback_start = pd.Timestamp("1970-01-01")
        day_dates = pd.Series(
            [calendar_map.get(col, fallback_start + pd.Timedelta(days=int(col.split("_")[1]) - 1)) for col in day_cols],
            index=day_cols,
        )

    entity_stats: Dict[str, Dict[str, Any]] = {}
    rows_seen = 0
    total_cells = 0
    missing_cells = 0
    feature_rows = []
    use_chunk_size = min(chunk_size, 5000)
    for chunk in pd.read_csv(sales_path, chunksize=use_chunk_size, low_memory=False):
        rows_seen += len(chunk)
        total_cells += int(chunk.shape[0] * chunk.shape[1])
        missing_cells += int(chunk.isna().sum().sum())
        entity_ids = base_scanner.build_entity_id_series(chunk, entity_cols)
        sales_block = chunk[day_cols].apply(pd.to_numeric, errors="coerce")
        values = sales_block.to_numpy(dtype=float)
        valid = ~np.isnan(values)
        sales_filled = np.nan_to_num(values, nan=0.0)
        valid_counts = valid.sum(axis=1)
        zero_counts = ((values == 0) & valid).sum(axis=1)
        nonzero_counts = ((values != 0) & valid).sum(axis=1)
        sales_sum = sales_filled.sum(axis=1)
        sales_sumsq = (sales_filled ** 2).sum(axis=1)
        sales_min = np.nanmin(np.where(valid, values, np.nan), axis=1)
        sales_max = np.nanmax(np.where(valid, values, np.nan), axis=1)
        for idx, entity_id in enumerate(entity_ids.astype(str).tolist()):
            stat = entity_stats.setdefault(entity_id, _empty_entity_stat())
            stat["row_count"] += int(len(day_cols))
            stat["min_date"] = day_dates.min()
            stat["max_date"] = day_dates.max()
            stat["observed_days"] += int(len(day_cols))
            stat["valid_sales_days"] += int(valid_counts[idx])
            stat["nonzero_sales_days"] += int(nonzero_counts[idx])
            stat["sales_missing_count"] += int(len(day_cols) - valid_counts[idx])
            stat["zero_sales_days"] += int(zero_counts[idx])
            stat["sales_non_missing_count"] += int(valid_counts[idx])
            stat["sales_sum"] += float(sales_sum[idx])
            stat["sales_sumsq"] += float(sales_sumsq[idx])
            if pd.notna(sales_min[idx]) and (pd.isna(stat["sales_min"]) or sales_min[idx] < stat["sales_min"]):
                stat["sales_min"] = float(sales_min[idx])
            if pd.notna(sales_max[idx]) and (pd.isna(stat["sales_max"]) or sales_max[idx] > stat["sales_max"]):
                stat["sales_max"] = float(sales_max[idx])

    ent_span, quality, sales_dist, gaps = _entity_frames_from_stats(entity_stats)
    for col in metadata_cols:
        lower = str(col).lower()
        role = "entity id column" if col in {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"} else "categorical feature"
        feature_rows.append({
            "feature_name": col,
            "role": role,
            "dtype": "metadata",
            "missing_count": np.nan,
            "missing_ratio": np.nan,
            "unique_count": np.nan,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "p1": None,
            "p99": None,
            "constant_feature": False,
            "leakage_risk": False,
            "recommend_normalization": False,
            "recommended_for_rfe": False,
            "shared_or_entity_specific": "entity_specific" if "id" in lower else "global_or_row_level",
        })
    feature_rows.append({
        "feature_name": "d_1...d_n",
        "role": "target column",
        "dtype": "wide daily sales columns",
        "missing_count": np.nan,
        "missing_ratio": missing_cells / max(total_cells, 1),
        "unique_count": len(day_cols),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "median": None,
        "p1": None,
        "p99": None,
        "constant_feature": False,
        "leakage_risk": False,
        "recommend_normalization": False,
        "recommended_for_rfe": False,
        "shared_or_entity_specific": "entity_specific",
    })
    features = pd.DataFrame(feature_rows)
    candidates = _source_target_candidates_from_entity_frames(ent_span, sales_dist)
    inference = {
        "inferred_date_col": "calendar.csv:date mapped from d_1...d_n",
        "inferred_sales_col": "d_1...d_n",
        "inferred_entity_cols": entity_cols,
    }
    summary = _build_aggregated_summary(
        dataset_name="",
        data_path=sales_path.parent,
        main_file=sales_path,
        files=[sales_path],
        aux_files=[calendar_path] if calendar_path else [],
        notebooks=[],
        inference=inference,
        ent_span=ent_span,
        quality=quality,
        sales_dist=sales_dist,
        features=features,
        scan_coverage="CHUNKED_FULL_SCAN",
        read_mode="m5_wide_chunked_full",
        rows_seen=rows_seen,
        total_cells=total_cells,
        missing_cells=missing_cells,
        entity_cols=entity_cols,
        warnings=[],
        zero_context={"note": "M5 宽表完整扫描，零销售按 d_ 列 observed zero sales 统计；未用库存字段区分缺货。"},
    )
    summary["main_table_cols"] = len(metadata_cols) + len(day_cols)
    return AggregatedProfile(summary, ent_span, quality, sales_dist, gaps, features, pd.DataFrame(), candidates)


def _safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        if math.isinf(float(value)) or math.isnan(float(value)):
            return None
        return float(value)
    except Exception:
        return None


def _quantiles(series: pd.Series, qs: Iterable[float]) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {f"p{int(q * 100)}": np.nan for q in qs}
    return {f"p{int(q * 100)}": float(clean.quantile(q)) for q in qs}


def infer_granularity(work_df: pd.DataFrame) -> str:
    if "_date" not in work_df or work_df["_date"].dropna().empty:
        return "unknown"
    dates = work_df["_date"].dropna().dt.normalize().drop_duplicates().sort_values()
    diffs = dates.diff().dt.days.dropna()
    if diffs.empty:
        return "unknown"
    median = diffs.median()
    regular_ratio = float((diffs == median).mean())
    if regular_ratio < 0.8:
        return "irregular"
    if median == 1:
        return "daily"
    if 6 <= median <= 8:
        return "weekly"
    if 28 <= median <= 31:
        return "monthly"
    return "irregular"


def entity_time_span(work_df: pd.DataFrame) -> pd.DataFrame:
    if not {"_entity_id", "_date"}.issubset(work_df.columns):
        return pd.DataFrame(columns=["entity_id"])
    rows = []
    for entity_id, group in work_df.groupby("_entity_id", sort=False):
        dates = group["_date"].dropna().dt.normalize()
        sales = group["_sales"] if "_sales" in group else pd.Series(dtype=float)
        rows.append({
            "entity_id": entity_id,
            "min_date": dates.min() if not dates.empty else pd.NaT,
            "max_date": dates.max() if not dates.empty else pd.NaT,
            "total_calendar_days": int((dates.max() - dates.min()).days + 1) if not dates.empty else 0,
            "observed_days": int(dates.nunique()) if not dates.empty else 0,
            "valid_sales_days": int(group.loc[sales.notna(), "_date"].dt.normalize().nunique()) if "_sales" in group else 0,
            "nonzero_sales_days": int(group.loc[sales.fillna(0) != 0, "_date"].dt.normalize().nunique()) if "_sales" in group else 0,
        })
    df = pd.DataFrame(rows)
    df["meets_210_days"] = (df["total_calendar_days"] >= REQUIRED_DAYS) & (df["valid_sales_days"] >= REQUIRED_DAYS)
    return df.sort_values(["meets_210_days", "valid_sales_days", "total_calendar_days"], ascending=[False, False, False])


def entity_gap_report(work_df: pd.DataFrame) -> pd.DataFrame:
    if not {"_entity_id", "_date"}.issubset(work_df.columns):
        return pd.DataFrame(columns=["entity_id"])
    rows = []
    for entity_id, group in work_df.groupby("_entity_id", sort=False):
        dates = group["_date"].dropna().dt.normalize().drop_duplicates().sort_values()
        if len(dates) < 2:
            rows.append({
                "entity_id": entity_id,
                "gap_count": 0,
                "max_gap_days": 0,
                "total_missing_calendar_days": 0,
                "longest_gap_start": None,
                "longest_gap_end": None,
            })
            continue
        diffs = dates.diff().dt.days.fillna(1).astype(int)
        gaps = diffs[diffs > 1]
        max_gap = int(gaps.max() - 1) if not gaps.empty else 0
        if max_gap:
            idx = gaps.idxmax()
            gap_end = dates.loc[idx]
            prev_pos = dates.index.get_loc(idx) - 1
            gap_start = dates.iloc[prev_pos]
        else:
            gap_start = gap_end = None
        rows.append({
            "entity_id": entity_id,
            "gap_count": int((diffs > 1).sum()),
            "max_gap_days": max_gap,
            "total_missing_calendar_days": int((diffs - 1).clip(lower=0).sum()),
            "longest_gap_start": gap_start,
            "longest_gap_end": gap_end,
        })
    return pd.DataFrame(rows).sort_values("max_gap_days", ascending=False)


def entity_data_quality(work_df: pd.DataFrame) -> pd.DataFrame:
    if "_entity_id" not in work_df:
        return pd.DataFrame(columns=["entity_id"])
    rows = []
    for entity_id, group in work_df.groupby("_entity_id", sort=False):
        sales = group["_sales"] if "_sales" in group else pd.Series(dtype=float)
        rows.append({
            "entity_id": entity_id,
            "row_count": int(len(group)),
            "missing_ratio": float(group.isna().mean().mean()),
            "sales_missing_ratio": float(sales.isna().mean()) if len(sales) else np.nan,
            "zero_sales_days": int((sales == 0).sum()) if len(sales) else 0,
            "zero_sales_ratio": float((sales == 0).sum() / max(int(sales.notna().sum()), 1)) if len(sales) else np.nan,
            "nonzero_sales_days": int((sales > 0).sum()) if len(sales) else 0,
            "nonzero_sales_ratio": float((sales > 0).sum() / max(int(sales.notna().sum()), 1)) if len(sales) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("missing_ratio", ascending=False)


def feature_profile(work_df: pd.DataFrame, inference: Dict[str, Any]) -> pd.DataFrame:
    date_col = inference.get("inferred_date_col")
    sales_col = inference.get("inferred_sales_col")
    entity_cols = set(inference.get("inferred_entity_cols") or [])
    skip = {"_date", "_sales", "_entity_id"}
    rows = []
    for col in work_df.columns:
        if col in skip:
            continue
        series = work_df[col]
        lower = str(col).lower()
        if col == sales_col:
            role = "target column"
        elif col == date_col:
            role = "date column"
        elif col in entity_cols:
            role = "entity id column"
        elif any(w in lower for w in PROMO_WORDS):
            role = "promo feature"
        elif any(w in lower for w in HOLIDAY_WORDS):
            role = "holiday/store status feature"
        elif any(w in lower for w in PRICE_WORDS):
            role = "price feature"
        elif any(w in lower for w in WEATHER_WORDS):
            role = "weather feature"
        elif pd.api.types.is_bool_dtype(series) or (
            not base_scanner._contains_array_like_values(series)
            and set(series.dropna().unique()[:3]).issubset({0, 1, True, False})
        ):
            role = "boolean feature"
        elif pd.api.types.is_numeric_dtype(series):
            role = "numeric feature"
        else:
            role = "categorical feature"
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric.notna().mean() > 0.8
        unique_count = base_scanner._safe_nunique(series)
        constant = unique_count >= 0 and unique_count <= 1
        leakage = col != sales_col and any(w in lower for w in ("target", "label", "future", "lead", "next"))
        rfe_candidate = (
            role not in {"target column", "date column", "entity id column"}
            and float(series.isna().mean()) <= 0.5
            and not constant
            and not leakage
        )
        rows.append({
            "feature_name": col,
            "role": role,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": float(series.isna().mean()),
            "unique_count": unique_count,
            "min": _safe_float(numeric.min()) if is_numeric else None,
            "max": _safe_float(numeric.max()) if is_numeric else None,
            "mean": _safe_float(numeric.mean()) if is_numeric else None,
            "std": _safe_float(numeric.std()) if is_numeric else None,
            "median": _safe_float(numeric.median()) if is_numeric else None,
            "p1": _safe_float(numeric.quantile(0.01)) if is_numeric else None,
            "p99": _safe_float(numeric.quantile(0.99)) if is_numeric else None,
            "constant_feature": constant,
            "leakage_risk": leakage,
            "recommend_normalization": bool(is_numeric and role not in {"target column", "date column", "entity id column"}),
            "recommended_for_rfe": bool(rfe_candidate),
            "shared_or_entity_specific": "entity_specific" if col in entity_cols else "global_or_row_level",
        })
    return pd.DataFrame(rows).sort_values(["recommended_for_rfe", "missing_ratio"], ascending=[False, True])


def sales_distribution_by_entity(work_df: pd.DataFrame) -> pd.DataFrame:
    if not {"_entity_id", "_sales"}.issubset(work_df.columns):
        return pd.DataFrame(columns=["entity_id"])
    rows = []
    for entity_id, group in work_df.groupby("_entity_id", sort=False):
        sales = pd.to_numeric(group["_sales"], errors="coerce").dropna()
        if sales.empty:
            continue
        mean = float(sales.mean())
        std = float(sales.std())
        rows.append({
            "entity_id": entity_id,
            "count": int(len(sales)),
            "mean": mean,
            "std": std,
            "median": float(sales.median()),
            "min": float(sales.min()),
            "max": float(sales.max()),
            "p25": float(sales.quantile(0.25)),
            "p75": float(sales.quantile(0.75)),
            "p90": float(sales.quantile(0.90)),
            "p95": float(sales.quantile(0.95)),
            "p99": float(sales.quantile(0.99)),
            "skewness": _safe_float(sales.skew()),
            "coefficient_of_variation": float(std / mean) if mean else np.nan,
            "outlier_days_count": int((sales > mean * 10).sum()) if mean else 0,
            "outlier_ratio": float((sales > mean * 10).mean()) if mean else 0.0,
            "max_to_mean_ratio": float(sales.max() / mean) if mean else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def source_target_candidate_report(
    work_df: pd.DataFrame,
    inference: Dict[str, Any],
    entity_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    st_info: Dict[str, Any],
) -> pd.DataFrame:
    if entity_df.empty:
        return pd.DataFrame()
    merged = entity_df.merge(
        sales_df[["entity_id", "mean", "std", "coefficient_of_variation"]], on="entity_id", how="left"
    ) if not sales_df.empty else entity_df.copy()
    merged["quality_score"] = (
        np.minimum(merged["valid_sales_days"] / REQUIRED_DAYS, 1.0) * 0.45
        + np.minimum(merged["total_calendar_days"] / REQUIRED_DAYS, 1.0) * 0.25
        + (1.0 - merged.get("coefficient_of_variation", pd.Series(1, index=merged.index)).fillna(1).clip(0, 5) / 5) * 0.15
        + merged["meets_210_days"].astype(float) * 0.15
    )
    merged = merged.sort_values("quality_score", ascending=False).copy()
    targets = set(st_info.get("inferred_target_entities") or [])
    sources = set(st_info.get("inferred_source_entities") or [])
    merged["candidate_role"] = np.where(
        merged["entity_id"].isin(targets),
        "existing_or_inferred_target",
        np.where(merged["entity_id"].isin(sources), "existing_or_inferred_source", "candidate_pool"),
    )
    merged["observed_train_days"] = COLD_START["train_days"]
    merged["observed_val_days"] = COLD_START["val_days"]
    merged["test_days"] = COLD_START["test_days"]
    merged["total_required_days"] = REQUIRED_DAYS
    merged["total_history_days"] = merged["total_calendar_days"]
    merged["cold_start_window_ratio"] = REQUIRED_DAYS / merged["total_history_days"].replace(0, np.nan)
    merged["observed_ratio"] = 30 / merged["total_history_days"].replace(0, np.nan)
    merged["test_ratio"] = 180 / merged["total_history_days"].replace(0, np.nan)
    return merged.head(50)


def similarity_summary(work_df: pd.DataFrame, sales_df: pd.DataFrame, max_entities: int = 40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not {"_entity_id", "_date", "_sales"}.issubset(work_df.columns) or sales_df.empty:
        empty = pd.DataFrame(columns=["entity_a", "entity_b"])
        return empty, empty
    top_entities = sales_df.sort_values("count", ascending=False)["entity_id"].head(max_entities).tolist()
    pivot = work_df[work_df["_entity_id"].isin(top_entities)].pivot_table(
        index=work_df["_date"].dt.normalize(),
        columns="_entity_id",
        values="_sales",
        aggfunc="mean",
    )
    rows = []
    columns = list(pivot.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            pair = pivot[[a, b]].dropna()
            if pair.empty:
                corr = np.nan
                mean_diff = np.nan
                std_diff = np.nan
            else:
                corr = pair[a].corr(pair[b]) if len(pair) > 1 else np.nan
                mean_diff = abs(float(pair[a].mean() - pair[b].mean()))
                std_diff = abs(float(pair[a].std() - pair[b].std()))
            rows.append({
                "entity_a": a,
                "entity_b": b,
                "mean_difference": mean_diff,
                "std_difference": std_diff,
                "correlation_aligned_by_date": corr,
                "similarity_score": (0 if pd.isna(corr) else corr) - math.log1p(0 if pd.isna(mean_diff) else mean_diff),
            })
    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df, pair_df
    return (
        pair_df.sort_values("similarity_score", ascending=False).head(10),
        pair_df.sort_values("similarity_score", ascending=True).head(10),
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_无数据_"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[col]):
            view[col] = view[col].dt.strftime("%Y-%m-%d")
    headers = list(view.columns)
    lines = ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def describe_span_distribution(entity_df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if entity_df.empty or col not in entity_df:
        return {}
    s = pd.to_numeric(entity_df[col], errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def detect_zero_context(work_df: pd.DataFrame) -> Dict[str, Any]:
    columns = {str(c).lower(): c for c in work_df.columns}
    open_col = next((columns[c] for c in columns if c in {"open", "is_open"}), None)
    closed_col = next((columns[c] for c in columns if "closed" in c), None)
    stock_col = next((columns[c] for c in columns if any(w in c for w in ("stockout", "inventory", "available"))), None)
    promo_col = next((columns[c] for c in columns if any(w in c for w in ("promo", "promotion", "onpromotion"))), None)
    tx_col = next((columns[c] for c in columns if "transaction" in c), None)
    sales = work_df["_sales"] if "_sales" in work_df else pd.Series(dtype=float)
    zero_mask = sales == 0
    result = {
        "observed_zero_sales": int(zero_mask.sum()) if len(sales) else 0,
        "closed_or_not_open_days": None,
        "possible_stockout_days": None,
        "note": "无法可靠区分真实零销售与缺货/关店，只能统计 observed zero sales。",
    }
    if open_col or closed_col or stock_col or promo_col or tx_col:
        closed_mask = pd.Series(False, index=work_df.index)
        if open_col:
            closed_mask |= pd.to_numeric(work_df[open_col], errors="coerce").fillna(1) == 0
        if closed_col:
            closed_mask |= pd.to_numeric(work_df[closed_col], errors="coerce").fillna(0) == 1
        stock_mask = pd.Series(False, index=work_df.index)
        if stock_col:
            stock_mask |= pd.to_numeric(work_df[stock_col], errors="coerce").fillna(1) <= 0
        if promo_col:
            stock_mask |= zero_mask & (pd.to_numeric(work_df[promo_col], errors="coerce").fillna(0) > 0)
        if tx_col:
            stock_mask |= zero_mask & (pd.to_numeric(work_df[tx_col], errors="coerce").fillna(0) > 0)
        result.update({
            "closed_or_not_open_days": int((zero_mask & closed_mask).sum()),
            "possible_stockout_days": int((zero_mask & stock_mask).sum()),
            "note": "存在辅助字段，已按 open/closed/stock/promo/transactions 启发式区分。",
        })
    return result


def month_missing_report(work_df: pd.DataFrame) -> pd.DataFrame:
    if "_date" not in work_df:
        return pd.DataFrame(columns=["month", "missing_ratio"])
    tmp = work_df.copy()
    tmp["_month"] = tmp["_date"].dt.to_period("M").astype(str)
    rows = []
    for month, group in tmp.groupby("_month"):
        rows.append({"month": month, "missing_ratio": float(group.isna().mean().mean()), "row_count": int(len(group))})
    return pd.DataFrame(rows)


def scan_one_dataset(
    dataset: DatasetPath,
    output_root: Path,
    max_rows: int,
    chunk_size: int,
    max_probe_rows: int,
) -> Dict[str, Any]:
    ds_dir = output_root / dataset.name
    ds_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    warnings = []

    def log(message: str) -> None:
        line = f"[{dataset.name}] {message}"
        logs.append(line)
        print(line, flush=True)

    def warn(message: str) -> None:
        warnings.append(message)
        log(f"WARNING: {message}")

    result: Dict[str, Any] = {"Dataset": dataset.name, "scan_status": "FAILED", "data_path": str(dataset.path)}
    try:
        log("loading files...")
        source_path = dataset.path
        if dataset.kind == "archive":
            source_path = _extract_zip(dataset.path, output_root / "_extracted_zips")
            warn(f"只发现压缩包，已解压备用扫描：{dataset.path}")
        files = collect_dataset_files(source_path)
        notebooks = list(source_path.rglob("*.ipynb")) if source_path.is_dir() else []
        if not files:
            raise FileNotFoundError(f"No supported data files found under {source_path}")
        main_file, aux_files, file_profiles = choose_main_table(files, max_probe_rows=max_probe_rows)
        if main_file is None:
            raise RuntimeError("No readable main table candidate found.")
        log(f"detected main table: {main_file}")

        probe_df = _read_probe(main_file, max_probe_rows)
        m5_days = _m5_day_columns(probe_df.columns)
        if dataset.name == "Dataset6" and len(m5_days) >= 30:
            calendar_file = next((p for p in aux_files if p.name.lower() == "calendar.csv"), None)
            log("detected M5 wide table; running chunked full aggregation...")
            profile = aggregate_m5_wide_full_scan(main_file, calendar_file, chunk_size=chunk_size)
            warnings.extend(profile.summary.get("warnings") or [])
            summary = profile.summary
            summary.update({
                "dataset": dataset.name,
                "data_path": str(dataset.path),
                "main_table_file": str(main_file),
                "file_count": len(files),
                "auxiliary_files": [str(p) for p in aux_files],
                "notebooks": [str(p) for p in notebooks],
                "warnings": warnings,
            })
            log("writing reports...")
            write_csv(profile.entity_time_span, ds_dir / "entity_time_span.csv")
            write_csv(profile.entity_data_quality, ds_dir / "entity_data_quality.csv")
            write_csv(profile.feature_profile, ds_dir / "feature_profile.csv")
            write_csv(profile.sales_distribution, ds_dir / "sales_distribution_by_entity.csv")
            write_csv(profile.gap_report, ds_dir / "entity_gap_report.csv")
            write_csv(profile.source_target_candidates, ds_dir / "source_target_candidate_report.csv")
            write_csv(pd.DataFrame(file_profiles.values()), ds_dir / "discovered_files.csv")
            write_csv(profile.monthly_missing, ds_dir / "monthly_missing_report.csv")
            write_csv(pd.DataFrame(), ds_dir / "source_similarity_top10.csv")
            write_csv(pd.DataFrame(), ds_dir / "source_dissimilarity_top10.csv")
            (ds_dir / "dataset_profile_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            write_dataset_markdown(
                ds_dir / "dataset_profile_summary.md",
                summary,
                profile.entity_time_span,
                profile.entity_data_quality,
                profile.feature_profile,
                profile.sales_distribution,
                profile.gap_report,
                profile.source_target_candidates,
                pd.DataFrame(),
            )
            result.update(summary)
            result["scan_status"] = "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS"
            return result

        if dataset.name in {"Dataset4", "Dataset5"} and main_file.suffix.lower() in {".csv", ".parquet"}:
            log("running chunked full aggregation...")
            date_guess = base_scanner.infer_date_col(probe_df)
            probe_for_inference = probe_df
            if date_guess.get("inferred_date_col") and base_scanner._detect_wide_qty_columns(probe_df.columns):
                probe_for_inference = base_scanner._melt_wide_sales(
                    probe_df, date_guess["inferred_date_col"], base_scanner.ScanLogger()
                )
            inference = base_scanner.infer_columns(probe_for_inference, dataset.name)
            log(f"detected date column: {inference.get('inferred_date_col')}")
            log(f"detected sales column: {inference.get('inferred_sales_col')}")
            log(f"detected entity columns: {', '.join(inference.get('inferred_entity_cols') or [])}")
            profile = aggregate_long_table_full_scan(
                main_file,
                inference,
                chunk_size=chunk_size,
                file_format=main_file.suffix.lower().lstrip("."),
            )
            warnings.extend(profile.summary.get("warnings") or [])
            summary = profile.summary
            summary.update({
                "dataset": dataset.name,
                "data_path": str(dataset.path),
                "main_table_file": str(main_file),
                "file_count": len(files),
                "auxiliary_files": [str(p) for p in aux_files],
                "notebooks": [str(p) for p in notebooks],
                "main_table_cols": len(probe_df.columns),
                "warnings": warnings,
            })
            summary["candidate_targets"] = profile.source_target_candidates.head(10)["entity_id"].tolist() if not profile.source_target_candidates.empty else []
            summary["candidate_sources"] = profile.source_target_candidates.iloc[1:11]["entity_id"].tolist() if len(profile.source_target_candidates) > 1 else []
            log("writing reports...")
            write_csv(profile.entity_time_span, ds_dir / "entity_time_span.csv")
            write_csv(profile.entity_data_quality, ds_dir / "entity_data_quality.csv")
            write_csv(profile.feature_profile, ds_dir / "feature_profile.csv")
            write_csv(profile.sales_distribution, ds_dir / "sales_distribution_by_entity.csv")
            write_csv(profile.gap_report, ds_dir / "entity_gap_report.csv")
            write_csv(profile.source_target_candidates, ds_dir / "source_target_candidate_report.csv")
            write_csv(pd.DataFrame(file_profiles.values()), ds_dir / "discovered_files.csv")
            write_csv(profile.monthly_missing, ds_dir / "monthly_missing_report.csv")
            write_csv(pd.DataFrame(), ds_dir / "source_similarity_top10.csv")
            write_csv(pd.DataFrame(), ds_dir / "source_dissimilarity_top10.csv")
            (ds_dir / "dataset_profile_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            write_dataset_markdown(
                ds_dir / "dataset_profile_summary.md",
                summary,
                profile.entity_time_span,
                profile.entity_data_quality,
                profile.feature_profile,
                profile.sales_distribution,
                profile.gap_report,
                profile.source_target_candidates,
                pd.DataFrame(),
            )
            result.update(summary)
            result["scan_status"] = "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS"
            return result

        raw_df, load_meta = read_table(main_file, max_rows=max_rows, chunk_size=chunk_size)
        if load_meta["read_mode"] != "full":
            warn(f"大文件使用 {load_meta['read_mode']}，本次最多扫描 {max_rows} 行；完整统计需提高 --max-rows 或离线分块扩展。")
        raw_df, wide_note = normalize_wide_day_sales(raw_df, aux_files)
        if wide_note:
            warn(wide_note)
        date_guess = base_scanner.infer_date_col(raw_df)
        if date_guess.get("inferred_date_col") and base_scanner._detect_wide_qty_columns(raw_df.columns):
            raw_df = base_scanner._melt_wide_sales(raw_df, date_guess["inferred_date_col"], base_scanner.ScanLogger())
        inference = base_scanner.infer_columns(raw_df, dataset.name)
        log(f"detected date column: {inference.get('inferred_date_col')}")
        log(f"detected sales column: {inference.get('inferred_sales_col')}")
        log(f"detected entity columns: {', '.join(inference.get('inferred_entity_cols') or [])}")
        work_df = base_scanner.apply_inferred_schema(raw_df, inference, base_scanner.ScanLogger())
        st_info = base_scanner.infer_source_target_entities(
            dataset.name, work_df, inference.get("inferred_entity_cols") or [], COLD_START, base_scanner.ScanLogger()
        )
        entity_count_before_cap = int(work_df["_entity_id"].nunique()) if "_entity_id" in work_df else 0
        if entity_count_before_cap > 10_000:
            top_entities = work_df["_entity_id"].value_counts().head(10_000).index
            work_df = work_df[work_df["_entity_id"].isin(top_entities)].copy()
            warn(
                f"实体数量 {entity_count_before_cap} 过大，详细实体级画像仅保留记录数最多的 10000 个实体；"
                "全量实体建议离线分块统计。"
            )

        log("scanning time span...")
        ent_span = entity_time_span(work_df)
        gaps = entity_gap_report(work_df)
        log("scanning data quality...")
        quality = entity_data_quality(work_df)
        features = feature_profile(work_df, inference)
        sales_dist = sales_distribution_by_entity(work_df)
        st_candidates = source_target_candidate_report(work_df, inference, ent_span, sales_dist, st_info)
        similar_top, dissimilar_top = similarity_summary(work_df, sales_dist)
        monthly_missing = month_missing_report(work_df)
        zero_context = detect_zero_context(work_df)
        granularity = infer_granularity(work_df)
        if granularity == "irregular":
            warn("时间粒度不规则。")
        if inference.get("needs_manual_review"):
            warn("日期列/销售列/实体列存在低置信度自动推断，需要人工复核。")

        log("writing reports...")
        write_csv(ent_span, ds_dir / "entity_time_span.csv")
        write_csv(quality, ds_dir / "entity_data_quality.csv")
        write_csv(features, ds_dir / "feature_profile.csv")
        write_csv(sales_dist, ds_dir / "sales_distribution_by_entity.csv")
        write_csv(gaps, ds_dir / "entity_gap_report.csv")
        write_csv(st_candidates, ds_dir / "source_target_candidate_report.csv")
        write_csv(pd.DataFrame(file_profiles.values()), ds_dir / "discovered_files.csv")
        write_csv(monthly_missing, ds_dir / "monthly_missing_report.csv")
        write_csv(similar_top, ds_dir / "source_similarity_top10.csv")
        write_csv(dissimilar_top, ds_dir / "source_dissimilarity_top10.csv")

        dates = work_df["_date"].dropna() if "_date" in work_df else pd.Series(dtype="datetime64[ns]")
        sales = work_df["_sales"].dropna() if "_sales" in work_df else pd.Series(dtype=float)
        span_dist = describe_span_distribution(ent_span, "total_calendar_days")
        valid_dist = describe_span_distribution(ent_span, "valid_sales_days")
        store_cols = [c for c in inference.get("inferred_entity_cols") or [] if "store" in str(c).lower() or "shop" in str(c).lower()]
        item_cols = [c for c in inference.get("inferred_entity_cols") or [] if any(w in str(c).lower() for w in ("item", "product", "sku"))]
        summary = {
            "dataset": dataset.name,
            "data_path": str(dataset.path),
            "main_table_file": str(main_file),
            "file_count": len(files),
            "auxiliary_files": [str(p) for p in aux_files],
            "notebooks": [str(p) for p in notebooks],
            "main_table_rows_scanned": int(len(work_df)),
            "main_table_cols": int(len(work_df.columns)),
            "read_mode": load_meta["read_mode"],
            "scan_coverage": "FULL_SCAN" if load_meta["read_mode"] == "full" else "CAPPED_SCAN",
            "date_col": inference.get("inferred_date_col"),
            "sales_col": inference.get("inferred_sales_col"),
            "entity_cols": inference.get("inferred_entity_cols") or [],
            "time_granularity": granularity,
            "global_min_date": str(dates.min().date()) if not dates.empty else None,
            "global_max_date": str(dates.max().date()) if not dates.empty else None,
            "global_total_days": int((dates.max() - dates.min()).days + 1) if not dates.empty else 0,
            "entity_count": int(ent_span["entity_id"].nunique()) if not ent_span.empty else 0,
            "store_count": int(work_df[store_cols[0]].nunique()) if store_cols else None,
            "item_count": int(work_df[item_cols[0]].nunique()) if item_cols else None,
            "sku_count": int(work_df[item_cols[0]].nunique()) if item_cols else None,
            "main_entity_grain": " + ".join(inference.get("inferred_entity_cols") or ["global_entity"]),
            "span_distribution": span_dist,
            "valid_sales_days_distribution": valid_dist,
            "entities_meeting_210_days": int(ent_span["meets_210_days"].sum()) if not ent_span.empty else 0,
            "entities_not_meeting_210_days": int((~ent_span["meets_210_days"]).sum()) if not ent_span.empty else 0,
            "missing_ratio": float(work_df.isna().mean().mean()),
            "zero_sales_ratio": float((sales == 0).sum() / max(int(sales.notna().sum()), 1)) if len(sales) else None,
            "feature_count": int(len(features)),
            "rfe_candidate_count": int(features["recommended_for_rfe"].sum()) if not features.empty else 0,
            "source_target_method": st_info.get("source_target_inference_method"),
            "source_target_reason": st_info.get("source_target_reason"),
            "candidate_targets": st_candidates.head(10)["entity_id"].tolist() if not st_candidates.empty else [],
            "candidate_sources": st_candidates.iloc[1:11]["entity_id"].tolist() if len(st_candidates) > 1 else [],
            "warnings": warnings,
            "zero_context": zero_context,
            "shortest_entities": ent_span.sort_values("total_calendar_days").head(10)["entity_id"].tolist() if not ent_span.empty else [],
            "longest_entities": ent_span.sort_values("total_calendar_days", ascending=False).head(10)["entity_id"].tolist() if not ent_span.empty else [],
        }
        (ds_dir / "dataset_profile_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        write_dataset_markdown(ds_dir / "dataset_profile_summary.md", summary, ent_span, quality, features, sales_dist, gaps, st_candidates, similar_top)

        result.update(summary)
        result["scan_status"] = "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS"
    except Exception as exc:
        result["error"] = str(exc)
        logs.append(traceback.format_exc())
        (ds_dir / "dataset_profile_summary.md").write_text(
            f"# {dataset.name} 数据集档案\n\n扫描失败：{exc}\n", encoding="utf-8"
        )
    finally:
        (ds_dir / "scan_log.txt").write_text("\n".join(logs), encoding="utf-8")
    return result


def write_dataset_markdown(
    path: Path,
    summary: Dict[str, Any],
    ent_span: pd.DataFrame,
    quality: pd.DataFrame,
    features: pd.DataFrame,
    sales_dist: pd.DataFrame,
    gaps: pd.DataFrame,
    st_candidates: pd.DataFrame,
    similar_top: pd.DataFrame,
) -> None:
    span = summary.get("span_distribution") or {}
    valid = summary.get("valid_sales_days_distribution") or {}
    target_ratio = ""
    if not st_candidates.empty:
        first = st_candidates.iloc[0]
        target_ratio = f"{first.get('cold_start_window_ratio', np.nan):.4f}"
    numeric_features = features[features["role"].eq("numeric feature")]["feature_name"].tolist() if not features.empty else []
    promo_features = features[features["role"].str.contains("promo", na=False)]["feature_name"].tolist() if not features.empty else []
    holiday_features = features[features["role"].str.contains("holiday", na=False)]["feature_name"].tolist() if not features.empty else []
    price_features = features[features["role"].str.contains("price", na=False)]["feature_name"].tolist() if not features.empty else []
    weather_features = features[features["role"].str.contains("weather", na=False)]["feature_name"].tolist() if not features.empty else []
    rfe_features = features[features["recommended_for_rfe"]]["feature_name"].head(30).tolist() if not features.empty else []
    sales_mean = _safe_float(sales_dist["mean"].mean()) if not sales_dist.empty else None
    sales_std = _safe_float(sales_dist["std"].mean()) if not sales_dist.empty else None
    skew = _safe_float(sales_dist["skewness"].mean()) if not sales_dist.empty and "skewness" in sales_dist else None
    has_outliers = bool((sales_dist.get("outlier_days_count", pd.Series(dtype=int)) > 0).any()) if not sales_dist.empty else False
    max_gap_raw = gaps["max_gap_days"].max() if not gaps.empty and "max_gap_days" in gaps else 0
    max_gap = int(max_gap_raw) if pd.notna(max_gap_raw) else 0
    has_gaps = max_gap > 0
    source_config_found = summary.get("source_target_method") == "from_existing_project_config"
    risks = []
    if summary.get("scan_coverage") == "CAPPED_SCAN":
        risks.append("CAPPED_SCAN：只抽样/截断扫描，不能给出最终实验建议，需要完整扫描后再判断。")
    if summary.get("entities_meeting_210_days", 0) == 0:
        risks.append("没有实体满足 210 天冷启动窗口。")
    if summary.get("time_granularity") == "irregular":
        risks.append("时间粒度不规则。")
    if summary.get("missing_ratio", 0) > 0.2:
        risks.append("缺失率偏高。")
    if not risks:
        risks.append("未发现阻断性风险，仍需人工确认 source/target 语义。")

    lines = [
        f"# {summary['dataset'].replace('Dataset', 'Dataset ')} 数据集档案",
        "",
        "## 1. 基本信息",
        "",
        f"- 数据路径：{summary.get('data_path')}",
        f"- 文件数量：{summary.get('file_count')}",
        f"- 主表文件：{summary.get('main_table_file')}",
        f"- 主表行数：{summary.get('main_table_rows_scanned')}",
        f"- 主表列数：{summary.get('main_table_cols')}",
        f"- 扫描覆盖：{summary.get('scan_coverage')}",
        f"- 自动识别日期列：{summary.get('date_col')}",
        f"- 自动识别销售列：{summary.get('sales_col')}",
        f"- 自动识别实体键：{summary.get('entity_cols')}",
        f"- 时间粒度：{summary.get('time_granularity')}",
        "",
        "## 2. 时间跨度",
        "",
        f"- 全局起止日期：{summary.get('global_min_date')} 至 {summary.get('global_max_date')}",
        f"- 全局总天数：{summary.get('global_total_days')}",
        "- 实体时间跨度：",
        f"  - min：{span.get('min')}",
        f"  - max：{span.get('max')}",
        f"  - median：{span.get('median')}",
        f"  - mean：{span.get('mean')}",
        "- 有效销售天数：",
        f"  - min：{valid.get('min')}",
        f"  - max：{valid.get('max')}",
        f"  - median：{valid.get('median')}",
        "",
        "## 3. 实体数量与划分",
        "",
        f"- 门店数：{summary.get('store_count')}",
        f"- 商品数：{summary.get('item_count')}",
        f"- SKU 数：{summary.get('sku_count')}",
        f"- 主实体数量：{summary.get('entity_count')}",
        f"- 当前是否发现 source / target 配置：{'是' if source_config_found else '未发现显式 source / target 配置'}",
        f"- 候选 target：{summary.get('candidate_targets')[:5]}",
        f"- 候选 source：{summary.get('candidate_sources')[:10]}",
        "",
        "## 4. 冷启动窗口可行性",
        "",
        "- 冷启动设定：15 train + 15 val + 180 test = 210 days",
        f"- 满足 210 天历史的实体数：{summary.get('entities_meeting_210_days')}",
        f"- 不满足 210 天历史的实体数：{summary.get('entities_not_meeting_210_days')}",
        f"- target 冷启动窗口占总历史比例：{target_ratio}",
        "",
        "## 5. 数据密度与质量",
        "",
        f"- 总缺失率：{summary.get('missing_ratio')}",
        f"- 零销售占比：{summary.get('zero_sales_ratio')}",
        f"- 是否存在断档：{'是' if has_gaps else '否'}",
        f"- 最大断档长度：{max_gap}",
        f"- 时间粒度是否一致：{'否' if summary.get('time_granularity') == 'irregular' else '是'}",
        f"- 真实零销售 / 缺货 / 关店是否可区分：{summary.get('zero_context', {}).get('note')}",
        "",
        "## 6. 特征维度",
        "",
        f"- 总特征列数：{summary.get('feature_count')}",
        f"- 数值特征：{numeric_features[:20]}",
        f"- 类别特征：{features[features['role'].eq('categorical feature')]['feature_name'].head(20).tolist() if not features.empty else []}",
        f"- 促销特征：{promo_features}",
        f"- 节假日特征：{holiday_features}",
        f"- 价格特征：{price_features}",
        f"- 天气特征：{weather_features}",
        f"- 推荐进入 RFE 的候选特征：{rfe_features}",
        "",
        "## 7. 销售量分布",
        "",
        f"- 销售均值：{sales_mean}",
        f"- 销售标准差：{sales_std}",
        f"- 偏度：{skew}",
        f"- 是否存在极端异常值：{'是' if has_outliers else '否'}",
        "- 源域实体之间分布差异：见 `source_similarity_top10.csv` 和 `source_dissimilarity_top10.csv`",
        "",
        "## 8. 风险与行动建议",
        "",
        f"- 主要风险：{'; '.join(risks)}",
        "- 对 source_history_days 的建议：优先选择不超过实体历史长度中位数且满足实验窗口的档位。",
        "- 对 source / target 划分的建议：target 优先选择历史足够且质量完整的实体，source 选择销售分布相似且缺失少的实体；本报告只给候选，不替代最终实验设定。",
        "- 对 RFE 特征选择的建议：从 `feature_profile.csv` 中 `recommended_for_rfe=True` 的字段开始，排除缺失高、常数和潜在泄漏字段。",
        "- 对后续实验配置的建议：先查看总览风险排序，再决定是否缩短 cold-start window 或清洗断档。",
        "",
        "## 附：文件结构",
        "",
        df_to_markdown(pd.DataFrame({"auxiliary_file": summary.get("auxiliary_files", [])}), max_rows=30),
        "",
        "## 附：Top source/target 候选",
        "",
        df_to_markdown(st_candidates, max_rows=20),
        "",
        "## 附：最相似实体对 Top 10",
        "",
        df_to_markdown(similar_top, max_rows=10),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_discovery_reports(discovery: DatasetDiscovery, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[discover] data_root = {discovery.data_root}")
    rows = []
    for idx in range(1, 7):
        name = f"Dataset{idx}"
        ds = discovery.datasets.get(name)
        if ds:
            print(f"[discover] {name} -> {ds.path}")
            rows.append({
                "dataset": name,
                "selected_path": str(ds.path),
                "kind": ds.kind,
                "alternatives": "; ".join(str(p) for p in ds.alternatives),
                "status": "FOUND",
            })
        else:
            print(f"[discover] {name} -> NOT FOUND")
            rows.append({"dataset": name, "selected_path": "", "kind": "", "alternatives": "", "status": "NOT_FOUND"})
    print("[discover] ignored files:")
    for path in discovery.ignored_files:
        print(f"- {path.name}")
    write_csv(pd.DataFrame(rows), output_dir / "dataset_path_discovery_report.csv")
    md = ["# Dataset Path Discovery Report", "", f"- data_root: `{discovery.data_root}`", "", "## D1-D6"]
    md.append(df_to_markdown(pd.DataFrame(rows), max_rows=20))
    md.extend(["", "## Ignored Files"])
    md.extend(f"- {p}" for p in discovery.ignored_files)
    (output_dir / "dataset_path_discovery_report.md").write_text("\n".join(md), encoding="utf-8")


def risk_level(row: Dict[str, Any]) -> str:
    if row.get("scan_status") == "FAILED":
        return "HIGH"
    if row.get("scan_coverage") == "CAPPED_SCAN":
        return "HIGH"
    score = 0
    if row.get("entities_meeting_210_days", 0) == 0:
        score += 2
    if (row.get("missing_ratio") or 0) > 0.2:
        score += 1
    if row.get("time_granularity") == "irregular":
        score += 1
    if row.get("scan_coverage") not in {"FULL_SCAN", "CHUNKED_FULL_SCAN"}:
        score += 1
    return "HIGH" if score >= 3 else "MEDIUM" if score else "LOW"


def write_global_reports(results: List[Dict[str, Any]], output_dir: Path) -> None:
    rows = []
    history_rows = []
    cold_rows = []
    risk_rows = []
    for result in results:
        entity_count = result.get("entity_count") or 0
        feasible = result.get("entities_meeting_210_days") or 0
        median_days = (result.get("span_distribution") or {}).get("median") or 0
        missing = result.get("missing_ratio")
        zero = result.get("zero_sales_ratio")
        level = risk_level(result)
        rows.append({
            "Dataset": result.get("Dataset") or result.get("dataset"),
            "scan_coverage": result.get("scan_coverage") or ("FAILED" if result.get("scan_status") == "FAILED" else None),
            "Date Range": f"{result.get('global_min_date')}~{result.get('global_max_date')}",
            "Total Days": result.get("global_total_days"),
            "Entity Count": entity_count,
            "Min Entity Days": (result.get("span_distribution") or {}).get("min"),
            "Median Entity Days": median_days,
            "Zero Ratio": zero,
            "Missing Ratio": missing,
            "Feature Count": result.get("feature_count"),
            "Cold Start Feasible": feasible > 0,
            "Scan Status": result.get("scan_status"),
        })
        hist = {"Dataset": result.get("Dataset") or result.get("dataset")}
        for days in (30, 60, 90, 180, 365, 730):
            hist[str(days)] = bool(median_days >= days)
        feasible_days = [d for d in (30, 60, 90, 180, 365, 730) if median_days >= d]
        hist["Recommended Max"] = max(feasible_days) if feasible_days else 0
        history_rows.append(hist)
        cold_rows.append({
            "Dataset": result.get("Dataset") or result.get("dataset"),
            "Required Days": REQUIRED_DAYS,
            "Entities >=210 Days": feasible,
            "Feasible Ratio": feasible / entity_count if entity_count else 0,
            "Risk Level": level,
        })
        risk_rows.append({
            "Dataset": result.get("Dataset") or result.get("dataset"),
            "Risk Level": level,
            "Reasons": "; ".join(result.get("warnings") or []) or result.get("error") or "未发现主要阻断风险",
            "Action": (
                "需要完整扫描后再判断"
                if result.get("scan_coverage") == "CAPPED_SCAN"
                else "先清洗/复核后实验"
                if level == "HIGH"
                else "复核 source/target 后可进入实验"
                if level == "MEDIUM"
                else "适合优先进入实验"
            ),
        })

    overview_df = pd.DataFrame(rows)
    history_df = pd.DataFrame(history_rows)
    cold_df = pd.DataFrame(cold_rows)
    risk_df = pd.DataFrame(risk_rows)
    write_csv(overview_df, output_dir / "d1_d6_complete_overview.csv")
    write_csv(cold_df, output_dir / "d1_d6_cold_start_window_comparison.csv")

    rfe_rows = []
    for result in results:
        rfe_rows.append({
            "Dataset": result.get("Dataset") or result.get("dataset"),
            "Feature Count": result.get("feature_count"),
            "Missing Feature Risk": "HIGH" if (result.get("missing_ratio") or 0) > 0.2 else "LOW",
            "Constant Feature Risk": "SEE feature_profile.csv",
            "Leakage Risk": "SEE feature_profile.csv",
            "RFE Recommendation": "Use recommended_for_rfe fields after manual review.",
        })
    rfe_df = pd.DataFrame(rfe_rows)

    md_lines = [
        "# D1-D6 数据集完整扫描总览",
        "",
        "## 1. 数据集总体比较表",
        "",
        df_to_markdown(overview_df, max_rows=20),
        "",
        "## 2. source_history_days 可行性",
        "",
        df_to_markdown(history_df, max_rows=20),
        "",
        "## 3. 冷启动窗口 15+15+180 可行性",
        "",
        df_to_markdown(cold_df, max_rows=20),
        "",
        "## 4. RFE 特征可靠性",
        "",
        df_to_markdown(rfe_df, max_rows=20),
        "",
        "## 5. 数据质量风险排序",
        "",
        df_to_markdown(risk_df.sort_values("Risk Level", ascending=True), max_rows=20),
        "",
        "## 6. 后续实验行动建议",
        "",
        "- 适合直接进入实验：scan_coverage 为 FULL_SCAN / CHUNKED_FULL_SCAN 且 Risk Level 为 LOW 的数据集。",
        "- 需要清洗：Risk Level 为 HIGH 或存在 irregular/granularity、缺失率高、断档严重的数据集。",
        "- source / target 划分风险较高：未发现显式配置且候选实体历史不足的数据集。",
        "- 不适合使用 15+15+180：Entities >=210 Days 为 0 的数据集。",
        "- 需要重新设计冷启动窗口：Recommended Max 小于 210 的数据集；CAPPED_SCAN 数据集需要完整扫描后再判断。",
    ]
    (output_dir / "d1_d6_complete_overview.md").write_text("\n".join(md_lines), encoding="utf-8")

    action_lines = ["# D1-D6 Risk And Action Items", "", df_to_markdown(risk_df, max_rows=20)]
    (output_dir / "d1_d6_risk_and_action_items.md").write_text("\n".join(action_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and scan Dataset1-Dataset6 under one root.")
    parser.add_argument("--data-root", required=True, help="Root directory containing D1-D6 raw datasets.")
    parser.add_argument("--output-dir", default="outputs/dataset_profiles", help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=100_000, help="Maximum rows loaded per main table for huge files.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="CSV chunk size for huge files.")
    parser.add_argument("--max-probe-rows", type=int, default=1000, help="Rows used to score candidate main tables.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    discovery = discover_dataset_paths(args.data_root)
    write_discovery_reports(discovery, output_dir)

    results: List[Dict[str, Any]] = []
    for idx in range(1, 7):
        name = f"Dataset{idx}"
        ds = discovery.datasets.get(name)
        if not ds:
            msg = f"[{name}] not found under {discovery.data_root}"
            errors.append(msg)
            results.append({"Dataset": name, "scan_status": "FAILED", "error": msg})
            continue
        result = scan_one_dataset(ds, output_dir, args.max_rows, args.chunk_size, args.max_probe_rows)
        if result.get("scan_status") == "FAILED":
            errors.append(f"[{name}] {result.get('error')}")
        results.append(result)

    write_global_reports(results, output_dir)
    if errors:
        (output_dir / "scan_errors.log").write_text("\n".join(errors), encoding="utf-8")
    else:
        (output_dir / "scan_errors.log").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
