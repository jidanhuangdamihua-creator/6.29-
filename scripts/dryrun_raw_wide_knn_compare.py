#!/usr/bin/env python3
"""Read-only D4–D6 narrow-versus-raw/wide KNN diagnostic.

The module deliberately keeps its first-layer diagnostics pure so the source
pool contract can be tested without loading the full raw datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import SOLIDIFIED_KNN_ROOT, SOURCE_HISTORY_DAYS
from src.source_selection.source_selector import SourceSelector
from src.utils.parquet_data_loader import (
    derive_d4_d6_runtime_knn_windows,
    load_parquet_source_target,
    read_dataset_windows,
)
from src.utils.source_domain_filter import apply_source_domain_policy, normalize_domain_filter


PROTECTED_OUTPUT_ROOTS = (
    PROJECT_ROOT / "configs" / "solidified" / "knn",
    PROJECT_ROOT / "数据集" / "固化数据",
    PROJECT_ROOT / "outputs" / "domain_adaptation",
    PROJECT_ROOT / "outputs" / "experiment_results",
    PROJECT_ROOT / "outputs" / "final_summary",
)
ALLOWED_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "feature_consistency"


class WideCleanUnavailable(RuntimeError):
    """No reusable wide-clean pool exists and raw construction cannot proceed."""


def _entity_count(frame: pd.DataFrame, group_cols: Sequence[str]) -> int:
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"entity group columns missing from source pool: {missing}")
    return int(frame.loc[:, list(group_cols)].drop_duplicates().shape[0])


def compute_pool_domain_diagnostics(
    source: pd.DataFrame,
    *,
    domain_column: str,
    domain_value: Any,
    group_cols: Sequence[str],
) -> dict[str, Any]:
    """Measure a scalar configured-domain filter without applying runtime mode."""
    if domain_column not in source.columns:
        raise ValueError(f"domain column missing from source pool: {domain_column}")
    before_rows = int(len(source))
    before_entities = _entity_count(source, group_cols)
    matching = source.loc[source[domain_column].eq(domain_value)].copy()
    after_rows = int(len(matching))
    after_entities = _entity_count(matching, group_cols)
    values = sorted(str(value) for value in source[domain_column].dropna().unique().tolist())
    domain_nunique = int(source[domain_column].nunique(dropna=True))
    vacuous = (
        domain_nunique == 1
        and values == [str(domain_value)]
        and before_rows == after_rows
        and before_entities == after_entities
    )
    return {
        "source_rows": before_rows,
        "source_entities": before_entities,
        "source_domain_nunique": domain_nunique,
        "source_domain_values": values,
        "after_filter_rows": after_rows,
        "after_filter_entities": after_entities,
        "domain_filter_vacuous": vacuous,
        "domain_filter_effective": bool(before_rows != after_rows or before_entities != after_entities),
    }


def _entity_key_series(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"cannot derive entity key; missing group_cols: {missing}")
    return frame.loc[:, list(group_cols)].astype("string").fillna("<NA>").agg("\x1f".join, axis=1)


def stable_cap_entities(
    source: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    cap: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministically retain whole entities using SHA-256 of JSON group_cols."""
    keys = _entity_key_series(source, group_cols)
    unique_keys = sorted(set(keys.tolist()), key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())
    pre_cap = len(unique_keys)
    if cap is not None and cap <= 0:
        raise ValueError("max-source-entities must be positive")
    selected = unique_keys if cap is None else unique_keys[:cap]
    result = source.loc[keys.isin(selected)].copy()
    return result, {
        "max_source_entities": cap,
        "pre_cap_source_entities": pre_cap,
        "post_cap_source_entities": len(selected),
        "cap_applied": cap is not None and pre_cap > cap,
        "hash_key_columns": list(group_cols),
    }


def compare_feature_schema(
    narrow_features: Iterable[str], wide_features: Iterable[str]
) -> dict[str, Any]:
    narrow = list(narrow_features)
    wide = list(wide_features)
    narrow_set = set(narrow)
    wide_set = set(wide)
    return {
        "narrow_feature_columns_count": len(narrow),
        "wide_feature_columns_count": len(wide),
        "feature_columns_missing_in_wide": [column for column in narrow if column not in wide_set],
        "feature_columns_extra_in_wide": [column for column in wide if column not in narrow_set],
        "feature_schema_match": narrow_set == wide_set,
    }


def topk_overlap_ratio(left: Iterable[str], right: Iterable[str], *, top_k: int) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return len(set(left).intersection(set(right))) / float(top_k)


def wide_with_non_configured_domain_count(
    selected_domains: Iterable[Any], configured_domain_value: Any
) -> int:
    return sum(str(value) != str(configured_domain_value) for value in selected_domains)


def reconstruct_runtime_window_metadata(windows: dict[str, Any]) -> dict[str, Any]:
    runtime = derive_d4_d6_runtime_knn_windows(windows, SOURCE_HISTORY_DAYS)
    target_start = pd.Timestamp(runtime["target_observed_start"])
    target_end = pd.Timestamp(runtime["target_observed_end"])
    source_start = pd.Timestamp(runtime["source_history_start"])
    source_end = pd.Timestamp(runtime["source_history_end"])
    return {
        "target_feature_window_days": int((target_end - target_start).days + 1),
        "source_history_days": int((source_end - source_start).days + 1),
        "target_observed_start": target_start.strftime("%Y-%m-%d"),
        "target_observed_end": target_end.strftime("%Y-%m-%d"),
        "source_history_start": source_start.strftime("%Y-%m-%d"),
        "source_history_end": source_end.strftime("%Y-%m-%d"),
        "runtime_window_metadata": runtime,
    }


def validate_output_dir(output_dir: Path) -> Path:
    """Reject every official or experimental output location before writing."""
    candidate = Path(output_dir)
    resolved = candidate.resolve()
    for protected in PROTECTED_OUTPUT_ROOTS:
        protected_resolved = protected.resolve()
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise ValueError(f"protected output path is not allowed: {candidate}")
    allowed = ALLOWED_OUTPUT_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(
            "output directory must be below outputs/feature_consistency: " f"{candidate}"
        )
    return candidate


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def _payloads(dataset_id: int, knn_root: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for mode in ("with", "without"):
        path = knn_root / f"Dataset{dataset_id}" / f"knn_{mode}_info_sharing.json"
        if not path.exists():
            raise FileNotFoundError(f"missing solidified KNN JSON: {path}")
        payload = _load_payload(path)
        if int(payload.get("dataset_id", -1)) != dataset_id:
            raise ValueError(f"KNN JSON dataset_id mismatch in {path}")
        payloads[mode] = payload
    return payloads


def _selected_features(payload: dict[str, Any]) -> list[str]:
    info = payload.get("feature_info")
    if isinstance(info, dict) and isinstance(info.get("selected_features"), list):
        features = [str(value) for value in info["selected_features"]]
    else:
        features = [str(value) for value in payload.get("feature_cols", [])]
    if not features:
        raise ValueError("KNN JSON has no feature_cols or feature_info.selected_features")
    return features


def _domain_filter(payload: dict[str, Any]) -> tuple[str, Any]:
    normalized = normalize_domain_filter(payload.get("domain_filter"))
    if len(normalized) != 1:
        raise ValueError(f"dry-run requires exactly one configured domain filter: {normalized}")
    return next(iter(normalized.items()))


def _entity_string(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(group_cols)].astype("string").fillna("<NA>").agg("_".join, axis=1)


def normalize_wide_clean_schema(
    frame: pd.DataFrame,
    *,
    payload: dict[str, Any],
    dataset_id: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize known aliases and apply JSON group_cols as the only identity rule."""
    out = frame.copy()
    mappings: dict[str, str] = {}
    aliases = {
        "stock_hour_6_22_cnt": "stock_hour6_22_cnt",
        "unit_sales": "sales",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
            mappings[old] = new
    group_cols = [str(value) for value in payload.get("group_cols", [])]
    if len(group_cols) != 2:
        raise ValueError(f"KNN JSON requires exactly two group_cols, got {group_cols}")
    required_features = _selected_features(payload)
    domain_column, _ = _domain_filter(payload)
    required = ["date", *group_cols, domain_column, *required_features]
    missing = [column for column in dict.fromkeys(required) if column not in out.columns]
    if missing:
        raise ValueError(
            f"D{dataset_id} wide-clean cannot meet JSON schema contract; missing columns: {missing}"
        )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError(f"D{dataset_id} wide-clean contains invalid date values")
    out["entity_id"] = _entity_string(out, group_cols)
    out["source_entity_key"] = out["entity_id"]
    duplicate_keys = out.duplicated(subset=[*group_cols, "date"], keep=False)
    if duplicate_keys.any():
        raise ValueError(
            f"D{dataset_id} wide-clean has duplicate group/date rows; cannot form runtime KNN signatures"
        )
    return out, {
        "json_group_cols": group_cols,
        "derived_entity_key_format": "_".join(group_cols),
        "normalized_column_name_mappings": mappings,
        "wide_clean_identity_columns": group_cols,
        "solidified_source_identity_columns": group_cols,
    }


def _window_wide_source(
    wide: pd.DataFrame, runtime: dict[str, Any], narrow_source: pd.DataFrame
) -> pd.DataFrame:
    source = wide.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    start = pd.Timestamp(runtime["source_history_start"])
    end = pd.Timestamp(runtime["source_history_end"])
    source = source.loc[source["date"].between(start, end, inclusive="both")].copy()
    source.attrs.update(narrow_source.attrs)
    return source


def _parquet_row_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"missing parquet for row-count diagnostic: {path}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def _build_d4_wide_clean(raw_root: Path, payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from scripts.preprocess_clean_datasets import clean_d4_dataframe

    base = raw_root / "Dataset 4叮咚数据集" / "data"
    train_path = base / "train.parquet"
    eval_path = base / "eval.parquet"
    paths = [path for path in (train_path, eval_path) if path.exists()]
    if not paths:
        raise FileNotFoundError(f"D4 raw train/eval parquet not found below {base}")
    raw_parts = [pd.read_parquet(path) for path in paths]
    raw = pd.concat(raw_parts, ignore_index=True)
    clean = clean_d4_dataframe(raw)
    normalized, schema = normalize_wide_clean_schema(clean, payload=payload, dataset_id=4)
    schema.update({
        "raw_train_rows": int(len(raw_parts[0])) if train_path.exists() else 0,
        "raw_eval_rows": int(len(raw_parts[-1])) if eval_path.exists() else 0,
    })
    return normalized, schema


def _build_d5_wide_clean(
    raw_root: Path,
    payload: dict[str, Any],
    runtime: dict[str, Any],
    intermediate_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build D5 features with the same per-item daily-completion contract."""
    from scripts.preprocess_clean_datasets import (
        _partition_d5_train_by_store,
        clean_d5_store_dataframe,
        preprocess_d5_holidays,
        preprocess_d5_oil,
    )

    base = raw_root / "Dataset 5Favorita"
    train_path = base / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"D5 raw train.csv not found: {train_path}")
    items = pd.read_csv(base / "items.csv")
    stores = pd.read_csv(base / "stores.csv")
    oil = preprocess_d5_oil(pd.read_csv(base / "oil.csv"))
    transactions = pd.read_csv(base / "transactions.csv")
    holidays = preprocess_d5_holidays(pd.read_csv(base / "holidays_events.csv"), stores)
    start = pd.Timestamp(runtime["source_history_start"])
    end = pd.Timestamp(runtime["source_history_end"])
    partition_dir = intermediate_dir / "d5_store_partitions"
    _partition_d5_train_by_store(train_path, partition_dir)
    pieces: list[pd.DataFrame] = []
    for store_row in stores.sort_values("store_nbr").itertuples(index=False):
        store = pd.Series(store_row._asdict())
        store_path = partition_dir / f"store_{int(store['store_nbr'])}.csv"
        if not store_path.exists():
            continue
        store_train = pd.read_csv(store_path, low_memory=False)
        store_train["date"] = pd.to_datetime(store_train["date"], errors="coerce")
        store_train = store_train.loc[store_train["date"].le(end)].copy()
        if store_train.empty:
            continue
        clean = clean_d5_store_dataframe(
            store_train,
            items=items,
            store_row=store,
            oil=oil,
            transactions=transactions,
            holidays_by_store=holidays,
            global_end_date=end,
        )
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
        pieces.append(clean.loc[clean["date"].between(start, end, inclusive="both")].copy())
    if not pieces:
        raise ValueError("D5 raw data produced no rows in the runtime source window")
    wide = pd.concat(pieces, ignore_index=True)
    normalized, schema = normalize_wide_clean_schema(wide, payload=payload, dataset_id=5)
    schema["d5_zero_fill_contract"] = "per-item dates are completed from first observation through source cutoff"
    return normalized, schema


def _build_d6_wide_clean(raw_root: Path, payload: dict[str, Any], runtime: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from scripts.preprocess_clean_datasets import clean_d6_chunk

    base = raw_root / "Dataset 6m5-forecasting-accuracy"
    sales_path = base / "sales_train_evaluation.csv"
    calendar_path = base / "calendar.csv"
    prices_path = base / "sell_prices.csv"
    if not all(path.exists() for path in (sales_path, calendar_path, prices_path)):
        raise FileNotFoundError(f"D6 raw inputs are incomplete below {base}")
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    start = pd.Timestamp(runtime["source_history_start"])
    end = pd.Timestamp(runtime["source_history_end"])
    calendar_dates = pd.to_datetime(calendar["date"], errors="coerce")
    d_columns = calendar.loc[calendar_dates.between(start, end, inclusive="both"), "d"].tolist()
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    usecols = [*id_columns, *d_columns]
    chunks: list[pd.DataFrame] = []
    for raw in pd.read_csv(sales_path, usecols=usecols, chunksize=1_000, low_memory=False):
        clean = clean_d6_chunk(raw, calendar, prices)
        chunks.append(clean)
    if not chunks:
        raise ValueError("D6 raw data produced no rows in the runtime source window")
    wide = pd.concat(chunks, ignore_index=True)
    return normalize_wide_clean_schema(wide, payload=payload, dataset_id=6)


def build_or_load_wide_clean(
    *,
    dataset_id: int,
    payload: dict[str, Any],
    raw_root: Path | None,
    clean_input_root: Path | None,
    run_dir: Path,
    runtime: dict[str, Any],
    reuse_existing_wide_clean: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    filename = f"dataset{dataset_id}-wide-clean.parquet"
    reusable = clean_input_root / filename if clean_input_root is not None else None
    if reusable is not None and reusable.exists():
        wide, schema = normalize_wide_clean_schema(pd.read_parquet(reusable), payload=payload, dataset_id=dataset_id)
        schema["wide_clean_source"] = "reused"
        return wide, schema, reusable
    if reuse_existing_wide_clean:
        expected = reusable if reusable is not None else Path(filename)
        raise WideCleanUnavailable(
            f"--reuse-existing-wide-clean requested but missing expected table: {expected}"
        )
    if raw_root is None:
        raise WideCleanUnavailable("wide-clean table missing and raw input root was not supplied")
    builders = {4: _build_d4_wide_clean, 5: _build_d5_wide_clean, 6: _build_d6_wide_clean}
    try:
        if dataset_id == 4:
            wide, schema = builders[dataset_id](raw_root, payload)
        elif dataset_id == 5:
            wide, schema = builders[dataset_id](raw_root, payload, runtime, run_dir / "intermediate")
        else:
            wide, schema = builders[dataset_id](raw_root, payload, runtime)
    except Exception as exc:
        raise WideCleanUnavailable(
            f"D{dataset_id} wide-clean construction failed: {type(exc).__name__}: {exc}"
        ) from exc
    destination = run_dir / "intermediate" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(destination, index=False)
    schema["wide_clean_source"] = "constructed"
    return wide, schema, destination


def _topk_for_targets(
    *,
    target: pd.DataFrame,
    source: pd.DataFrame,
    payload: dict[str, Any],
    features: Sequence[str],
    top_k: int,
    domain_column: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    selector = SourceSelector()
    group_cols = tuple(str(value) for value in payload["group_cols"])
    source_domains = source.loc[:, [*group_cols, domain_column]].drop_duplicates(subset=list(group_cols))
    result: dict[str, list[str]] = {}
    domains: dict[str, list[str]] = {}
    for target_key in payload.get("results", {}):
        target_frame = target.loc[target["entity_id"].astype(str).eq(str(target_key))].copy()
        if target_frame.empty:
            raise ValueError(f"target entity missing from target parquet: {target_key}")
        selected = selector.select_top_k_sources(
            target_df=target_frame,
            source_df=source,
            feature_cols=list(features),
            k=top_k,
            group_cols=group_cols,
        )
        keys = [tuple(row["source_key"]) for row in selected["sources"]]
        result[str(target_key)] = ["_".join(str(value) for value in key) for key in keys]
        domains[str(target_key)] = [
            str(source_domains.loc[(source_domains[list(group_cols)] == list(key)).all(axis=1), domain_column].iloc[0])
            for key in keys
        ]
    return result, domains


def _existing_json_topk(payload: dict[str, Any], target_key: str) -> list[str]:
    return [str(row.get("source_entity", "")) for row in payload.get("results", {}).get(target_key, [])]


def _runtime_windows_match_json(payloads: dict[str, dict[str, Any]], runtime: dict[str, Any]) -> bool | None:
    """Compare explicit runtime metadata only when a current JSON carries it."""
    expected = {
        "target_observed_start": runtime["target_observed_start"],
        "target_observed_end": runtime["target_observed_end"],
        "source_history_start": runtime["source_history_start"],
        "source_history_end": runtime["source_history_end"],
    }
    metadata_rows: list[dict[str, Any]] = []
    for payload in payloads.values():
        selection_metadata = payload.get("selection_metadata", {})
        if isinstance(selection_metadata, dict):
            metadata_rows.extend(value for value in selection_metadata.values() if isinstance(value, dict))
    comparable = [row for row in metadata_rows if all(key in row for key in expected)]
    if not comparable:
        return None
    return all(all(str(row[key]) == value for key, value in expected.items()) for row in comparable)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _write_dataset_artifacts(run_dir: Path, dataset_id: int, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    safe_rows = [{key: _csv_value(value) for key, value in row.items()} for row in rows]
    pd.DataFrame(safe_rows).to_csv(run_dir / f"dataset{dataset_id}_raw_wide_knn_compare.csv", index=False)
    (run_dir / f"dataset{dataset_id}_raw_wide_knn_compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _markdown_report(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Raw/Wide KNN dry-run comparison",
        "",
        "Diagnostic only: official parquet, KNN JSON, experiment results, and final summaries were not modified.",
        "",
        "## Executive summary",
        "",
        "This report compares the existing narrow source pool with a temporary wide-clean source pool. It is evidence for deciding whether a full wide-source ablation is meaningful; it does not replace any official artifact.",
        "",
        "## Dataset-level verdict",
        "",
        "| Dataset | Status | Narrow filter vacuous | Wide filter effective | Recommendation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        narrow = summary.get("narrow_diagnostics", {})
        wide = summary.get("wide_diagnostics", {})
        rows = summary.get("rows", [])
        selected_other = any(row.get("wide_with_selected_non_configured_domain_count", 0) > 0 for row in rows)
        low_overlap = any(row.get("overlap_narrow_vs_wide_with", 1.0) < 1.0 for row in rows)
        recommendation = "consider full wide-source ablation" if selected_other and low_overlap else "keep current narrow mainline results"
        lines.append(
            f"| D{summary['dataset']} | {summary.get('status')} | {narrow.get('domain_filter_vacuous')} | {wide.get('domain_filter_effective')} | {recommendation} |"
        )
    lines.extend([
        "",
        "## Runtime-window comparability check",
        "",
    ])
    for summary in summaries:
        runtime = summary.get("runtime_window", {})
        lines.extend([
            f"### D{summary['dataset']}",
            f"- Target feature window: {runtime.get('target_feature_window_days')} days ({runtime.get('target_observed_start')} to {runtime.get('target_observed_end')})",
            f"- Source history: {runtime.get('source_history_days')} days ({runtime.get('source_history_start')} to {runtime.get('source_history_end')})",
            f"- Matches current KNN JSON runtime metadata: {summary.get('runtime_windows_match_current_knn_metadata')}",
            "- KNN used windowed features rather than raw full history: true",
            "",
            "## Source-pool diagnostics",
            "",
            f"- Current narrow domain distribution: {summary.get('narrow_diagnostics', {}).get('source_domain_values')}",
            f"- Wide-clean domain distribution: {summary.get('wide_diagnostics', {}).get('source_domain_values')}",
            f"- Narrow filter vacuous: {summary.get('narrow_diagnostics', {}).get('domain_filter_vacuous')}",
            f"- Wide filter effective: {summary.get('wide_diagnostics', {}).get('domain_filter_effective')}",
            f"- Constructed wide-clean rows / solidified narrow rows: {summary.get('constructed_wide_clean_rows')} / {summary.get('solidified_source_rows')}",
            f"- Wide vs solidified row delta / ratio: {summary.get('wide_vs_solidified_row_count_delta')} / {summary.get('wide_vs_solidified_row_count_ratio')}",
            f"- Raw D4 train rows / eval rows (when applicable): {summary.get('schema_contract', {}).get('raw_train_rows')} / {summary.get('schema_contract', {}).get('raw_eval_rows')}",
            "",
            "## Schema-contract check",
            "",
            f"- JSON group_cols: {summary.get('schema_contract', {}).get('json_group_cols')}",
            f"- Derived entity key: {summary.get('schema_contract', {}).get('derived_entity_key_format')}",
            f"- Column mappings: {summary.get('schema_contract', {}).get('normalized_column_name_mappings')}",
            f"- Feature schema comparable: {summary.get('schema_contract', {}).get('feature_schema_match')}",
            "",
            "## Top-K overlap summary",
            "",
        ])
        for row in summary.get("rows", []):
            lines.append(
                f"- {row.get('target_entity_key')} ({row.get('mode')}): "
                f"existing/narrow={row.get('overlap_existing_json_vs_recomputed_narrow')}; "
                f"narrow/wide-with={row.get('overlap_narrow_vs_wide_with')}; "
                f"wide-with/wide-without={row.get('overlap_wide_with_vs_wide_without')}; "
                f"non-configured wide-with selections={row.get('wide_with_selected_non_configured_domain_count')}"
            )
        lines.append("")
    lines.extend([
        "## Final recommendation",
        "",
        "Keep current narrow mainline results. Run a full wide-source ablation only when wide-with selects non-configured domains and Top-K overlap is low. Do not replace official KNN JSON or official parquet based only on this dry-run.",
        "",
    ])
    (run_dir / "raw_wide_knn_compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset_comparison(
    *,
    dataset_id: int,
    modes: Sequence[str],
    top_k: int,
    max_source_entities: int | None,
    run_dir: Path,
    raw_root: Path | None,
    clean_input_root: Path | None,
    reuse_existing_wide_clean: bool,
    parquet_root: Path,
    knn_root: Path,
) -> dict[str, Any]:
    payloads = _payloads(dataset_id, knn_root)
    payload = payloads["with"]
    features = _selected_features(payload)
    if features != _selected_features(payloads["without"]):
        raise ValueError(f"D{dataset_id} with/without JSON feature contracts differ")
    if payload.get("group_cols") != payloads["without"].get("group_cols"):
        raise ValueError(f"D{dataset_id} with/without JSON group_cols differ")
    domain_column, domain_value = _domain_filter(payload)
    windows = read_dataset_windows(dataset_id, knn_root / f"Dataset{dataset_id}")
    runtime = reconstruct_runtime_window_metadata(windows)
    narrow_source, target = load_parquet_source_target(
        dataset_id=dataset_id,
        parquet_dir=parquet_root,
        windows=windows,
        source_history_days=SOURCE_HISTORY_DAYS,
    )
    wide, schema_contract, wide_path = build_or_load_wide_clean(
        dataset_id=dataset_id,
        payload=payload,
        raw_root=raw_root,
        clean_input_root=clean_input_root,
        run_dir=run_dir,
        runtime=runtime["runtime_window_metadata"],
        reuse_existing_wide_clean=reuse_existing_wide_clean,
    )
    schema_contract["wide_clean_rows_before_runtime_window"] = int(len(wide))
    solidified_source_full_rows = _parquet_row_count(parquet_root / f"dataset{dataset_id}-source.parquet")
    wide = _window_wide_source(wide, runtime["runtime_window_metadata"], narrow_source)
    group_cols = [str(value) for value in payload["group_cols"]]
    schema = compare_feature_schema(features, wide.columns.tolist())
    schema_contract.update(schema)
    if schema["feature_columns_missing_in_wide"]:
        raise ValueError(
            f"D{dataset_id} wide-clean missing required JSON features: {schema['feature_columns_missing_in_wide']}"
        )
    narrow_diag = compute_pool_domain_diagnostics(
        narrow_source, domain_column=domain_column, domain_value=domain_value, group_cols=group_cols
    )
    wide_diag = compute_pool_domain_diagnostics(
        wide, domain_column=domain_column, domain_value=domain_value, group_cols=group_cols
    )
    wide_without_eligible = apply_source_domain_policy(
        wide, payload.get("domain_filter"), information_sharing="without", entity_group_cols=group_cols
    ).frame
    wide_with_eligible = apply_source_domain_policy(
        wide, payload.get("domain_filter"), information_sharing="with", entity_group_cols=group_cols
    ).frame
    wide_with, wide_with_cap = stable_cap_entities(
        wide_with_eligible, group_cols, cap=max_source_entities
    )
    wide_without, wide_without_cap = stable_cap_entities(
        wide_without_eligible, group_cols, cap=max_source_entities
    )
    all_rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_payload = payloads[mode]
        narrow_eligible = apply_source_domain_policy(
            narrow_source, mode_payload.get("domain_filter"), information_sharing=mode, entity_group_cols=group_cols
        ).frame
        narrow_mode, narrow_cap = stable_cap_entities(
            narrow_eligible, group_cols, cap=max_source_entities
        )
        narrow_topk, _ = _topk_for_targets(target=target, source=narrow_mode, payload=mode_payload, features=features, top_k=top_k, domain_column=domain_column)
        wide_with_topk, wide_with_domains = _topk_for_targets(target=target, source=wide_with, payload=mode_payload, features=features, top_k=top_k, domain_column=domain_column)
        wide_without_topk, wide_without_domains = _topk_for_targets(target=target, source=wide_without, payload=mode_payload, features=features, top_k=top_k, domain_column=domain_column)
        for target_key in narrow_topk:
            existing = _existing_json_topk(mode_payload, target_key)
            narrow_keys = narrow_topk[target_key]
            with_keys = wide_with_topk[target_key]
            without_keys = wide_without_topk[target_key]
            all_rows.append({
                "dataset": dataset_id,
                "mode": mode,
                "target_entity_key": target_key,
                "configured_domain_filter_column": domain_column,
                "configured_domain_filter_value": domain_value,
                "narrow_source_pool_rows": narrow_diag["source_rows"],
                "narrow_source_pool_entities": narrow_diag["source_entities"],
                "narrow_source_domain_nunique": narrow_diag["source_domain_nunique"],
                "narrow_source_domain_values": narrow_diag["source_domain_values"],
                "narrow_after_filter_rows": narrow_diag["after_filter_rows"],
                "narrow_after_filter_entities": narrow_diag["after_filter_entities"],
                "domain_filter_vacuous_on_narrow": narrow_diag["domain_filter_vacuous"],
                "wide_source_pool_rows": wide_diag["source_rows"],
                "wide_source_pool_entities": wide_diag["source_entities"],
                "wide_source_domain_nunique": wide_diag["source_domain_nunique"],
                "wide_source_domain_values": wide_diag["source_domain_values"],
                "wide_after_filter_rows": wide_diag["after_filter_rows"],
                "wide_after_filter_entities": wide_diag["after_filter_entities"],
                "domain_filter_effective_on_wide": wide_diag["domain_filter_effective"],
                **schema,
                **{f"narrow_{key}": value for key, value in narrow_cap.items()},
                **{f"wide_with_{key}": value for key, value in wide_with_cap.items()},
                **{f"wide_without_{key}": value for key, value in wide_without_cap.items()},
                **runtime,
                "existing_json_topk_entity_keys": existing,
                "recomputed_narrow_topk_entity_keys": narrow_keys,
                "wide_with_topk_entity_keys": with_keys,
                "wide_without_topk_entity_keys": without_keys,
                "overlap_existing_json_vs_recomputed_narrow": topk_overlap_ratio(existing, narrow_keys, top_k=top_k),
                "overlap_narrow_vs_wide_with": topk_overlap_ratio(narrow_keys, with_keys, top_k=top_k),
                "overlap_narrow_vs_wide_without": topk_overlap_ratio(narrow_keys, without_keys, top_k=top_k),
                "overlap_wide_with_vs_wide_without": topk_overlap_ratio(with_keys, without_keys, top_k=top_k),
                "wide_with_selected_domain_values": wide_with_domains[target_key],
                "wide_without_selected_domain_values": wide_without_domains[target_key],
                "wide_with_selected_non_configured_domain_count": wide_with_non_configured_domain_count(wide_with_domains[target_key], domain_value),
                "raw_train_rows": schema_contract.get("raw_train_rows"),
                "raw_eval_rows": schema_contract.get("raw_eval_rows"),
                "constructed_wide_clean_rows": schema_contract["wide_clean_rows_before_runtime_window"],
                "solidified_source_rows": solidified_source_full_rows,
                "wide_vs_solidified_row_count_delta": int(schema_contract["wide_clean_rows_before_runtime_window"] - solidified_source_full_rows),
                "wide_vs_solidified_row_count_ratio": float(schema_contract["wide_clean_rows_before_runtime_window"] / solidified_source_full_rows) if solidified_source_full_rows else None,
                "status": "ok",
                "error_message": "",
                "notes": "diagnostic dry-run only",
            })
    summary = {
        "dataset": dataset_id,
        "status": "ok",
        "wide_clean_path": str(wide_path),
        "runtime_window": runtime,
        "runtime_windows_match_current_knn_metadata": _runtime_windows_match_json(payloads, runtime),
        "schema_contract": schema_contract,
        "narrow_diagnostics": narrow_diag,
        "wide_diagnostics": wide_diag,
        "rows": all_rows,
        "constructed_wide_clean_rows": schema_contract["wide_clean_rows_before_runtime_window"],
        "solidified_source_rows": solidified_source_full_rows,
        "wide_vs_solidified_row_count_delta": int(schema_contract["wide_clean_rows_before_runtime_window"] - solidified_source_full_rows),
        "wide_vs_solidified_row_count_ratio": float(schema_contract["wide_clean_rows_before_runtime_window"] / solidified_source_full_rows) if solidified_source_full_rows else None,
    }
    _write_dataset_artifacts(run_dir, dataset_id, all_rows, summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, choices=[4, 5, 6], required=True)
    parser.add_argument("--mode", choices=["with", "without", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-source-entities", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--raw-input-root", type=Path, default=PROJECT_ROOT / "数据集" / "原始数据")
    parser.add_argument("--clean-input-root", type=Path, default=None)
    parser.add_argument("--reuse-existing-wide-clean", action="store_true")
    parser.add_argument("--fail-if-wide-clean-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--parquet-root", type=Path, default=PROJECT_ROOT / "数据集" / "固化数据", help=argparse.SUPPRESS)
    parser.add_argument("--knn-root", type=Path, default=SOLIDIFIED_KNN_ROOT, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ALLOWED_OUTPUT_ROOT / f"raw_wide_knn_dryrun_{timestamp}")
    output_dir = validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    modes = ["with", "without"] if args.mode == "both" else [args.mode]
    summaries: list[dict[str, Any]] = []
    try:
        if args.reuse_existing_wide_clean and args.clean_input_root is None:
            raise ValueError("--reuse-existing-wide-clean requires --clean-input-root")
        summaries.append(run_dataset_comparison(
            dataset_id=args.dataset,
            modes=modes,
            top_k=args.top_k,
            max_source_entities=args.max_source_entities,
            run_dir=output_dir,
            raw_root=args.raw_input_root,
            clean_input_root=args.clean_input_root,
            reuse_existing_wide_clean=args.reuse_existing_wide_clean,
            parquet_root=args.parquet_root,
            knn_root=args.knn_root,
        ))
    except Exception as exc:
        summary = {
            "dataset": args.dataset,
            "status": "failed",
            "error_message": f"{type(exc).__name__}: {exc}",
            "runtime_window": {},
            "schema_contract": {},
            "rows": [],
        }
        _write_dataset_artifacts(output_dir, args.dataset, [{
            "dataset": args.dataset, "mode": args.mode, "status": "failed",
            "error_message": summary["error_message"], "notes": "diagnostic dry-run only",
        }], summary)
        summaries.append(summary)
    _markdown_report(output_dir, summaries)
    print(f"raw/wide KNN diagnostic output: {output_dir}")
    print("Diagnostic dry-run only; no official parquet, KNN JSON, or experiment result was modified.")
    if summaries[-1].get("status") == "failed":
        if not args.fail_if_wide_clean_missing and "WideCleanUnavailable" in summaries[-1].get("error_message", ""):
            return
        raise SystemExit(1)


if __name__ == "__main__":
    main()
