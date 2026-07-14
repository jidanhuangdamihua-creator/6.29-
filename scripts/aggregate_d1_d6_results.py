#!/usr/bin/env python3
"""Aggregate D1-D6 experiment result CSVs into final summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.constants import (
    RESULT_CONTRACT_VERSION,
    RESULT_SCHEMA_COLUMNS,
    SCHEMA_FAMILY_D1_D3,
    SCHEMA_FAMILY_D4_D6,
    preferred_columns_with_extras,
)
from src.evaluation.metric_contract import (
    build_formal_smape_aggregates,
)
from src.protocols.experiment_protocol import FORMAL_SEEDS
from src.protocols.experiment_protocol import FORMAL_HORIZONS, FORMAL_METHODS
from src.utils.result_acceptance import (
    AcceptanceScope,
    AggregateProfile,
    ExpectedResultContract,
)
from src.utils.run_artifacts import (
    CodeIdentity,
    discover_code_identity,
    publish_global_aggregate,
)
from src.utils.run_layout import RunLayout
from src.utils.result_validation import (
    annotate_silent_metric_failure,
    classify_protocol_result,
    confirmed_baseline_rows,
    promote_complete_baseline_groups,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_summary"
EXPECTED_DATASET_IDS = tuple(range(1, 7))
EXPECTED_MODES = ("without", "with")

# Legacy completed runs kept for explicit fallback only. They are not used by default.
SOURCE_CSVS: Dict[int, Path] = {
    1: PROJECT_ROOT / "outputs/runs/20260625_224541/results/dataset1_results.csv",
    2: PROJECT_ROOT / "outputs/runs/20260625_224541/results/dataset2_results.csv",
    3: PROJECT_ROOT / "outputs/runs/20260627_213402/results/dataset3_results.csv",
    4: PROJECT_ROOT / "outputs/runs/20260627_164818_D4_300d_without/results/dataset4_results.csv",
    5: PROJECT_ROOT / "outputs/runs/20260627_151252_D5_300d_without/results/dataset5_results.csv",
    6: PROJECT_ROOT / "outputs/runs/20260627_160244_D6_300d_without/results/dataset6_results.csv",
}

PREFERRED_COLUMNS = [
    "dataset_id",
    "information_sharing",
    "target_entity_key",
    "target_entity_id",
    "target_store_id",
    "target_item_id",
    "method",
    "scenario",
    "smape",
    "rmse",
    "mae",
    "metric_space_used",
    "paper_metric_aligned",
    "valid_source_count",
    "skipped_source_count",
    "failed_source_count",
    "failed_source_keys",
    "skipped_nonfinite_source_count",
    "failed_sources",
    "selected_features",
    "date_alignment_mode",
    "source_csv_path",
]

_CsvDataFrameCache = Dict[Path, Tuple[int, int, pd.DataFrame]]


def aggregate_formal_smape(
    frame: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Public formal aggregation entry point shared with tests and reports."""
    return build_formal_smape_aggregates(frame, expected_seeds=expected_seeds)


def _read_csv_dataframe(path: Path, csv_cache: Optional[_CsvDataFrameCache] = None) -> pd.DataFrame:
    if csv_cache is None:
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    path = Path(path)
    stat = path.stat()
    cached = csv_cache.get(path)
    if cached is not None:
        cached_mtime_ns, cached_size, cached_df = cached
        if cached_mtime_ns == stat.st_mtime_ns and cached_size == stat.st_size:
            return cached_df.copy()

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    csv_cache[path] = (stat.st_mtime_ns, stat.st_size, df)
    return df.copy()


def normalize_information_sharing(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    if "without" in text or text in {"no_information", "no_info", "none"}:
        return "without"
    if text in {"with", "with_information_sharing", "info_sharing"} or "with_info" in text:
        return "with"
    return text


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _dataset_id_from_row(row: Dict[str, str], fallback: int) -> int:
    for key in ("dataset_id", "dataset"):
        raw = row.get(key, "")
        if not raw:
            continue
        match = re.search(r"(\d+)", str(raw))
        if match:
            return int(match.group(1))
    return fallback


def _dataset_id_from_path(path: Path) -> Optional[int]:
    match = re.search(r"dataset(\d+)", path.name.lower())
    return int(match.group(1)) if match else None


def _mode_from_path(path: Path, dataset_id: int) -> str:
    text = str(path).lower()
    dataset = f"dataset{dataset_id}"
    if f"d{dataset_id}_without" in text or f"{dataset}_without" in text:
        return "without"
    if f"d{dataset_id}_with" in text or f"{dataset}_with" in text:
        return "with"
    return ""


def _modes_from_csv(path: Path, csv_cache: Optional[_CsvDataFrameCache] = None) -> List[str]:
    try:
        df = _read_csv_dataframe(path, csv_cache=csv_cache)
    except Exception:
        return []
    modes = set()
    for column in ("information_sharing", "scenario"):
        if column not in df.columns:
            continue
        for value in df[column].dropna().astype(str).unique().tolist():
            mode = normalize_information_sharing(value)
            if mode in EXPECTED_MODES:
                modes.add(mode)
    return sorted(modes)


def _candidate_score(path: Path, dataset_id: int, mode: str) -> int:
    text = str(path).lower()
    name = path.name.lower()
    if f"d{dataset_id}_{mode}" in text:
        return 4
    if f"dataset{dataset_id}_{mode}" in name:
        return 3
    if f"_{mode}" in text:
        return 2
    return 1


def _choose_candidate(
    candidates: Sequence[Path],
    dataset_id: int,
    mode: str,
    strict: bool,
) -> Tuple[Path, str]:
    ranked = sorted(
        candidates,
        key=lambda p: (_candidate_score(p, dataset_id, mode), p.stat().st_mtime),
        reverse=True,
    )
    if len(ranked) == 1:
        return ranked[0], ""
    top_score = _candidate_score(ranked[0], dataset_id, mode)
    tied = [p for p in ranked if _candidate_score(p, dataset_id, mode) == top_score]
    if strict and len(tied) > 1:
        raise ValueError(
            f"Multiple result CSV candidates for D{dataset_id} {mode}: "
            + ", ".join(str(p) for p in tied)
        )
    return ranked[0], (
        f"multiple candidates for D{dataset_id} {mode}; selected newest/highest-priority path {ranked[0]}"
    )


def discover_source_csvs(
    run_dir: Path,
    *,
    strict: bool = False,
    allow_missing: bool = False,
    csv_cache: Optional[_CsvDataFrameCache] = None,
) -> Tuple[Dict[Tuple[int, str], Path], List[Dict[str, Any]]]:
    run_dir = Path(run_dir)
    paths = sorted(run_dir.rglob("results/dataset*.csv")) if run_dir.exists() else []
    buckets: Dict[Tuple[int, str], List[Path]] = defaultdict(list)

    for path in paths:
        dataset_id = _dataset_id_from_path(path)
        if dataset_id is None or dataset_id not in EXPECTED_DATASET_IDS:
            continue
        mode = _mode_from_path(path, dataset_id)
        modes = [mode] if mode in EXPECTED_MODES else _modes_from_csv(path, csv_cache=csv_cache)
        if not modes:
            modes = [""]
        for candidate_mode in modes:
            buckets[(dataset_id, candidate_mode)].append(path)

    selected: Dict[Tuple[int, str], Path] = {}
    audit_rows: List[Dict[str, Any]] = []
    for dataset_id in EXPECTED_DATASET_IDS:
        for mode in EXPECTED_MODES:
            candidates = list(dict.fromkeys(buckets.get((dataset_id, mode), []) + buckets.get((dataset_id, ""), [])))
            if not candidates:
                row = {
                    "dataset_id": dataset_id,
                    "information_sharing": mode,
                    "source_csv_path": "",
                    "status": "missing",
                    "warning": f"Missing result CSV for D{dataset_id} {mode}",
                }
                audit_rows.append(row)
                if strict and not allow_missing:
                    raise FileNotFoundError(row["warning"])
                continue
            path, warning = _choose_candidate(candidates, dataset_id, mode, strict=strict)
            selected[(dataset_id, mode)] = path
            audit_rows.append(
                {
                    "dataset_id": dataset_id,
                    "information_sharing": mode,
                    "source_csv_path": str(path),
                    "status": "selected",
                    "warning": warning,
                }
            )
    return selected, audit_rows


def _assert_dataset3_result_target_is_store10(df: pd.DataFrame, path: Path) -> None:
    """Reject Dataset3 result CSVs whose target identity cannot be proven store 10."""
    target_identity_columns = [
        column
        for column in ("target_entity_id", "target_store_id", "entity_id", "store_id")
        if column in df.columns
    ]
    if not target_identity_columns:
        raise ValueError(
            "D3 result file has no target identity columns and cannot be safely aggregated: "
            f"{path}"
        )

    for column in target_identity_columns:
        values = sorted(df[column].astype(str).unique().tolist())
        if values != ["10"]:
            raise ValueError(
                f"D3 wrong/stale result {column}: expected ['10'], got {values} in {path}"
            )


def _normalize_row(row: Dict[str, str], dataset_hint: int, source_path: Path) -> Dict[str, str]:
    out = dict(row)
    dataset_id = _dataset_id_from_row(out, dataset_hint)
    out["dataset_id"] = str(dataset_id)
    out.setdefault("result_contract_version", RESULT_CONTRACT_VERSION)
    if not str(out.get("result_contract_version", "")).strip():
        out["result_contract_version"] = RESULT_CONTRACT_VERSION
    out.setdefault(
        "schema_family",
        SCHEMA_FAMILY_D1_D3 if dataset_id in {1, 2, 3} else SCHEMA_FAMILY_D4_D6,
    )
    if not str(out.get("schema_family", "")).strip():
        out["schema_family"] = SCHEMA_FAMILY_D1_D3 if dataset_id in {1, 2, 3} else SCHEMA_FAMILY_D4_D6
    mode = normalize_information_sharing(out.get("information_sharing") or out.get("scenario"))
    if mode in EXPECTED_MODES:
        out["information_sharing"] = mode
        out["scenario"] = mode
    elif not out.get("scenario") and out.get("information_sharing"):
        out["scenario"] = out["information_sharing"]
    if not out.get("target_entity_key") and not out.get("target_entity_id"):
        if dataset_id == 3:
            raise ValueError(
                "D3 missing target identity; refusing to set target_entity_key=GLOBAL"
            )
        out["target_entity_key"] = "GLOBAL"
    if not out.get("valid_source_count"):
        for alt in ("actual_pretrained_model_count", "source_count", "effective_k"):
            if out.get(alt) not in (None, ""):
                out["valid_source_count"] = out[alt]
                break
    if not out.get("selected_features"):
        for alt in ("rfe_selected_features", "feature_cols_final"):
            if out.get(alt) not in (None, ""):
                out["selected_features"] = out[alt]
                break
    out["source_csv_path"] = str(source_path)
    for column in RESULT_SCHEMA_COLUMNS:
        out.setdefault(column, "")
    normalized = annotate_silent_metric_failure(out)
    normalized["result_status"] = classify_protocol_result(normalized)
    return normalized


def _read_source(
    path: Path,
    dataset_hint: int,
    csv_cache: Optional[_CsvDataFrameCache] = None,
) -> List[Dict[str, str]]:
    df = _read_csv_dataframe(path, csv_cache=csv_cache)
    if df.columns.empty:
        raise ValueError(f"Invalid CSV (no header): {path}")
    if dataset_hint == 3:
        _assert_dataset3_result_target_is_store10(df, path)
    return [_normalize_row(row, dataset_hint, path) for row in df.to_dict(orient="records")]


def _union_fieldnames(rows: Sequence[Dict[str, str]]) -> List[str]:
    discovered = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                discovered.append(key)
                seen.add(key)
    preferred = list(dict.fromkeys(list(RESULT_SCHEMA_COLUMNS) + PREFERRED_COLUMNS))
    return preferred_columns_with_extras(discovered, preferred)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _target_key(row: Dict[str, str]) -> str:
    return row.get("target_entity_key") or row.get("target_entity_id") or "GLOBAL"


def _metric_stats(rows: Sequence[Dict[str, str]]) -> Tuple[int, int, int, int]:
    smape_nan = smape_inf = rmse_nan = rmse_inf = 0
    for row in rows:
        for raw, counter_nan, counter_inf in (
            (row.get("smape"), "smape_nan", "smape_inf"),
            (row.get("rmse"), "rmse_nan", "rmse_inf"),
        ):
            text = "" if raw is None else str(raw).strip()
            if not text:
                if counter_nan == "smape_nan":
                    smape_nan += 1
                else:
                    rmse_nan += 1
                continue
            try:
                value = float(text)
            except ValueError:
                if counter_nan == "smape_nan":
                    smape_nan += 1
                else:
                    rmse_nan += 1
                continue
            if math.isnan(value):
                if counter_nan == "smape_nan":
                    smape_nan += 1
                else:
                    rmse_nan += 1
            elif math.isinf(value):
                if counter_inf == "smape_inf":
                    smape_inf += 1
                else:
                    rmse_inf += 1
    return smape_nan, smape_inf, rmse_nan, rmse_inf


def _mean_metric(rows: Iterable[Dict[str, str]], metric: str) -> Tuple[Optional[float], int]:
    values = []
    for row in rows:
        parsed = _parse_float(row.get(metric))
        if parsed is not None:
            values.append(parsed)
    if not values:
        return None, 0
    return statistics.fmean(values), len(values)


def _distribution(rows: Sequence[Dict[str, str]], column: str) -> str:
    counter = Counter(str(row.get(column, "")).strip() or "<empty>" for row in rows)
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _duplicate_count(rows: Sequence[Dict[str, str]]) -> int:
    keys = [
        (
            row.get("dataset_id", ""),
            row.get("method", ""),
            _target_key(row),
            row.get("information_sharing") or row.get("scenario", ""),
        )
        for row in rows
    ]
    counter = Counter(keys)
    return sum(count - 1 for count in counter.values() if count > 1)


def _build_audit_rows(
    all_rows: Sequence[Dict[str, str]],
    discovery_audit: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        mode = normalize_information_sharing(row.get("information_sharing") or row.get("scenario"))
        if mode in EXPECTED_MODES:
            grouped[(int(row["dataset_id"]), mode)].append(row)

    audit_rows: List[Dict[str, Any]] = []
    for audit in discovery_audit:
        dataset_id = int(audit["dataset_id"])
        mode = str(audit["information_sharing"])
        group = grouped.get((dataset_id, mode), [])
        if audit["status"] == "missing":
            audit_rows.append(dict(audit, rows=0))
            continue
        audit_rows.append(
            {
                **audit,
                "rows": len(group),
                "error_rows": sum(1 for row in group if str(row.get("error", "")).strip()),
                "missing_rmse": sum(1 for row in group if _parse_float(row.get("rmse")) is None),
                "missing_smape": sum(1 for row in group if _parse_float(row.get("smape")) is None),
                "metric_space_used_distribution": _distribution(group, "metric_space_used"),
                "paper_metric_aligned_distribution": _distribution(group, "paper_metric_aligned"),
                "duplicate_dataset_method_target_mode_rows": _duplicate_count(group),
            }
        )
    return audit_rows


def _write_metric_summaries(
    output: Path,
    all_rows: Sequence[Dict[str, str]],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> List[Path]:
    aggregates = aggregate_formal_smape(
        pd.DataFrame(all_rows),
        expected_seeds=expected_seeds,
    )
    dataset_method_rows = aggregates["dataset_macro"].rename(
        columns={"dataset": "dataset_id", "smape": "mean_smape"}
    ).to_dict(orient="records")
    for row in dataset_method_rows:
        row.update({"row_count": 1, "smape_valid_count": 1, "rmse_valid_count": 0, "mean_rmse": ""})

    dataset_method_path = output.with_name(f"{output.stem}_dataset_method_metrics.csv")
    _write_csv(
        dataset_method_path,
        dataset_method_rows,
        [
            "dataset_id",
            "method",
            "horizon",
            "sharing_scenario",
            "row_count",
            "smape_valid_count",
            "rmse_valid_count",
            "mean_smape",
            "mean_rmse",
        ],
    )

    method_mean_rows = aggregates["cross_dataset_macro"].rename(
        columns={"smape": "mean_smape"}
    ).to_dict(orient="records")
    for row in method_mean_rows:
        row.update({"row_count": 1, "smape_valid_count": 1, "rmse_valid_count": 0, "mean_rmse": ""})

    method_mean_path = output.with_name(f"{output.stem}_method_mean_metrics.csv")
    _write_csv(
        method_mean_path,
        method_mean_rows,
        [
            "method",
            "horizon",
            "sharing_scenario",
            "rank",
            "row_count",
            "smape_valid_count",
            "rmse_valid_count",
            "mean_smape",
            "mean_rmse",
        ],
    )
    return [dataset_method_path, method_mean_path]


def _write_best_method_outputs(
    output: Path,
    all_rows: Sequence[Dict[str, str]],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> List[Path]:
    aggregates = aggregate_formal_smape(
        pd.DataFrame(all_rows),
        expected_seeds=expected_seeds,
    )
    seed_detail = aggregates["eligible_rows"].copy()
    detail_columns = [
        "dataset",
        "target",
        "method",
        "horizon",
        "sharing_scenario",
        "seed",
        "smape",
        "seed_rank",
    ]
    if seed_detail.empty:
        seed_detail = pd.DataFrame(columns=detail_columns)
    else:
        seed_detail["seed_rank"] = seed_detail.groupby(
            ["dataset", "target", "horizon", "sharing_scenario", "seed"],
            dropna=False,
        )["smape"].rank(method="average", ascending=True)
    seed_detail_path = output.with_name(
        f"{output.stem}_method_results_by_target_seed.csv"
    )
    _write_csv(
        seed_detail_path,
        seed_detail.to_dict(orient="records"),
        list(seed_detail.columns),
    )
    best_by_target_rows: List[Dict[str, Any]] = []
    wins_by_dataset: Dict[int, Counter] = defaultdict(Counter)
    target_group = ["dataset", "target", "horizon", "sharing_scenario"]
    for key, group in aggregates["seed_mean"].groupby(
        target_group,
        dropna=False,
        sort=True,
    ):
        dataset, target, horizon, scenario = key
        ranked = group.sort_values(["smape", "method"], kind="stable")
        best_row = ranked.iloc[0]
        best_smape = float(best_row["smape"])
        method = str(best_row["method"])
        dataset_id = _dataset_id_from_row({"dataset": str(dataset)}, fallback=0)
        best_by_target_rows.append(
            {
                "dataset_id": dataset_id,
                "target_entity_key": target,
                "horizon": horizon,
                "information_sharing": scenario,
                "best_method": method,
                "best_smape": best_smape,
                "best_rmse": "",
                "candidate_method_count": int(ranked["method"].nunique()),
            }
        )
        wins_by_dataset[dataset_id][method] += 1

    best_target_path = output.with_name(f"{output.stem}_best_method_by_target.csv")
    _write_csv(
        best_target_path,
        best_by_target_rows,
        [
            "dataset_id",
            "target_entity_key",
            "horizon",
            "information_sharing",
            "best_method",
            "best_smape",
            "best_rmse",
            "candidate_method_count",
        ],
    )

    md_lines = [
        "# D1-D6 Best Method Summary by Dataset",
        "",
        "Best method per target entity is chosen by lowest complete-seed mean sMAPE "
        "within each `(dataset_id, target_entity_key, horizon, information_sharing)` group.",
        "",
    ]
    for dataset_id in sorted(wins_by_dataset):
        counter = wins_by_dataset[dataset_id]
        total_targets = sum(counter.values())
        md_lines.append(f"## Dataset {dataset_id}")
        md_lines.append("")
        md_lines.append(f"- Targets evaluated: {total_targets}")
        md_lines.append("- Win counts by method:")
        for method, wins in counter.most_common():
            md_lines.append(f"  - {method}: {wins}")
        if counter:
            top_method, top_wins = counter.most_common(1)[0]
            md_lines.append(f"- Most frequent winner: **{top_method}** ({top_wins}/{total_targets} targets)")
        md_lines.append("")

    best_dataset_md_path = output.with_name(f"{output.stem}_best_method_by_dataset.md")
    best_dataset_md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    return [best_target_path, best_dataset_md_path, seed_detail_path]


def aggregate(
    *,
    run_dir: Path,
    output: Path | None = None,
    strict: bool = False,
    allow_missing: bool = False,
    legacy_fallback: bool = False,
) -> Dict[str, Any]:
    """Legacy exploratory audit; never a formal acceptance or sealing path."""
    run_dir = Path(run_dir)
    if output is None:
        output = run_dir / "aggregate" / "legacy_audit_d1_d6_all_results.csv"
    csv_cache: _CsvDataFrameCache = {}
    selected, discovery_audit = discover_source_csvs(
        run_dir,
        strict=strict,
        allow_missing=allow_missing,
        csv_cache=csv_cache,
    )
    if not selected and legacy_fallback:
        selected = {(dataset_id, ""): path for dataset_id, path in SOURCE_CSVS.items()}

    unique_paths = list(dict.fromkeys(selected.values()))
    all_rows: List[Dict[str, str]] = []
    for path in unique_paths:
        dataset_hint = _dataset_id_from_path(path) or 0
        rows = _read_source(path, dataset_hint, csv_cache=csv_cache)
        all_rows.extend(rows)

    if not all_rows:
        if not allow_missing:
            raise FileNotFoundError(f"No result CSV rows found under {run_dir}")
        output.parent.mkdir(parents=True, exist_ok=True)
        empty_columns = list(dict.fromkeys(list(RESULT_SCHEMA_COLUMNS) + PREFERRED_COLUMNS))
        _write_csv(output, [], empty_columns)
        audit_rows = _build_audit_rows([], discovery_audit)
        audit_path = output.with_name(f"{output.stem}_audit.csv")
        _write_csv(audit_path, audit_rows, sorted({key for row in audit_rows for key in row}))
        return {"all_results_path": output, "audit_path": audit_path, "audit_rows": audit_rows}

    promoted_frame = promote_complete_baseline_groups(pd.DataFrame(all_rows))
    all_rows = promoted_frame.to_dict(orient="records")
    output = Path(output)
    all_fieldnames = _union_fieldnames(all_rows)
    _write_csv(output, all_rows, all_fieldnames)

    audit_rows = _build_audit_rows(all_rows, discovery_audit)
    audit_path = output.with_name(f"{output.stem}_audit.csv")
    _write_csv(audit_path, audit_rows, sorted({key for row in audit_rows for key in row}))

    baseline_rows = confirmed_baseline_rows(promoted_frame).to_dict(orient="records")
    extra_paths = _write_metric_summaries(
        output,
        baseline_rows,
        expected_seeds=FORMAL_SEEDS,
    )
    extra_paths.extend(
        _write_best_method_outputs(
            output,
            baseline_rows,
            expected_seeds=FORMAL_SEEDS,
        )
    )

    row_counts = Counter(int(row["dataset_id"]) for row in all_rows)
    smape_nan, smape_inf, rmse_nan, rmse_inf = _metric_stats(all_rows)

    print("=== D1-D6 Aggregation Complete ===")
    print("\nSource CSV paths used:")
    for path in unique_paths:
        print(f"  {path}")
    for row in audit_rows:
        if row.get("warning"):
            print(f"WARNING: {row['warning']}")
    print(f"\nTotal row count: {len(all_rows)}")
    print("\nRow count by dataset:")
    for dataset_id in sorted(row_counts):
        print(f"  D{dataset_id}: {row_counts[dataset_id]}")
    print("\nNaN/inf metric counts:")
    print(f"  smape NaN/empty: {smape_nan}")
    print(f"  smape inf: {smape_inf}")
    print(f"  rmse NaN/empty: {rmse_nan}")
    print(f"  rmse inf: {rmse_inf}")
    print("\nOutput file paths:")
    for path in [output, audit_path] + extra_paths:
        print(f"  {path}")

    return {
        "all_results_path": output,
        "audit_path": audit_path,
        "extra_paths": extra_paths,
        "audit_rows": audit_rows,
    }


def aggregate_accepted_formal_run(
    *,
    run_dir: Path,
    profile: AggregateProfile,
) -> Dict[str, Any]:
    """Re-publish only manifest-backed accepted modes through shared authority."""
    from scripts.run_strict_protocol_baseline import build_mode_expected_contract

    run_root = Path(run_dir)
    plan_path = run_root / "run_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"formal aggregation requires run plan: {plan_path}")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_identity = payload.get("code_identity")
    if not isinstance(raw_identity, dict):
        raise ValueError("formal run plan is missing code_identity")
    plan_identity = CodeIdentity(
        git_commit=str(raw_identity.get("git_commit", "")),
        dirty=bool(raw_identity.get("dirty", True)),
        worktree_digest=str(raw_identity.get("worktree_digest", "")),
    )
    current_identity = discover_code_identity(PROJECT_ROOT)
    if current_identity != plan_identity or current_identity.dirty:
        raise ValueError("formal aggregation code identity does not match the clean run plan")

    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("formal run plan contains no cells")
    groups = {
        (int(cell["dataset_id"]), str(cell["mode"]))
        for cell in cells
        if isinstance(cell, dict)
    }
    full_groups = {
        (dataset_id, mode)
        for dataset_id in range(1, 7)
        for mode in ("without", "with")
    }
    if profile is AggregateProfile.FULL_D1_D6_BASELINE and groups != full_groups:
        raise ValueError("full formal profile requires all 12 dataset-mode groups")

    layout = RunLayout(run_root)
    mode_contracts = [
        build_mode_expected_contract(dataset=f"d{dataset_id}", scenario=mode)
        for dataset_id, mode in sorted(groups)
    ]
    targets = {
        key: value
        for contract in mode_contracts
        for key, value in contract.targets_by_dataset_mode.items()
    }
    dataset_ids = tuple(sorted({dataset_id for dataset_id, _ in groups}))
    modes = tuple(
        mode for mode in ("without", "with") if any(item_mode == mode for _, item_mode in groups)
    )
    expected = ExpectedResultContract(
        scope=AcceptanceScope.GLOBAL_AGGREGATE,
        formal=True,
        dataset_ids=dataset_ids,
        modes=modes,
        protocol_tracks=("strict_paper",),
        targets_by_dataset_mode=targets,
        methods=FORMAL_METHODS,
        horizons=FORMAL_HORIZONS,
        seeds=FORMAL_SEEDS,
        confirmation_eligible=True,
        aggregate_profile=profile,
    )
    mode_paths = [layout.mode_result(dataset_id, mode) for dataset_id, mode in sorted(groups)]
    manifest = publish_global_aggregate(
        mode_paths,
        stable_path=layout.aggregate_result,
        expected=expected,
        code_identity=plan_identity,
    )
    return {
        "all_results_path": layout.aggregate_result,
        "manifest_path": layout.aggregate_manifest,
        "sha256": manifest["sha256"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate accepted formal modes, or explicitly run the legacy audit aggregator."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--legacy-fallback", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--formal-profile",
        choices=[profile.value for profile in AggregateProfile],
    )
    mode.add_argument("--legacy-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.formal_profile is not None:
        if args.output is not None or args.strict or args.allow_missing or args.legacy_fallback:
            raise SystemExit(
                "formal aggregation uses RunLayout and accepted manifests; legacy options are forbidden"
            )
        aggregate_accepted_formal_run(
            run_dir=args.run_dir,
            profile=AggregateProfile(args.formal_profile),
        )
        return
    aggregate(
        run_dir=args.run_dir,
        output=args.output,
        strict=bool(args.strict),
        allow_missing=bool(args.allow_missing),
        legacy_fallback=bool(args.legacy_fallback),
    )


if __name__ == "__main__":
    main()
