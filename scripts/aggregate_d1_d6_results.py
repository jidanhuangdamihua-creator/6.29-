#!/usr/bin/env python3
"""Aggregate D1-D6 experiment result CSVs into final summary tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

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


def _modes_from_csv(path: Path) -> List[str]:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
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
) -> Tuple[Dict[Tuple[int, str], Path], List[Dict[str, Any]]]:
    run_dir = Path(run_dir)
    paths = sorted(run_dir.rglob("results/dataset*.csv")) if run_dir.exists() else []
    buckets: Dict[Tuple[int, str], List[Path]] = defaultdict(list)

    for path in paths:
        dataset_id = _dataset_id_from_path(path)
        if dataset_id is None or dataset_id not in EXPECTED_DATASET_IDS:
            continue
        mode = _mode_from_path(path, dataset_id)
        modes = [mode] if mode in EXPECTED_MODES else _modes_from_csv(path)
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
    return out


def _read_source(path: Path, dataset_hint: int) -> List[Dict[str, str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.columns.empty:
        raise ValueError(f"Invalid CSV (no header): {path}")
    if dataset_hint == 3:
        _assert_dataset3_result_target_is_store10(df, path)
    return [_normalize_row(row, dataset_hint, path) for row in df.to_dict(orient="records")]


def _union_fieldnames(rows: Sequence[Dict[str, str]]) -> List[str]:
    columns: List[str] = []
    seen = set()
    for name in PREFERRED_COLUMNS:
        if any(name in row for row in rows):
            columns.append(name)
            seen.add(name)
    extras = sorted({key for row in rows for key in row.keys() if key not in seen})
    return columns + extras


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


def _write_metric_summaries(output: Path, all_rows: Sequence[Dict[str, str]]) -> List[Path]:
    dataset_method_rows: List[Dict[str, Any]] = []
    method_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"smape": [], "rmse": []})
    dataset_method_groups: Dict[Tuple[int, str], List[Dict[str, str]]] = defaultdict(list)

    for row in all_rows:
        dataset_id = int(row["dataset_id"])
        method = row.get("method", "")
        key = (dataset_id, method)
        dataset_method_groups[key].append(row)
        for metric in ("smape", "rmse"):
            parsed = _parse_float(row.get(metric))
            if parsed is not None:
                method_values[method][metric].append(parsed)

    for (dataset_id, method), group in sorted(dataset_method_groups.items()):
        mean_smape, smape_n = _mean_metric(group, "smape")
        mean_rmse, rmse_n = _mean_metric(group, "rmse")
        dataset_method_rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "row_count": len(group),
                "smape_valid_count": smape_n,
                "rmse_valid_count": rmse_n,
                "mean_smape": "" if mean_smape is None else mean_smape,
                "mean_rmse": "" if mean_rmse is None else mean_rmse,
            }
        )

    dataset_method_path = output.with_name(f"{output.stem}_dataset_method_metrics.csv")
    _write_csv(
        dataset_method_path,
        dataset_method_rows,
        ["dataset_id", "method", "row_count", "smape_valid_count", "rmse_valid_count", "mean_smape", "mean_rmse"],
    )

    method_mean_rows: List[Dict[str, Any]] = []
    for method in sorted(method_values):
        smape_vals = method_values[method]["smape"]
        rmse_vals = method_values[method]["rmse"]
        method_mean_rows.append(
            {
                "method": method,
                "row_count": len(smape_vals) + len(rmse_vals),
                "smape_valid_count": len(smape_vals),
                "rmse_valid_count": len(rmse_vals),
                "mean_smape": "" if not smape_vals else statistics.fmean(smape_vals),
                "mean_rmse": "" if not rmse_vals else statistics.fmean(rmse_vals),
            }
        )

    method_mean_path = output.with_name(f"{output.stem}_method_mean_metrics.csv")
    _write_csv(
        method_mean_path,
        method_mean_rows,
        ["method", "row_count", "smape_valid_count", "rmse_valid_count", "mean_smape", "mean_rmse"],
    )
    return [dataset_method_path, method_mean_path]


def _write_best_method_outputs(output: Path, all_rows: Sequence[Dict[str, str]]) -> List[Path]:
    best_by_target_rows: List[Dict[str, Any]] = []
    wins_by_dataset: Dict[int, Counter] = defaultdict(Counter)
    target_groups: Dict[Tuple[int, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        key = (int(row["dataset_id"]), _target_key(row), row.get("information_sharing") or row.get("scenario", ""))
        target_groups[key].append(row)

    for (dataset_id, target, scenario), group in sorted(target_groups.items()):
        ranked = []
        for row in group:
            smape = _parse_float(row.get("smape"))
            if smape is not None:
                ranked.append((smape, row))
        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0])
        best_smape, best_row = ranked[0]
        method = best_row.get("method", "")
        best_by_target_rows.append(
            {
                "dataset_id": dataset_id,
                "target_entity_key": target,
                "information_sharing": scenario,
                "best_method": method,
                "best_smape": best_smape,
                "best_rmse": _parse_float(best_row.get("rmse")) or "",
                "candidate_method_count": len(ranked),
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
        "Best method per target entity is chosen by lowest sMAPE within each "
        "`(dataset_id, target_entity_key, information_sharing)` group.",
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
    return [best_target_path, best_dataset_md_path]


def aggregate(
    *,
    run_dir: Path = PROJECT_ROOT / "outputs" / "runs",
    output: Path = OUTPUT_DIR / "d1_d6_all_results.csv",
    strict: bool = False,
    allow_missing: bool = False,
    legacy_fallback: bool = False,
) -> Dict[str, Any]:
    selected, discovery_audit = discover_source_csvs(run_dir, strict=strict, allow_missing=allow_missing)
    if not selected and legacy_fallback:
        selected = {(dataset_id, ""): path for dataset_id, path in SOURCE_CSVS.items()}

    unique_paths = list(dict.fromkeys(selected.values()))
    all_rows: List[Dict[str, str]] = []
    for path in unique_paths:
        dataset_hint = _dataset_id_from_path(path) or 0
        rows = _read_source(path, dataset_hint)
        all_rows.extend(rows)

    if not all_rows:
        if not allow_missing:
            raise FileNotFoundError(f"No result CSV rows found under {run_dir}")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output, [], PREFERRED_COLUMNS)
        audit_rows = _build_audit_rows([], discovery_audit)
        audit_path = output.with_name(f"{output.stem}_audit.csv")
        _write_csv(audit_path, audit_rows, sorted({key for row in audit_rows for key in row}))
        return {"all_results_path": output, "audit_path": audit_path, "audit_rows": audit_rows}

    output = Path(output)
    all_fieldnames = _union_fieldnames(all_rows)
    _write_csv(output, all_rows, all_fieldnames)

    audit_rows = _build_audit_rows(all_rows, discovery_audit)
    audit_path = output.with_name(f"{output.stem}_audit.csv")
    _write_csv(audit_path, audit_rows, sorted({key for row in audit_rows for key in row}))

    extra_paths = _write_metric_summaries(output, all_rows)
    extra_paths.extend(_write_best_method_outputs(output, all_rows))

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate D1-D6 result CSVs from a run directory.")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "d1_d6_all_results.csv")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--legacy-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aggregate(
        run_dir=args.run_dir,
        output=args.output,
        strict=bool(args.strict),
        allow_missing=bool(args.allow_missing),
        legacy_fallback=bool(args.legacy_fallback),
    )


if __name__ == "__main__":
    main()
