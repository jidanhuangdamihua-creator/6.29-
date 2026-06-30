#!/usr/bin/env python3
"""Audit solidified parquet files under 数据集/固化数据/."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.constants import SOLIDIFIED_TARGET_WINDOWS

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "数据集" / "固化数据"
OUTPUT_PATH = ROOT / "outputs" / "parquet_audit_report.md"

DATASET_IDS = tuple(range(1, 7))
ROLES = ("source", "target")

D123_DATE_BASELINE: dict[str, tuple[str, str]] = {
    "dataset1-source": ("2013-01-01", "2017-12-31"),
    "dataset1-target": ("2013-01-01", "2017-12-31"),
    "dataset2-source": ("2014-01-01", "2018-12-31"),
    "dataset2-target": ("2014-01-01", "2018-12-31"),
    "dataset3-source": ("2013-01-01", "2015-07-31"),
    "dataset3-target": ("2013-01-01", "2015-07-31"),
}

D123_ENTITY_BASELINE: dict[str, int | None] = {
    "dataset1-source": 27,
    "dataset1-target": 3,
    "dataset2-source": 9,
    "dataset2-target": 1,
    "dataset3-source": 29,
    "dataset3-target": 1,
}

LARGE_SOURCE = {"dataset4-source", "dataset5-source"}


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class FileReport:
    stem: str
    path: Path | None
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, label: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(label=label, passed=passed, detail=detail))

    @property
    def fail_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


def parquet_stem(dataset_id: int, role: str) -> str:
    return f"dataset{dataset_id}-{role}"


def parquet_path(dataset_id: int, role: str) -> Path:
    return PARQUET_DIR / f"{parquet_stem(dataset_id, role)}.parquet"


def format_check_line(label: str, passed: bool, detail: str = "") -> str:
    if passed:
        if detail:
            return f"- {label}: [PASS] ({detail})"
        return f"- {label}: [PASS]"
    reason = detail or "check failed"
    return f"- {label}: [FAIL: {reason}]"


def normalize_date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date value: {value!r}")
    return pd.Timestamp(parsed).normalize()


def date_str(value: Any) -> str:
    return normalize_date(value).strftime("%Y-%m-%d")


def read_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def read_schema_columns(path: Path) -> list[tuple[str, str]]:
    schema = pq.read_schema(path)
    return [(name, str(schema.field(name).type)) for name in schema.names]


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    missing = [col for col in columns if col not in available]
    if missing:
        raise KeyError(f"missing columns {missing} in {path.name}")
    return pd.read_parquet(path, columns=columns)


def read_date_bounds(path: Path) -> tuple[str, str]:
    dates = read_columns(path, ["date"])["date"]
    return date_str(dates.min()), date_str(dates.max())


def read_entity_values(path: Path) -> tuple[int, list[Any]]:
    frame = read_columns(path, ["entity_id"])
    values = sorted(frame["entity_id"].dropna().unique(), key=lambda value: str(value))
    return int(frame["entity_id"].nunique()), values


def load_target_selection_json(dataset_id: int) -> dict[str, Any]:
    json_path = (
        ROOT
        / "outputs"
        / "domain_adaptation"
        / f"Dataset{dataset_id}"
        / "target_selection"
        / "target_selection_result.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(str(json_path))
    return json.loads(json_path.read_text(encoding="utf-8"))


def extract_d456_window_info(data: dict[str, Any], dataset_id: int) -> dict[str, Any]:
    if dataset_id not in SOLIDIFIED_TARGET_WINDOWS:
        raise ValueError(f"unsupported dataset id: {dataset_id}")
    solidified_window = SOLIDIFIED_TARGET_WINDOWS[dataset_id]

    if dataset_id == 4:
        source_window = data["source_window"]
        return {
            "train_start": solidified_window["train_start"],
            "test_end": solidified_window["test_end"],
            "source_end": source_window["source_history_end"],
            "target_skus": list(data.get("target_skus") or []),
            "target_store": data.get("target_store_id"),
        }
    if dataset_id == 5:
        windows = data["time_windows"]
        return {
            "train_start": solidified_window["train_start"],
            "test_end": solidified_window["test_end"],
            "source_end": windows["source_end"],
            "target_skus": list(data.get("target_skus") or []),
            "target_store": data.get("target_store"),
        }
    if dataset_id == 6:
        return {
            "train_start": solidified_window["train_start"],
            "test_end": solidified_window["test_end"],
            "source_end": data["source_end"],
            "target_skus": list(data.get("target_skus") or []),
            "target_store": data.get("target_store"),
            "target_entities": list(data.get("target_entities") or []),
        }
    raise ValueError(f"unsupported dataset id: {dataset_id}")


def build_expected_target_entities(dataset_id: int, info: dict[str, Any]) -> set[str]:
    store = info.get("target_store")
    skus = info.get("target_skus") or []
    if dataset_id == 6:
        entities = info.get("target_entities")
        if isinstance(entities, list) and entities:
            return {f"{row['store_id']}_{row['item_id']}" for row in entities}
        return {f"{store}_{sku}" for sku in skus}
    return {f"{store}_{sku}" for sku in skus}


def check_existence(reports: dict[str, FileReport]) -> tuple[bool, str]:
    missing = [stem for stem, report in reports.items() if report.path is None]
    status = len(missing) == 0
    detail = "all 12 files present" if status else f"missing: {', '.join(sorted(missing))}"
    for report in reports.values():
        report.add("file existence", status, detail)
    return status, detail


def check_date_range(report: FileReport, dataset_id: int, role: str, d456_info: dict[int, dict[str, Any]] | None) -> None:
    if report.path is None:
        report.add("date range", False, "file missing")
        return
    try:
        actual_min, actual_max = read_date_bounds(report.path)
    except Exception as exc:  # noqa: BLE001 - audit should continue
        report.add("date range", False, str(exc))
        return

    stem = report.stem
    if dataset_id <= 3:
        expected_min, expected_max = D123_DATE_BASELINE[stem]
        passed = actual_min == expected_min and actual_max == expected_max
        detail = (
            f"actual={actual_min}~{actual_max}, expected={expected_min}~{expected_max}"
            if not passed
            else f"{actual_min}~{actual_max}"
        )
        report.add("date range", passed, detail)
        return

    info = (d456_info or {}).get(dataset_id)
    if info is None:
        report.add("date range", False, "missing target_selection_result.json")
        return

    train_start = normalize_date(info["train_start"])
    test_end = normalize_date(info["test_end"])
    source_end = normalize_date(info["source_end"])
    actual_min_ts = normalize_date(actual_min)
    actual_max_ts = normalize_date(actual_max)

    if role == "target":
        passed = actual_min_ts <= train_start and actual_max_ts >= test_end
        detail = (
            f"actual={actual_min}~{actual_max}, required coverage {info['train_start']}~{info['test_end']}"
            if not passed
            else f"covers {info['train_start']}~{info['test_end']} (actual {actual_min}~{actual_max})"
        )
    else:
        passed = actual_max_ts <= source_end
        detail = (
            f"actual_max={actual_max}, source_end={info['source_end']}"
            if not passed
            else f"date_max={actual_max} <= source_end={info['source_end']}"
        )
    report.add("date range", passed, detail)


def check_entity_composition(
    report: FileReport,
    dataset_id: int,
    role: str,
    d456_info: dict[int, dict[str, Any]] | None,
) -> None:
    if report.path is None:
        report.add("entity composition", False, "file missing")
        return
    try:
        nunique, values = read_entity_values(report.path)
    except Exception as exc:  # noqa: BLE001
        report.add("entity composition", False, str(exc))
        return

    stem = report.stem
    value_preview = ", ".join(str(value) for value in values)
    if role == "target":
        entity_detail = f"nunique={nunique}; values=[{value_preview}]"
    else:
        entity_detail = f"nunique={nunique}"

    if dataset_id <= 3:
        expected = D123_ENTITY_BASELINE.get(stem)
        if expected is None:
            report.add("entity composition", True, entity_detail)
            return
        passed = nunique == expected
        detail = entity_detail if passed else f"{entity_detail}; expected nunique={expected}"
        report.add("entity composition", passed, detail)
        return

    if role == "source":
        report.add("entity composition", True, entity_detail)
        return

    info = (d456_info or {}).get(dataset_id)
    if info is None:
        report.add("entity composition", False, f"{entity_detail}; missing JSON reference")
        return

    expected_entities = build_expected_target_entities(dataset_id, info)
    actual_entities = {str(value) for value in values}
    passed = actual_entities == expected_entities
    detail = entity_detail if passed else (
        f"{entity_detail}; expected values=[{', '.join(sorted(expected_entities))}]"
    )
    report.add("entity composition", passed, detail)


def check_schema(report: FileReport, dataset_id: int) -> None:
    if report.path is None:
        report.add("schema columns", False, "file missing")
        return
    try:
        columns = read_schema_columns(report.path)
    except Exception as exc:  # noqa: BLE001
        report.add("schema columns", False, str(exc))
        return

    column_lines = ", ".join(f"{name}:{dtype}" for name, dtype in columns)
    report.add("schema columns", True, column_lines)

    names = {name for name, _ in columns}
    if dataset_id == 2:
        forbidden = sorted(name for name in ("brand_code", "entity_id_code") if name in names)
        passed = not forbidden
        detail = "forbidden columns absent" if passed else f"forbidden columns present: {forbidden}"
        report.add("D2 forbidden columns", passed, detail)
    if dataset_id == 3:
        passed = "store_id" not in names
        detail = "store_id absent" if passed else "store_id must not appear in D3 parquet"
        report.add("D3 forbidden store_id column", passed, detail)


def check_d3_special(report: FileReport) -> None:
    if report.path is None:
        for label in (
            "D3 entity_id dtype",
            "D3 TODO_REGION_UNAVAILABLE scan",
            "D3 item_id equals 1",
        ):
            report.add(label, False, "file missing")
        return

    try:
        entity_frame = read_columns(report.path, ["entity_id"])
        dtype = entity_frame["entity_id"].dtype
        dtype_pass = pd.api.types.is_integer_dtype(dtype)
        report.add(
            "D3 entity_id dtype",
            dtype_pass,
            f"dtype={dtype}" if dtype_pass else f"dtype={dtype}, expected numeric int",
        )

        full_frame = pd.read_parquet(report.path)
        object_cols = full_frame.select_dtypes(include=["object", "string"]).columns.tolist()
        bad_hits: list[str] = []
        for col in object_cols:
            count = int((full_frame[col].astype(str) == "TODO_REGION_UNAVAILABLE").sum())
            if count:
                bad_hits.append(f"{col}={count}")
        todo_pass = not bad_hits
        todo_detail = "no TODO_REGION_UNAVAILABLE in object columns" if todo_pass else "; ".join(bad_hits)
        report.add("D3 TODO_REGION_UNAVAILABLE scan", todo_pass, todo_detail)

        item_frame = read_columns(report.path, ["item_id"])
        unique_items = sorted(item_frame["item_id"].dropna().unique().tolist())
        item_pass = unique_items == [1]
        item_detail = f"unique item_id={unique_items}" if item_pass else f"unique item_id={unique_items}, expected [1]"
        report.add("D3 item_id equals 1", item_pass, item_detail)
    except Exception as exc:  # noqa: BLE001
        report.add("D3 entity_id dtype", False, str(exc))
        report.add("D3 TODO_REGION_UNAVAILABLE scan", False, str(exc))
        report.add("D3 item_id equals 1", False, str(exc))


def check_d5_sales_non_negative(report: FileReport) -> None:
    if report.path is None:
        report.add("D5 sales non-negative", False, "file missing")
        return
    try:
        sales = read_columns(report.path, ["sales"])["sales"]
        min_sales = float(sales.min())
        passed = min_sales >= 0
        detail = f"min sales={min_sales}" if passed else f"min sales={min_sales} < 0"
        report.add("D5 sales non-negative", passed, detail)
    except Exception as exc:  # noqa: BLE001
        report.add("D5 sales non-negative", False, str(exc))


def check_row_count(report: FileReport) -> None:
    if report.path is None:
        report.add("row count", False, "file missing")
        return
    try:
        rows = read_row_count(report.path)
        report.add("row count", True, f"{rows:,} rows")
    except Exception as exc:  # noqa: BLE001
        report.add("row count", False, str(exc))


def load_d456_info() -> dict[int, dict[str, Any]]:
    info: dict[int, dict[str, Any]] = {}
    for dataset_id in (4, 5, 6):
        try:
            data = load_target_selection_json(dataset_id)
            info[dataset_id] = extract_d456_window_info(data, dataset_id)
        except Exception as exc:  # noqa: BLE001
            info[dataset_id] = {"_error": str(exc)}
    return info


def render_report(reports: list[FileReport], existence_ok: bool, existence_detail: str) -> str:
    lines = [
        "# Solidified Parquet Audit Report",
        "",
        f"- Input directory: `{PARQUET_DIR}`",
        f"- Generated by: `{Path(__file__).name}`",
        "",
        "## Global Checks",
        "",
        format_check_line("file existence (12 files)", existence_ok, existence_detail),
        "",
    ]

    total_checks = 1
    total_failures = 0 if existence_ok else 1

    for report in reports:
        lines.append(f"## {report.stem}")
        if report.path is None:
            lines.append(f"- path: `{PARQUET_DIR / (report.stem + '.parquet')}` (missing)")
        else:
            lines.append(f"- path: `{report.path}`")
        lines.append("")

        for check in report.checks:
            if check.label == "file existence":
                continue
            total_checks += 1
            if not check.passed:
                total_failures += 1
            lines.append(format_check_line(check.label, check.passed, check.detail))
        lines.append("")

    passed_checks = total_checks - total_failures
    lines.extend(
        [
            "## Summary",
            "",
            f"- Files checked: {len(reports)}",
            f"- Checks: {passed_checks}/{total_checks} PASS, {total_failures} FAIL",
            f"- Overall: {'PASS' if total_failures == 0 else 'FAIL'}",
            "",
        ]
    )
    return "\n".join(lines)


def audit_all() -> int:
    reports: dict[str, FileReport] = {}
    for dataset_id in DATASET_IDS:
        for role in ROLES:
            stem = parquet_stem(dataset_id, role)
            path = parquet_path(dataset_id, role)
            reports[stem] = FileReport(stem=stem, path=path if path.exists() else None)

    existence_ok, existence_detail = check_existence(reports)

    d456_info_raw = load_d456_info()
    d456_info = {
        dataset_id: info
        for dataset_id, info in d456_info_raw.items()
        if "_error" not in info
    }

    ordered_reports: list[FileReport] = []
    for dataset_id in DATASET_IDS:
        for role in ROLES:
            stem = parquet_stem(dataset_id, role)
            report = reports[stem]
            check_date_range(report, dataset_id, role, d456_info)
            check_entity_composition(report, dataset_id, role, d456_info)
            check_schema(report, dataset_id)
            if dataset_id == 3 and role == "source":
                check_d3_special(report)
            if dataset_id == 5 and role == "source":
                check_d5_sales_non_negative(report)
            check_row_count(report)
            ordered_reports.append(report)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_report(ordered_reports, existence_ok, existence_detail), encoding="utf-8")
    print(f"Wrote audit report to {OUTPUT_PATH}")

    total_failures = sum(report.fail_count for report in ordered_reports)
    return 0 if total_failures == 0 else 1


def main() -> int:
    try:
        return audit_all()
    except Exception as exc:  # noqa: BLE001
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
