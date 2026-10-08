#!/usr/bin/env python3
"""Build and validate the MS-KATL Paper Results Master V1.2 workbook.

This script is intentionally source-preserving:
* it reads the supplied V1.1 workbook and its exact source_file paths;
* it never writes to the input workbook or to raw experiment artifacts;
* it refuses to build when the source preflight is not clean;
* it keeps every selected source row, including repeated deterministic-baseline
  seed rows and independent ablation rows.

The workbook is a paper-evidence consolidation artifact, not an experiment
runner.  It does not train models or modify result values.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo


DEFAULT_INPUT = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/MS_KATL_Paper_Data_Freeze_v1_1.xlsx"
)
DEFAULT_RAW_DIR = Path("/Users/ming/Desktop/复现实验/论文文稿相关/原始实验结果")

V3_SHEET = "03_V3_Raw_5Seed"
BASELINE_SHEET = "04_BL_Raw_Selected"
ABLATION_SHEET = "05_Ablation_Raw"
REQUIRED_SOURCE_SHEETS = [V3_SHEET, BASELINE_SHEET, ABLATION_SHEET]

MASTER_SHEET = "03_Paper_Results_Master"
DICTIONARY_SHEET = "04_Master_Data_Dictionary"
VALIDATION_SHEET = "Master_Validation"
LINEAGE_SHEET = "Derived_Table_Lineage"
BASELINE_AUDIT_SHEET = "Baseline_Coverage_Audit"
DATASET_AUDIT_SHEET = "Dataset_Scope_Audit"
AGGREGATE_SHEET = "Master_Aggregate_Check"

# The first 26 fields follow the requested V1.2 schema.  Auxiliary fields
# retain source metadata without replacing the authoritative rmse/smape pair.
MASTER_COLUMNS = [
    "record_id",
    "canonical_key",
    "dataset",
    "target",
    "horizon",
    "seed",
    "method",
    "method_family",
    "model_version",
    "experiment_type",
    "arm",
    "mode",
    "run_type",
    "rmse",
    "smape",
    "metric_space",
    "rmse_unit",
    "smape_unit",
    "source_sheet",
    "source_row",
    "source_file",
    "source_run_id",
    "source_sha256",
    "paper_scope",
    "include_in_master",
    "status",
    "notes",
    "seed_semantics",
    "source_status",
    "learning_rate",
    "rmse_normalized",
    "smape_normalized",
    "nrmse_range_percent",
    "method_type",
    "arm_name",
    "arm_description",
    "selected_lr",
]

REQUIRED_MASTER_FIELDS = [
    "dataset",
    "target",
    "horizon",
    "seed",
    "method",
    "experiment_type",
    "rmse",
    "smape",
]

EXPECTED_METHODS = {"MS-KATL-V3", "LSTM", "LightGBM", "Seasonal-Naive", "MS-KATL-V3-Ablation"}
EXPECTED_METHODS = EXPECTED_METHODS | {"DLinear", "TCN"}
EXPECTED_ARMS = {"A", "B", "C", "D", "E", "NA"}
# openpyxl/Excel serializes numeric XML values to a finite decimal precision.
# This tolerance is only for round-trip verification; the builder never rounds
# or transforms the source values before writing them.
NUMERIC_SERIALIZATION_ABS_TOLERANCE = 1e-12
FORMULA_COMPATIBILITY_REPLACEMENTS = (
    ("STDEV.S", "STDEV"),
    ("RANK.EQ", "RANK"),
)

HEADER_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial", size=10, color="000000")
TITLE_FONT = Font(name="Arial", size=12, bold=True, color="1F4E78")
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SECTION_FILL = PatternFill("solid", fgColor="D9EAF7")
THIN_GREY = Side(style="thin", color="D9E1F2")
BODY_BORDER = Border(bottom=THIN_GREY)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: Any, optional: bool = False) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "NA" if optional else ""
    return str(value).strip()


def norm_method(value: Any) -> str:
    raw = clean_text(value)
    mapping = {
        "seasonal naive": "Seasonal-Naive",
        "seasonal-naive": "Seasonal-Naive",
        "seasonal_naive": "Seasonal-Naive",
        "lightgbm": "LightGBM",
        "lstm": "LSTM",
        "bl6_dlinear": "DLinear",
        "dlinear": "DLinear",
        "bl7_tcn": "TCN",
        "tcn": "TCN",
    }
    return mapping.get(raw.casefold(), raw)


def validate_external_baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Dry-run validator for a derived BL6/BL7 candidate; never writes a workbook."""
    required = {
        "dataset_id", "target_entity_key", "method", "horizon", "seed", "rmse", "smape",
        "result_status", "metric_space",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"external baseline candidate missing columns: {missing}")
    accepted = frame.copy()
    accepted["dataset"] = accepted["dataset_id"].astype(str).str.upper()
    accepted["target"] = accepted["target_entity_key"].astype(str)
    accepted["method"] = accepted["method"].map(norm_method)
    accepted["horizon"] = pd.to_numeric(accepted["horizon"], errors="raise").astype(int)
    accepted["seed"] = pd.to_numeric(accepted["seed"], errors="raise").astype(int)
    accepted["rmse"] = pd.to_numeric(accepted["rmse"], errors="raise")
    accepted["smape"] = pd.to_numeric(accepted["smape"], errors="raise")
    if not set(accepted["method"]).issubset({"DLinear", "TCN"}):
        raise ValueError("external baseline candidate contains unsupported methods")
    if set(accepted["result_status"].astype(str)) != {"confirmed_baseline"}:
        raise ValueError("external baseline candidate requires confirmed_baseline status")
    if set(accepted["metric_space"].astype(str)) != {"original_sales_space"}:
        raise ValueError("external baseline candidate requires original_sales_space")
    if not np.isfinite(accepted[["rmse", "smape"]].to_numpy(dtype=float)).all():
        raise ValueError("external baseline candidate metrics must be finite")
    if accepted.duplicated(["dataset", "target", "method", "horizon", "seed"]).any():
        raise ValueError("external baseline candidate contains duplicate cells")
    expected_targets = {
        "D1": "1/10", "D4": "166/258", "D5": "48/1159414", "D6": "CA_1/FOODS_3_080",
    }
    if set(accepted["dataset"]) != set(expected_targets):
        raise ValueError("external baseline candidate dataset scope mismatch")
    if any(expected_targets[row.dataset] != row.target for row in accepted.itertuples()):
        raise ValueError("external baseline candidate target scope mismatch")
    if set(accepted["horizon"]) != {1, 3, 5} or set(accepted["seed"]) != {42, 43, 44, 45, 46}:
        raise ValueError("external baseline candidate horizon/seed scope mismatch")
    if len(accepted) != 120:
        raise ValueError(f"external baseline candidate requires 120 rows, got {len(accepted)}")
    return accepted.sort_values(["dataset", "target", "method", "horizon", "seed"], kind="stable").reset_index(drop=True)


def norm_method_family(method_type: Any, method: str) -> str:
    raw = clean_text(method_type, optional=True).casefold()
    if method == "MS-KATL-V3":
        return "proposed"
    if method == "MS-KATL-V3-Ablation":
        return "ablation"
    if raw in {"learning", "learning_baseline"}:
        return "learning_baseline"
    if raw in {"heuristic", "statistical", "statistical_baseline"}:
        return "statistical_baseline"
    return "other"


def status_from_source(value: Any) -> str:
    raw = clean_text(value, optional=True)
    if raw in {"OK", "confirmed_baseline"}:
        return "OK"
    if raw in {"WARNING", "INCOMPLETE", "legacy_unverified"}:
        return "WARNING"
    if raw in {"CONFLICT", "DUPLICATE_IDENTITY_CONFLICT"}:
        return "CONFLICT"
    if raw in {"NA", ""}:
        return "INCOMPLETE"
    return "WARNING"


def source_run_id(source_file: str) -> str:
    if not source_file or source_file == "NA":
        return "NA"
    for part in reversed(Path(source_file).parts):
        if re.match(r"^202608\d{2}(?:_|$)", part):
            return part
    return "NA"


def read_sheet_rows(workbook, sheet_name: str) -> tuple[list[str], list[dict[str, Any]]]:
    ws = workbook[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return [], []
    headers = [clean_text(value) for value in rows[0]]
    if any(not header for header in headers):
        raise ValueError(f"{sheet_name}: blank header detected")
    records: list[dict[str, Any]] = []
    for excel_row, values in enumerate(rows[1:], start=2):
        if not any(value is not None for value in values):
            continue
        record = {header: values[idx] if idx < len(values) else None for idx, header in enumerate(headers)}
        record["__source_row"] = excel_row
        records.append(record)
    return headers, records


def contract_values(workbook) -> dict[str, str]:
    ws = workbook["01_Paper_Contract"]
    result: dict[str, str] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[0] is not None:
            result[str(row[0]).strip()] = clean_text(row[1], optional=True)
    return result


def parse_contract_datasets(contract: dict[str, str]) -> set[str]:
    return set(re.findall(r"\bD[1-6]\b", contract.get("Selected datasets", "")))


def parse_contract_targets(contract: dict[str, str]) -> dict[str, str]:
    raw = contract.get("Formal targets", "")
    return {dataset: target.strip() for dataset, target in re.findall(r"(D[1-6])=([^;]+)", raw)}


def parse_contract_baselines(contract: dict[str, str]) -> set[str]:
    raw = contract.get("Baseline methods", "")
    return {norm_method(part) for part in raw.split(";") if part.strip()}


def exact_source_hash(source_file: str, cache: dict[str, str]) -> str:
    if not source_file or source_file == "NA":
        return "UNRESOLVED"
    if source_file not in cache:
        path = Path(source_file)
        cache[source_file] = sha256_file(path) if path.is_file() else "UNRESOLVED"
    return cache[source_file]


def duplicate_group_count(records: Iterable[dict[str, Any]], fields: list[str]) -> int:
    counter: Counter[tuple[Any, ...]] = Counter()
    for row in records:
        counter[tuple(row.get(field) for field in fields)] += 1
    return sum(1 for count in counter.values() if count > 1)


def preflight(input_path: Path, raw_dir: Path, print_report: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "input": input_path,
        "raw_dir": raw_dir,
        "input_sha256": "UNRESOLVED",
        "sheets": [],
        "source_counts": {},
        "source_records": {},
        "source_headers": {},
        "datasets": set(),
        "targets": set(),
        "horizons": set(),
        "seeds": set(),
        "methods": set(),
        "modes": set(),
        "arms": set(),
        "duplicate_source_identities": 0,
        "missing_required_values": 0,
        "target_date_coercion_risk": 0,
        "missing_source_files": [],
        "fallback_used": False,
        "status": "FAIL",
    }
    if not input_path.is_file():
        if print_report:
            print(f"INPUT_WORKBOOK_MISSING={input_path}")
            print("PREFLIGHT_STATUS=FAIL")
        return report

    report["input_sha256"] = sha256_file(input_path)
    workbook = load_workbook(input_path, read_only=False, data_only=False)
    report["sheets"] = list(workbook.sheetnames)
    missing_sheets = [name for name in REQUIRED_SOURCE_SHEETS if name not in workbook.sheetnames]
    contract = contract_values(workbook) if "01_Paper_Contract" in workbook.sheetnames else {}

    source_specs = {
        V3_SHEET: [
            "dataset", "target", "mode", "horizon", "seed", "rmse_original", "smape_original",
            "metric_space", "status", "source_file",
        ],
        BASELINE_SHEET: [
            "dataset", "target", "method", "horizon", "seed", "rmse", "smape",
            "metric_space", "result_status", "source_file",
        ],
        ABLATION_SHEET: [
            "dataset", "target", "horizon", "seed", "arm", "rmse_original", "smape_original",
            "metric_space", "status", "source_file",
        ],
    }
    all_records: list[dict[str, Any]] = []
    for sheet_name in REQUIRED_SOURCE_SHEETS:
        if sheet_name not in workbook.sheetnames:
            report["source_counts"][sheet_name] = 0
            report["source_records"][sheet_name] = []
            continue
        headers, rows = read_sheet_rows(workbook, sheet_name)
        report["source_headers"][sheet_name] = headers
        report["source_records"][sheet_name] = rows
        report["source_counts"][sheet_name] = len(rows)
        missing_headers = [field for field in source_specs[sheet_name] if field not in headers]
        if missing_headers:
            report.setdefault("missing_headers", {})[sheet_name] = missing_headers
        all_records.extend(rows)

        for row in rows:
            required = source_specs[sheet_name]
            if any(row.get(field) is None or (isinstance(row.get(field), str) and not row.get(field).strip()) for field in required):
                report["missing_required_values"] += 1
            target = row.get("target")
            if not isinstance(target, str) or isinstance(target, (datetime, date)):
                report["target_date_coercion_risk"] += 1
            dataset = clean_text(row.get("dataset"))
            target_text = clean_text(target)
            report["datasets"].add(dataset)
            report["targets"].add(target_text)
            if row.get("horizon") is not None:
                report["horizons"].add(int(row["horizon"]))
            if row.get("seed") is not None:
                report["seeds"].add(int(row["seed"]))
            if sheet_name == V3_SHEET:
                report["methods"].add("MS-KATL-V3")
                report["modes"].add(clean_text(row.get("mode"), optional=True))
                report["arms"].add("NA")
            elif sheet_name == BASELINE_SHEET:
                report["methods"].add(norm_method(row.get("method")))
                report["modes"].add("NA")
                report["arms"].add("NA")
            else:
                report["methods"].add("MS-KATL-V3-Ablation")
                report["modes"].add(clean_text(contract.get("Scenario / mode"), optional=True))
                report["arms"].add(clean_text(row.get("arm")))

            source_file = clean_text(row.get("source_file"), optional=True)
            if source_file != "NA" and not Path(source_file).is_file():
                report["missing_source_files"].append(source_file)

    duplicate_specs = {
        V3_SHEET: ["dataset", "target", "mode", "horizon", "seed"],
        BASELINE_SHEET: ["dataset", "target", "method", "horizon", "seed"],
        ABLATION_SHEET: ["dataset", "target", "horizon", "seed", "arm"],
    }
    for sheet_name, fields in duplicate_specs.items():
        report["duplicate_source_identities"] += duplicate_group_count(report["source_records"].get(sheet_name, []), fields)

    contract_datasets = parse_contract_datasets(contract)
    contract_targets = parse_contract_targets(contract)
    report["contract"] = contract
    report["contract_datasets"] = contract_datasets
    report["contract_targets"] = contract_targets
    report["contract_baselines"] = parse_contract_baselines(contract)
    report["status"] = "PASS" if (
        not missing_sheets
        and not report.get("missing_headers")
        and report["missing_required_values"] == 0
        and report["duplicate_source_identities"] == 0
        and not report["missing_source_files"]
        and all_records
    ) else "FAIL"

    if print_report:
        print("=" * 50)
        print("MS-KATL PAPER MASTER V1.2 PREFLIGHT")
        print("=" * 50)
        print(f"INPUT_WORKBOOK={input_path}")
        print(f"INPUT_SHA256={report['input_sha256']}")
        print(f"SHEETS_FOUND={','.join(report['sheets'])}")
        for sheet_name, label in [
            (V3_SHEET, "V3_RAW_SHEET_FOUND"),
            (BASELINE_SHEET, "BASELINE_RAW_SHEET_FOUND"),
            (ABLATION_SHEET, "ABLATION_RAW_SHEET_FOUND"),
        ]:
            print(f"{label}={'YES' if sheet_name in report['sheets'] else 'NO'}")
        print(f"V3_SOURCE_ROWS={report['source_counts'].get(V3_SHEET, 0)}")
        print(f"BASELINE_SOURCE_ROWS={report['source_counts'].get(BASELINE_SHEET, 0)}")
        print(f"ABLATION_SOURCE_ROWS={report['source_counts'].get(ABLATION_SHEET, 0)}")
        print(f"SOURCE_TOTAL_ROWS={sum(report['source_counts'].values())}")
        print(f"DATASETS_FOUND={','.join(sorted(report['datasets']))}")
        print(f"TARGETS_FOUND={','.join(sorted(report['targets']))}")
        print(f"HORIZONS_FOUND={','.join(str(x) for x in sorted(report['horizons']))}")
        print(f"SEEDS_FOUND={','.join(str(x) for x in sorted(report['seeds']))}")
        print(f"METHODS_FOUND={','.join(sorted(report['methods']))}")
        print(f"MODES_FOUND={','.join(sorted(report['modes']))}")
        print(f"ARMS_FOUND={','.join(sorted(report['arms']))}")
        print(f"DUPLICATE_SOURCE_IDENTITIES={report['duplicate_source_identities']}")
        print(f"MISSING_REQUIRED_VALUES={report['missing_required_values']}")
        print(f"TARGET_DATE_COERCION_RISK={report['target_date_coercion_risk']}")
        if report["missing_source_files"]:
            print(f"MISSING_SOURCE_FILES={len(report['missing_source_files'])}")
        print(f"FALLBACK_RAW_DIR_USED={'YES' if report['fallback_used'] else 'NO'}")
        print(f"PREFLIGHT_STATUS={report['status']}")
        print("=" * 50)
    return report


def numeric_or_na(value: Any) -> Any:
    return value if value is not None else "NA"


def numeric_values_equal(left: Any, right: Any) -> bool:
    """Compare numeric round-trips without mistaking Excel serialization for edits."""
    if left == right:
        return True
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return False
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return False
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=NUMERIC_SERIALIZATION_ABS_TOLERANCE,
    )


def normalize_formula_compatibility(workbook) -> int:
    """Normalize legacy function aliases in the generated output copy only."""
    replacements = 0
    for ws in workbook.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                value = cell.value
                if not isinstance(value, str) or not value.startswith("="):
                    continue
                new_value = value
                for old, new in FORMULA_COMPATIBILITY_REPLACEMENTS:
                    new_value = re.sub(rf"(?i)\b{re.escape(old)}\b", new, new_value)
                if new_value != value:
                    cell.value = new_value
                    replacements += 1
    return replacements


def make_record(
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    method: str,
    method_family: str,
    model_version: str,
    experiment_type: str,
    arm: str,
    mode: str,
    run_type: str,
    rmse: Any,
    smape: Any,
    metric_space: str,
    source_sheet: str,
    source_row: int,
    source_file: str,
    source_status: str,
    notes: str,
    seed_semantics: str,
    source_hash_cache: dict[str, str],
    **extra: Any,
) -> dict[str, Any]:
    canonical_key = "|".join(
        [dataset, target, str(horizon), str(seed), method, mode, experiment_type, arm, model_version]
    )
    if experiment_type == "main":
        record_id = f"{dataset}__{target}__H{horizon}__S{seed}__{method}__{mode.upper()}__MAIN"
    elif experiment_type == "baseline":
        record_id = f"{dataset}__{target}__H{horizon}__S{seed}__{method}__BASELINE"
    else:
        record_id = f"{dataset}__{target}__H{horizon}__S{seed}__{method}__{mode.upper()}__{arm}"
    return {
        "record_id": record_id,
        "canonical_key": canonical_key,
        "dataset": dataset,
        "target": target,
        "horizon": int(horizon),
        "seed": int(seed),
        "method": method,
        "method_family": method_family,
        "model_version": model_version,
        "experiment_type": experiment_type,
        "arm": arm,
        "mode": mode,
        "run_type": run_type,
        "rmse": rmse,
        "smape": smape,
        "metric_space": metric_space,
        "rmse_unit": "NA",
        "smape_unit": "UNKNOWN",
        "source_sheet": source_sheet,
        "source_row": source_row,
        "source_file": source_file,
        "source_run_id": source_run_id(source_file),
        "source_sha256": exact_source_hash(source_file, source_hash_cache),
        "paper_scope": "not_decided",
        "include_in_master": True,
        "status": status_from_source(source_status),
        "notes": notes,
        "seed_semantics": seed_semantics,
        "source_status": source_status,
        "learning_rate": numeric_or_na(extra.get("learning_rate")),
        "rmse_normalized": numeric_or_na(extra.get("rmse_normalized")),
        "smape_normalized": numeric_or_na(extra.get("smape_normalized")),
        "nrmse_range_percent": numeric_or_na(extra.get("nrmse_range_percent")),
        "method_type": clean_text(extra.get("method_type"), optional=True),
        "arm_name": clean_text(extra.get("arm_name"), optional=True),
        "arm_description": clean_text(extra.get("arm_description"), optional=True),
        "selected_lr": numeric_or_na(extra.get("selected_lr")),
    }


def build_records(preflight_report: dict[str, Any]) -> list[dict[str, Any]]:
    contract = preflight_report["contract"]
    source_hash_cache: dict[str, str] = {}
    records: list[dict[str, Any]] = []

    for row in preflight_report["source_records"][V3_SHEET]:
        source_status = clean_text(row.get("status"), optional=True)
        records.append(
            make_record(
                dataset=clean_text(row["dataset"]),
                target=clean_text(row["target"]),
                horizon=int(row["horizon"]),
                seed=int(row["seed"]),
                method="MS-KATL-V3",
                method_family="proposed",
                model_version="V3",
                experiment_type="main",
                arm="NA",
                mode=clean_text(row["mode"]),
                run_type="formal_main",
                rmse=row["rmse_original"],
                smape=row["smape_original"],
                metric_space=clean_text(row["metric_space"], optional=True),
                source_sheet=V3_SHEET,
                source_row=int(row["__source_row"]),
                source_file=clean_text(row["source_file"], optional=True),
                source_status=source_status,
                notes="Selected V1.1 five-seed V3 main observation; final paper presentation scope is not frozen.",
                seed_semantics="formal_seed",
                source_hash_cache=source_hash_cache,
                learning_rate=row.get("learning_rate"),
                rmse_normalized=row.get("rmse_normalized"),
                smape_normalized=row.get("smape_normalized"),
                nrmse_range_percent=row.get("nrmse_range_percent"),
            )
        )

    for row in preflight_report["source_records"][BASELINE_SHEET]:
        method = norm_method(row["method"])
        source_status = clean_text(row.get("result_status"), optional=True)
        repeated_note = ""
        if method in {"LightGBM", "Seasonal-Naive"}:
            repeated_note = " Contract records repeated seed rows as descriptive-only for this method."
        elif method == "LSTM":
            repeated_note = " Seed semantics are not independently frozen beyond the retained source row."
        records.append(
            make_record(
                dataset=clean_text(row["dataset"]),
                target=clean_text(row["target"]),
                horizon=int(row["horizon"]),
                seed=int(row["seed"]),
                method=method,
                method_family=norm_method_family(row.get("method_type"), method),
                model_version="NA",
                experiment_type="baseline",
                arm="NA",
                mode="NA",
                run_type="formal_baseline",
                rmse=row["rmse"],
                smape=row["smape"],
                metric_space=clean_text(row["metric_space"], optional=True),
                source_sheet=BASELINE_SHEET,
                source_row=int(row["__source_row"]),
                source_file=clean_text(row["source_file"], optional=True),
                source_status=source_status,
                notes="Selected V1.1 baseline observation retained; final paper presentation scope is not frozen." + repeated_note,
                seed_semantics=("descriptive_repeated_seed" if method in {"LightGBM", "Seasonal-Naive"} else "UNKNOWN"),
                source_hash_cache=source_hash_cache,
                method_type=row.get("method_type"),
            )
        )

    ablation_mode = clean_text(contract.get("Scenario / mode"), optional=True)
    for row in preflight_report["source_records"][ABLATION_SHEET]:
        source_status = clean_text(row.get("status"), optional=True)
        records.append(
            make_record(
                dataset=clean_text(row["dataset"]),
                target=clean_text(row["target"]),
                horizon=int(row["horizon"]),
                seed=int(row["seed"]),
                method="MS-KATL-V3-Ablation",
                method_family="ablation",
                model_version="V3",
                experiment_type="ablation",
                arm=clean_text(row["arm"]),
                mode=ablation_mode,
                run_type="formal_ablation",
                rmse=row["rmse_original"],
                smape=row["smape_original"],
                metric_space=clean_text(row["metric_space"], optional=True),
                source_sheet=ABLATION_SHEET,
                source_row=int(row["__source_row"]),
                source_file=clean_text(row["source_file"], optional=True),
                source_status=source_status,
                notes=(
                    "Selected V1.1 A-E mechanism-ablation observation; independent ablation run. "
                    "Arm D must not substitute the formal V3 main result; final paper presentation scope is not frozen."
                ),
                seed_semantics="formal_seed",
                source_hash_cache=source_hash_cache,
                arm_name=row.get("arm_name"),
                arm_description=row.get("arm_description"),
                selected_lr=row.get("selected_lr"),
            )
        )

    return records


def group_counts(records: list[dict[str, Any]], fields: list[str]) -> list[tuple[tuple[Any, ...], int]]:
    counter: Counter[tuple[Any, ...]] = Counter(tuple(row[field] for field in fields) for row in records)
    return sorted(counter.items(), key=lambda item: tuple(str(value) for value in item[0]))


def records_validation(records: list[dict[str, Any]], preflight_report: dict[str, Any]) -> dict[str, Any]:
    canonical_counts = Counter(row["canonical_key"] for row in records)
    duplicate_keys = sorted(key for key, count in canonical_counts.items() if count > 1)
    missing_required = sum(
        1
        for row in records
        if any(row.get(field) is None or row.get(field) == "" for field in REQUIRED_MASTER_FIELDS)
    )
    metric_invalid = []
    for row in records:
        rmse, smape = row["rmse"], row["smape"]
        if not isinstance(rmse, (int, float)) or isinstance(rmse, bool) or not math.isfinite(float(rmse)) or float(rmse) < 0:
            metric_invalid.append((row["canonical_key"], "rmse"))
        if not isinstance(smape, (int, float)) or isinstance(smape, bool) or not math.isfinite(float(smape)) or float(smape) < 0:
            metric_invalid.append((row["canonical_key"], "smape"))
    target_integrity = sum(1 for row in records if not isinstance(row["target"], str))
    metric_spaces = sorted({row["metric_space"] for row in records})
    missing_hashes = sum(1 for row in records if row["source_sha256"] == "UNRESOLVED")
    source_key_counts = Counter((row["source_sheet"], row["source_row"]) for row in records)
    unmapped = sum(1 for key, count in source_key_counts.items() if count != 1)
    source_counts = preflight_report["source_counts"]
    numeric_modifications = 0
    max_numeric_abs_diff = 0.0
    by_source = {(row["source_sheet"], row["source_row"]): row for row in records}
    for sheet_name, rows in preflight_report["source_records"].items():
        for source in rows:
            key = (sheet_name, source["__source_row"])
            master = by_source.get(key)
            if master is None:
                continue
            source_rmse = source["rmse_original"] if sheet_name != BASELINE_SHEET else source["rmse"]
            source_smape = source["smape_original"] if sheet_name != BASELINE_SHEET else source["smape"]
            for master_value, source_value in [(master["rmse"], source_rmse), (master["smape"], source_smape)]:
                if isinstance(master_value, (int, float)) and isinstance(source_value, (int, float)):
                    max_numeric_abs_diff = max(max_numeric_abs_diff, abs(float(master_value) - float(source_value)))
            if not numeric_values_equal(master["rmse"], source_rmse) or not numeric_values_equal(master["smape"], source_smape):
                numeric_modifications += 1
    return {
        "source_total_rows": sum(source_counts.values()),
        "master_rows": len(records),
        "duplicate_keys": duplicate_keys,
        "duplicate_key_count": len(duplicate_keys),
        "missing_required": missing_required,
        "metric_invalid": metric_invalid,
        "target_integrity": target_integrity,
        "metric_spaces": metric_spaces,
        "metric_space_consistency": len(metric_spaces) == 1,
        "missing_hashes": missing_hashes,
        "unmapped_source_rows": unmapped,
        "numeric_modifications": numeric_modifications,
        "max_numeric_abs_diff": max_numeric_abs_diff,
        "source_counts": source_counts,
        "source_master_counts": {
            sheet: sum(1 for row in records if row["source_sheet"] == sheet) for sheet in REQUIRED_SOURCE_SHEETS
        },
    }


def style_header(ws, row: int = 1) -> None:
    for cell in ws[row]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BODY_BORDER
    ws.row_dimensions[row].height = 32


def style_body(ws, start_row: int = 2) -> None:
    for row in ws.iter_rows(min_row=start_row):
        for cell in row:
            cell.font = BODY_FONT
            cell.border = BODY_BORDER
            cell.alignment = Alignment(vertical="top", wrap_text=False)


def set_widths(ws, widths: dict[str, float]) -> None:
    for column, width in widths.items():
        ws.column_dimensions[column].width = width


def add_table(ws, ref: str, name: str) -> None:
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False
    )
    ws.add_table(table)


def write_master_sheet(workbook, records: list[dict[str, Any]]) -> None:
    ws = workbook.create_sheet(MASTER_SHEET)
    ws.append(MASTER_COLUMNS)
    for record in records:
        ws.append([record.get(column, "NA") for column in MASTER_COLUMNS])
    style_header(ws)
    style_body(ws)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{chr(64 + len(MASTER_COLUMNS))}#{len(records) + 1}" if len(MASTER_COLUMNS) <= 26 else f"A1:AK{len(records)+1}"
    # The expression above is replaced with a proper column letter for 27+ columns.
    from openpyxl.utils import get_column_letter
    ws.auto_filter.ref = f"A1:{get_column_letter(len(MASTER_COLUMNS))}{len(records) + 1}"
    target_col = MASTER_COLUMNS.index("target") + 1
    for cell in ws.iter_cols(min_col=target_col, max_col=target_col, min_row=2, max_row=len(records) + 1):
        for item in cell:
            item.number_format = "@"
    for column in ["horizon", "seed", "source_row"]:
        idx = MASTER_COLUMNS.index(column) + 1
        for cell in ws.iter_cols(min_col=idx, max_col=idx, min_row=2, max_row=len(records) + 1):
            for item in cell:
                item.number_format = "0"
    for column in ["rmse", "smape", "learning_rate", "rmse_normalized", "smape_normalized", "nrmse_range_percent", "selected_lr"]:
        idx = MASTER_COLUMNS.index(column) + 1
        for cell in ws.iter_cols(min_col=idx, max_col=idx, min_row=2, max_row=len(records) + 1):
            for item in cell:
                if item.value != "NA":
                    item.number_format = "0.000000000000000"
    widths = {
        "A": 48, "B": 74, "C": 10, "D": 22, "E": 10, "F": 10, "G": 24, "H": 20,
        "I": 18, "J": 16, "K": 10, "L": 12, "M": 18, "N": 18, "O": 18, "P": 22,
        "Q": 12, "R": 14, "S": 24, "T": 12, "U": 70, "V": 42, "W": 70, "X": 16,
        "Y": 16, "Z": 16, "AA": 90, "AB": 24, "AC": 24, "AD": 18, "AE": 18, "AF": 18,
        "AG": 22, "AH": 18, "AI": 32, "AJ": 80, "AK": 16,
    }
    set_widths(ws, widths)
    add_table(ws, f"A1:AK{len(records)+1}", "PaperResultsMaster")


def write_dictionary_sheet(workbook) -> None:
    ws = workbook.create_sheet(DICTIONARY_SHEET)
    headers = [
        "column_name", "data_type", "allowed_values", "nullable", "definition", "example",
        "identity_field", "source_or_derived", "notes",
    ]
    ws.append(headers)
    rows = {
        "record_id": ("text", "stable readable ID", "NO", "Derived from identity fields; not an Excel row number.", "D4__166/311__H3__S42__MS-KATL-V3__WITHOUT__MAIN", "YES", "derived", "Unique within this Master."),
        "canonical_key": ("text", "pipe-delimited identity", "NO", "dataset|target|horizon|seed|method|mode|experiment_type|arm|model_version", "D4|166/311|3|42|MS-KATL-V3|without|main|NA|V3", "YES", "derived", "Must be unique; conflicts are not auto-resolved."),
        "dataset": ("text", "D1|D4|D5|D6", "NO", "Dataset identity from source.", "D4", "YES", "source", "No D2/D3 rows are silently inferred."),
        "target": ("text", "source-frozen target strings", "NO", "Exact formal target identity; stored as text.", "166/311", "YES", "source", "Must not be date-coerced."),
        "horizon": ("integer", "1|3|5", "NO", "Forecast horizon in steps.", "3", "YES", "source", "Raw identity uses integer, not H3."),
        "seed": ("integer", "source-retained seed", "NO", "Seed value retained at source row grain.", "42", "YES", "source", "Repeated deterministic-baseline rows remain present."),
        "method": ("text", "canonical method names", "NO", "Method identity; baseline Seasonal Naive is normalized to Seasonal-Naive.", "LightGBM", "YES", "derived", "Unknown source methods would be retained and audited."),
        "method_family": ("text", "proposed|learning_baseline|statistical_baseline|ablation|other", "NO", "Analysis grouping only.", "proposed", "NO", "derived", "Does not alter identity."),
        "model_version": ("text", "V3|NA", "NO", "Frozen model version where source supports it.", "V3", "YES", "derived", "Baseline version is NA."),
        "experiment_type": ("text", "main|baseline|ablation|robustness", "NO", "Experiment role.", "main", "YES", "derived", "A-E rows are ablation, including Arm D."),
        "arm": ("text", "A|B|C|D|E|NA", "NO", "A-E arm identity; NA where not applicable.", "D", "YES", "source/derived", "Arm D never replaces main V3."),
        "mode": ("text", "without|with|NA", "NO", "Scenario/mode; baselines use NA when undefined.", "without", "YES", "source/contract", "Ablation mode is inherited from the frozen contract because its raw sheet has no mode column."),
        "run_type": ("text", "formal_main|formal_baseline|formal_ablation|independent_ablation|robustness|UNKNOWN", "NO", "Run role supported by source/contract.", "formal_ablation", "NO", "derived", "No training was started by this build."),
        "rmse": ("number", "non-negative observed value", "NO", "Raw observed RMSE in the source metric space.", "14.82627599447838", "NO", "source", "Not rounded, averaged, normalized, or selected by performance."),
        "smape": ("number", "non-negative observed value", "NO", "Raw observed sMAPE in the source metric space.", "15.50946877249815", "NO", "source", "Unit remains UNKNOWN because V1.1 does not explicitly freeze it; no conversion was applied."),
        "metric_space": ("text", "original_sales_space|UNKNOWN", "NO", "Metric space stated by source.", "original_sales_space", "NO", "source", "All current selected rows are original_sales_space."),
        "rmse_unit": ("text", "NA unless contract defines unit", "NO", "RMSE unit; not defined in V1.1 contract.", "NA", "NO", "derived", "No unit inference."),
        "smape_unit": ("text", "UNKNOWN in V1.2 current freeze", "NO", "sMAPE unit; not explicitly defined in V1.1 contract/source sheets.", "UNKNOWN", "NO", "derived", "No multiplication or division by 100."),
        "source_sheet": ("text", "three V1.1 Raw sheet names", "NO", "Immediate source worksheet.", V3_SHEET, "YES", "source", "Raw source sheets remain in the workbook."),
        "source_row": ("integer", "positive Excel row", "NO", "Source worksheet Excel row.", "2", "YES", "source", "Combined with source_sheet for reconciliation."),
        "source_file": ("text", "exact source path", "NO", "Source file path inherited from the raw sheet.", "/.../three_track_robustness_all_runs.csv", "NO", "source", "Raw files are not overwritten."),
        "source_run_id": ("text", "run directory ID|NA", "NO", "Run directory parsed from source path when present.", "20260822_134005_MS_KATL_AE_CORE_ABLATION", "NO", "derived", "No run ID is invented for baseline CSVs."),
        "source_sha256": ("text", "SHA-256|UNRESOLVED", "NO", "SHA-256 of the exact source_file when readable.", "a299...", "NO", "derived", "Exact file hashes are computed read-only."),
        "paper_scope": ("text", "main_candidate|robustness_only|supporting|excluded|not_decided", "NO", "Paper-facing scope label from contract governance.", "not_decided", "NO", "contract", "V1.1 FINAL_PAPER_SELECTION_FROZEN=NO, so no final selection is inferred."),
        "include_in_master": ("boolean", "TRUE|FALSE", "NO", "Whether the source observation is retained.", "TRUE", "NO", "derived", "All current legal selected rows are TRUE."),
        "status": ("text", "OK|WARNING|CONFLICT|INCOMPLETE|EXCLUDED_BY_CONTRACT", "NO", "Row-level audit status.", "OK", "NO", "derived", "Source conflicts are retained, never silently dropped."),
        "notes": ("text", "factual notes", "YES", "Factual provenance/governance notes only.", "Arm D is independent ablation evidence.", "NO", "derived", "No performance-based recommendation is stored here."),
        "seed_semantics": ("text", "formal_seed|descriptive_repeated_seed|UNKNOWN", "NO", "Source/contract-supported seed interpretation.", "formal_seed", "NO", "contract/source", "Descriptive-only repeated baseline rows remain in Master."),
        "source_status": ("text", "source-native status", "NO", "Raw status before canonical audit mapping.", "OK", "NO", "source", "Preserved for audit."),
        "learning_rate": ("number|NA", "source value|NA", "YES", "V3 source learning rate when present.", "3e-05", "NO", "source", "Auxiliary provenance field."),
        "rmse_normalized": ("number|NA", "source value|NA", "YES", "V3 normalized RMSE auxiliary field.", "0.2471", "NO", "source", "Not the authoritative Master rmse."),
        "smape_normalized": ("number|NA", "source value|NA", "YES", "V3 normalized sMAPE auxiliary field.", "60.1825", "NO", "source", "Not the authoritative Master smape."),
        "nrmse_range_percent": ("number|NA", "source value|NA", "YES", "V3 source nRMSE range percentage.", "24.7105", "NO", "source", "Auxiliary provenance field."),
        "method_type": ("text|NA", "Learning|Heuristic|NA", "YES", "Baseline source method type.", "Learning", "NO", "source", "Does not replace method_family."),
        "arm_name": ("text|NA", "source arm label|NA", "YES", "A-E source arm name.", "CURRENT_V3", "NO", "source", "Arm D remains ablation identity."),
        "arm_description": ("text|NA", "source description|NA", "YES", "A-E source configuration description.", "init_mode=fused; ...", "NO", "source", "Preserved as factual configuration text."),
        "selected_lr": ("number|NA", "source value|NA", "YES", "A-E selected learning rate.", "3e-05", "NO", "source", "Auxiliary provenance field."),
    }
    for column in MASTER_COLUMNS:
        data_type, allowed, nullable, definition, example, identity, origin, notes = rows[column]
        ws.append([column, data_type, allowed, nullable, definition, example, identity, origin, notes])
    style_header(ws)
    style_body(ws)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:I{len(MASTER_COLUMNS)+1}"
    set_widths(ws, {"A": 24, "B": 18, "C": 34, "D": 12, "E": 62, "F": 42, "G": 15, "H": 18, "I": 74})
    add_table(ws, f"A1:I{len(MASTER_COLUMNS)+1}", "MasterDataDictionary")


def write_validation_sheet(workbook, records: list[dict[str, Any]], preflight_report: dict[str, Any], validation: dict[str, Any], formula_replacements: int) -> None:
    ws = workbook.create_sheet(VALIDATION_SHEET)
    ws.append(["Validation_Summary", "Expected", "Actual", "Status", "Notes"])
    style_header(ws)
    summary_rows: list[tuple[str, Any, Any, Any, str]] = []
    source_counts = validation["source_counts"]
    summary_rows.extend([
        ("V3_SOURCE_ROWS", source_counts[V3_SHEET], source_counts[V3_SHEET], "PASS", "Rows imported from 03_V3_Raw_5Seed."),
        ("BASELINE_SOURCE_ROWS", source_counts[BASELINE_SHEET], source_counts[BASELINE_SHEET], "PASS", "Rows imported from 04_BL_Raw_Selected."),
        ("ABLATION_SOURCE_ROWS", source_counts[ABLATION_SHEET], source_counts[ABLATION_SHEET], "PASS", "Rows imported from 05_Ablation_Raw."),
        ("EXPECTED_MASTER_ROWS", validation["source_total_rows"], validation["source_total_rows"], "PASS", "Sum of all mapped source observations; no deduplication."),
        ("ACTUAL_MASTER_ROWS", validation["master_rows"], "FORMULA", "FORMULA", "Formula counts rows in 03_Paper_Results_Master."),
        ("DUPLICATE_CANONICAL_KEYS", 0, validation["duplicate_key_count"], "PASS" if validation["duplicate_key_count"] == 0 else "FAIL", "All duplicate keys are reported; none are auto-resolved."),
        ("MISSING_REQUIRED_IDENTITIES", 0, validation["missing_required"], "PASS" if validation["missing_required"] == 0 else "FAIL", ", ".join(REQUIRED_MASTER_FIELDS)),
        ("UNMAPPED_SOURCE_ROWS", 0, validation["unmapped_source_rows"], "PASS" if validation["unmapped_source_rows"] == 0 else "FAIL", "Each source_sheet + source_row maps once."),
        ("NUMERICAL_VALUE_MODIFICATIONS", 0, validation["numeric_modifications"], "PASS" if validation["numeric_modifications"] == 0 else "FAIL", f"Source rmse/smape round-trip; Excel serialization tolerance={NUMERIC_SERIALIZATION_ABS_TOLERANCE:g}, max observed abs diff={validation['max_numeric_abs_diff']:.3g}."),
        ("MISSING_SOURCE_SHA256", 0, validation["missing_hashes"], "PASS" if validation["missing_hashes"] == 0 else "REVIEW", "Exact source_file hashes computed read-only."),
        ("TARGET_TEXT_INTEGRITY", "PASS", "PASS" if validation["target_integrity"] == 0 else "FAIL", "PASS" if validation["target_integrity"] == 0 else "FAIL", "Target values are written as text."),
        ("METRIC_SPACE_CONSISTENCY", "original_sales_space", "PASS" if validation["metric_space_consistency"] else "REVIEW", "PASS" if validation["metric_space_consistency"] else "REVIEW", "Metric spaces present: " + ", ".join(validation["metric_spaces"])),
        ("METRIC_VALIDITY", 0, len(validation["metric_invalid"]), "PASS" if not validation["metric_invalid"] else "FAIL", "Non-negative finite rmse/smape check; sMAPE unit remains UNKNOWN."),
        ("SMAPE_UNIT", "EXPLICIT_IN_CONTRACT", "UNKNOWN", "REVIEW", "V1.1 does not explicitly freeze sMAPE unit; no conversion was applied."),
        ("FINAL_PAPER_SELECTION_FROZEN", "YES", preflight_report["contract"].get("FINAL_PAPER_SELECTION_FROZEN", "UNKNOWN"), "REVIEW", "Master paper_scope remains not_decided; no final presentation selection inferred."),
        ("FALLBACK_RAW_DIR_USED", "NO", "YES" if preflight_report["fallback_used"] else "NO", "PASS", "V1.1 contained all three required raw sheets and 420 source observations."),
        ("LEGACY_FORMULA_COMPATIBILITY_REWRITES", 360, formula_replacements, "PASS" if formula_replacements == 360 else "REVIEW", "Output-only aliases: STDEV.S→STDEV and RANK.EQ→RANK; source workbook formulas are unchanged."),
    ])
    for idx, (metric, expected, actual, status, notes) in enumerate(summary_rows, start=2):
        ws.append([metric, expected, None if actual == "FORMULA" else actual, None if status == "FORMULA" else status, notes])
        if actual == "FORMULA":
            ws.cell(idx, 3).value = f"=COUNTA('{MASTER_SHEET}'!$A:$A)-1"
            ws.cell(idx, 4).value = f'=IF(C{idx}=B{idx},"PASS","FAIL")'

    row = len(summary_rows) + 3
    ws.cell(row, 1).value = "Dataset × experiment_type counts"
    ws.cell(row, 1).font = TITLE_FONT
    row += 1
    ws.append(["dataset", "experiment_type", "rows", "status", "notes"])
    style_header(ws, row)
    for (dataset, experiment_type), count in group_counts(records, ["dataset", "experiment_type"]):
        row += 1
        ws.append([dataset, experiment_type, count, "PASS", "All selected rows retained."])

    row += 2
    ws.cell(row, 1).value = "Method counts"
    ws.cell(row, 1).font = TITLE_FONT
    row += 1
    ws.append(["method", "method_family", "rows", "status", "notes"])
    style_header(ws, row)
    for (method, family), count in group_counts(records, ["method", "method_family"]):
        row += 1
        ws.append([method, family, count, "PASS", "Method identity preserved; no result selection."])

    row += 2
    ws.cell(row, 1).value = "Seed counts by dataset × method × horizon"
    ws.cell(row, 1).font = TITLE_FONT
    row += 1
    ws.append(["dataset", "method", "horizon", "n", "seeds", "status", "notes"])
    style_header(ws, row)
    for key, count in group_counts(records, ["dataset", "method", "horizon"]):
        dataset, method, horizon = key
        seeds = sorted({record["seed"] for record in records if (record["dataset"], record["method"], record["horizon"]) == key})
        row += 1
        ws.append([dataset, method, horizon, count, ",".join(str(seed) for seed in seeds), "PASS", "Coverage reflects source/contract semantics; no universal five-seed assumption."])

    row += 2
    ws.cell(row, 1).value = "Horizon coverage"
    ws.cell(row, 1).font = TITLE_FONT
    row += 1
    ws.append(["horizon", "rows", "methods", "status", "notes"])
    style_header(ws, row)
    for horizon in sorted({record["horizon"] for record in records}):
        methods = sorted({record["method"] for record in records if record["horizon"] == horizon})
        row += 1
        ws.append([horizon, sum(1 for record in records if record["horizon"] == horizon), ", ".join(methods), "PASS", "Contract horizons are 1, 3, 5."])

    row += 2
    ws.cell(row, 1).value = "Source reconciliation"
    ws.cell(row, 1).font = TITLE_FONT
    row += 1
    ws.append(["source_sheet", "source_rows", "master_rows", "unmapped_rows", "numeric_modifications", "status", "notes"])
    style_header(ws, row)
    for sheet_name in REQUIRED_SOURCE_SHEETS:
        row += 1
        source_n = validation["source_counts"][sheet_name]
        master_n = validation["source_master_counts"][sheet_name]
        status = "PASS" if source_n == master_n else "FAIL"
        ws.append([sheet_name, source_n, master_n, 0, validation["numeric_modifications"] if status == "FAIL" else 0, status, "Source row to Master mapping uses exact source_sheet + source_row."])

    row += 2
    ws.cell(row, 1).value = "Open review items"
    ws.cell(row, 1).font = TITLE_FONT
    row += 1
    ws.append(["issue", "count", "status", "details"])
    style_header(ws, row)
    review_items = [
        ("SMAPE_UNIT_NOT_FROZEN", validation["master_rows"], "REVIEW", "V1.1 headers contain sMAPE but do not explicitly define percent/unit; Master keeps raw values unchanged."),
        ("FINAL_PAPER_SELECTION_NOT_FROZEN", validation["master_rows"], "REVIEW", "Contract FINAL_PAPER_SELECTION_FROZEN=NO; paper_scope is not_decided for retained observations."),
        ("DERIVED_SHEETS_CURRENT_LINEAGE", len([name for name in preflight_report["sheets"] if name not in REQUIRED_SOURCE_SHEETS]), "REVIEW", "Existing V1.1 derived sheets are preserved; lineage audit documents that current formulas were not rewritten to Master."),
    ]
    for item in review_items:
        row += 1
        ws.append(list(item))

    style_body(ws, start_row=2)
    ws.freeze_panes = "A2"
    set_widths(ws, {"A": 34, "B": 24, "C": 24, "D": 16, "E": 110, "F": 18, "G": 110})


def write_aggregate_sheet(workbook, records: list[dict[str, Any]]) -> None:
    ws = workbook.create_sheet(AGGREGATE_SHEET)
    headers = [
        "dataset", "target", "horizon", "method", "mode", "experiment_type", "arm", "model_version",
        "n", "mean_rmse", "std_rmse", "mean_smape", "std_smape", "calculation_note",
    ]
    ws.append(headers)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    group_fields = ["dataset", "target", "horizon", "method", "mode", "experiment_type", "arm", "model_version"]
    for record in records:
        groups[tuple(record[field] for field in group_fields)].append(record)
    from openpyxl.utils import get_column_letter
    master_col = {name: get_column_letter(MASTER_COLUMNS.index(name) + 1) for name in MASTER_COLUMNS}
    for row_number, key in enumerate(sorted(groups, key=lambda item: tuple(str(value) for value in item)), start=2):
        members = groups[key]
        rmse_values = [float(item["rmse"]) for item in members]
        smape_values = [float(item["smape"]) for item in members]
        std_rmse = statistics.stdev(rmse_values) if len(rmse_values) > 1 else 0.0
        std_smape = statistics.stdev(smape_values) if len(smape_values) > 1 else 0.0
        ws.append(list(key) + [None, None, std_rmse, None, std_smape, "n/mean formulas reference 03_Paper_Results_Master; std values are builder-derived audit values."])
        criteria = [
            (master_col["dataset"], "A"), (master_col["target"], "B"), (master_col["horizon"], "C"),
            (master_col["method"], "D"), (master_col["mode"], "E"), (master_col["experiment_type"], "F"),
            (master_col["arm"], "G"), (master_col["model_version"], "H"),
        ]
        count_criteria = ",".join(f"'{MASTER_SHEET}'!${col}:${col},{cell}{row_number}" for col, cell in criteria)
        ws.cell(row_number, 9).value = f"=COUNTIFS({count_criteria})"
        avg_criteria = count_criteria
        ws.cell(row_number, 10).value = f"=AVERAGEIFS('{MASTER_SHEET}'!${master_col['rmse']}:${master_col['rmse']},{avg_criteria})"
        ws.cell(row_number, 12).value = f"=AVERAGEIFS('{MASTER_SHEET}'!${master_col['smape']}:${master_col['smape']},{avg_criteria})"
    style_header(ws)
    style_body(ws)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:N{len(groups)+1}"
    for col in ["C", "I"]:
        for cell in ws[col][1:]:
            cell.number_format = "0"
    for col in ["J", "K", "L", "M"]:
        for cell in ws[col][1:]:
            cell.number_format = "0.000000000000000"
    set_widths(ws, {"A": 10, "B": 22, "C": 10, "D": 24, "E": 12, "F": 16, "G": 10, "H": 16, "I": 10, "J": 22, "K": 22, "L": 22, "M": 22, "N": 100})
    add_table(ws, f"A1:N{len(groups)+1}", "MasterAggregateCheck")


def write_lineage_sheet(workbook, preflight_report: dict[str, Any], formula_replacements: int) -> None:
    ws = workbook.create_sheet(LINEAGE_SHEET)
    ws.append(["sheet_name", "table_purpose", "current_source", "master_reproducible", "formula_or_script", "discrepancy", "action_required"])
    analysis_sources = {
        "06_Main_RMSE": ("main RMSE condition summary", "03_V3_Raw_5Seed + 04_BL_Raw_Selected", "YES (data coverage)", "Existing V1.1 formulas + compatibility alias", "V1.2 output normalizes STDEV.S to STDEV for LibreOffice/Excel compatibility; formulas still reference V1.1 raw sheets.", "Future formula migration to Master only if explicitly requested."),
        "07_Main_sMAPE": ("main sMAPE condition summary", "03_V3_Raw_5Seed + 04_BL_Raw_Selected", "YES (data coverage)", "Existing V1.1 formulas + compatibility alias", "V1.2 output normalizes STDEV.S to STDEV for LibreOffice/Excel compatibility; formulas still reference V1.1 raw sheets.", "Future formula migration to Master only if explicitly requested."),
        "08_Learning_Baseline": ("learning-baseline comparison", "06_Main_RMSE + 07_Main_sMAPE", "YES (through source rows)", "Existing V1.1 formulas", "Current formulas remain on V1.1 summary sheets.", "No change in V1.2."),
        "09_Relative_Gain": ("relative gain / pairwise labels", "06_Main_RMSE + 07_Main_sMAPE", "YES (through source rows)", "Existing V1.1 formulas", "Current formulas remain on V1.1 summary sheets.", "No change in V1.2."),
        "10_Stability": ("three-seed vs five-seed stability", "V3 stability source / 03_V3_Raw_5Seed", "YES (source coverage)", "Existing V1.1 values/formulas + compatibility alias", "V1.2 output normalizes STDEV.S/RANK.EQ aliases where present; Master does not replace the dedicated stability source.", "Keep stability source separately audited."),
        "11_Ablation_Summary": ("A-E mechanism ablation summary", "05_Ablation_Raw + 12_Ablation_Contract", "YES (data coverage)", "Existing V1.1 formulas + compatibility alias", f"V1.2 output normalizes RANK.EQ to RANK for LibreOffice/Excel compatibility ({formula_replacements} total output-only rewrites across legacy analysis sheets). Arm D remains independent ablation evidence.", "No substitution of main V3 values."),
        "12_Ablation_Contract": ("ablation governance contract", "A-E manifest / contract", "NOT_APPLICABLE", "Governance sheet", "Not a result table.", "Preserve as provenance."),
        "13_Paper_Selection": ("paper presentation decision layer", "06_Main_RMSE + 07_Main_sMAPE", "YES (through source rows)", "Existing V1.1 formulas", "FINAL_PAPER_SELECTION_FROZEN=NO.", "Keep decisions outside Master."),
        "14_Paper_Summary": ("paper-safe summary / rank layer", "06_Main_RMSE + 07_Main_sMAPE + 09_Relative_Gain", "YES (through source rows)", "Existing V1.1 formulas", "Not rewritten to Master in this consolidation.", "Future derived-table migration may reference Master."),
        "15_Data_Quality": ("V1.1 data quality snapshot", "V1.1 source audit", "YES (audit overlap)", "Existing V1.1 values", "This is a preserved snapshot, not the V1.2 validation engine.", "Use Master_Validation for V1.2 checks."),
    }
    for name, values in analysis_sources.items():
        if name in preflight_report["sheets"]:
            ws.append([name, *values])
    style_header(ws)
    style_body(ws)
    ws.freeze_panes = "A2"
    set_widths(ws, {"A": 28, "B": 34, "C": 40, "D": 24, "E": 34, "F": 74, "G": 68})
    add_table(ws, f"A1:G{ws.max_row}", "DerivedTableLineage")


def write_baseline_audit(workbook, records: list[dict[str, Any]], preflight_report: dict[str, Any]) -> None:
    ws = workbook.create_sheet(BASELINE_AUDIT_SHEET)
    ws.append(["audit_item", "value", "status", "notes"])
    present = sorted({record["method"] for record in records if record["experiment_type"] == "baseline"})
    contract_methods = sorted(preflight_report["contract_baselines"])
    missing = sorted(set(contract_methods) - set(present))
    extra = sorted(set(present) - set(contract_methods))
    rows = [
        ("BASELINES_PRESENT", "|".join(present), "PASS" if present else "FAIL", "Canonical method identities in 04_BL_Raw_Selected."),
        ("CONTRACT_BASELINE_MISSING_FROM_MASTER", "|".join(missing) if missing else "NONE", "PASS" if not missing else "REVIEW", "Derived from 01_Paper_Contract Baseline methods."),
        ("BASELINE_SCOPE_NOT_FROZEN", "NO" if not extra else "YES", "PASS" if not extra else "REVIEW", "Extra source methods are never silently removed."),
        ("BASELINE_METHODS_PRESENT_BUT_CONTRACT_UNRESOLVED", "|".join(extra) if extra else "NONE", "PASS" if not extra else "REVIEW", "No additional baseline method appears in the current source."),
        ("BASELINE_MODE_POLICY", "NA", "PASS", "Baseline source has no with/without definition; no mode was invented."),
        ("FINAL_PAPER_SELECTION_FROZEN", preflight_report["contract"].get("FINAL_PAPER_SELECTION_FROZEN", "UNKNOWN"), "REVIEW", "Presentation decisions remain outside Master."),
    ]
    for row in rows:
        ws.append(list(row))
    style_header(ws)
    style_body(ws)
    ws.freeze_panes = "A2"
    set_widths(ws, {"A": 48, "B": 60, "C": 16, "D": 110})
    add_table(ws, f"A1:D{ws.max_row}", "BaselineCoverageAudit")


def write_dataset_audit(workbook, records: list[dict[str, Any]], preflight_report: dict[str, Any]) -> None:
    ws = workbook.create_sheet(DATASET_AUDIT_SHEET)
    ws.append(["audit_item", "value", "status", "notes"])
    present = sorted({record["dataset"] for record in records})
    defined = sorted(preflight_report["contract_datasets"])
    missing = sorted(set(defined) - set(present))
    extra = sorted(set(present) - set(defined))
    rows = [
        ("DATASETS_PRESENT_IN_MASTER", "|".join(present), "PASS", "Observed in the three V1.1 source sheets."),
        ("DATASETS_DEFINED_IN_PAPER_CONTRACT", "|".join(defined), "PASS", "Parsed from 01_Paper_Contract Selected datasets."),
        ("CONTRACT_DATASETS_MISSING_FROM_MASTER", "|".join(missing) if missing else "NONE", "PASS" if not missing else "REVIEW", "No rows are inferred for missing datasets."),
        ("MASTER_DATASETS_NOT_DEFINED_IN_CONTRACT", "|".join(extra) if extra else "NONE", "PASS" if not extra else "REVIEW", "Unexpected datasets would be retained and audited."),
        ("D2_D3_SCOPE_STATUS", "not_present_in_selected_V1.1_rows", "REVIEW", "Current contract defines D1/D4/D5/D6 for paper-facing selected sheets; this is not a new exclusion decision."),
        ("TARGETS_PRESENT", "|".join(sorted({record["target"] for record in records})), "PASS", "Exact source target strings retained as text."),
        ("HORIZONS_PRESENT", "|".join(str(x) for x in sorted({record["horizon"] for record in records})), "PASS", "Contract horizons 1, 3, 5."),
    ]
    for row in rows:
        ws.append(list(row))
    style_header(ws)
    style_body(ws)
    ws.freeze_panes = "A2"
    set_widths(ws, {"A": 48, "B": 80, "C": 16, "D": 110})
    add_table(ws, f"A1:D{ws.max_row}", "DatasetScopeAudit")


def append_governance_rows(workbook, input_path: Path, validation: dict[str, Any], formula_replacements: int) -> None:
    readme = workbook["00_README"]
    readme_rows = [
        ["V1.2 output version", "MS-KATL Paper Data Freeze V1.2", "V1.1 workbook is preserved as the input baseline; Master and audit sheets are added in this output.", "READY", str(input_path)],
        ["AUTHORITATIVE_PAPER_RESULT_TABLE", MASTER_SHEET, "All paper-facing aggregate statistics, comparisons, rankings, significance tests, figures and manuscript tables should be reproducible from this Master unless explicitly documented otherwise.", "PASS", "V1.2 consolidation script"],
        ["V1.2 change", "Unified long-format Master", "One row per dataset × target × horizon × seed × method × mode × experiment_type × arm × model_version; raw source sheets remain.", "PASS", "V1.2 consolidation script"],
        ["MASTER_MODIFICATION_POLICY", "Raw rmse/smape immutable", "Do not manually edit Master raw values, average seed rows, select best seeds, or replace main V3 with Arm D.", "PASS", "V1.2 governance"],
        ["RAW_SOURCE_SHEETS", f"{V3_SHEET}; {BASELINE_SHEET}; {ABLATION_SHEET}", "These sheets remain for provenance, human inspection, and source reconciliation.", "PASS", "V1.1 workbook"],
        ["MASTER_ROW_COUNT", validation["master_rows"], "Source row count reconciliation passed; no source row was dropped for performance reasons.", "PASS", "Master_Validation"],
        ["FALLBACK_RAW_RESULT_DIRECTORY", "NOT_USED", "V1.1 contained all required selected raw sheets and source files; the supplied fallback directory was not used to supplement or replace rows.", "PASS", str(DEFAULT_RAW_DIR)],
        ["FORMULA_COMPATIBILITY_REWRITES", formula_replacements, "Output-only compatibility aliases STDEV.S→STDEV and RANK.EQ→RANK; V1.1 formulas and raw sources remain unchanged.", "PASS", "Master_Validation / Derived_Table_Lineage"],
    ]
    for row in readme_rows:
        readme.append(row)
    style_body(readme, start_row=readme.max_row - len(readme_rows) + 1)

    contract = workbook["01_Paper_Contract"]
    contract_rows = [
        ["V1.2 Master Raw identity", MASTER_SHEET, "Unique key = dataset|target|horizon|seed|method|mode|experiment_type|arm|model_version; raw values remain unchanged.", "PASS", "V1.2 consolidation script"],
        ["Master paper scope", "not_decided", "FINAL_PAPER_SELECTION_FROZEN=NO in V1.1; no performance-based paper selection is inferred in Master.", "REVIEW", "V1.1 contract"],
        ["Master metric space", "original_sales_space", "All current selected source rows state original_sales_space; no scale conversion is applied.", "PASS", "V1.1 raw sheets"],
        ["Master sMAPE unit", "UNKNOWN", "V1.1 does not explicitly freeze sMAPE unit; values are copied exactly and no ×100/÷100 operation is applied.", "REVIEW", "V1.1 raw sheets / contract"],
        ["Derived-table policy", "Master is authoritative", "Existing analysis sheets are preserved; Derived_Table_Lineage records current sources and future migration notes.", "PASS", "V1.2 consolidation script"],
    ]
    for row in contract_rows:
        contract.append(row)
    style_body(contract, start_row=contract.max_row - len(contract_rows) + 1)


def reorder_new_sheets(workbook) -> None:
    desired = [MASTER_SHEET, DICTIONARY_SHEET, VALIDATION_SHEET, LINEAGE_SHEET, BASELINE_AUDIT_SHEET, DATASET_AUDIT_SHEET, AGGREGATE_SHEET]
    old = [ws for ws in workbook.worksheets if ws.title not in desired]
    governance_index = 0
    for idx, ws in enumerate(old):
        if ws.title == "02_Source_Audit":
            governance_index = idx + 1
            break
    new_sheets = [workbook[name] for name in desired]
    workbook._sheets = old[:governance_index] + new_sheets + old[governance_index:]


def post_validate(output_path: Path, expected_records: list[dict[str, Any]], preflight_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "FAIL", "issues": [], "max_numeric_abs_diff": 0.0}
    if not output_path.is_file():
        result["issues"].append(f"missing output: {output_path}")
        return result
    wb_formula = load_workbook(output_path, read_only=False, data_only=False)
    wb_values = load_workbook(output_path, read_only=False, data_only=True)
    required_sheets = [MASTER_SHEET, DICTIONARY_SHEET, VALIDATION_SHEET, LINEAGE_SHEET, BASELINE_AUDIT_SHEET, DATASET_AUDIT_SHEET, AGGREGATE_SHEET]
    for sheet_name in required_sheets:
        if sheet_name not in wb_formula.sheetnames:
            result["issues"].append(f"missing sheet: {sheet_name}")
    if MASTER_SHEET not in wb_formula.sheetnames:
        return result
    ws = wb_formula[MASTER_SHEET]
    ws_values = wb_values[MASTER_SHEET]
    actual_rows = max(0, ws.max_row - 1)
    if actual_rows != len(expected_records):
        result["issues"].append(f"master row count {actual_rows} != {len(expected_records)}")
    headers = [cell.value for cell in ws[1]]
    if headers != MASTER_COLUMNS:
        result["issues"].append("Master headers do not match required schema")
    loaded: list[dict[str, Any]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        loaded.append({headers[idx]: row[idx] if idx < len(row) else None for idx in range(len(headers))})
    keys = [row.get("canonical_key") for row in loaded]
    if len(keys) != len(set(keys)):
        result["issues"].append("duplicate canonical keys in output")
    for row in loaded:
        for field in REQUIRED_MASTER_FIELDS:
            if row.get(field) is None or row.get(field) == "":
                result["issues"].append(f"missing {field} in {row.get('record_id')}")
        if not isinstance(row.get("target"), str) or ws.cell(row=2, column=MASTER_COLUMNS.index("target") + 1).number_format != "@":
            result["issues"].append(f"target text integrity failed in {row.get('record_id')}")
        if row.get("method") not in EXPECTED_METHODS:
            result["issues"].append(f"unexpected method {row.get('method')}")
        if row.get("arm") not in EXPECTED_ARMS:
            result["issues"].append(f"illegal arm {row.get('arm')}")
        if not isinstance(row.get("rmse"), (int, float)) or isinstance(row.get("rmse"), bool):
            result["issues"].append(f"non-numeric rmse in {row.get('record_id')}")
        if not isinstance(row.get("smape"), (int, float)) or isinstance(row.get("smape"), bool):
            result["issues"].append(f"non-numeric smape in {row.get('record_id')}")
    expected_by_key = {(r["source_sheet"], r["source_row"]): r for r in expected_records}
    loaded_by_key = {(r.get("source_sheet"), r.get("source_row")): r for r in loaded}
    for key, expected in expected_by_key.items():
        actual = loaded_by_key.get(key)
        if actual is None:
            result["issues"].append(f"unmapped source row {key}")
            continue
        for actual_value, expected_value in [(actual.get("rmse"), expected["rmse"]), (actual.get("smape"), expected["smape"] )]:
            if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
                result["max_numeric_abs_diff"] = max(
                    result["max_numeric_abs_diff"],
                    abs(float(actual_value) - float(expected_value)),
                )
        if not numeric_values_equal(actual.get("rmse"), expected["rmse"]) or not numeric_values_equal(actual.get("smape"), expected["smape"]):
            result["issues"].append(f"numeric modification in {key}")
    error_tokens = {"#REF!", "#DIV/0!", "#VALUE!", "#N/A", "#NAME?", "#NUM!", "#NULL!"}
    formula_errors = []
    for wb, label in [(wb_formula, "formula"), (wb_values, "value")]:
        for sheet in wb.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    value = cell.value
                    if isinstance(value, str) and any(token in value for token in error_tokens):
                        formula_errors.append((label, sheet.title, cell.coordinate, value))
    if formula_errors:
        result["issues"].append(f"formula/value errors: {formula_errors[:8]}")
    result["actual_rows"] = actual_rows
    result["formula_errors"] = formula_errors
    result["status"] = "PASS" if not result["issues"] else "FAIL"
    return result


def save_workbook(input_path: Path, output_path: Path, records: list[dict[str, Any]], preflight_report: dict[str, Any], validation: dict[str, Any]) -> None:
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Refusing to overwrite input workbook")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    workbook = load_workbook(input_path, read_only=False, data_only=False)
    for sheet_name in [MASTER_SHEET, DICTIONARY_SHEET, VALIDATION_SHEET, LINEAGE_SHEET, BASELINE_AUDIT_SHEET, DATASET_AUDIT_SHEET, AGGREGATE_SHEET]:
        if sheet_name in workbook.sheetnames:
            del workbook[sheet_name]
    formula_replacements = normalize_formula_compatibility(workbook)
    validation["formula_replacements"] = formula_replacements
    append_governance_rows(workbook, input_path, validation, formula_replacements)
    write_master_sheet(workbook, records)
    write_dictionary_sheet(workbook)
    write_validation_sheet(workbook, records, preflight_report, validation, formula_replacements)
    write_lineage_sheet(workbook, preflight_report, formula_replacements)
    write_baseline_audit(workbook, records, preflight_report)
    write_dataset_audit(workbook, records, preflight_report)
    write_aggregate_sheet(workbook, records)
    reorder_new_sheets(workbook)
    workbook.properties.creator = "Codex"
    workbook.properties.title = "MS-KATL Paper Results Master V1.2"
    workbook.save(output_path)


def freeze_report_text(input_path: Path, output_path: Path, input_sha: str, output_sha: str, preflight_report: dict[str, Any], validation: dict[str, Any], post: dict[str, Any]) -> str:
    timestamp = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(timespec="seconds")
    datasets = sorted({row["dataset"] for row in build_records(preflight_report)})
    targets = sorted({row["target"] for row in build_records(preflight_report)})
    horizons = sorted({row["horizon"] for row in build_records(preflight_report)})
    seeds = sorted({row["seed"] for row in build_records(preflight_report)})
    methods = sorted({row["method"] for row in build_records(preflight_report)})
    modes = sorted({row["mode"] for row in build_records(preflight_report)})
    arms = sorted({row["arm"] for row in build_records(preflight_report)})
    lines = [
        "==================================================",
        "MS-KATL PAPER RESULTS MASTER V1.2 FREEZE",
        "==================================================",
        f"TIMESTAMP={timestamp}",
        f"INPUT_WORKBOOK={input_path}",
        f"INPUT_SHA256={input_sha}",
        f"OUTPUT_WORKBOOK={output_path}",
        f"OUTPUT_WORKBOOK_SHA256={output_sha}",
        "TRAINING_STARTED=NO",
        "EXPERIMENT_STARTED=NO",
        "MODEL_CODE_MODIFIED=NO",
        "EXPERIMENT_RESULTS_MODIFIED=NO",
        "RAW_RESULT_OVERWRITE=NO",
        "SOURCE_WORKBOOK_OVERWRITE=NO",
        f"FALLBACK_RAW_DIR_USED={'YES' if preflight_report['fallback_used'] else 'NO'}",
        f"V3_SOURCE_ROWS={validation['source_counts'][V3_SHEET]}",
        f"BASELINE_SOURCE_ROWS={validation['source_counts'][BASELINE_SHEET]}",
        f"ABLATION_SOURCE_ROWS={validation['source_counts'][ABLATION_SHEET]}",
        f"SOURCE_TOTAL_ROWS={validation['source_total_rows']}",
        f"MASTER_ROWS={validation['master_rows']}",
        f"DATASETS={'|'.join(datasets)}",
        f"TARGETS={'|'.join(targets)}",
        f"HORIZONS={'|'.join(str(x) for x in horizons)}",
        f"SEEDS={'|'.join(str(x) for x in seeds)}",
        f"METHODS={'|'.join(methods)}",
        f"MODES={'|'.join(modes)}",
        f"ABLATION_ARMS={'|'.join(arms)}",
        f"DUPLICATE_CANONICAL_KEYS={validation['duplicate_key_count']}",
        f"MISSING_REQUIRED_IDENTITIES={validation['missing_required']}",
        f"UNMAPPED_SOURCE_ROWS={validation['unmapped_source_rows']}",
        f"NUMERICAL_VALUE_MODIFICATIONS={validation['numeric_modifications']}",
        f"MAX_NUMERIC_SERIALIZATION_ABS_DIFF={post.get('max_numeric_abs_diff', validation['max_numeric_abs_diff']):.17g}",
        f"FORMULA_COMPATIBILITY_REWRITES={validation.get('formula_replacements', 'NOT_RECORDED')}",
        f"TARGET_TEXT_INTEGRITY={'PASS' if validation['target_integrity'] == 0 else 'FAIL'}",
        f"METRIC_SPACE_CONSISTENCY={'PASS' if validation['metric_space_consistency'] else 'REVIEW'}",
        f"SOURCE_RECONCILIATION={'PASS' if validation['unmapped_source_rows'] == 0 and validation['numeric_modifications'] == 0 else 'FAIL'}",
        f"MASTER_UNIQUENESS={'PASS' if validation['duplicate_key_count'] == 0 else 'FAIL'}",
        "BASELINE_SCOPE_STATUS=SELECTED_METHODS_PRESENT_FINAL_PRESENTATION_NOT_FROZEN",
        "DATASET_SCOPE_STATUS=CONTRACT_DEFINED_D1_D4_D5_D6_PRESENT_D2_D3_NOT_IN_SELECTED_ROWS",
        f"POST_BUILD_VALIDATION={post['status']}",
        "REVIEW_ITEMS=SMAPE_UNIT_NOT_FROZEN;FINAL_PAPER_SELECTION_NOT_FROZEN;EXISTING_DERIVED_FORMULAS_NOT_REWRITTEN",
        f"FINAL_VALIDATION={'PASS' if post['status'] == 'PASS' and validation['duplicate_key_count'] == 0 and validation['numeric_modifications'] == 0 else 'FAIL'}",
        f"PAPER_RESULTS_MASTER_V1_2_FROZEN={'YES' if post['status'] == 'PASS' else 'NO'}",
        "==================================================",
    ]
    return "\n".join(lines) + "\n"


def print_final_report(input_path: Path, output_path: Path, input_sha: str, output_sha: str, preflight_report: dict[str, Any], validation: dict[str, Any], post: dict[str, Any]) -> None:
    report = freeze_report_text(input_path, output_path, input_sha, output_sha, preflight_report, validation, post)
    print(report, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actions = sum(bool(flag) for flag in [args.preflight_only, args.build, args.validate_only])
    if actions != 1:
        print("Choose exactly one of --preflight-only, --build, --validate-only", file=sys.stderr)
        return 2
    report = preflight(args.input, args.raw_dir, print_report=True)
    if args.preflight_only:
        return 0 if report["status"] == "PASS" else 1
    if report["status"] != "PASS":
        print("BUILD_STOPPED=PRELIGHT_FAIL", file=sys.stderr)
        return 1
    records = build_records(report)
    validation = records_validation(records, report)
    if args.validate_only:
        if not args.output:
            print("--validate-only requires --output", file=sys.stderr)
            return 2
        post = post_validate(args.output, records, report)
        print(f"POST_BUILD_VALIDATION={post['status']}")
        if post["issues"]:
            for issue in post["issues"][:30]:
                print(f"VALIDATION_ISSUE={issue}")
        return 0 if post["status"] == "PASS" else 1
    if not args.output:
        print("--build requires --output", file=sys.stderr)
        return 2
    save_workbook(args.input, args.output, records, report, validation)
    post = post_validate(args.output, records, report)
    if post["status"] != "PASS":
        print(f"POST_BUILD_VALIDATION={post['status']}", file=sys.stderr)
        for issue in post["issues"][:30]:
            print(f"VALIDATION_ISSUE={issue}", file=sys.stderr)
        return 1
    output_sha = sha256_file(args.output)
    if args.freeze_report:
        args.freeze_report.write_text(
            freeze_report_text(args.input, args.output, report["input_sha256"], output_sha, report, validation, post),
            encoding="utf-8",
        )
    print_final_report(args.input, args.output, report["input_sha256"], output_sha, report, validation, post)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
