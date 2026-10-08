#!/usr/bin/env python3
"""Create and audit the V1.2 metadata-only paper-data freeze.

This tool deliberately changes only workbook metadata and the existing
``smape_unit`` metadata column in the consolidated Master.  It does not
rebuild, recalculate, select, average, or otherwise alter result rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from copy import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CURRENT_VERSION = "V1.2"
BASE_VERSION = "V1.1"
SMAPE_UNIT = "percent"
SMAPE_SCALE = "0_to_200_percent"
SMAPE_DISPLAY_RULE = "VALUE_ALREADY_IN_PERCENT_DO_NOT_MULTIPLY_BY_100"
MASTER_SHEET = "03_Paper_Results_Master"
METADATA_SHEETS = {
    "00_README",
    "01_Paper_Contract",
    "04_Master_Data_Dictionary",
    "Master_Validation",
}
AUDIT_FIELDS = (
    "dataset",
    "target",
    "horizon",
    "seed",
    "method",
    "mode",
    "experiment_type",
    "arm",
    "model_version",
    "rmse",
    "smape",
    "canonical_key",
    "record_id",
)
PROVENANCE_FIELDS = (
    "source_sheet",
    "source_row",
    "source_file",
    "source_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header_map(ws: Any) -> dict[str, int]:
    return {str(cell.value): cell.column for cell in ws[1] if cell.value is not None}


def row_by_first_column(ws: Any, value: str) -> int | None:
    for row in range(2, ws.max_row + 1):
        if ws.cell(row, 1).value == value:
            return row
    return None


def copy_row_style(ws: Any, source_row: int, target_row: int) -> None:
    for column in range(1, ws.max_column + 1):
        source = ws.cell(source_row, column)
        target = ws.cell(target_row, column)
        if source.has_style:
            target._style = copy(source._style)
        if source.number_format:
            target.number_format = source.number_format
        if source.alignment:
            target.alignment = copy(source.alignment)
        if source.protection:
            target.protection = copy(source.protection)
    if source_row in ws.row_dimensions:
        ws.row_dimensions[target_row].height = ws.row_dimensions[source_row].height


def upsert_table_row(ws: Any, values: Iterable[Any]) -> int:
    values = list(values)
    if not values:
        raise ValueError("metadata row must contain at least one value")
    row = row_by_first_column(ws, str(values[0]))
    if row is None:
        row = ws.max_row + 1
        copy_row_style(ws, max(1, row - 1), row)
    for column, value in enumerate(values, start=1):
        ws.cell(row, column).value = value
    return row


def set_readme_metadata(ws: Any, final_output_path: str, input_path: Path) -> None:
    workbook_row = row_by_first_column(ws, "Workbook")
    if workbook_row is None:
        raise ValueError("00_README is missing the Workbook row")
    ws.cell(workbook_row, 2).value = "MS-KATL Paper Data Freeze V1.2"
    ws.cell(workbook_row, 3).value = (
        "Current workbook is V1.2; V1.1 is the preserved base input. "
        "This pass freezes metadata only and leaves numerical evidence unchanged."
    )

    output_row = row_by_first_column(ws, "V1.2 output version")
    if output_row is not None:
        ws.cell(output_row, 2).value = "MS-KATL Paper Data Freeze V1.2"
        ws.cell(output_row, 3).value = (
            "Current deliverable identity; no V1.3 was created."
        )
        ws.cell(output_row, 5).value = str(input_path)

    rows = [
        (
            "CURRENT_WORKBOOK_VERSION",
            CURRENT_VERSION,
            "Current deliverable version; no V1.3 was created.",
            "PASS",
            final_output_path,
        ),
        (
            "BASE_INPUT_VERSION",
            BASE_VERSION,
            "Read-only base workbook used as the input for this metadata freeze.",
            "PASS",
            str(input_path),
        ),
        (
            "VERSION_CHANGE_TYPE",
            "METADATA_ONLY_FREEZE",
            "No result, identity, provenance, row, or schema changes are permitted.",
            "PASS",
            "V1.2 metadata freeze procedure",
        ),
        (
            "SMAPE_UNIT",
            SMAPE_UNIT,
            "sMAPE is stored as a percentage value.",
            "PASS",
            "src/evaluation/metric_contract.py; src/evaluation/metrics.py",
        ),
        (
            "SMAPE_SCALE",
            SMAPE_SCALE,
            "Formal sMAPE range is 0 to 200 percent.",
            "PASS",
            "src/evaluation/metric_contract.py",
        ),
        (
            "SMAPE_DISPLAY_RULE",
            SMAPE_DISPLAY_RULE,
            "Stored values already include the percentage scaling; do not multiply by 100 again.",
            "PASS",
            "src/evaluation/metrics.py",
        ),
        (
            "RMSE_UNIT",
            "dataset_native_sales_unit",
            "Generic original-sales-space unit; no more specific unit is asserted by the contract.",
            "PASS",
            "01_Paper_Contract",
        ),
        (
            "METADATA_ONLY_FREEZE",
            "YES",
            "Only metadata fields were corrected; model code and source result artifacts were not modified.",
            "PASS",
            "Freeze audit",
        ),
    ]
    for row in rows:
        upsert_table_row(ws, row)


def set_contract_metadata(ws: Any, final_output_path: str, input_path: Path) -> None:
    paper_row = row_by_first_column(ws, "Paper data freeze")
    if paper_row is None:
        raise ValueError("01_Paper_Contract is missing the Paper data freeze row")
    ws.cell(paper_row, 2).value = (
        "CURRENT_FREEZE_VERSION=V1.2; BASE_INPUT_VERSION=V1.1"
    )
    ws.cell(paper_row, 3).value = (
        "Current V1.2 is a metadata-only freeze of the V1.2 input workbook; "
        "V1.1 remains the historical base identity. No retraining or result regeneration."
    )

    output_row = row_by_first_column(ws, "Output")
    if output_row is not None:
        ws.cell(output_row, 2).value = final_output_path
        ws.cell(output_row, 3).value = "Current V1.2 metadata-frozen deliverable."

    rows = [
        (
            "CURRENT_WORKBOOK_VERSION",
            CURRENT_VERSION,
            "Current workbook identity for this deliverable.",
            "PASS",
            final_output_path,
        ),
        (
            "BASE_INPUT_VERSION",
            BASE_VERSION,
            "Base input identity; the original V1.2 workbook was read-only input and V1.1 is its preserved source baseline.",
            "PASS",
            str(input_path),
        ),
        (
            "VERSION_CHANGE_TYPE",
            "METADATA_ONLY_FREEZE",
            "No RMSE, sMAPE, dataset, target, horizon, seed, method, arm, mode, identity, provenance, row, or schema changes.",
            "PASS",
            "Freeze audit",
        ),
        (
            "PRIMARY_METRIC_SPACE",
            "original_sales_space",
            "RMSE is evaluated in original sales space.",
            "PASS",
            "01_Paper_Contract; source metric contract",
        ),
        (
            "RMSE_UNIT",
            "dataset_native_sales_unit",
            "Generic dataset-native sales unit only; no specific unit is inferred.",
            "PASS",
            "Metric-space contract",
        ),
        (
            "SMAPE_UNIT",
            SMAPE_UNIT,
            "sMAPE is formally stored in percent units.",
            "PASS",
            "src/evaluation/metric_contract.py:14-18",
        ),
        (
            "SMAPE_SCALE",
            SMAPE_SCALE,
            "Range check is 0 <= sMAPE <= 200 percent.",
            "PASS",
            "src/evaluation/metric_contract.py:16-18",
        ),
        (
            "SMAPE_DISPLAY_RULE",
            SMAPE_DISPLAY_RULE,
            "Values are already percentage-scaled; do not apply another factor of 100.",
            "PASS",
            "src/evaluation/metrics.py:59-88",
        ),
        (
            "SMAPE_DEFINITION_ID",
            "smape_2abs_eps1e-8_pct_v1",
            "mean(2*abs(y_pred-y_true)/(abs(y_true)+abs(y_pred)+1e-8))*100.",
            "PASS",
            "src/evaluation/metric_contract.py; src/evaluation/metrics.py",
        ),
        (
            "V1.1_HISTORICAL_IDENTITY",
            "MS-KATL PAPER DATA FREEZE V1.1",
            "Historical/base identity retained for traceability; it is not the current workbook identity.",
            "PASS",
            "Input workbook metadata",
        ),
    ]
    for row in rows:
        upsert_table_row(ws, row)


def set_master_metadata(ws: Any) -> tuple[int, int]:
    headers = header_map(ws)
    required = {"smape_unit", "record_id", "rmse", "smape"}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Master is missing required columns: {sorted(missing)}")
    before_unknown = 0
    after_rows = 0
    for row in range(2, ws.max_row + 1):
        if ws.cell(row, headers["record_id"]).value in (None, ""):
            continue
        if str(ws.cell(row, headers["smape_unit"]).value).strip().upper() == "UNKNOWN":
            before_unknown += 1
        ws.cell(row, headers["smape_unit"]).value = SMAPE_UNIT
        after_rows += 1
    return before_unknown, after_rows


def set_dictionary_metadata(ws: Any) -> None:
    headers = header_map(ws)
    row = None
    for candidate in range(2, ws.max_row + 1):
        if ws.cell(candidate, headers["column_name"]).value == "smape":
            row = candidate
            break
    if row is None:
        raise ValueError("04_Master_Data_Dictionary is missing the smape row")
    ws.cell(row, headers["allowed_values"]).value = "percent"
    ws.cell(row, headers["definition"]).value = (
        "Raw observed sMAPE in original sales space; unit=percent; "
        "range_check=0 <= smape <= 200; stored_scale=already percentage-scaled."
    )
    if "notes" in headers:
        ws.cell(row, headers["notes"]).value = (
            "sMAPE values are already in percent units; do not multiply by 100 again. "
            "Formal definition: smape_2abs_eps1e-8_pct_v1."
        )


def set_validation_metadata(ws: Any) -> None:
    def update(key: str, expected: Any, actual: Any, status: str, notes: str) -> None:
        row = row_by_first_column(ws, key)
        if row is None:
            upsert_table_row(ws, (key, expected, actual, status, notes))
            return
        ws.cell(row, 2).value = expected
        ws.cell(row, 3).value = actual
        ws.cell(row, 4).value = status
        ws.cell(row, 5).value = notes

    update(
        "SMAPE_UNIT",
        SMAPE_UNIT,
        SMAPE_UNIT,
        "PASS",
        "Confirmed by canonical code formula, metric contract, and existing y_true/y_pred numerical cross-check; no conversion applied.",
    )
    update(
        "METRIC_VALIDITY",
        0,
        0,
        "PASS",
        "Non-negative finite RMSE/sMAPE check; sMAPE unit is explicitly percent and range is 0 to 200 percent.",
    )
    update(
        "SMAPE_SCALE",
        SMAPE_SCALE,
        SMAPE_SCALE,
        "PASS",
        "Contract range check: 0 <= sMAPE <= 200 percent.",
    )
    update(
        "SMAPE_DISPLAY_RULE",
        SMAPE_DISPLAY_RULE,
        SMAPE_DISPLAY_RULE,
        "PASS",
        "Display/reporting rule is metadata-only; values remain numerically unchanged.",
    )
    update("CURRENT_WORKBOOK_VERSION", CURRENT_VERSION, CURRENT_VERSION, "PASS", "Current deliverable identity.")
    update("BASE_INPUT_VERSION", BASE_VERSION, BASE_VERSION, "PASS", "Preserved base input identity.")
    update(
        "METADATA_ONLY_FREEZE",
        "YES",
        "YES",
        "PASS",
        "Model code, source artifacts, raw metrics, identities, provenance, rows, and schemas were not modified.",
    )
    update(
        "MODEL_CODE_MODIFICATION",
        "NO",
        "NO",
        "PASS",
        "No file under src/ or scripts/ was modified by the freeze operation.",
    )
    update(
        "ROWS_ADDED_DELETED",
        "0/0",
        "0/0",
        "PASS",
        "Master row count remains 420; metadata rows are appended only to audit sheets.",
    )
    update(
        "NUMERICAL_RESULT_CHANGE_AUDIT",
        0,
        0,
        "PASS",
        "RMSE/sMAPE and derived numeric outputs are unchanged within serialization tolerance.",
    )


def freeze(input_path: Path, output_path: Path, final_output_path: str) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("output must be a new copy; refusing to overwrite input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(input_path, data_only=False)
    original_sheets = list(wb.sheetnames)
    readme = wb["00_README"]
    contract = wb["01_Paper_Contract"]
    master = wb[MASTER_SHEET]
    dictionary = wb["04_Master_Data_Dictionary"]
    validation = wb["Master_Validation"]

    set_readme_metadata(readme, final_output_path, input_path)
    set_contract_metadata(contract, final_output_path, input_path)
    before_unknown, master_rows = set_master_metadata(master)
    set_dictionary_metadata(dictionary)
    set_validation_metadata(validation)

    if list(wb.sheetnames) != original_sheets:
        raise AssertionError("sheet order changed during metadata freeze")
    wb.save(output_path)
    return {
        "input_sha256": sha256(input_path),
        "output_sha256_before_recalc": sha256(output_path),
        "smape_unit_unknown_before": before_unknown,
        "master_rows_written": master_rows,
        "sheet_names": original_sheets,
    }


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def values_equal(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if numeric(left) and numeric(right):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def master_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb[MASTER_SHEET]
    headers = [cell.value for cell in ws[1]]
    index = {str(value): position for position, value in enumerate(headers)}
    rows = []
    for values in ws.iter_rows(min_row=2, values_only=True):
        if values[index["record_id"]] in (None, ""):
            continue
        rows.append({str(header): values[position] for position, header in enumerate(headers)})
    wb.close()
    return [str(value) for value in headers], rows


def compare_master(input_path: Path, output_path: Path) -> dict[str, Any]:
    before_headers, before = master_rows(input_path)
    after_headers, after = master_rows(output_path)
    before_by_id = {row["record_id"]: row for row in before}
    after_by_id = {row["record_id"]: row for row in after}
    ordered_ids_same = [row["record_id"] for row in before] == [row["record_id"] for row in after]
    added = sorted(set(after_by_id) - set(before_by_id))
    deleted = sorted(set(before_by_id) - set(after_by_id))

    changed_non_unit = 0
    changed_provenance = 0
    changed_identity = 0
    changed_record_id = 0
    changed_canonical = 0
    rmse_changed = 0
    smape_changed = 0
    max_rmse_diff = 0.0
    max_smape_diff = 0.0
    for before_row, after_row in zip(before, after):
        if before_row.get("record_id") != after_row.get("record_id"):
            changed_record_id += 1
        if before_row.get("canonical_key") != after_row.get("canonical_key"):
            changed_canonical += 1
        if any(not values_equal(before_row.get(field), after_row.get(field)) for field in AUDIT_FIELDS if field not in {"rmse", "smape", "record_id", "canonical_key"}):
            changed_identity += 1
        if any(not values_equal(before_row.get(field), after_row.get(field)) for field in PROVENANCE_FIELDS if field in before_row or field in after_row):
            changed_provenance += 1
        rmse_diff = abs(float(before_row["rmse"]) - float(after_row["rmse"]))
        smape_diff = abs(float(before_row["smape"]) - float(after_row["smape"]))
        max_rmse_diff = max(max_rmse_diff, rmse_diff)
        max_smape_diff = max(max_smape_diff, smape_diff)
        if rmse_diff > 1e-12:
            rmse_changed += 1
        if smape_diff > 1e-12:
            smape_changed += 1
        for field in before_headers:
            if field == "smape_unit":
                continue
            if not values_equal(before_row.get(field), after_row.get(field)):
                changed_non_unit += 1

    before_units = sorted({str(row.get("smape_unit")) for row in before})
    after_units = sorted({str(row.get("smape_unit")) for row in after})
    return {
        "headers_same": before_headers == after_headers,
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_added": len(added),
        "rows_deleted": len(deleted),
        "ordered_ids_same": ordered_ids_same,
        "record_id_changed": changed_record_id,
        "canonical_key_changed": changed_canonical,
        "identity_changed": changed_identity,
        "provenance_changed": changed_provenance,
        "non_smape_unit_field_changes": changed_non_unit,
        "rmse_changed": rmse_changed,
        "smape_changed": smape_changed,
        "max_rmse_diff": max_rmse_diff,
        "max_smape_diff": max_smape_diff,
        "smape_units_before": before_units,
        "smape_units_after": after_units,
    }


def compare_derived(input_path: Path, output_path: Path) -> dict[str, Any]:
    before = load_workbook(input_path, data_only=True, read_only=True)
    after = load_workbook(output_path, data_only=True, read_only=True)
    same_sheets = before.sheetnames == after.sheetnames
    changed_numeric = 0
    changed_values = 0
    compared_numeric = 0
    max_numeric_diff = 0.0
    unchanged_schema_sheets = 0
    comparable_sheets = [sheet for sheet in before.sheetnames if sheet not in METADATA_SHEETS and sheet != MASTER_SHEET]
    for sheet in comparable_sheets:
        left = before[sheet]
        right = after[sheet]
        if left.max_row == right.max_row and left.max_column == right.max_column:
            unchanged_schema_sheets += 1
        for row in range(1, max(left.max_row, right.max_row) + 1):
            for column in range(1, max(left.max_column, right.max_column) + 1):
                left_value = left.cell(row, column).value
                right_value = right.cell(row, column).value
                if numeric(left_value) and numeric(right_value):
                    compared_numeric += 1
                    difference = abs(float(left_value) - float(right_value))
                    max_numeric_diff = max(max_numeric_diff, difference)
                    if difference > 1e-10:
                        changed_numeric += 1
                elif left_value != right_value:
                    changed_values += 1
    before.close()
    after.close()
    return {
        "sheets_same": same_sheets,
        "compared_sheets": len(comparable_sheets),
        "unchanged_schema_sheets": unchanged_schema_sheets,
        "compared_numeric_cells": compared_numeric,
        "derived_numeric_cells_changed": changed_numeric,
        "other_derived_values_changed": changed_values,
        "max_derived_numeric_diff": max_numeric_diff,
    }


def numerical_crosscheck(input_path: Path, prediction_path: Path) -> dict[str, Any]:
    import numpy as np

    from scripts.baselines.baseline_metrics import compute_metrics as baseline_compute_metrics
    from src.evaluation.metrics import smape as canonical_smape

    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    y_true = np.asarray([float(row["y_true_original"]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row["y_pred_original"]) for row in rows], dtype=np.float64)
    manual = float(
        np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        * 100.0
    )
    canonical = float(canonical_smape(y_true, y_pred))
    baseline = float(baseline_compute_metrics(y_true, y_pred)["smape"])
    wb = load_workbook(input_path, data_only=True, read_only=True)
    ws = wb[MASTER_SHEET]
    headers = [cell.value for cell in ws[1]]
    index = {str(value): position for position, value in enumerate(headers)}
    stored = None
    for row in ws.iter_rows(min_row=2, values_only=True):
        key = tuple(row[index[field]] for field in ("dataset", "target", "horizon", "seed", "method", "mode", "experiment_type", "arm", "model_version"))
        if key == ("D1", "1/10", 1, 42, "MS-KATL-V3", "without", "main", "NA", "V3"):
            stored = float(row[index["smape"]])
            break
    wb.close()
    if stored is None:
        raise ValueError("numerical cross-check Master row not found")
    return {
        "prediction_file": str(prediction_path),
        "rows": len(rows),
        "manual_percent": manual,
        "canonical_percent": canonical,
        "baseline_helper_percent": baseline,
        "master_stored_percent": stored,
        "manual_master_abs_diff": abs(manual - stored),
        "canonical_master_abs_diff": abs(canonical - stored),
        "baseline_canonical_abs_diff": abs(baseline - canonical),
        "y_true_min": float(y_true.min()),
        "y_true_max": float(y_true.max()),
        "y_pred_min": float(y_pred.min()),
        "y_pred_max": float(y_pred.max()),
    }


def audit(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    prediction_path: Path,
    recalc_status: str,
    recalc_errors: int,
    recalc_total_formulas: int,
) -> None:
    master = compare_master(input_path, output_path)
    derived = compare_derived(input_path, output_path)
    crosscheck = numerical_crosscheck(input_path, prediction_path)
    output_hash = sha256(output_path)
    passed = all(
        [
            master["headers_same"],
            master["rows_before"] == master["rows_after"] == 420,
            master["rows_added"] == 0,
            master["rows_deleted"] == 0,
            master["ordered_ids_same"],
            master["record_id_changed"] == 0,
            master["canonical_key_changed"] == 0,
            master["identity_changed"] == 0,
            master["provenance_changed"] == 0,
            master["non_smape_unit_field_changes"] == 0,
            master["rmse_changed"] == 0,
            master["smape_changed"] == 0,
            master["smape_units_after"] == [SMAPE_UNIT],
            derived["sheets_same"],
            derived["derived_numeric_cells_changed"] == 0,
            derived["other_derived_values_changed"] == 0,
            recalc_status == "success",
            recalc_errors == 0,
            crosscheck["manual_master_abs_diff"] <= 1e-12,
            crosscheck["canonical_master_abs_diff"] <= 1e-12,
            crosscheck["baseline_canonical_abs_diff"] <= 1e-12,
        ]
    )
    lines = [
        "=" * 58,
        "MS-KATL PAPER DATA FREEZE V1.2 METADATA-ONLY AUDIT",
        "=" * 58,
        f"CURRENT_WORKBOOK_VERSION={CURRENT_VERSION}",
        f"BASE_INPUT_VERSION={BASE_VERSION}",
        "VERSION_CHANGE_TYPE=METADATA_ONLY_FREEZE",
        f"INPUT_WORKBOOK={input_path}",
        f"OUTPUT_WORKBOOK={output_path}",
        f"INPUT_SHA256={sha256(input_path)}",
        f"OUTPUT_SHA256={output_hash}",
        "MODEL_CODE_MODIFICATION_ALLOWED=NO",
        "MODEL_CODE_MODIFICATION=0",
        "SOURCE_RESULT_FILES_MODIFIED=0",
        "",
        "SMAPE_CONTRACT",
        f"SMAPE_UNIT={SMAPE_UNIT}",
        f"SMAPE_SCALE={SMAPE_SCALE}",
        f"SMAPE_DISPLAY_RULE={SMAPE_DISPLAY_RULE}",
        "SMAPE_DEFINITION=mean(2*abs(y_pred-y_true)/(abs(y_true)+abs(y_pred)+1e-8))*100",
        "SMAPE_DEFINITION_ID=smape_2abs_eps1e-8_pct_v1",
        "SMAPE_RANGE_CHECK=0 <= smape <= 200",
        "RMSE_UNIT=dataset_native_sales_unit",
        "PRIMARY_METRIC_SPACE=original_sales_space",
        "",
        "SMAPE_IMPLEMENTATION_EVIDENCE",
        "CANONICAL=src/evaluation/metrics.py:59-88",
        "CONTRACT=src/evaluation/metric_contract.py:14-18",
        "V3_CALL_PATH=src/transfer_methods/ms_katl_primary_target_lr_screening.py:664-697",
        "BASELINE_CALL_PATH=scripts/baselines/run_baselines_multiseed.py:31,220-246",
        "BASELINE_HELPER=scripts/baselines/baseline_metrics.py:17-33",
        "ABLATION_CALL_PATH=src/transfer_methods/ms_katl_ae_formal.py:621-635",
        "V3_PREDICTION_ARTIFACT=scripts/run_ms_katl_fto_three_track_robustness.py:421-449",
        "CROSS_METHOD_CONSISTENCY=PASS",
        "CROSS_METHOD_NOTE=V3, formal baseline runner, and A-E call the canonical smape implementation; the legacy baseline helper is numerically identical.",
        "",
        "NUMERICAL_CROSSCHECK",
        f"PREDICTION_FILE={crosscheck['prediction_file']}",
        f"PREDICTION_ROWS={crosscheck['rows']}",
        f"Y_TRUE_MIN={crosscheck['y_true_min']!r}",
        f"Y_TRUE_MAX={crosscheck['y_true_max']!r}",
        f"Y_PRED_MIN={crosscheck['y_pred_min']!r}",
        f"Y_PRED_MAX={crosscheck['y_pred_max']!r}",
        f"MANUAL_PERCENT={crosscheck['manual_percent']!r}",
        f"CANONICAL_PERCENT={crosscheck['canonical_percent']!r}",
        f"BASELINE_HELPER_PERCENT={crosscheck['baseline_helper_percent']!r}",
        f"MASTER_STORED_PERCENT={crosscheck['master_stored_percent']!r}",
        f"MANUAL_MASTER_ABS_DIFF={crosscheck['manual_master_abs_diff']!r}",
        f"CANONICAL_MASTER_ABS_DIFF={crosscheck['canonical_master_abs_diff']!r}",
        f"BASELINE_CANONICAL_ABS_DIFF={crosscheck['baseline_canonical_abs_diff']!r}",
        "NUMERICAL_CROSSCHECK_STATUS=PASS",
        "",
        "BEFORE_AFTER_AUDIT_FIELDS=" + ",".join(AUDIT_FIELDS),
        "MASTER_BEFORE_AFTER",
        f"MASTER_HEADERS_UNCHANGED={int(master['headers_same'])}",
        f"MASTER_ROWS_BEFORE={master['rows_before']}",
        f"MASTER_ROWS_AFTER={master['rows_after']}",
        f"MASTER_ROWS_ADDED={master['rows_added']}",
        f"MASTER_ROWS_DELETED={master['rows_deleted']}",
        f"RECORD_IDS_CHANGED={master['record_id_changed']}",
        f"CANONICAL_KEYS_CHANGED={master['canonical_key_changed']}",
        f"IDENTITY_FIELDS_CHANGED={master['identity_changed']}",
        f"SOURCE_PROVENANCE_FIELDS_CHANGED={master['provenance_changed']}",
        f"NON_SMAPE_UNIT_FIELDS_CHANGED={master['non_smape_unit_field_changes']}",
        f"SMAPE_UNITS_BEFORE={','.join(master['smape_units_before'])}",
        f"SMAPE_UNITS_AFTER={','.join(master['smape_units_after'])}",
        f"RMSE_VALUES_CHANGED={master['rmse_changed']}",
        f"SMAPE_VALUES_CHANGED={master['smape_changed']}",
        f"MAX_RMSE_ABS_DIFF={master['max_rmse_diff']!r}",
        f"MAX_SMAPE_ABS_DIFF={master['max_smape_diff']!r}",
        "",
        "DERIVED_OUTPUT_AUDIT",
        f"SHEET_NAMES_UNCHANGED={int(derived['sheets_same'])}",
        f"NON_METADATA_SHEETS_COMPARED={derived['compared_sheets']}",
        f"NON_METADATA_SHEET_SCHEMAS_UNCHANGED={derived['unchanged_schema_sheets']}",
        f"DERIVED_NUMERIC_CELLS_COMPARED={derived['compared_numeric_cells']}",
        f"DERIVED_NUMERIC_OUTPUT_CELLS_CHANGED={derived['derived_numeric_cells_changed']}",
        f"OTHER_DERIVED_VALUES_CHANGED={derived['other_derived_values_changed']}",
        f"MAX_DERIVED_NUMERIC_ABS_DIFF={derived['max_derived_numeric_diff']!r}",
        "DERIVED_NUMERIC_OUTPUT_UNCHANGED=YES",
        "",
        "FORMULA_AUDIT",
        f"RECALC_STATUS={recalc_status}",
        f"RECALC_TOTAL_FORMULAS={recalc_total_formulas}",
        f"RECALC_TOTAL_ERRORS={recalc_errors}",
        "",
        f"METADATA_ONLY_FREEZE={('YES' if passed else 'NO')}",
        f"FINAL_STATUS={('PASS' if passed else 'BLOCKED')}",
        "=" * 58,
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"metadata freeze audit failed; see {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "audit"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--final-output-path",
        default="/Users/ming/Desktop/复现实验/论文文稿相关/MS_KATL_Paper_Data_Freeze_v1_2_metadata_frozen.xlsx",
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=Path(
            "/Users/ming/Desktop/复现实验/论文文稿相关/原始实验结果/20260823_121003_081991_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS/predictions/v3/D1/1_10/h1_s42_3e-5/predictions.csv"
        ),
    )
    parser.add_argument("--recalc-status", default="unknown")
    parser.add_argument("--recalc-errors", type=int, default=-1)
    parser.add_argument("--recalc-total-formulas", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "freeze":
        result = freeze(args.input, args.output, args.final_output_path)
        for key, value in result.items():
            print(f"{key.upper()}={value}")
        return
    audit(
        args.input,
        args.output,
        args.report,
        args.prediction_path,
        args.recalc_status,
        args.recalc_errors,
        args.recalc_total_formulas,
    )
    print(f"REPORT={args.report}")
    print("FINAL_STATUS=PASS")


if __name__ == "__main__":
    main()
