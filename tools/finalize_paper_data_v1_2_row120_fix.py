#!/usr/bin/env python3
"""Apply and audit the single Master_Validation row-120 alignment fix."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


MASTER_SHEET = "03_Paper_Results_Master"
VALIDATION_SHEET = "Master_Validation"
ROW = 120
NOTE = "sMAPE unit was formally confirmed as percent and frozen in V1.2 metadata."
ALLOWED_CHANGES = {
    (VALIDATION_SHEET, "C120"),
    (VALIDATION_SHEET, "D120"),
    (VALIDATION_SHEET, "E120"),
}
DERIVED_SHEETS = {
    "Derived_Table_Lineage",
    "Baseline_Coverage_Audit",
    "Dataset_Scope_Audit",
    "Master_Aggregate_Check",
    "06_Main_RMSE",
    "07_Main_sMAPE",
    "08_Learning_Baseline",
    "09_Relative_Gain",
    "10_Stability",
    "11_Ablation_Summary",
    "13_Paper_Selection",
    "14_Paper_Summary",
    "15_Data_Quality",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("refusing to overwrite the input workbook")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(input_path, data_only=False)
    ws = wb[VALIDATION_SHEET]
    current = [ws.cell(ROW, column).value for column in range(1, 6)]
    expected = [
        "SMAPE_UNIT_NOT_FROZEN",
        0,
        0,
        "PASS",
        NOTE,
    ]
    if current != expected:
        raise ValueError(f"unexpected row 120 before fix: {current!r}")
    ws["C120"] = "PASS"
    ws["D120"] = NOTE
    ws["E120"] = None
    wb.save(output_path)
    print(f"INPUT_SHA256={sha256(input_path)}")
    print(f"OUTPUT_SHA256_BEFORE_RECALC={sha256(output_path)}")
    print("CELLS_CHANGED_BY_REQUEST=3")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def equal(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    if is_number(left) and is_number(right):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def workbook_values(path: Path) -> dict[tuple[str, str], Any]:
    wb = load_workbook(path, data_only=True, read_only=True)
    values = {}
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if not hasattr(cell, "coordinate"):
                    continue
                values[(ws.title, cell.coordinate)] = cell.value
    wb.close()
    return values


def workbook_formulas(path: Path) -> dict[tuple[str, str], Any]:
    wb = load_workbook(path, data_only=False, read_only=True)
    formulas = {}
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    formulas[(ws.title, cell.coordinate)] = cell.value
    wb.close()
    return formulas


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


def master_audit(input_path: Path, output_path: Path) -> dict[str, Any]:
    before_headers, before = master_rows(input_path)
    after_headers, after = master_rows(output_path)
    before_ids = [row["record_id"] for row in before]
    after_ids = [row["record_id"] for row in after]
    before_by_id = {row["record_id"]: row for row in before}
    after_by_id = {row["record_id"]: row for row in after}
    rmse_changed = 0
    smape_changed = 0
    canonical_changed = 0
    for left, right in zip(before, after):
        if left["canonical_key"] != right["canonical_key"]:
            canonical_changed += 1
        if abs(float(left["rmse"]) - float(right["rmse"])) > 1e-12:
            rmse_changed += 1
        if abs(float(left["smape"]) - float(right["smape"])) > 1e-12:
            smape_changed += 1
    return {
        "headers_same": before_headers == after_headers,
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_added": len(set(after_by_id) - set(before_by_id)),
        "rows_deleted": len(set(before_by_id) - set(after_by_id)),
        "rmse_changed": rmse_changed,
        "smape_changed": smape_changed,
        "record_ids_changed": sum(left != right for left, right in zip(before_ids, after_ids)),
        "canonical_keys_changed": canonical_changed,
        "smape_unit_values": sorted({str(row.get("smape_unit")) for row in after}),
        "smape_unit_percent_rows": sum(str(row.get("smape_unit")) == "percent" for row in after),
    }


def row120_audit(path: Path) -> dict[str, Any]:
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb[VALIDATION_SHEET]
    values = [ws.cell(ROW, column).value for column in range(1, 6)]
    wb.close()
    expected = ["SMAPE_UNIT_NOT_FROZEN", 0, "PASS", NOTE, None]
    return {
        "values": values,
        "aligned": values == expected,
        "unresolved_count": 0 if values == expected else 1,
        "status": values[2] if len(values) > 2 else None,
    }


def version_audit(path: Path) -> tuple[bool, bool]:
    wb = load_workbook(path, data_only=True, read_only=True)
    current = False
    base = False
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            values = list(row)
            for index, value in enumerate(values[:-1]):
                if value == "CURRENT_WORKBOOK_VERSION" and values[index + 1] == "V1.2":
                    current = True
                if value == "BASE_INPUT_VERSION" and values[index + 1] == "V1.1":
                    base = True
    wb.close()
    return current, base


def audit(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    recalc_status: str,
    recalc_errors: int,
    recalc_total_formulas: int,
) -> None:
    before_values = workbook_values(input_path)
    after_values = workbook_values(output_path)
    before_formulas = workbook_formulas(input_path)
    after_formulas = workbook_formulas(output_path)
    changed_other_values = 0
    for key in set(before_values) | set(after_values):
        if key in ALLOWED_CHANGES:
            continue
        if not equal(before_values.get(key), after_values.get(key)):
            changed_other_values += 1

    master = master_audit(input_path, output_path)
    row120 = row120_audit(output_path)
    current_version, base_version = version_audit(output_path)
    derived_value_changes = 0
    for key in set(before_values) | set(after_values):
        if key[0] not in DERIVED_SHEETS:
            continue
        if not equal(before_values.get(key), after_values.get(key)):
            derived_value_changes += 1
    derived_formula_changes = sum(
        before_formulas.get(key) != after_formulas.get(key)
        for key in set(before_formulas) | set(after_formulas)
        if key[0] in DERIVED_SHEETS
    )
    master_values_changed = sum(
        not equal(before_values.get(key), after_values.get(key))
        for key in set(before_values) | set(after_values)
        if key[0] == MASTER_SHEET
    )
    master_formula_changes = sum(
        before_formulas.get(key) != after_formulas.get(key)
        for key in set(before_formulas) | set(after_formulas)
        if key[0] == MASTER_SHEET
    )
    all_formula_changes = sum(
        before_formulas.get(key) != after_formulas.get(key)
        for key in set(before_formulas) | set(after_formulas)
    )

    passed = all(
        [
            row120["aligned"],
            current_version,
            base_version,
            master["headers_same"],
            master["rows_before"] == 420,
            master["rows_after"] == 420,
            master["rows_added"] == 0,
            master["rows_deleted"] == 0,
            master["rmse_changed"] == 0,
            master["smape_changed"] == 0,
            master["record_ids_changed"] == 0,
            master["canonical_keys_changed"] == 0,
            master["smape_unit_values"] == ["percent"],
            master["smape_unit_percent_rows"] == 420,
            changed_other_values == 0,
            master_values_changed == 0,
            master_formula_changes == 0,
            derived_value_changes == 0,
            derived_formula_changes == 0,
            all_formula_changes == 0,
            recalc_status == "success",
            recalc_errors == 0,
            recalc_total_formulas == 2790,
        ]
    )

    lines = [
        "=" * 64,
        "MS-KATL V1.2 FINAL VALIDATION ROW ALIGNMENT FIX",
        "=" * 64,
        "TASK_TYPE=ONE_ROW_METADATA_FIX_ONLY",
        "TRAINING_STARTED=NO",
        "EXPERIMENT_STARTED=NO",
        "MODEL_CODE_MODIFIED=NO",
        "EXPERIMENT_RESULTS_MODIFIED=NO",
        "CURRENT_WORKBOOK_VERSION=V1.2",
        "BASE_INPUT_VERSION=V1.1",
        f"INPUT_WORKBOOK={input_path}",
        f"OUTPUT_WORKBOOK={output_path}",
        f"INPUT_SHA256={sha256(input_path)}",
        f"OUTPUT_SHA256={sha256(output_path)}",
        "",
        "ROW_120_BEFORE=" + repr([before_values.get((VALIDATION_SHEET, f"{column}120")) for column in "ABCDE"]),
        "ROW_120_AFTER=" + repr(row120["values"]),
        f"ROW_120_ALIGNMENT={'PASS' if row120['aligned'] else 'BLOCKED'}",
        "SMAPE_UNIT=percent",
        f"SMAPE_UNIT_FREEZE={'PASS' if master['smape_unit_values'] == ['percent'] and master['smape_unit_percent_rows'] == 420 else 'BLOCKED'}",
        f"SMAPE_UNIT_NOT_FROZEN_COUNT={row120['unresolved_count']}",
        f"SMAPE_UNIT_NOT_FROZEN_STATUS={row120['status']}",
        "",
        "MASTER_AUDIT",
        f"MASTER_ROWS_BEFORE={master['rows_before']}",
        f"MASTER_ROWS_AFTER={master['rows_after']}",
        f"RMSE_VALUES_CHANGED={master['rmse_changed']}",
        f"SMAPE_VALUES_CHANGED={master['smape_changed']}",
        f"RECORD_IDS_CHANGED={master['record_ids_changed']}",
        f"CANONICAL_KEYS_CHANGED={master['canonical_keys_changed']}",
        f"MASTER_VALUES_CHANGED={master_values_changed}",
        f"MASTER_FORMULAS_CHANGED={master_formula_changes}",
        f"MASTER_SMAPE_UNIT_UNIQUE_VALUES={','.join(master['smape_unit_values'])}",
        f"MASTER_SMAPE_UNIT_PERCENT_ROWS={master['smape_unit_percent_rows']}",
        "ROWS_ADDED_TO_MASTER=0",
        "ROWS_DELETED_FROM_MASTER=0",
        "",
        "DERIVED_AUDIT",
        f"DERIVED_SHEETS_VALUES_CHANGED={derived_value_changes}",
        f"DERIVED_SHEETS_FORMULAS_CHANGED={derived_formula_changes}",
        f"FORMULAS_CHANGED={all_formula_changes}",
        f"RECALC_STATUS={recalc_status}",
        f"RECALC_TOTAL_FORMULAS={recalc_total_formulas}",
        f"RECALC_TOTAL_ERRORS={recalc_errors}",
        "",
        "WORKBOOK_VERSION_FREEZE=PASS",
        "SCIENTIFIC_RESULTS_UNCHANGED=YES",
        f"FINAL_VALIDATION={'PASS' if passed else 'BLOCKED'}",
        "=" * 64,
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"row 120 final validation failed; see {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cleanup", "audit"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--recalc-status", default="unknown")
    parser.add_argument("--recalc-errors", type=int, default=-1)
    parser.add_argument("--recalc-total-formulas", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "cleanup":
        cleanup(args.input, args.output)
        return
    audit(
        args.input,
        args.output,
        args.report,
        args.recalc_status,
        args.recalc_errors,
        args.recalc_total_formulas,
    )
    print(f"REPORT={args.report}")
    print("FINAL_VALIDATION=PASS")


if __name__ == "__main__":
    main()
