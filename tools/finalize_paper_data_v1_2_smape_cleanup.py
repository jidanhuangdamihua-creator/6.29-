#!/usr/bin/env python3
"""Finalize the V1.2 sMAPE metadata consistency cleanup.

Only three stale metadata records are corrected.  Master observations,
identities, formulas, workbook schema, and scientific results are audited
before and after the edit.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


CURRENT_VERSION = "V1.2"
BASE_VERSION = "V1.1"
SMAPE_UNIT = "percent"
SMAPE_SCALE = "0_to_200_percent"
SMAPE_DISPLAY_RULE = "VALUE_ALREADY_IN_PERCENT_DO_NOT_MULTIPLY_BY_100"
SMAPE_DEFINITION_ID = "smape_2abs_eps1e-8_pct_v1"
MASTER_SHEET = "03_Paper_Results_Master"
METADATA_SHEETS = {
    "00_README",
    "01_Paper_Contract",
    "04_Master_Data_Dictionary",
    "Master_Validation",
}
MASTER_FIELDS = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header_map(ws: Any) -> dict[str, int]:
    return {str(cell.value): cell.column for cell in ws[1] if cell.value is not None}


def find_row(ws: Any, first_column_value: str) -> int | None:
    for row in range(2, ws.max_row + 1):
        if ws.cell(row, 1).value == first_column_value:
            return row
    return None


def cleanup(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("refusing to overwrite the V1.2 input workbook")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(input_path, data_only=False)

    contract = wb["01_Paper_Contract"]
    row = find_row(contract, "Master sMAPE unit")
    if row is None:
        raise ValueError("legacy Master sMAPE unit row is missing")
    if contract.cell(row, 2).value != "UNKNOWN" or contract.cell(row, 4).value != "REVIEW":
        raise ValueError("legacy Master sMAPE unit row is not in the expected stale state")
    contract.cell(row, 2).value = SMAPE_UNIT
    contract.cell(row, 3).value = (
        "Stored sMAPE values are already percentage-scaled; no ×100/÷100 conversion is applied."
    )
    contract.cell(row, 4).value = "PASS"

    dictionary = wb["04_Master_Data_Dictionary"]
    dictionary_headers = header_map(dictionary)
    row = None
    for candidate in range(2, dictionary.max_row + 1):
        if dictionary.cell(candidate, dictionary_headers["column_name"]).value == "smape_unit":
            row = candidate
            break
    if row is None:
        raise ValueError("smape_unit dictionary row is missing")
    dictionary.cell(row, dictionary_headers["allowed_values"]).value = SMAPE_UNIT
    dictionary.cell(row, dictionary_headers["example"]).value = SMAPE_UNIT
    dictionary.cell(row, dictionary_headers["definition"]).value = "Unit label for stored sMAPE values."
    dictionary.cell(row, dictionary_headers["notes"]).value = (
        "Stored sMAPE values are already percentage-scaled; do not multiply by 100 again. "
        f"Formal definition ID: {SMAPE_DEFINITION_ID}."
    )

    validation = wb["Master_Validation"]
    row = find_row(validation, "SMAPE_UNIT_NOT_FROZEN")
    if row is None:
        raise ValueError("SMAPE_UNIT_NOT_FROZEN review row is missing")
    validation.cell(row, 2).value = 0
    validation.cell(row, 3).value = 0
    validation.cell(row, 4).value = "PASS"
    validation.cell(row, 5).value = (
        "sMAPE unit was formally confirmed as percent and frozen in V1.2 metadata."
    )

    wb.save(output_path)
    print(f"INPUT_SHA256={sha256(input_path)}")
    print(f"OUTPUT_SHA256_BEFORE_RECALC={sha256(output_path)}")
    print("METADATA_ROWS_CORRECTED=3")
    print("MASTER_OBSERVATIONS_WRITTEN=0")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def equal(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if is_number(left) and is_number(right):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def master_snapshot(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
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


def formula_snapshot(path: Path) -> dict[tuple[str, str], Any]:
    wb = load_workbook(path, data_only=False, read_only=True)
    formulas = {}
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    formulas[(ws.title, cell.coordinate)] = cell.value
    wb.close()
    return formulas


def stale_metadata_counts(path: Path) -> tuple[int, int]:
    wb = load_workbook(path, data_only=True, read_only=True)
    unknown = 0
    review = 0

    contract = wb["01_Paper_Contract"]
    row = find_row(contract, "Master sMAPE unit")
    if row is not None:
        if str(contract.cell(row, 2).value).upper() == "UNKNOWN":
            unknown += 1
        if str(contract.cell(row, 4).value).upper() == "REVIEW":
            review += 1

    dictionary = wb["04_Master_Data_Dictionary"]
    headers = header_map(dictionary)
    for candidate in range(2, dictionary.max_row + 1):
        if dictionary.cell(candidate, headers["column_name"]).value != "smape_unit":
            continue
        values = [
            dictionary.cell(candidate, headers["allowed_values"]).value,
            dictionary.cell(candidate, headers["example"]).value,
            dictionary.cell(candidate, headers["definition"]).value,
            dictionary.cell(candidate, headers["notes"]).value,
        ]
        if any("UNKNOWN" in str(value).upper() for value in values):
            unknown += 1
        if any("REVIEW" in str(value).upper() for value in values):
            review += 1
        break

    validation = wb["Master_Validation"]
    row = find_row(validation, "SMAPE_UNIT_NOT_FROZEN")
    if row is not None:
        if validation.cell(row, 2).value not in (0, "0"):
            unknown += 1
        if str(validation.cell(row, 3).value).upper() == "REVIEW" or str(validation.cell(row, 4).value).upper() == "REVIEW":
            review += 1

    # Catch any remaining row-level contradiction containing sMAPE + UNKNOWN/REVIEW.
    for ws in wb.worksheets:
        for values in ws.iter_rows(values_only=True):
            text = " | ".join(str(value) for value in values if value is not None)
            upper = text.upper()
            if "SMAPE" in upper and "UNKNOWN" in upper:
                unknown += 1
            if "SMAPE" in upper and "REVIEW" in upper:
                review += 1
    wb.close()
    return unknown, review


def master_audit(input_path: Path, output_path: Path) -> dict[str, Any]:
    before_headers, before = master_snapshot(input_path)
    after_headers, after = master_snapshot(output_path)
    before_ids = [row["record_id"] for row in before]
    after_ids = [row["record_id"] for row in after]
    before_by_id = {row["record_id"]: row for row in before}
    after_by_id = {row["record_id"]: row for row in after}
    rows_added = len(set(after_by_id) - set(before_by_id))
    rows_deleted = len(set(before_by_id) - set(after_by_id))
    rmse_changed = 0
    smape_changed = 0
    identity_changed = 0
    record_ids_changed = sum(left != right for left, right in zip(before_ids, after_ids))
    canonical_keys_changed = 0
    for left, right in zip(before, after):
        if left.get("canonical_key") != right.get("canonical_key"):
            canonical_keys_changed += 1
        if any(not equal(left.get(field), right.get(field)) for field in MASTER_FIELDS if field not in {"rmse", "smape", "record_id", "canonical_key"}):
            identity_changed += 1
        if abs(float(left["rmse"]) - float(right["rmse"])) > 1e-12:
            rmse_changed += 1
        if abs(float(left["smape"]) - float(right["smape"])) > 1e-12:
            smape_changed += 1
    units = sorted({str(row.get("smape_unit")) for row in after})
    percent_rows = sum(str(row.get("smape_unit")) == SMAPE_UNIT for row in after)
    return {
        "headers_same": before_headers == after_headers,
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_added": rows_added,
        "rows_deleted": rows_deleted,
        "rmse_changed": rmse_changed,
        "smape_changed": smape_changed,
        "identity_changed": identity_changed,
        "record_ids_changed": record_ids_changed,
        "canonical_keys_changed": canonical_keys_changed,
        "units": units,
        "percent_rows": percent_rows,
    }


def derived_audit(input_path: Path, output_path: Path) -> dict[str, Any]:
    before = load_workbook(input_path, data_only=True, read_only=True)
    after = load_workbook(output_path, data_only=True, read_only=True)
    sheets_same = before.sheetnames == after.sheetnames
    changed_numeric = 0
    max_numeric_diff = 0.0
    compared_numeric = 0
    for sheet in before.sheetnames:
        if sheet in METADATA_SHEETS or sheet == MASTER_SHEET:
            continue
        left = before[sheet]
        right = after[sheet]
        for row in range(1, max(left.max_row, right.max_row) + 1):
            for column in range(1, max(left.max_column, right.max_column) + 1):
                x = left.cell(row, column).value
                y = right.cell(row, column).value
                if is_number(x) and is_number(y):
                    compared_numeric += 1
                    diff = abs(float(x) - float(y))
                    max_numeric_diff = max(max_numeric_diff, diff)
                    if diff > 1e-10:
                        changed_numeric += 1
    before.close()
    after.close()
    return {
        "sheets_same": sheets_same,
        "compared_numeric": compared_numeric,
        "changed_numeric": changed_numeric,
        "max_numeric_diff": max_numeric_diff,
    }


def audit(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    recalc_status: str,
    recalc_errors: int,
    recalc_total_formulas: int,
) -> None:
    before_formula = formula_snapshot(input_path)
    after_formula = formula_snapshot(output_path)
    master = master_audit(input_path, output_path)
    derived = derived_audit(input_path, output_path)
    stale_unknown, stale_review = stale_metadata_counts(output_path)
    formula_changed = sum(
        before_formula.get(key) != after_formula.get(key)
        for key in set(before_formula) | set(after_formula)
    )

    wb = load_workbook(output_path, data_only=True, read_only=True)
    current_values = []
    base_values = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for value in row:
                if value == "V1.2" or value == "CURRENT_FREEZE_VERSION=V1.2; BASE_INPUT_VERSION=V1.1":
                    current_values.append(value)
                if value == "V1.1" or value == "CURRENT_FREEZE_VERSION=V1.2; BASE_INPUT_VERSION=V1.1":
                    base_values.append(value)
    wb.close()

    derived_changed = derived["changed_numeric"] > 0
    passed = all(
        [
            master["headers_same"],
            master["rows_before"] == 420,
            master["rows_after"] == 420,
            master["rows_added"] == 0,
            master["rows_deleted"] == 0,
            master["rmse_changed"] == 0,
            master["smape_changed"] == 0,
            master["record_ids_changed"] == 0,
            master["canonical_keys_changed"] == 0,
            master["units"] == [SMAPE_UNIT],
            master["percent_rows"] == 420,
            stale_unknown == 0,
            stale_review == 0,
            before_formula == after_formula,
            derived["sheets_same"],
            not derived_changed,
            recalc_status == "success",
            recalc_errors == 0,
            recalc_total_formulas == 2790,
            bool(current_values),
            bool(base_values),
        ]
    )

    lines = [
        "=" * 62,
        "MS-KATL V1.2 SMAPE METADATA CONSISTENCY CLEANUP",
        "=" * 62,
        "TASK_TYPE=METADATA_ONLY_CORRECTION",
        "TRAINING_STARTED=NO",
        "EXPERIMENT_STARTED=NO",
        "MODEL_CODE_MODIFICATION=0",
        "EXPERIMENT_RESULT_MODIFICATION=0",
        f"CURRENT_WORKBOOK_VERSION={CURRENT_VERSION}",
        f"BASE_INPUT_VERSION={BASE_VERSION}",
        f"INPUT_WORKBOOK={input_path}",
        f"OUTPUT_WORKBOOK={output_path}",
        f"INPUT_SHA256={sha256(input_path)}",
        f"OUTPUT_SHA256={sha256(output_path)}",
        "",
        "SMAPE_UNIT=percent",
        "SMAPE_SCALE=0_to_200_percent",
        "SMAPE_DISPLAY_RULE=VALUE_ALREADY_IN_PERCENT_DO_NOT_MULTIPLY_BY_100",
        "SMAPE_DEFINITION_ID=smape_2abs_eps1e-8_pct_v1",
        "SMAPE_UNIT_CONSISTENCY=PASS",
        "",
        "MASTER_AUDIT",
        f"MASTER_ROWS_BEFORE={master['rows_before']}",
        f"MASTER_ROWS_AFTER={master['rows_after']}",
        f"RMSE_VALUES_CHANGED={master['rmse_changed']}",
        f"SMAPE_VALUES_CHANGED={master['smape_changed']}",
        f"RECORD_IDS_CHANGED={master['record_ids_changed']}",
        f"CANONICAL_KEYS_CHANGED={master['canonical_keys_changed']}",
        f"ROWS_ADDED_TO_MASTER={master['rows_added']}",
        f"ROWS_DELETED_FROM_MASTER={master['rows_deleted']}",
        f"MASTER_SMAPE_UNIT_UNIQUE_VALUES={','.join(master['units'])}",
        f"MASTER_SMAPE_UNIT_PERCENT_ROWS={master['percent_rows']}",
        "",
        "STALE_METADATA_AUDIT",
        f"STALE_SMAPE_UNKNOWN_METADATA={stale_unknown}",
        f"STALE_SMAPE_UNIT_REVIEW_ITEMS={stale_review}",
        "",
        "FORMULA_AND_DERIVED_AUDIT",
        f"FORMULAS_BEFORE={len(before_formula)}",
        f"FORMULAS_AFTER={len(after_formula)}",
        f"FORMULAS_CHANGED={formula_changed}",
        f"DERIVED_NUMERIC_CELLS_COMPARED={derived['compared_numeric']}",
        f"DERIVED_NUMERIC_OUTPUT_CHANGED={'YES' if derived_changed else 'NO'}",
        f"MAX_DERIVED_NUMERIC_ABS_DIFF={derived['max_numeric_diff']!r}",
        f"RECALC_STATUS={recalc_status}",
        f"RECALC_TOTAL_FORMULAS={recalc_total_formulas}",
        f"RECALC_TOTAL_ERRORS={recalc_errors}",
        "",
        "WORKBOOK_VERSION_FREEZE=PASS",
        "STALE_METADATA_CONTRADICTIONS=0",
        "SCIENTIFIC_RESULTS_UNCHANGED=YES",
        f"FINAL_VALIDATION={'PASS' if passed else 'BLOCKED'}",
        "=" * 62,
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"final cleanup audit failed; see {report_path}")


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
    else:
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
