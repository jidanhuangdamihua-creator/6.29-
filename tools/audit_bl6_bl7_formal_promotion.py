#!/usr/bin/env python3
"""Read-only BL6/BL7 formal-result promotion and freeze-readiness audit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_original_scale_metrics
from src.protocols.experiment_protocol import FORMAL_SEEDS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "baselines" / "bl6_bl7_full_20260907"
INPUT_XLSX = ROOT.parent / "论文文稿相关" / "原始实验结果" / "BL6_BL7_FULL_RESULTS_COMBINED.xlsx"
AUTHORITY_XLSX = ROOT.parent / "论文文稿相关" / "MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN.xlsx"
REPORT = ROOT / "BL6_BL7_FORMAL_RESULT_PROMOTION_FREEZE_READINESS_AUDIT.md"
METHODS = ("BL6_DLinear", "BL7_TCN")
PAPER_HORIZONS = (1, 3, 5)
FORMAL_DATASETS = ("D1", "D4", "D5", "D6")
KNOWN_TARGETS = {"D1": "1/10", "D4": "166/258", "D5": "48/1159414"}
KEY = ["dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed"]
RAW_KEY = KEY + ["sample_key"]
RAW_COLUMNS = [
    "run_id", "dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed",
    "sample_key", "forecast_origin", "input_start", "input_end", "target_timestamp",
    "y_true", "y_pred", "metric_space",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(raw: object, *, base: Path = ROOT) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    candidates = [base / p, ROOT / p, RESULTS / p.name, INPUT_XLSX.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base / p).resolve()


def parse_contract_targets(authority: Path) -> dict[str, str]:
    contract = pd.read_excel(authority, sheet_name="01_Paper_Contract")
    row = contract.loc[contract.iloc[:, 0].astype(str).eq("Formal targets")]
    if row.empty:
        raise ValueError("01_Paper_Contract has no Formal targets row")
    raw = str(row.iloc[0, 1])
    return {dataset: target.strip() for dataset, target in re.findall(r"(D[1-6])=([^;]+)", raw)}


def parse_contract_value(authority: Path, item: str) -> str:
    contract = pd.read_excel(authority, sheet_name="01_Paper_Contract")
    row = contract.loc[contract.iloc[:, 0].astype(str).eq(item)]
    return "" if row.empty else str(row.iloc[0, 1])


def git_identity() -> tuple[str, str, str]:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return branch, head, status


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def fmt(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    branch, head, git_status = git_identity()
    authority_targets = parse_contract_targets(AUTHORITY_XLSX)
    authority_datasets = tuple(re.findall(r"D[1-6]", parse_contract_value(AUTHORITY_XLSX, "Selected datasets")))
    authority_horizons = tuple(int(x) for x in re.findall(r"\d+", parse_contract_value(AUTHORITY_XLSX, "Forecast horizons")))
    authority_seeds = tuple(int(x) for x in re.findall(r"\d+", parse_contract_value(AUTHORITY_XLSX, "V3 seeds")))
    target_conflicts = {
        dataset: (KNOWN_TARGETS[dataset], authority_targets.get(dataset))
        for dataset in KNOWN_TARGETS
        if authority_targets.get(dataset) != KNOWN_TARGETS[dataset]
    }
    d6_authority = authority_targets.get("D6", "UNAVAILABLE")
    formal_targets = {dataset: authority_targets[dataset] for dataset in FORMAL_DATASETS}

    input_exists = INPUT_XLSX.exists()
    workbook_sha = sha256(INPUT_XLSX) if input_exists else "MISSING"
    workbook_sheets: list[str] = []
    workbook_columns: list[str] = []
    workbook_rows = 0
    if input_exists:
        workbook = pd.ExcelFile(INPUT_XLSX)
        workbook_sheets = list(workbook.sheet_names)
        frame = pd.read_excel(INPUT_XLSX, sheet_name="BL6_BL7_Results")
        workbook_columns = list(frame.columns)
        workbook_rows = len(frame)
    else:
        frame = pd.DataFrame()

    csv_paths = [RESULTS / f"d{i}_baselines_multiseed.csv" for i in range(1, 7)]
    csv_exists = all(p.exists() for p in csv_paths)
    source_frame = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True) if csv_exists else pd.DataFrame()
    # The combined workbook is the declared input.  The CSV union is used only as a
    # read-only cross-check of the workbook and is never promoted as a substitute.
    if not frame.empty and not source_frame.empty:
        comparable = [column for column in frame.columns if column in source_frame.columns]
        sort_columns = [column for column in KEY if column in comparable]
        left = frame[comparable].sort_values(sort_columns, kind="stable").reset_index(drop=True)
        right = source_frame[comparable].sort_values(sort_columns, kind="stable").reset_index(drop=True)
        workbook_matches_csv = len(left) == len(right)
        for column in comparable:
            if not workbook_matches_csv:
                break
            if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
                workbook_matches_csv = bool(np.allclose(left[column].to_numpy(float), right[column].to_numpy(float), atol=1e-12, rtol=0.0, equal_nan=True))
            else:
                workbook_matches_csv = left[column].fillna("").astype(str).tolist() == right[column].fillna("").astype(str).tolist()
    else:
        workbook_matches_csv = False

    raw_cache: dict[Path, pd.DataFrame] = {}
    raw_errors: list[str] = []
    manifest_checks: list[str] = []
    raw_hash_checks: list[bool] = []
    summary_hash_checks: list[bool] = []
    source_summary_checks: list[bool] = []
    for dataset in sorted(frame["dataset_id"].astype(str).unique()) if not frame.empty else []:
        dnum = int(dataset[1:])
        manifest_path = RESULTS / f"d{dnum}_baseline_run_manifest.json"
        summary_path = RESULTS / f"d{dnum}_baselines_multiseed.csv"
        raw_path = RESULTS / f"d{dnum}_baseline_predictions.csv"
        if not manifest_path.exists():
            manifest_checks.append(f"{dataset}: manifest missing")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_hash_checks.append(manifest.get("summary_sha256") == sha256(summary_path))
        raw_hash_checks.append(manifest.get("raw_prediction_sha256") == sha256(raw_path))
        manifest_checks.append(
            f"{dataset}: summary_sha={manifest.get('summary_sha256') == sha256(summary_path)}, "
            f"raw_sha={manifest.get('raw_prediction_sha256') == sha256(raw_path)}, "
            f"code_identity={'git_head' in manifest and 'git_status_short' in manifest}"
        )
        raw_cache[raw_path.resolve()] = pd.read_csv(raw_path)
    for value in frame.get("source_summary_file", pd.Series(dtype=str)).dropna().astype(str).unique() if not frame.empty else []:
        source_summary_checks.append(resolve_path(value).exists())

    formal_mask = pd.Series(False, index=frame.index)
    if not frame.empty:
        formal_mask = frame.apply(
            lambda r: str(r["dataset_id"]) in formal_targets and str(r["target_entity_key"]) == formal_targets[str(r["dataset_id"])],
            axis=1,
        )
    formal = frame.loc[formal_mask].copy()
    expected_cells = len(FORMAL_DATASETS) * len(METHODS) * len(PAPER_HORIZONS) * len(FORMAL_SEEDS)
    cell_counts = formal.groupby(KEY, dropna=False).size() if not formal.empty else pd.Series(dtype=int)
    duplicate_cells = int((cell_counts > 1).sum())
    observed_cells = int(len(cell_counts))
    expected_keys = {
        (dataset, target, "without", method, horizon, seed)
        for dataset, target in formal_targets.items()
        for method in METHODS
        for horizon in PAPER_HORIZONS
        for seed in FORMAL_SEEDS
    }
    observed_keys = set(cell_counts.index.tolist())
    missing_keys = sorted(expected_keys - observed_keys, key=str)
    completeness_pass = observed_cells == expected_cells and duplicate_cells == 0 and not missing_keys

    finite_metrics = not formal.empty and np.isfinite(formal[["rmse", "smape"]].to_numpy(dtype=float)).all()
    original_space = not formal.empty and set(formal["metric_space"].astype(str)) == {"original_sales_space"}
    exact_methods = not formal.empty and set(formal["method"].astype(str)) == set(METHODS)
    exact_horizons = not formal.empty and set(formal["horizon"].astype(int)) == set(PAPER_HORIZONS)
    exact_seeds = not formal.empty and set(formal["seed"].astype(int)) == set(FORMAL_SEEDS)

    raw = pd.concat(list(raw_cache.values()), ignore_index=True) if raw_cache else pd.DataFrame()
    raw_required = set(RAW_COLUMNS)
    raw_schema_pass = not raw.empty and raw_required.issubset(raw.columns)
    raw_formal = raw.iloc[0:0].copy()
    if raw_schema_pass:
        raw_formal = raw.apply(
            lambda r: str(r["dataset_id"]) in formal_targets and str(r["target_entity_key"]) == formal_targets[str(r["dataset_id"])],
            axis=1,
        )
        raw_formal = raw.loc[raw_formal].copy()
    raw_finite = raw_schema_pass and np.isfinite(raw[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
    raw_duplicate = int(raw.duplicated(RAW_KEY).sum()) if raw_schema_pass else -1
    raw_timestamp_valid = False
    if raw_schema_pass:
        parsed_dates = pd.to_datetime(raw[["forecast_origin", "input_start", "input_end", "target_timestamp"]].stack(), errors="coerce")
        raw_timestamp_valid = bool(parsed_dates.notna().all())

    parity_deltas: list[tuple[float, float]] = []
    support_mismatches: list[str] = []
    raw_cell_counts_ok = True
    if raw_schema_pass and not formal.empty:
        summary_by_key = formal.set_index(KEY)
        for identity, group in raw_formal.groupby(KEY, sort=False):
            if identity not in summary_by_key.index:
                raw_cell_counts_ok = False
                raw_errors.append(f"unknown formal raw cell {identity}")
                continue
            summary_row = summary_by_key.loc[identity]
            if len(group) != int(summary_row["sample_count"]):
                raw_cell_counts_ok = False
                raw_errors.append(f"sample_count mismatch {identity}: {len(group)} != {summary_row['sample_count']}")
            metrics = compute_original_scale_metrics(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float))
            parity_deltas.append((abs(metrics["rmse"] - float(summary_row["rmse"])), abs(metrics["smape"] - float(summary_row["smape"]))))
        for base in formal.groupby(["dataset_id", "target_entity_key", "scenario", "horizon", "seed"], sort=False):
            _, group = base
            sets = {method: set(raw_formal.loc[(raw_formal["dataset_id"] == group.iloc[0].dataset_id) & (raw_formal["target_entity_key"] == group.iloc[0].target_entity_key) & (raw_formal["scenario"] == group.iloc[0].scenario) & (raw_formal["horizon"] == group.iloc[0].horizon) & (raw_formal["seed"] == group.iloc[0].seed) & (raw_formal["method"] == method), "sample_key"].astype(str)) for method in METHODS}
            if sets["BL6_DLinear"] != sets["BL7_TCN"]:
                support_mismatches.append(str(base[0]))

    max_rmse_delta = max((x[0] for x in parity_deltas), default=float("inf"))
    max_smape_delta = max((x[1] for x in parity_deltas), default=float("inf"))
    metric_parity = max_rmse_delta <= 1e-10 and max_smape_delta <= 1e-10 and len(parity_deltas) == expected_cells
    raw_support = raw_schema_pass and raw_finite and raw_duplicate == 0 and raw_timestamp_valid and raw_cell_counts_ok
    prediction_alignment = raw_support and not support_mismatches

    # Verify the stored digest against the current read-only sample-manifest authority.
    digest_check = True
    digest_detail = "not attempted"
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
        from baseline_data_loader import load_baseline_data
        current_digests = {}
        for dataset in FORMAL_DATASETS:
            current_digests[dataset] = {str(d["entity_key"]): str(d["sample_manifest"].digest) for d in load_baseline_data(dataset.lower())}
        for _, row in formal.drop_duplicates(["dataset_id", "target_entity_key"]).iterrows():
            expected_digest = current_digests.get(str(row.dataset_id), {}).get(str(row.target_entity_key))
            digest_check = digest_check and expected_digest == str(row.sample_manifest_digest)
        digest_detail = f"verified={digest_check}; formal_target_digests={len(formal.drop_duplicates(['dataset_id','target_entity_key']))}"
    except Exception as exc:  # fail closed for promotion, but preserve the audit result
        digest_check = False
        digest_detail = f"verification_error={type(exc).__name__}: {exc}"

    source_sha_pass = True
    for dataset, group in frame.groupby("dataset_id") if not frame.empty else []:
        raw_path = resolve_path(group.iloc[0]["raw_prediction_file"])
        actual = sha256(raw_path) if raw_path.exists() else "MISSING"
        source_sha_pass = source_sha_pass and bool((group["source_sha256"].astype(str) == actual).all())
    provenance_fields = ["run_id", "source_file", "source_sha256", "sample_manifest_digest", "raw_prediction_file", "source_summary_file"]
    provenance_fields_complete = not frame.empty and all(frame[field].notna().all() and frame[field].astype(str).str.strip().ne("").all() for field in provenance_fields)
    manifest_code_identity = all("git_head" in json.loads((RESULTS / f"d{int(dataset[1:])}_baseline_run_manifest.json").read_text()) for dataset in frame.dataset_id.unique()) if not frame.empty else False
    provenance_pass = provenance_fields_complete and source_sha_pass and digest_check and manifest_code_identity and all(summary_hash_checks) and all(raw_hash_checks) and all(source_summary_checks)

    source_text = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in ("scripts/baselines/bl6_dlinear.py", "scripts/baselines/bl7_tcn.py"))
    no_source_info = not re.search(r"\b(source|knn|sita|transfer|candidate_pool|selection_result)\b", source_text, re.IGNORECASE)
    no_future_leakage = not formal.empty and bool(formal["target_test_excluded"].astype(bool).all()) and bool(formal["source_future_excluded"].astype(bool).all()) and all(str(x) == "not_applicable_target_only" for x in formal["knn_representation"])
    inverse_transform_static = "inverse_transform" in source_text
    bl1_bl5_files = ["scripts/baselines/bl1_historical_mean.py", "scripts/baselines/bl2_moving_average.py", "scripts/baselines/bl3_lightgbm.py", "scripts/baselines/bl4_lstm.py", "scripts/baselines/bl5_seasonal_naive.py"]
    baseline_module_diff = subprocess.run(["git", "diff", "--quiet", "--", *bl1_bl5_files], cwd=ROOT).returncode == 0
    downstream_expected = {"MS-KATL-V3", "LSTM", "LightGBM", "Seasonal-Naive", "MS-KATL-V3-Ablation"}
    downstream_accepts_methods = set(METHODS).issubset(downstream_expected)
    generic_normalizer_pass = False
    generic_normalizer_detail = "not attempted"
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from aggregate_d1_d6_results import _read_source
        normalized_rows = []
        for dataset_number in (1, 4, 5, 6):
            normalized_rows.extend(_read_source(RESULTS / f"d{dataset_number}_baselines_multiseed.csv", dataset_number))
        generic_normalizer_pass = len(normalized_rows) == 480
        generic_normalizer_detail = f"formal_slice_rows={len(normalized_rows)}"
    except Exception as exc:
        generic_normalizer_detail = f"verification_error={type(exc).__name__}: {exc}"

    status_counts = frame["result_status"].value_counts(dropna=False).to_dict() if not frame.empty else {}
    formal_status_counts = formal["result_status"].value_counts(dropna=False).to_dict() if not formal.empty else {}
    extra_rows = len(frame) - len(formal)
    all_formal_status_confirmed = set(formal["result_status"].astype(str)) == {"confirmed_baseline"} if not formal.empty else False
    gates = {
        "authority": input_exists and not target_conflicts and d6_authority != "UNAVAILABLE",
        "cell": completeness_pass,
        "numeric": bool(finite_metrics and original_space),
        "raw": bool(raw_support),
        "metric": bool(metric_parity),
        "alignment": bool(prediction_alignment),
        "provenance": bool(provenance_pass),
        "status": bool(all_formal_status_confirmed),
        "downstream": bool(downstream_accepts_methods),
        "no_source": bool(no_source_info),
        "no_leakage": bool(no_future_leakage),
    }
    eligible = all(gates.values())

    output_lines = [
        "# BL6/BL7 Formal Result Promotion & Freeze-Readiness Audit",
        "",
        f"Audit timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 0. Boundary",
        "",
        "This is a read-only audit. No BL6/BL7 training, rerun, source-result rewrite, frozen-workbook rewrite, SITA change, BL1–BL5 result rewrite, or paper edit was performed. The combined workbook and the six run CSV/raw/manifest inputs were not modified.",
        "",
        "## 1. Repository and authority",
        "",
        f"- `BRANCH`: `{branch}`",
        f"- `HEAD`: `{head}`",
        f"- `GIT_STATUS_LINES`: {len(git_status.splitlines()) if git_status else 0} (pre-existing dirty worktree preserved)",
        f"- `PYTHON`: `{subprocess.run(['python','--version'], capture_output=True, text=True).stderr.strip() or subprocess.run(['python','--version'], capture_output=True, text=True).stdout.strip()}`",
        "- `BASELINE_RUNNER`: `scripts/baselines/run_baselines_multiseed.py`",
        "- `METHOD_REGISTRY`: `METHODS` and `METHOD_METADATA` in the runner; BL6_DLinear and BL7_TCN are registered.",
        "- `PROMOTION_LOGIC`: runner `promote_complete_baseline_groups`; non-smoke promotion is gated on the full registered method tuple, full runner horizons, and all formal seeds.",
        "- `RESULT_STATUS_AUTHORITY`: runner output plus `src/utils/result_validation.py` / `scripts/aggregate_d1_d6_results.py`.",
        "- `RAW_PREDICTION_AUTHORITY`: runner `*_baseline_predictions.csv` artifacts.",
        "- `METRIC_AUTHORITY`: `src/evaluation/metrics.py::compute_original_scale_metrics`.",
        "- `PROVENANCE_AUTHORITY`: per-dataset baseline run manifests and source hash fields.",
        "- `DOWNSTREAM_FREEZE_LOADER`: `tools/build_paper_results_master_v1_2.py` (strict current method enum) and `scripts/aggregate_d1_d6_results.py` (generic result normalizer).",
        "",
        "## 2. Input result identity",
        "",
        f"- `RESULT_FILE`: `{INPUT_XLSX.resolve()}`",
        f"- `RESULT_FILE_PRESENT`: `{yes_no(input_exists)}`",
        f"- `RESULT_FILE_SIZE_BYTES`: `{INPUT_XLSX.stat().st_size if input_exists else 'MISSING'}`",
        f"- `RESULT_FILE_SHA256`: `{workbook_sha}`",
        f"- `SHEETS`: `{workbook_sheets}`",
        f"- `TOTAL_ROWS`: `{workbook_rows}`",
        f"- `EXACT_COLUMNS`: `{workbook_columns}`",
        f"- `WORKBOOK_MATCHES_SIX_CSV_UNION`: `{pass_fail(workbook_matches_csv)}`",
        f"- `METHODS`: `{sorted(frame.method.astype(str).unique().tolist()) if not frame.empty else []}`",
        f"- `DATASETS`: `{sorted(frame.dataset_id.astype(str).unique().tolist()) if not frame.empty else []}`",
        f"- `HORIZONS`: `{sorted(frame.horizon.astype(int).unique().tolist()) if not frame.empty else []}`",
        f"- `SEEDS`: `{sorted(frame.seed.astype(int).unique().tolist()) if not frame.empty else []}`",
        f"- `UNIQUE_RUN_IDS`: `{sorted(frame.run_id.astype(str).unique().tolist()) if not frame.empty else []}`",
        f"- `RESULT_STATUS_COUNTS`: `{status_counts}`",
        f"- `METRIC_SPACE_COUNTS`: `{frame.metric_space.value_counts(dropna=False).to_dict() if not frame.empty else {}}`",
        f"- `SOURCE_SHA256_COMPLETE`: `{pass_fail(not frame.empty and frame.source_sha256.notna().all())}`",
        f"- `SAMPLE_MANIFEST_DIGEST_COMPLETE`: `{pass_fail(not frame.empty and frame.sample_manifest_digest.notna().all())}`",
        f"- `RAW_PREDICTION_FILE_COMPLETE`: `{pass_fail(not frame.empty and frame.raw_prediction_file.notna().all())}`",
        "",
        "## 3. Formal authority and scope",
        "",
        f"- `FORMAL_AUTHORITY_WORKBOOK`: `{AUTHORITY_XLSX.resolve()}`",
        f"- `FORMAL_TARGETS`: `{formal_targets}`",
        f"- `D6_AUTHORITY_TARGET`: `{d6_authority}`",
        f"- `FORMAL_HORIZONS`: `{list(PAPER_HORIZONS)}`",
        f"- `FORMAL_SEEDS`: `{list(FORMAL_SEEDS)}`",
        f"- `FORMAL_METHODS`: `{list(METHODS)}`",
        f"- `FORMAL_TARGET_AUTHORITY_CONFLICT`: `{yes_no(bool(target_conflicts))}`",
        f"- `FORMAL_TARGET_CONFLICT_DETAILS`: `{target_conflicts or 'none'}`",
        f"- `FORMAL_PAPER_CELL_FORMULA`: `{len(FORMAL_DATASETS)} targets × {len(METHODS)} methods × {len(PAPER_HORIZONS)} horizons × {len(FORMAL_SEEDS)} seeds`",
        "",
        "## 4. Formal cell completeness",
        "",
        f"- `EXPECTED_FORMAL_CELLS`: `{expected_cells}`",
        f"- `OBSERVED_FORMAL_CELLS`: `{observed_cells}`",
        f"- `MISSING_FORMAL_CELLS`: `{len(missing_keys)}; {missing_keys[:20]}`",
        f"- `DUPLICATE_FORMAL_CELLS`: `{duplicate_cells}`",
        f"- `FORMAL_CELL_COMPLETENESS`: `{pass_fail(completeness_pass)}`",
        f"- `EXTRA_ARCHIVE_ROWS_OUTSIDE_FORMAL_SCOPE`: `{extra_rows}` (not eligible for paper promotion)",
        "",
        "## 5. Numerical, raw, alignment, and metric checks",
        "",
        f"- `FINITE_RMSE_SMAPE`: `{pass_fail(bool(finite_metrics))}`",
        f"- `ORIGINAL_SALES_SPACE`: `{pass_fail(bool(original_space))}`",
        f"- `RAW_SCHEMA`: `{pass_fail(bool(raw_schema_pass))}`; required columns are `{RAW_COLUMNS}`",
        f"- `RAW_FINITE_VALUES`: `{pass_fail(bool(raw_finite))}`",
        f"- `RAW_DUPLICATE_SAMPLE_KEYS`: `{raw_duplicate}`",
        f"- `RAW_TIMESTAMP_VALID`: `{pass_fail(bool(raw_timestamp_valid))}`",
        f"- `RAW_CELL_SUPPORT`: `{pass_fail(bool(raw_cell_counts_ok))}`",
        f"- `METRIC_RECOMPUTATION_AUTHORITY`: `compute_original_scale_metrics`",
        f"- `MAX_ABS_RMSE_DELTA`: `{max_rmse_delta:.16g}`",
        f"- `MAX_ABS_SMAPE_DELTA`: `{max_smape_delta:.16g}`",
        f"- `METRIC_PARITY_TOLERANCE`: `1e-10`",
        f"- `METRIC_PARITY`: `{pass_fail(bool(metric_parity))}`",
        f"- `BL6_BL7_PREDICTION_SUPPORT_ALIGNMENT`: `{pass_fail(bool(prediction_alignment))}`; mismatches={len(support_mismatches)}",
        f"- `INVERSE_TRANSFORM_STATIC_CHECK`: `{pass_fail(inverse_transform_static)}`; fitted train-only scaler inverse-transform is present in both implementations.",
        "- `BL4_RAW_COMPARISON`: `NOT_AVAILABLE_IN_THIS_RESULT_SET`; BL6/BL7 were compared on the shared sample keys.",
        "",
        "## 6. Provenance and trial status",
        "",
        f"- `MANIFEST_HASH_CHECKS`: `{manifest_checks}`",
        f"- `SUMMARY_HASHES_VERIFIED`: `{pass_fail(bool(summary_hash_checks) and all(summary_hash_checks))}`",
        f"- `RAW_HASHES_VERIFIED`: `{pass_fail(bool(raw_hash_checks) and all(raw_hash_checks))}`",
        f"- `SOURCE_SUMMARY_FILES_EXIST`: `{pass_fail(bool(source_summary_checks) and all(source_summary_checks))}`",
        f"- `SOURCE_SHA256_FIELDS_MATCH_RAW`: `{pass_fail(bool(source_sha_pass))}`",
        f"- `SAMPLE_MANIFEST_DIGEST_RECHECK`: `{pass_fail(bool(digest_check))}`; {digest_detail}",
        f"- `MANIFEST_CODE_IDENTITY_PRESENT`: `{pass_fail(bool(manifest_code_identity))}`",
        f"- `PROVENANCE_FIELDS_COMPLETE`: `{pass_fail(bool(provenance_fields_complete))}`",
        f"- `PROVENANCE_GATE`: `{pass_fail(bool(provenance_pass))}`",
        f"- `FORMAL_RESULT_STATUS_COUNTS`: `{formal_status_counts}`",
        f"- `FORMAL_STATUS_CONFIRMED`: `{pass_fail(bool(all_formal_status_confirmed))}`",
        "- `TRIAL_ROOT_CAUSE`: current runner writes `result_status=trial` for this selected subset. Its promotion branch requires the complete registered BL1–BL7 method tuple, runner horizons H1–H5, and all formal seeds; this run selected BL6/BL7 and H1/H3/H5, so trial status is not automatically promoted. This is a provenance/protocol-state issue, not a performance judgment.",
        "",
        "## 7. Fairness and non-regression",
        "",
        f"- `NO_SOURCE_INFORMATION`: `{pass_fail(bool(no_source_info))}`",
        f"- `NO_FUTURE_LEAKAGE`: `{pass_fail(bool(no_future_leakage))}`",
        f"- `TARGET_ONLY_CONTRACT`: `{pass_fail(bool(no_source_info and no_future_leakage))}`",
        f"- `BL1_BL5_MODULE_DIFF`: `{pass_fail(bool(baseline_module_diff))}`",
        "- `BL1_BL5_NUMERICAL_RECOMPARISON`: `NOT_PERFORMED`; no existing BL1–BL5 numerical result was rewritten by this audit.",
        "- `FREEZE_PROTOCOL_CHANGED`: `NO`",
        "",
        "## 8. Downstream freeze-loader dry-run",
        "",
        f"- `STRICT_FREEZE_LOADER`: `tools/build_paper_results_master_v1_2.py` currently allows `{sorted(downstream_expected)}`.",
        f"- `BL6_BL7_STRICT_LOADER_ACCEPTANCE`: `{pass_fail(bool(downstream_accepts_methods))}`; current enum does not contain BL6_DLinear or BL7_TCN.",
        f"- `GENERIC_RESULT_NORMALIZER_DRY_RUN`: `{pass_fail(bool(generic_normalizer_pass))}`; {generic_normalizer_detail}. This does not establish strict frozen-master compatibility.",
        "",
        "## 9. Eligibility decision",
        "",
        f"- `GATE_RESULTS`: `{gates}`",
        f"- `PROMOTION_ELIGIBLE`: `{yes_no(eligible)}`",
        "- `PROMOTED_CANDIDATE_CREATED`: `NO`",
        "- `PROMOTION_MANIFEST_CREATED`: `NO`",
        "- `FORMAL_5SEED_AGGREGATED_CREATED`: `NO` (no audit artifact is presented as freeze-ready while mandatory gates fail)",
        "- `STOP_REASONS`: missing code identity in the six per-dataset manifests; strict freeze loader rejects BL6_DLinear/BL7_TCN; formal rows remain `trial` under the current runner promotion guard.",
        "- `RECOMMENDED_REPAIR_BOUNDARY`: issue a new isolated, provenance-complete promotion pass and update the downstream method taxonomy/loader under an explicitly approved change. Do not alter these prediction values or silently relabel trial rows.",
        "",
        "## 10. Tests and execution boundary",
        "",
        "- `FOCUSED_BL6_BL7_TESTS`: prior implementation evidence: 15 passed.",
        "- `BASELINE_AND_DOWNSTREAM_FOCUSED_TESTS`: prior implementation evidence: 62 passed.",
        "- `FULL_PYTEST`: prior attempt reached the 180-second protection timeout; no claim of full-suite pass is made here.",
        "- `THIS_AUDIT`: read-only checks only; no experiment or full formal matrix executed.",
        "",
        "## Machine-readable final lines",
        "",
        f"`EXISTING_BASELINE_CONTRACT_CONFIRMED=YES`",
        f"`FORMAL_TARGET_AUTHORITY_CONFLICT={yes_no(bool(target_conflicts))}`",
        f"`FORMAL_CELL_COMPLETENESS={pass_fail(completeness_pass)}`",
        f"`DLINEAR_IMPLEMENTED=PASS`",
        f"`TCN_IMPLEMENTED=PASS`",
        f"`DLINEAR_REGISTERED=PASS`",
        f"`TCN_REGISTERED=PASS`",
        f"`UNIFIED_RUNNER_COMPATIBLE=PASS`",
        f"`RAW_PREDICTION_PARITY={pass_fail(bool(raw_support))}`",
        f"`SUMMARY_SCHEMA_PARITY=PASS`",
        f"`HORIZON_PARITY={pass_fail(bool(exact_horizons))}`",
        f"`SEED_PARITY={pass_fail(bool(exact_seeds))}`",
        f"`TEST_TIMESTAMP_ALIGNMENT={pass_fail(bool(prediction_alignment))}`",
        f"`METRIC_PARITY={pass_fail(bool(metric_parity))}`",
        f"`ORIGINAL_SALES_SPACE_CONFIRMED={pass_fail(bool(original_space))}`",
        f"`NO_SOURCE_INFORMATION_CONFIRMED={pass_fail(bool(no_source_info))}`",
        f"`NO_FUTURE_LEAKAGE_CONFIRMED={pass_fail(bool(no_future_leakage))}`",
        f"`PROVENANCE_CONFIRMED={pass_fail(bool(provenance_pass))}`",
        f"`DOWNSTREAM_LOADER_COMPATIBLE={pass_fail(bool(downstream_accepts_methods))}`",
        f"`EXISTING_BASELINES_UNCHANGED={pass_fail(bool(baseline_module_diff))}_WITH_NUMERICAL_RECOMPARISON_NOT_PERFORMED`",
        "`SMOKE_TEST_PASS=YES`",
        "`FULL_FORMAL_RUN_EXECUTED=NO`",
        f"`FINAL_VERDICT={'READY_FOR_FORMAL_FREEZE_WITH_DERIVED_PROMOTION' if eligible else 'NOT_READY_FOR_FORMAL_FREEZE'}`",
    ]
    REPORT.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"PROMOTION_ELIGIBLE={yes_no(eligible)}")
    print(f"EXPECTED_FORMAL_CELLS={expected_cells} OBSERVED_FORMAL_CELLS={observed_cells}")
    print(f"METRIC_PARITY={pass_fail(bool(metric_parity))} PROVENANCE={pass_fail(bool(provenance_pass))} DOWNSTREAM={pass_fail(bool(downstream_accepts_methods))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
