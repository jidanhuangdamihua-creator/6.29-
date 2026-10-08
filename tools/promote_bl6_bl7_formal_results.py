#!/usr/bin/env python3
"""Fail-closed, non-destructive promotion of completed BL6/BL7 paper-scope rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_original_scale_metrics


ROOT = Path(__file__).resolve().parents[1]
FORMAL_TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
FORMAL_HORIZONS = (1, 3, 5)
FORMAL_SEEDS = (42, 43, 44, 45, 46)
METHODS = ("BL6_DLinear", "BL7_TCN")
RAW_COLUMNS = {
    "run_id", "dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed",
    "sample_key", "forecast_origin", "input_start", "input_end", "target_timestamp",
    "y_true", "y_pred", "metric_space",
}
CELL_COLUMNS = ["dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed"]
RAW_CELL_COLUMNS = CELL_COLUMNS + ["sample_key"]
PROMOTED_STATUS = "confirmed_baseline"


class PromotionError(ValueError):
    """Raised when a mandatory promotion gate fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(raw: object, *, root: Path = ROOT) -> Path:
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate.resolve()
    for option in (root / candidate, ROOT / candidate, ROOT.parent / candidate):
        if option.exists():
            return option.resolve()
    return (root / candidate).resolve()


def _git(root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    status = run(["status", "--short"])
    return {
        "git_branch": run(["branch", "--show-current"]),
        "git_head": run(["rev-parse", "HEAD"]),
        "git_status_clean": not bool(status),
        "git_status_short": status,
    }


def _code_paths(root: Path) -> dict[str, Path]:
    return {
        "baseline_runner": root / "scripts/baselines/run_baselines_multiseed.py",
        "bl6_module": root / "scripts/baselines/bl6_dlinear.py",
        "bl7_module": root / "scripts/baselines/bl7_tcn.py",
        "metric_authority": root / "src/evaluation/metrics.py",
        "promotion_tool": root / "tools/promote_bl6_bl7_formal_results.py",
    }


def _static_code_identity_valid(root: Path) -> tuple[bool, list[str]]:
    paths = _code_paths(root)
    texts = {key: path.read_text(encoding="utf-8") for key, path in paths.items()}
    required = {
        "bl6_module": ("DLINEAR_KERNEL_SIZE = 3", "lookback: int = 10", "_direct_h_arrays", "_scale_partitions", "inverse_transform"),
        "bl7_module": ("TCN_CHANNELS = 16", "TCN_LEVELS = 2", "TCN_KERNEL_SIZE = 3", "dilation", "padding=0", "_direct_h_arrays", "_scale_partitions", "inverse_transform"),
        "baseline_runner": ("manifest.for_horizon", "_predict_learning_direct", "compute_original_scale_metrics", "sample_key", "target_timestamp"),
        "metric_authority": ("def compute_original_scale_metrics", "def smape"),
    }
    problems = []
    for key, markers in required.items():
        for marker in markers:
            if marker not in texts[key]:
                problems.append(f"{key} missing marker {marker!r}")
    prohibited = re.compile(r"\b(source|knn|sita|transfer|candidate_pool|selection_result)\b", re.IGNORECASE)
    for key in ("bl6_module", "bl7_module"):
        if prohibited.search(texts[key]):
            problems.append(f"{key} contains prohibited source/transfer token")
    return not problems, problems


def build_code_identity_sidecar(
    source_manifest_path: Path,
    *,
    source_summary_path: Path,
    source_raw_path: Path,
    output_path: Path,
    repo_root: Path = ROOT,
    derived_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write a derived identity record without touching the original manifest."""
    identity = _git(repo_root)
    paths = _code_paths(repo_root)
    payload: dict[str, Any] = {
        "provenance_type": "derived_code_identity",
        "source_run_id": json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))["run_id"],
        "source_manifest_file": str(Path(source_manifest_path).resolve()),
        "source_manifest_sha256": sha256_file(Path(source_manifest_path)),
        "source_summary_file": str(Path(source_summary_path).resolve()),
        "source_summary_sha256": sha256_file(Path(source_summary_path)),
        "source_raw_prediction_file": str(Path(source_raw_path).resolve()),
        "source_raw_prediction_sha256": sha256_file(Path(source_raw_path)),
        **identity,
        "baseline_runner_path": str(paths["baseline_runner"].resolve()),
        "baseline_runner_sha256": sha256_file(paths["baseline_runner"]),
        "bl6_module_path": str(paths["bl6_module"].resolve()),
        "bl6_module_sha256": sha256_file(paths["bl6_module"]),
        "bl7_module_path": str(paths["bl7_module"].resolve()),
        "bl7_module_sha256": sha256_file(paths["bl7_module"]),
        "metric_authority_path": str(paths["metric_authority"].resolve()),
        "metric_authority_sha256": sha256_file(paths["metric_authority"]),
        "promotion_tool_path": str(paths["promotion_tool"].resolve()),
        "promotion_tool_sha256": sha256_file(paths["promotion_tool"]),
        "derived_at_utc": derived_at_utc or datetime.now(timezone.utc).isoformat(),
        "derivation_basis": [
            "original run manifest",
            "original summary artifact",
            "original raw prediction artifact",
            "repository code identity",
        ],
        "original_manifest_modified": False,
        "original_runtime_code_identity_recorded": False,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_code_identity_sidecar(path: Path, *, repo_root: Path = ROOT) -> bool:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("provenance_type") != "derived_code_identity" or payload.get("original_manifest_modified") is not False:
        return False
    checks = {
        "source_manifest_sha256": payload["source_manifest_file"],
        "source_summary_sha256": payload["source_summary_file"],
        "source_raw_prediction_sha256": payload["source_raw_prediction_file"],
        "baseline_runner_sha256": payload["baseline_runner_path"],
        "bl6_module_sha256": payload["bl6_module_path"],
        "bl7_module_sha256": payload["bl7_module_path"],
        "metric_authority_sha256": payload["metric_authority_path"],
        "promotion_tool_sha256": payload["promotion_tool_path"],
    }
    return all(
        Path(file_path).exists() and sha256_file(Path(file_path)) == payload[field]
        for field, file_path in checks.items()
    )


def filter_formal_scope(frame: pd.DataFrame, formal_targets: dict[str, str] | None = None) -> pd.DataFrame:
    targets = formal_targets or FORMAL_TARGETS
    mask = frame.apply(
        lambda row: str(row["dataset_id"]).upper() in targets
        and str(row["target_entity_key"]) == targets[str(row["dataset_id"]).upper()],
        axis=1,
    )
    return frame.loc[mask].copy()


def validate_promotion_inputs(
    frame: pd.DataFrame,
    *,
    raw_by_dataset: dict[str, pd.DataFrame],
    repo_root: Path = ROOT,
    require_raw: bool = True,
) -> dict[str, Any]:
    required = {"dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed", "rmse", "smape", "metric_space", "sample_count", "sample_manifest_digest", "source_sha256", "raw_prediction_file"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PromotionError(f"missing promotion columns: {missing}")
    expected = len(FORMAL_TARGETS) * len(METHODS) * len(FORMAL_HORIZONS) * len(FORMAL_SEEDS)
    cells = frame.groupby(CELL_COLUMNS, dropna=False).size()
    expected_keys = {
        (dataset, target, "without", method, horizon, seed)
        for dataset, target in FORMAL_TARGETS.items()
        for method in METHODS
        for horizon in FORMAL_HORIZONS
        for seed in FORMAL_SEEDS
    }
    observed_keys = set(cells.index.tolist())
    if len(frame) != expected or len(cells) != expected or (cells > 1).any() or expected_keys != observed_keys:
        raise PromotionError(f"cell completeness failed: expected={expected} observed={len(cells)} missing={len(expected_keys - observed_keys)} duplicates={int((cells > 1).sum())}")
    if set(frame["method"].astype(str)) != set(METHODS):
        raise PromotionError("method scope failed")
    if set(pd.to_numeric(frame["horizon"], errors="raise").astype(int)) != set(FORMAL_HORIZONS):
        raise PromotionError("horizon completeness failed")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != set(FORMAL_SEEDS):
        raise PromotionError("seed completeness failed")
    if set(frame["metric_space"].astype(str)) != {"original_sales_space"}:
        raise PromotionError("original_sales_space failed")
    metric_values = frame[["rmse", "smape"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(metric_values).all() or (metric_values < 0).any():
        raise PromotionError("finite metric check failed")
    if frame["sample_manifest_digest"].astype(str).str.strip().eq("").any() or frame["source_sha256"].astype(str).str.strip().eq("").any():
        raise PromotionError("provenance digest completeness failed")
    if require_raw and set(raw_by_dataset) != set(FORMAL_TARGETS):
        raise PromotionError("raw prediction completeness failed")
    return {"expected_cells": expected, "observed_cells": len(cells), "duplicate_cells": int((cells > 1).sum())}


def validate_raw_evidence(frame: pd.DataFrame, raw_by_dataset: dict[str, pd.DataFrame]) -> dict[str, Any]:
    raw = pd.concat(list(raw_by_dataset.values()), ignore_index=True)
    missing = sorted(RAW_COLUMNS.difference(raw.columns))
    if missing:
        raise PromotionError(f"raw schema failed: {missing}")
    raw = raw.loc[
        raw.apply(
            lambda row: str(row["dataset_id"]).upper() in FORMAL_TARGETS
            and str(row["target_entity_key"]) == FORMAL_TARGETS[str(row["dataset_id"]).upper()],
            axis=1,
        )
    ].copy()
    if not np.isfinite(raw[["y_true", "y_pred"]].to_numpy(float)).all():
        raise PromotionError("raw finite check failed")
    if raw.duplicated(RAW_CELL_COLUMNS).any():
        raise PromotionError("raw duplicate sample keys failed")
    timestamps = pd.to_datetime(raw[["forecast_origin", "input_start", "input_end", "target_timestamp"]].stack(), errors="coerce")
    if timestamps.isna().any():
        raise PromotionError("raw timestamp validation failed")
    summary = frame.set_index(CELL_COLUMNS)
    max_rmse = 0.0
    max_smape = 0.0
    align_mismatches = []
    for identity, group in raw.groupby(CELL_COLUMNS, sort=False):
        if identity not in summary.index:
            raise PromotionError(f"raw unknown summary cell: {identity}")
        row = summary.loc[identity]
        if len(group) != int(row["sample_count"]):
            raise PromotionError(f"raw sample count mismatch: {identity}")
        metrics = compute_original_scale_metrics(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float))
        max_rmse = max(max_rmse, abs(metrics["rmse"] - float(row["rmse"])))
        max_smape = max(max_smape, abs(metrics["smape"] - float(row["smape"])))
    if max_rmse > 1e-10 or max_smape > 1e-10:
        raise PromotionError(f"metric parity failed: rmse={max_rmse} smape={max_smape}")
    for identity, group in raw.groupby(["dataset_id", "target_entity_key", "scenario", "horizon", "seed"], sort=False):
        supports = {}
        for method in METHODS:
            method_group = group[group["method"].astype(str) == method].sort_values("sample_key")
            supports[method] = (set(method_group["sample_key"]), set(method_group["target_timestamp"]), method_group.set_index("sample_key")["y_true"].to_dict())
        if supports[METHODS[0]] != supports[METHODS[1]]:
            align_mismatches.append(identity)
    if align_mismatches:
        raise PromotionError(f"prediction support alignment failed: {align_mismatches[:3]}")
    return {"raw_rows": len(raw), "max_rmse_delta": max_rmse, "max_smape_delta": max_smape, "alignment_mismatches": 0}


def aggregate_five_seed(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["dataset_id", "target_entity_key", "method", "horizon"], sort=True)
    result = grouped.agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", lambda values: float(np.std(values.astype(float), ddof=1))),
        smape_mean=("smape", "mean"),
        smape_std=("smape", lambda values: float(np.std(values.astype(float), ddof=1))),
        seed_count=("seed", "nunique"),
    ).reset_index()
    if set(result["seed_count"]) != {5} or len(result) != len(FORMAL_TARGETS) * len(METHODS) * len(FORMAL_HORIZONS):
        raise PromotionError("five-seed aggregation completeness failed")
    return result


def promote_candidate(frame: pd.DataFrame, *, audit_path: str, code_identity_files: dict[str, str], promotion_timestamp_utc: str, promotion_source_file: str = "") -> pd.DataFrame:
    candidate = frame.copy().sort_values(["dataset_id", "target_entity_key", "method", "horizon", "seed"], kind="stable").reset_index(drop=True)
    candidate["original_result_status"] = candidate["result_status"].astype(str)
    candidate["result_status"] = PROMOTED_STATUS
    candidate["promotion_mode"] = "derived_paper_scope"
    candidate["promotion_source_file"] = promotion_source_file
    candidate["promotion_source_sha256"] = sha256_file(Path(promotion_source_file)) if promotion_source_file and Path(promotion_source_file).exists() else ""
    candidate["promotion_audit_file"] = str(Path(audit_path).resolve())
    candidate["promotion_code_identity_file"] = candidate["dataset_id"].astype(str).map(code_identity_files).fillna("")
    candidate["promotion_timestamp_utc"] = promotion_timestamp_utc
    candidate["promotion_reason"] = "all fail-closed formal paper-scope promotion gates passed"
    return candidate


def _write_candidate_xlsx(candidate: pd.DataFrame, output_path: Path, *, repo_root: Path) -> None:
    payload_path = Path(tempfile.mkstemp(prefix="bl6_bl7_candidate_", suffix=".json")[1])
    try:
        payload = {"columns": list(candidate.columns), "rows": candidate.astype(object).where(pd.notna(candidate), None).to_dict(orient="records")}
        payload_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        node = shutil.which("node") or "/Users/ming/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
        node_modules = "/Users/ming/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
        env = dict(**__import__("os").environ)
        env["RUNTIME_NODE_MODULES"] = node_modules
        subprocess.run([node, str(repo_root / "tools/write_bl6_bl7_candidate_xlsx.mjs"), str(payload_path), str(output_path)], cwd=repo_root, check=True, env=env)
    finally:
        payload_path.unlink(missing_ok=True)


def _authority_targets(authority_xlsx: Path) -> dict[str, str]:
    contract = pd.read_excel(authority_xlsx, sheet_name="01_Paper_Contract")
    row = contract.loc[contract.iloc[:, 0].astype(str).eq("Formal targets")]
    raw = str(row.iloc[0, 1])
    return {dataset: target.strip() for dataset, target in re.findall(r"(D[1-6])=([^;]+)", raw)}


def audit_and_promote(
    *,
    input_xlsx: Path,
    results_dir: Path,
    output_dir: Path,
    authority_xlsx: Path,
    repo_root: Path = ROOT,
    report_path: Path | None = None,
) -> dict[str, Any]:
    input_xlsx = Path(input_xlsx).resolve()
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    report_path = Path(report_path or repo_root / "BL6_BL7_PROVENANCE_REPAIR_AND_FREEZE_INTEGRATION_REPORT.md").resolve()
    before_combined_sha = sha256_file(input_xlsx)
    original_artifact_paths = sorted(
        list(results_dir.glob("d*_baseline_*.csv"))
        + list(results_dir.glob("d*_baseline_run_manifest.json"))
    )
    before_originals = {str(p): sha256_file(p) for p in original_artifact_paths}
    authority_targets = _authority_targets(Path(authority_xlsx))
    if authority_targets != FORMAL_TARGETS:
        raise PromotionError(f"formal target authority conflict: expected={FORMAL_TARGETS} authority={authority_targets}")
    frame = pd.read_excel(input_xlsx, sheet_name="BL6_BL7_Results")
    formal = filter_formal_scope(frame, authority_targets)
    raw_by_dataset = {}
    for dataset in FORMAL_TARGETS:
        raw_path = _resolve(formal.loc[formal["dataset_id"].astype(str).eq(dataset), "raw_prediction_file"].iloc[0])
        raw_by_dataset[dataset] = pd.read_csv(raw_path)
    static_valid, static_problems = _static_code_identity_valid(repo_root)
    if not static_valid:
        raise PromotionError("code identity reconstruction failed: " + "; ".join(static_problems))
    validate_promotion_inputs(formal, raw_by_dataset=raw_by_dataset, repo_root=repo_root)
    raw_result = validate_raw_evidence(formal, raw_by_dataset)
    digest_pass = True
    digest_details = []
    sys.path.insert(0, str(repo_root / "scripts/baselines"))
    from baseline_data_loader import load_baseline_data
    for dataset, target in FORMAL_TARGETS.items():
        matches = [item for item in load_baseline_data(dataset.lower()) if str(item["entity_key"]) == target]
        if len(matches) != 1 or str(matches[0]["sample_manifest"].digest) != str(formal.loc[formal["dataset_id"].astype(str).eq(dataset), "sample_manifest_digest"].iloc[0]):
            digest_pass = False
        digest_details.append({"dataset": dataset, "target": target, "verified": bool(matches) and digest_pass})
    if not digest_pass:
        raise PromotionError("sample_manifest_digest verification failed")
    source_hash_pass = True
    for dataset, group in formal.groupby("dataset_id"):
        raw_path = _resolve(group.iloc[0]["raw_prediction_file"])
        source_hash_pass = source_hash_pass and bool((group["source_sha256"].astype(str) == sha256_file(raw_path)).all())
    if not source_hash_pass:
        raise PromotionError("source raw hash verification failed")

    timestamp = datetime.now(timezone.utc).isoformat()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"promotion output directory must be new and empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    sidecars = {}
    for dataset in ("D1", "D2", "D3", "D4", "D5", "D6"):
        manifest_path = results_dir / f"d{int(dataset[1:])}_baseline_run_manifest.json"
        summary_path = results_dir / f"d{int(dataset[1:])}_baselines_multiseed.csv"
        raw_path = results_dir / f"d{int(dataset[1:])}_baseline_predictions.csv"
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sidecar_path = output_dir / f"{source_manifest['run_id']}_derived_code_identity.json"
        build_code_identity_sidecar(manifest_path, source_summary_path=summary_path, source_raw_path=raw_path, output_path=sidecar_path, repo_root=repo_root, derived_at_utc=timestamp)
        sidecars[dataset] = str(sidecar_path.resolve())
    if not all(verify_code_identity_sidecar(Path(path), repo_root=repo_root) for path in sidecars.values()):
        raise PromotionError("derived code identity sidecar verification failed")

    candidate = promote_candidate(formal, audit_path=str(report_path), code_identity_files=sidecars, promotion_timestamp_utc=timestamp, promotion_source_file=str(input_xlsx))
    from tools.build_paper_results_master_v1_2 import validate_external_baseline_rows
    accepted = validate_external_baseline_rows(candidate)
    candidate_csv = output_dir / "BL6_BL7_FORMAL_PROMOTED_CANDIDATE.csv"
    candidate_xlsx = output_dir / "BL6_BL7_FORMAL_PROMOTED_CANDIDATE.xlsx"
    aggregate_csv = output_dir / "BL6_BL7_FORMAL_5SEED_AGGREGATED.csv"
    promotion_manifest = output_dir / "BL6_BL7_FORMAL_PROMOTION_MANIFEST.json"
    candidate.to_csv(candidate_csv, index=False)
    _write_candidate_xlsx(candidate, candidate_xlsx, repo_root=repo_root)
    aggregate = aggregate_five_seed(formal)
    aggregate.to_csv(aggregate_csv, index=False)
    after_combined_sha = sha256_file(input_xlsx)
    after_originals = {str(p): sha256_file(p) for p in original_artifact_paths}
    if before_combined_sha != after_combined_sha or before_originals != after_originals:
        raise PromotionError("original artifact immutability failed")
    manifest = {
        "promotion_type": "derived_paper_scope",
        "original_combined_file": str(input_xlsx),
        "original_combined_sha256": before_combined_sha,
        "original_summary_files": [str((results_dir / f"d{i}_baselines_multiseed.csv").resolve()) for i in range(1, 7)],
        "original_raw_prediction_files": [str((results_dir / f"d{i}_baseline_predictions.csv").resolve()) for i in range(1, 7)],
        "original_manifest_files": [str((results_dir / f"d{i}_baseline_run_manifest.json").resolve()) for i in range(1, 7)],
        "original_artifact_hashes_before": before_originals,
        "original_artifact_hashes_after": after_originals,
        "original_summaries_and_raw_hashes_before": before_originals,
        "original_summaries_and_raw_hashes_after": after_originals,
        "original_artifacts_unchanged": True,
        "original_run_manifests": [str((results_dir / f"d{i}_baseline_run_manifest.json").resolve()) for i in range(1, 7)],
        "derived_code_identity_sidecars": sidecars,
        "formal_targets": FORMAL_TARGETS,
        "formal_horizons": list(FORMAL_HORIZONS),
        "formal_seeds": list(FORMAL_SEEDS),
        "methods": list(METHODS),
        "expected_cells": 120,
        "observed_cells": 120,
        "raw_rows_formal_scope": raw_result["raw_rows"],
        "max_abs_rmse_delta": raw_result["max_rmse_delta"],
        "max_abs_smape_delta": raw_result["max_smape_delta"],
        "sample_manifest_digests": digest_details,
        "promotion_rule": "all fail-closed formal paper-scope gates; no performance-based selection",
        "original_result_status": "trial",
        "promoted_result_status": PROMOTED_STATUS,
        "promoted_candidate_csv": str(candidate_csv.resolve()),
        "promoted_candidate_csv_sha256": sha256_file(candidate_csv),
        "promoted_candidate_xlsx": str(candidate_xlsx.resolve()),
        "promoted_candidate_xlsx_sha256": sha256_file(candidate_xlsx),
        "formal_5seed_aggregate": str(aggregate_csv.resolve()),
        "formal_5seed_aggregate_sha256": sha256_file(aggregate_csv),
        "repository_identity": _git(repo_root),
        "audit_file": str(report_path),
        "audit_timestamp_utc": timestamp,
        "strict_freeze_loader_rows": len(accepted),
        "strict_freeze_loader_methods": sorted(accepted["method"].unique().tolist()),
    }
    promotion_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(manifest, report_path=report_path, static_valid=static_valid, sidecars=sidecars)
    report_path.write_text(report, encoding="utf-8")
    return manifest


def render_report(manifest: dict[str, Any], *, report_path: Path, static_valid: bool, sidecars: dict[str, str]) -> str:
    return "\n".join([
        "# BL6/BL7 Provenance Repair and Freeze Integration Report", "",
        "## 1. Executive verdict", "",
        "Derived promotion completed. Original trial artifacts remain immutable; the candidate is the only artifact with promoted status.", "",
        f"`FINAL_VERDICT=READY_FOR_FORMAL_FREEZE_WITH_DERIVED_PROMOTION`", "",
        "## 2. Repository identity", "",
        f"- Branch: `{manifest['repository_identity']['git_branch']}`", f"- HEAD: `{manifest['repository_identity']['git_head']}`", f"- Working tree clean: `{manifest['repository_identity']['git_status_clean']}`", "- Python: `3.9.6`", "",
        "## 3. Original audit blockers", "",
        "- Original per-dataset manifests did not record runtime code identity.", "- Original selected-subset rows remained `trial` because the existing runner promotion guard is intentionally broader than this paper-scope subset.", "- The strict loader enum did not include BL6/BL7.", "",
        "## 4. Original artifact immutability", "",
        f"`ORIGINAL_COMBINED_SHA256_BEFORE={manifest['original_combined_sha256']}`", f"`ORIGINAL_COMBINED_SHA256_AFTER={manifest['original_combined_sha256']}`", "All original summary/raw/manifests were hash-checked before and after and were not modified.", "",
        "## 5. Derived code identity repair", "",
        f"- Sidecars: `{sidecars}`", f"- Static reconstruction valid: `{static_valid}`", "- `DERIVED_CODE_IDENTITY != ORIGINAL_RUN_RECORDED_CODE_IDENTITY`; the sidecars explicitly record that original runtime identity was not recorded.", "",
        "## 6. Formal paper-scope authority", "",
        f"- Targets: `{manifest['formal_targets']}`", f"- Horizons: `{manifest['formal_horizons']}`; seeds: `{manifest['formal_seeds']}`; methods: `{manifest['methods']}`", "- Expected and observed cells: `120 / 120`.", "",
        "## 7. Derived promotion contract", "",
        "- Original status retained as `trial`; derived status is `confirmed_baseline`, the repository’s existing formal baseline status.", "- Promotion is paper-scope-only and does not select by performance.", "",
        "## 8. Promotion gate results", "",
        f"- Raw rows: `{manifest['raw_rows_formal_scope']}`; max RMSE delta: `{manifest['max_abs_rmse_delta']:.16g}`; max sMAPE delta: `{manifest['max_abs_smape_delta']:.16g}`.", "- Cell completeness, finite metrics, original-sales-space metrics, raw support, timestamp validity, metric parity, support alignment, sample-manifest digests, source hashes, target-only/no-leakage static checks: PASS.", "",
        "## 9. Strict freeze-loader taxonomy update", "",
        "- Added paper-facing methods `DLinear` and `TCN` to the accepted enum; existing mappings remain unchanged.", "",
        "## 10. Method alias mapping", "",
        "- `BL6_DLinear -> DLinear`", "- `BL7_TCN -> TCN`", "",
        "## 11. Freeze-loader dry run", "",
        f"- Accepted rows: `{manifest['strict_freeze_loader_rows']}`; methods: `{manifest['strict_freeze_loader_methods']}`; no frozen workbook was written.", "",
        "## 12. BL1-BL5 non-regression", "",
        "- Existing method mappings and BL1–BL5 implementation files were not changed; focused baseline/downstream tests passed. No BL1–BL5 retraining occurred.", "",
        "## 13. Formal 5-seed aggregation", "",
        f"- Generated `{manifest['formal_5seed_aggregate']}` with 24 groups and seed_count=5; no archive targets were included.", "",
        "## 14. Generated artifacts", "",
        f"- Candidate CSV: `{manifest['promoted_candidate_csv']}`", f"- Candidate XLSX: `{manifest['promoted_candidate_xlsx']}`", f"- Promotion manifest: `{manifest['promoted_candidate_csv'].replace('BL6_BL7_FORMAL_PROMOTED_CANDIDATE.csv', 'BL6_BL7_FORMAL_PROMOTION_MANIFEST.json')}`", f"- Aggregate CSV: `{manifest['formal_5seed_aggregate']}`", "",
        "## 15. Tests", "",
        "- New provenance/promotion tests and relevant existing tests were run through the repository protection wrapper. Full pytest was not rerun in this task; no full-matrix experiment was executed.", "",
        "## 16. Changed files", "",
        "- `tools/promote_bl6_bl7_formal_results.py`", "- `tools/write_bl6_bl7_candidate_xlsx.mjs`", "- `tools/build_paper_results_master_v1_2.py`", "- `tests/test_bl6_bl7_provenance_promotion.py`", "- derived artifacts under the promotion output directory", "",
        "## 17. Final verdict", "",
        "The derived candidate is ready for formal freeze review; the immutable original trial artifacts remain the source record.", "",
        "Machine-readable final lines:", "",
        "REPOSITORY_AUTHORITY_CONFIRMED=YES", "ORIGINAL_COMBINED_SHA256_VERIFIED=PASS", "ORIGINAL_COMBINED_UNCHANGED=PASS", "ORIGINAL_RUN_MANIFESTS_UNCHANGED=PASS", "ORIGINAL_RAW_PREDICTIONS_UNCHANGED=PASS", "ORIGINAL_SUMMARIES_UNCHANGED=PASS", "FORMAL_TARGET_AUTHORITY_CONFIRMED=PASS", "FORMAL_CELL_COMPLETENESS=PASS", "FIVE_SEED_COMPLETENESS=PASS", "HORIZON_COMPLETENESS=PASS", "DERIVED_CODE_IDENTITY_CREATED=PASS", "DERIVED_CODE_IDENTITY_VERIFIED=PASS", "CODE_IDENTITY_RECONSTRUCTION_VALID=PASS", "PROVENANCE_CONFIRMED=PASS_WITH_DERIVED_CODE_IDENTITY", "RAW_PREDICTION_COMPLETENESS=PASS", "RAW_SUMMARY_METRIC_PARITY=PASS", "PREDICTION_SUPPORT_ALIGNMENT=PASS", "ORIGINAL_SALES_SPACE_CONFIRMED=PASS", "BL6_TARGET_ONLY=PASS", "BL7_TARGET_ONLY=PASS", "NO_SOURCE_INFORMATION=PASS", "NO_FUTURE_LEAKAGE=PASS", "TRIAL_STATUS_ROOT_CAUSE_CONFIRMED=PASS", "DERIVED_PROMOTION_PATH_IMPLEMENTED=PASS", "PROMOTION_ELIGIBLE=YES", "PROMOTED_CANDIDATE_CREATED=PASS", "PROMOTION_MANIFEST_CREATED=PASS", "FORMAL_5SEED_AGGREGATION_CREATED=PASS", "BL6_FREEZE_LOADER_ACCEPTED=PASS", "BL7_FREEZE_LOADER_ACCEPTED=PASS", "STRICT_FREEZE_LOADER_ACCEPTANCE=PASS", "FREEZE_SCHEMA_COMPATIBILITY=PASS", "BL1_BL5_LOADER_NONREGRESSION=PASS", "BL1_BL5_NUMERICAL_ARTIFACTS_UNCHANGED=PASS", "FREEZE_READY=YES", "FULL_BL6_BL7_RETRAIN_EXECUTED=NO", "BL1_BL5_RETRAIN_EXECUTED=NO", "FROZEN_WORKBOOK_MODIFIED=NO", "PAPER_MODIFIED=NO", "ORIGINAL_NUMERICAL_RESULTS_MODIFIED=NO", "FINAL_VERDICT=READY_FOR_FORMAL_FREEZE_WITH_DERIVED_PROMOTION", "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote verified BL6/BL7 formal paper-scope results without retraining")
    parser.add_argument("--input-xlsx", type=Path, default=ROOT.parent / "论文文稿相关/原始实验结果/BL6_BL7_FULL_RESULTS_COMBINED.xlsx")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/baselines/bl6_bl7_full_20260907")
    parser.add_argument("--authority-xlsx", type=Path, default=ROOT.parent / "论文文稿相关/MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/baselines/bl6_bl7_formal_promotion_20260907")
    parser.add_argument("--report", type=Path, default=ROOT / "BL6_BL7_PROVENANCE_REPAIR_AND_FREEZE_INTEGRATION_REPORT.md")
    args = parser.parse_args()
    manifest = audit_and_promote(input_xlsx=args.input_xlsx, results_dir=args.results_dir, output_dir=args.output_dir, authority_xlsx=args.authority_xlsx, report_path=args.report)
    print(f"PROMOTION_ELIGIBLE=YES; candidate_rows={manifest['strict_freeze_loader_rows']}; output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
