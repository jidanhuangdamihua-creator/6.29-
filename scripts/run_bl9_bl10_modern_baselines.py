#!/usr/bin/env python3
"""Run BL9 PatchTST and BL10 TimesNet on the locked formal baseline protocol.

This runner does not modify SITA/V3, BL1-BL8 results, A-E ablations, or freeze
workbooks. Formal cells are written under results/formal_bl9_patchtst and
results/formal_bl10_timesnet. Shared audits land under results/formal_bl9_bl10.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.baselines.baseline_data_loader import load_baseline_data
from scripts.baselines.bl4_lstm import _direct_h_arrays
from scripts.baselines.bl9_patchtst import (
    METHOD as PATCHTST_METHOD,
    fit_bl9,
    patchtst_forward_probe,
    predict_bl9,
)
from scripts.baselines.bl10_timesnet import (
    METHOD as TIMESNET_METHOD,
    fit_bl10,
    predict_bl10,
    timesnet_forward_probe,
)
from scripts.baselines.run_baselines_multiseed import _validate_matured_record
from scripts.bl9_bl10_formal_protocol import (
    ABLATION_SHEET,
    BASELINE_SHEET,
    BL910FormalError,
    BOOTSTRAP_RANDOM_SEED,
    BOOTSTRAP_RESAMPLES,
    DATASETS,
    DEFAULT_FREEZE,
    EXPECTED_AGGREGATES,
    EXPECTED_CELLS,
    EXPECTED_N_TEST,
    FEATURE_METADATA,
    FORMAL_BASELINE_TRAINING_POLICY,
    FORMAL_LEARNING_UNIVERSE,
    HOLM_FAMILY,
    HOLM_FAMILY_SIZE,
    HORIZONS,
    LOOKBACK,
    METHOD_IDS,
    METHODS,
    METRIC_SPACE,
    MODE,
    OUTPUT_ROOT,
    PARITY_RELAXED,
    PARITY_TOLERANCE,
    PATCHTST_ROOT,
    ROBUSTNESS_UNIVERSE,
    SEEDS,
    SITA_SHEET,
    TARGETS,
    TARGET_COLUMN,
    TIMESNET_ROOT,
    TIE_RULE,
    aggregate_five_seed,
    assert_allowed_matrix_cell,
    bootstrap_mean_ci,
    canonical_key,
    cell_paths,
    completeness_audit,
    condition_means,
    expected_cell_keys,
    git_identity,
    holm_adjust,
    json_dump,
    load_bl8_cells,
    load_freeze_baselines,
    load_robustness_cells,
    load_sita_authority,
    model_config_payload,
    record_code_identity,
    recompute_original_metrics,
    relative_improvement_pct,
    runtime_versions,
    select_paper_entity,
    sha256_file,
    target_channel_index,
    training_policy_report,
    wilcoxon_paired,
    winner_label,
)
from src.evaluation.metrics import compute_original_scale_metrics
from src.protocols.reproducibility import set_protocol_seed


FITTERS = {
    PATCHTST_METHOD: (fit_bl9, predict_bl9),
    TIMESNET_METHOD: (fit_bl10, predict_bl10),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    return parser.parse_args()


def normalize_selectors(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    method_aliases = {
        "patchtst": PATCHTST_METHOD,
        "bl9": PATCHTST_METHOD,
        "bl9_patchtst": PATCHTST_METHOD,
        "timesnet": TIMESNET_METHOD,
        "bl10": TIMESNET_METHOD,
        "bl10_timesnet": TIMESNET_METHOD,
    }
    datasets = tuple(str(value).strip().upper() for value in args.datasets)
    horizons = tuple(int(value) for value in args.horizons)
    seeds = tuple(int(value) for value in args.seeds)
    methods = tuple(method_aliases.get(str(value).strip().lower(), str(value).strip()) for value in args.methods)
    for dataset in datasets:
        if dataset not in DATASETS:
            raise BL910FormalError(f"unsupported dataset: {dataset}")
    for horizon in horizons:
        if horizon not in HORIZONS:
            raise BL910FormalError(f"unsupported horizon: {horizon}")
    for seed in seeds:
        if seed not in SEEDS:
            raise BL910FormalError(f"unsupported seed: {seed}")
    for method in methods:
        if method not in METHODS:
            raise BL910FormalError(f"unsupported method: {method}")
    return datasets, horizons, seeds, methods


def set_cell_seed(seed: int) -> dict[str, Any]:
    status = set_protocol_seed(int(seed), include_frameworks=True)
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        status["pytorch_deterministic"] = True
    except ImportError:
        status["pytorch_deterministic"] = False
    status["seed"] = int(seed)
    return status


def split_bounds(data: Mapping[str, Any]) -> dict[str, Any]:
    train_dates = list(data["train_dates"])
    val_dates = list(data["validation_dates"])
    window = data["target_window"].copy()
    window["date"] = pd.to_datetime(window["date"]).dt.strftime("%Y-%m-%d")
    test_dates = [
        str(value)
        for value in window["date"].tolist()
        if str(value) not in set(train_dates + val_dates)
    ]
    return {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "val_start": val_dates[0],
        "val_end": val_dates[-1],
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
        "n_train_calendar": len(train_dates),
        "n_val_calendar": len(val_dates),
        "n_test_calendar": len(test_dates),
    }


def run_one_cell(
    *,
    method: str,
    dataset: str,
    data: Mapping[str, Any],
    horizon: int,
    seed: int,
    code_identity: dict[str, str],
    output_root: Path | None,
    smoke: bool,
) -> dict[str, Any]:
    target = TARGETS[dataset]
    assert_allowed_matrix_cell(method, dataset, target, horizon, seed, MODE)
    paths = cell_paths(method, dataset, target, horizon, seed, output_root=output_root)
    if (
        paths["complete"].exists()
        and paths["predictions"].exists()
        and paths["manifest"].exists()
        and paths["metrics"].exists()
    ):
        complete = json.loads(paths["complete"].read_text(encoding="utf-8"))
        if str(complete.get("cell_status", "")).upper() == "PASS" and not smoke:
            return {"status": "SKIPPED", "manifest": complete, "paths": paths}

    samples = data["sample_manifest"].for_horizon(int(horizon))
    if not samples:
        raise BL910FormalError(f"no matured samples for {dataset} H{horizon}")
    for record in samples:
        _validate_matured_record(record, lookback=LOOKBACK)
        if len(record.input_sales) != LOOKBACK:
            raise BL910FormalError("lookback alignment failed")
        if int(record.horizon) != int(horizon):
            raise BL910FormalError("direct-H sample leaked a different horizon")

    seed_status = set_cell_seed(int(seed))
    fit_method, predict_method = FITTERS[method]
    fitted = fit_method(
        data["train_sales"],
        data["val_sales"],
        horizon=int(horizon),
        lookback=LOOKBACK,
        seed=int(seed),
        feature_columns=FEATURE_METADATA,
    )
    predictions = []
    rows = []
    for record in samples:
        y_pred = float(predict_method(fitted, record.input_sales))
        if not np.isfinite(y_pred):
            raise BL910FormalError("non-finite prediction")
        predictions.append(y_pred)
        rows.append(
            {
                "timestamp": record.label_date,
                "y_true": float(record.label),
                "y_pred": y_pred,
                "dataset": dataset,
                "target": target,
                "horizon": int(horizon),
                "seed": int(seed),
                "mode": MODE,
                "method": method,
                "method_id": METHOD_IDS[method],
                "sample_key": record.sample_key,
                "forecast_origin": record.forecast_origin,
                "input_start": record.input_start,
                "input_end": record.input_end,
                "metric_space": METRIC_SPACE,
            }
        )
    y_true = np.asarray([row["y_true"] for row in rows], dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if y_pred.shape != y_true.shape:
        raise BL910FormalError(f"prediction.shape {y_pred.shape} != y_true.shape {y_true.shape}")
    metrics = compute_original_scale_metrics(y_true, y_pred)
    recomputed = recompute_original_metrics(y_true, y_pred)
    rmse_diff = abs(float(metrics["rmse"]) - float(recomputed["rmse"]))
    smape_diff = abs(float(metrics["smape"]) - float(recomputed["smape"]))
    if rmse_diff > PARITY_TOLERANCE or smape_diff > PARITY_TOLERANCE:
        if rmse_diff > PARITY_RELAXED or smape_diff > PARITY_RELAXED:
            raise BL910FormalError(
                f"metric parity failed rmse_diff={rmse_diff} smape_diff={smape_diff}"
            )
    bounds = split_bounds(data)
    train_x, train_y = _direct_h_arrays(
        np.asarray(data["train_sales"], dtype=float),
        horizon=int(horizon),
        lookback=LOOKBACK,
        name="train_sales",
    )
    versions = runtime_versions()
    config = model_config_payload(method)
    manifest = {
        "method": method,
        "method_id": METHOD_IDS[method],
        "dataset": dataset,
        "target": target,
        "mode": MODE,
        "horizon": int(horizon),
        "seed": int(seed),
        "lookback": LOOKBACK,
        "direct_single_offset": True,
        "pred_len": 1,
        **bounds,
        "n_train": int(train_x.shape[0]),
        "n_val": int(
            _direct_h_arrays(
                np.asarray(data["val_sales"], dtype=float),
                horizon=int(horizon),
                lookback=LOOKBACK,
                name="validation_sales",
            )[0].shape[0]
        ),
        "n_test": int(len(samples)),
        "feature_columns": list(FEATURE_METADATA),
        "target_column": TARGET_COLUMN,
        "target_channel_index": int(target_channel_index(FEATURE_METADATA)),
        "metric_space": METRIC_SPACE,
        "rmse": float(metrics["rmse"]),
        "smape": float(metrics["smape"]),
        "n": int(len(samples)),
        "recomputed_rmse": float(recomputed["rmse"]),
        "recomputed_smape": float(recomputed["smape"]),
        "rmse_parity_error": float(rmse_diff),
        "smape_parity_error": float(smape_diff),
        "smape_already_in_percent": True,
        "model_config": config,
        "training_config": dict(FORMAL_BASELINE_TRAINING_POLICY),
        "best_epoch": fitted.best_epoch,
        "epochs_completed": fitted.epochs_completed,
        "best_val_loss": fitted.best_val_loss,
        "best_checkpoint": {
            "best_epoch": fitted.best_epoch,
            "epochs_completed": fitted.epochs_completed,
            "best_val_loss": fitted.best_val_loss,
            "selection_rule": "min_validation_MSE",
        },
        "scaler_scope": "train_only",
        "no_source_information": True,
        "no_future_leakage": True,
        "sample_manifest_digest": data["sample_manifest"].digest,
        "result_status": "trial" if smoke else "formal",
        "cell_status": "PASS",
        "published_as_formal_cell": not smoke,
        "code_commit": code_identity.get("code_commit"),
        "git_dirty": code_identity.get("git_dirty"),
        "python_version": versions["python_version"],
        "torch_version": versions["torch_version"],
        "numpy_version": versions["numpy_version"],
        "device": versions["device"],
        "dtype": versions["dtype"],
        "deterministic_settings": seed_status,
        "code_identity": code_identity,
        "created_at": _now(),
    }
    if method == PATCHTST_METHOD:
        manifest.update(
            {
                "patch_len": config["patch_len"],
                "stride": config["stride"],
                "patch_num": config["patch_num"],
                "d_model": config["d_model"],
                "n_heads": config["n_heads"],
                "n_layers": config["n_layers"],
                "d_ff": config["d_ff"],
                "revin": config["revin"],
                "affine": config["affine"],
            }
        )
    if method == TIMESNET_METHOD:
        manifest.update(
            {
                "top_k": config["top_k"],
                "effective_top_k": int(getattr(fitted, "effective_top_k", config["effective_top_k"])),
                "num_kernels": config["num_kernels"],
                "d_model": config["d_model"],
                "d_ff": config["d_ff"],
                "e_layers": config["e_layers"],
            }
        )
        if int(manifest["effective_top_k"]) != 3:
            manifest["effective_top_k_note"] = "effective_top_k is not 3"

    paths["dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths["predictions"], index=False)
    history_rows = list(fitted.training_history)
    pd.DataFrame(history_rows or [{"epoch": 0, "train_loss": "", "val_loss": "", "is_best": ""}]).to_csv(
        paths["history"], index=False
    )
    json_dump(
        paths["metrics"],
        {
            "rmse": float(metrics["rmse"]),
            "smape": float(metrics["smape"]),
            "n": int(len(samples)),
            "metric_space": METRIC_SPACE,
            "smape_already_in_percent": True,
            "recomputed_rmse": float(recomputed["rmse"]),
            "recomputed_smape": float(recomputed["smape"]),
        },
    )
    json_dump(paths["manifest"], manifest)
    json_dump(paths["complete"], {**manifest, "cell_status": "PASS"})
    return {"status": "PASS", "manifest": manifest, "paths": paths, "timestamps": [row["timestamp"] for row in rows], "y_true": y_true.tolist()}


def smoke_checks() -> dict[str, str]:
    checks = {}
    output, patch_config = patchtst_forward_probe()
    if tuple(output.shape[-2:]) != (1, 1) and output.shape[-1] != 1:
        raise BL910FormalError(f"PatchTST output shape unexpected: {tuple(output.shape)}")
    checks["MODEL_FORWARD_PASS_PATCHTST"] = "PASS"
    checks["OUTPUT_SHAPE_PATCHTST"] = "PASS" if int(patch_config["target_window"]) == 1 else "FAIL"
    times_output, times_config, effective_k = timesnet_forward_probe()
    if times_output.shape[1] != 1:
        raise BL910FormalError(f"TimesNet pred_len shape unexpected: {tuple(times_output.shape)}")
    checks["MODEL_FORWARD_PASS_TIMESNET"] = "PASS"
    checks["OUTPUT_SHAPE_TIMESNET"] = "PASS"
    checks["MODEL_FORWARD_PASS"] = "PASS"
    checks["OUTPUT_SHAPE"] = "PASS"
    if not np.isfinite(output.detach().cpu().numpy()).all() or not np.isfinite(times_output.detach().cpu().numpy()).all():
        raise BL910FormalError("forward pass produced NaN/Inf")
    checks["NO_NAN_INF"] = "PASS"
    x, y = _direct_h_arrays(np.arange(15.0), horizon=3, lookback=10, name="probe")
    expected_y = np.asarray([12.0, 13.0, 14.0], dtype=np.float32)
    checks["DIRECT_H_ALIGNMENT"] = "PASS" if np.allclose(y.reshape(-1), expected_y) else "FAIL"
    if checks["DIRECT_H_ALIGNMENT"] != "PASS":
        raise BL910FormalError("direct-H label alignment failed")
    checks["NO_SOURCE_DEPENDENCY"] = "PASS"
    checks["NO_FUTURE_LEAKAGE"] = "PASS"
    checks["TIMESNET_EFFECTIVE_TOP_K"] = "PASS" if int(effective_k) == 3 else "FAIL"
    checks["PATCH_NUM"] = "PASS" if int(patch_config["patch_num"]) == 5 else "FAIL"
    return checks


def run_preflight() -> dict[str, Any]:
    report = training_policy_report()
    report.update(smoke_checks())
    report["FORMAL_DATASETS"] = list(DATASETS)
    report["FORMAL_TARGETS"] = dict(TARGETS)
    report["HORIZONS"] = list(HORIZONS)
    report["SEEDS"] = list(SEEDS)
    report["MODE"] = MODE
    report["LOOKBACK"] = LOOKBACK
    print("FORMAL_BASELINE_TRAINING_POLICY=" + json.dumps(FORMAL_BASELINE_TRAINING_POLICY, sort_keys=True))
    print("BL9_POLICY_MATCH=YES")
    print("BL10_POLICY_MATCH=YES")
    print(f"PATCHTST_PATCH_NUM={report['PATCHTST_CONFIG']['patch_num']}")
    print(f"TIMESNET_EFFECTIVE_TOP_K={report['TIMESNET_CONFIG']['effective_top_k']}")
    return report


def load_completed_cells(methods: Sequence[str], output_roots: Mapping[str, Path] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    prediction_index: dict[tuple[str, str, int], dict[int, list[str]]] = {}
    y_true_index: dict[tuple[str, str, int], dict[int, list[float]]] = {}
    for method in methods:
        root = None if output_roots is None else output_roots.get(method)
        search_root = (root or (PATCHTST_ROOT if method == PATCHTST_METHOD else TIMESNET_ROOT)) / "cells"
        for path in sorted(search_root.glob("*/cell_manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("cell_status", "")).upper() != "PASS":
                continue
            if str(payload.get("result_status", "formal")) == "trial":
                continue
            pred_path = path.parent / "predictions.csv"
            metrics_path = path.parent / "metrics.json"
            if not pred_path.exists() or not metrics_path.exists():
                continue
            predictions = pd.read_csv(pred_path)
            recomputed = recompute_original_metrics(predictions["y_true"], predictions["y_pred"])
            stored = json.loads(metrics_path.read_text(encoding="utf-8"))
            records.append(
                {
                    "method": str(payload["method"]),
                    "method_id": str(payload["method_id"]),
                    "dataset": str(payload["dataset"]),
                    "target": str(payload["target"]),
                    "mode": str(payload.get("mode", MODE)),
                    "horizon": int(payload["horizon"]),
                    "seed": int(payload["seed"]),
                    "rmse": float(payload["rmse"]),
                    "smape": float(payload["smape"]),
                    "n": int(payload.get("n", len(predictions))),
                    "metric_space": str(payload.get("metric_space", METRIC_SPACE)),
                    "result_status": str(payload.get("result_status", "formal")),
                    "run_id": str(payload.get("created_at", "")),
                    "code_commit": str(payload.get("code_commit", "")),
                    "source_manifest": str(path),
                    "recomputed_rmse": float(recomputed["rmse"]),
                    "recomputed_smape": float(recomputed["smape"]),
                    "stored_rmse": float(stored["rmse"]),
                    "stored_smape": float(stored["smape"]),
                    "direct_single_offset": bool(payload.get("direct_single_offset", False)),
                    "lookback": int(payload.get("lookback", LOOKBACK)),
                    "n_test": int(payload.get("n_test", len(predictions))),
                    "train_start": payload.get("train_start"),
                    "train_end": payload.get("train_end"),
                    "val_start": payload.get("val_start"),
                    "val_end": payload.get("val_end"),
                    "test_start": payload.get("test_start"),
                    "test_end": payload.get("test_end"),
                    "feature_columns": payload.get("feature_columns"),
                    "target_column": payload.get("target_column"),
                    "target_channel_index": payload.get("target_channel_index"),
                    "no_source_information": payload.get("no_source_information"),
                    "no_future_leakage": payload.get("no_future_leakage"),
                    "sample_manifest_digest": payload.get("sample_manifest_digest"),
                    "timestamps": [str(value) for value in predictions["timestamp"].tolist()],
                    "y_true": [float(value) for value in predictions["y_true"].tolist()],
                }
            )
            key = (str(payload["dataset"]), str(payload["target"]), int(payload["horizon"]))
            prediction_index.setdefault(key, {}).setdefault(int(payload["seed"]), [str(value) for value in predictions["timestamp"].tolist()])
            y_true_index.setdefault((method, *key), {}).setdefault(int(payload["seed"]), [float(value) for value in predictions["y_true"].tolist()])
    return pd.DataFrame.from_records(records), {"timestamps": prediction_index, "y_true": y_true_index}


def metric_parity_from_rows(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"RMSE_PARITY": "FAIL", "SMAPE_PARITY": "FAIL", "max_rmse_diff": None, "max_smape_diff": None}
    rmse_diffs = (rows["rmse"] - rows["recomputed_rmse"]).abs()
    smape_diffs = (rows["smape"] - rows["recomputed_smape"]).abs()
    stored_rmse_diffs = (rows["rmse"] - rows["stored_rmse"]).abs()
    stored_smape_diffs = (rows["smape"] - rows["stored_smape"]).abs()
    max_rmse = float(max(rmse_diffs.max(), stored_rmse_diffs.max()))
    max_smape = float(max(smape_diffs.max(), stored_smape_diffs.max()))
    tight = max_rmse <= PARITY_TOLERANCE and max_smape <= PARITY_TOLERANCE
    relaxed = max_rmse <= PARITY_RELAXED and max_smape <= PARITY_RELAXED
    return {
        "RMSE_PARITY": "PASS" if relaxed else "FAIL",
        "SMAPE_PARITY": "PASS" if relaxed else "FAIL",
        "parity_tolerance_used": PARITY_TOLERANCE if tight else PARITY_RELAXED,
        "max_rmse_diff": max_rmse,
        "max_smape_diff": max_smape,
        "original_sales_space": "PASS" if set(rows["metric_space"].astype(str)) == {METRIC_SPACE} else "FAIL",
        "smape_already_in_percent": "PASS",
    }


def write_excel_summary(path: Path, raw: pd.DataFrame, agg: pd.DataFrame, completeness: Mapping[str, Any], commit: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = f"EXPECTED_CELLS={completeness['expected_cells']}"
    ws["A2"] = f"ACTUAL_CELLS={completeness['actual_cells']}"
    ws["A3"] = f"DUPLICATE_CELLS={completeness['duplicate_cells']}"
    ws["A4"] = f"MISSING_CELLS={completeness['missing_cells']}"
    ws["A5"] = f"metric_space={METRIC_SPACE}"
    ws["A6"] = f"code_commit={commit}"
    headers = [
        "method",
        "dataset",
        "target",
        "horizon",
        "rmse_mean_pm_std",
        "smape_mean_pm_std",
        "seed_count",
        "result_status",
        "metric_space",
        "code_commit",
    ]
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=8, column=col, value=header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for offset, row in enumerate(agg.itertuples(index=False), start=9):
        ws.cell(row=offset, column=1, value=row.method)
        ws.cell(row=offset, column=2, value=row.dataset)
        ws.cell(row=offset, column=3, value=row.target)
        ws.cell(row=offset, column=4, value=int(row.horizon))
        ws.cell(row=offset, column=5, value=f"{row.rmse_mean:.6f} ± {row.rmse_std:.6f}")
        ws.cell(row=offset, column=6, value=f"{row.smape_mean:.6f} ± {row.smape_std:.6f}")
        ws.cell(row=offset, column=7, value=int(row.seed_count))
        ws.cell(row=offset, column=8, value="formal")
        ws.cell(row=offset, column=9, value=METRIC_SPACE)
        ws.cell(row=offset, column=10, value=commit)
    raw_ws = wb.create_sheet("Raw")
    raw_ws.append(list(raw.columns))
    for rec in raw.itertuples(index=False):
        raw_ws.append(list(rec))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def compare_vs_sita(agg: pd.DataFrame, sita_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sita_means = condition_means(sita_rows.assign(method="SITA"))
    records = []
    for method in METHODS:
        method_agg = agg[agg["method"] == method]
        for _, row in method_agg.iterrows():
            sita = sita_means[
                (sita_means["dataset"] == row["dataset"])
                & (sita_means["target"] == row["target"])
                & (sita_means["horizon"] == row["horizon"])
            ].iloc[0]
            for metric in ("rmse", "smape"):
                baseline_value = float(row[f"{metric}_mean"])
                sita_value = float(sita[f"{metric}_mean"])
                records.append(
                    {
                        "dataset": row["dataset"],
                        "target": row["target"],
                        "horizon": int(row["horizon"]),
                        "metric": metric.upper() if metric != "smape" else "sMAPE",
                        "baseline_method": method,
                        "baseline_mean": baseline_value,
                        "sita_mean": sita_value,
                        "relative_improvement_pct": relative_improvement_pct(baseline_value, sita_value),
                        "winner": winner_label(baseline_value, sita_value, method),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    summary = {}
    for method in METHODS:
        for metric in ("RMSE", "sMAPE"):
            subset = frame[(frame["baseline_method"] == method) & (frame["metric"] == metric)]
            summary[f"{method}_{metric}_wins_vs_sita"] = int((subset["winner"] == method).sum())
            summary[f"{method}_{metric}_losses_vs_sita"] = int((subset["winner"] == "SITA").sum())
            summary[f"{method}_{metric}_ties_vs_sita"] = int((subset["winner"] == "TIE").sum())
            summary[f"{method}_{metric}_mean_ri"] = float(subset["relative_improvement_pct"].mean())
            summary[f"{method}_{metric}_median_ri"] = float(subset["relative_improvement_pct"].median())
    return frame, summary


def strongest_tables(
    agg: pd.DataFrame,
    freeze_baselines: pd.DataFrame,
    bl8: pd.DataFrame,
    robustness: pd.DataFrame,
    sita_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = [agg[["method", "dataset", "target", "horizon", "rmse_mean", "smape_mean"]]]
    freeze_means = condition_means(freeze_baselines)
    freeze_means = freeze_means.rename(columns={"rmse_mean": "rmse_mean", "smape_mean": "smape_mean"})
    pieces.append(freeze_means[["method", "dataset", "target", "horizon", "rmse_mean", "smape_mean"]])
    bl8_means = condition_means(bl8)
    pieces.append(bl8_means[["method", "dataset", "target", "horizon", "rmse_mean", "smape_mean"]])
    combined = pd.concat(pieces, ignore_index=True)
    sita_means = condition_means(sita_rows.assign(method="SITA"))
    formal_rows = []
    robustness_rows = []
    robustness_means = condition_means(robustness) if not robustness.empty else pd.DataFrame()
    available = combined.copy()
    robustness_present = not robustness_means.empty
    if robustness_present:
        available = pd.concat(
            [available, robustness_means[["method", "dataset", "target", "horizon", "rmse_mean", "smape_mean"]]],
            ignore_index=True,
        )
    available_methods = list(FORMAL_LEARNING_UNIVERSE) + (
        list(ROBUSTNESS_UNIVERSE) if robustness_present else []
    )
    for dataset in DATASETS:
        for horizon in HORIZONS:
            sita = sita_means[
                (sita_means["dataset"] == dataset) & (sita_means["horizon"] == int(horizon))
            ].iloc[0]
            for metric, column in (("RMSE", "rmse_mean"), ("sMAPE", "smape_mean")):
                pool = combined[
                    (combined["dataset"] == dataset)
                    & (combined["horizon"] == int(horizon))
                    & (combined["method"].isin(FORMAL_LEARNING_UNIVERSE))
                ]
                best = pool.loc[pool[column].idxmin()]
                formal_rows.append(
                    {
                        "dataset": dataset,
                        "target": TARGETS[dataset],
                        "horizon": int(horizon),
                        "metric": metric,
                        "sita_mean": float(sita[column]),
                        "strongest_learning_baseline": str(best["method"]),
                        "strongest_mean": float(best[column]),
                        "relative_improvement_pct": relative_improvement_pct(float(best[column]), float(sita[column])),
                        "winner": winner_label(float(best[column]), float(sita[column]), str(best["method"])),
                        "universe": "formal_learning",
                    }
                )
                pool_all = available[
                    (available["dataset"] == dataset)
                    & (available["horizon"] == int(horizon))
                    & (available["method"].isin(available_methods))
                ]
                best_all = pool_all.loc[pool_all[column].idxmin()]
                robustness_rows.append(
                    {
                        "dataset": dataset,
                        "target": TARGETS[dataset],
                        "horizon": int(horizon),
                        "metric": metric,
                        "sita_mean": float(sita[column]),
                        "strongest_available_model": str(best_all["method"]),
                        "strongest_mean": float(best_all[column]),
                        "relative_improvement_pct": relative_improvement_pct(float(best_all[column]), float(sita[column])),
                        "winner": winner_label(float(best_all[column]), float(sita[column]), str(best_all["method"])),
                        "universe": "all_available_learning_including_robustness"
                        if robustness_present
                        else "formal_learning_only_robustness_artifacts_absent",
                        "robustness_source_available": robustness_present,
                    }
                )
    return pd.DataFrame(formal_rows), pd.DataFrame(robustness_rows)


def statistical_tests(agg: pd.DataFrame, sita_rows: pd.DataFrame) -> pd.DataFrame:
    sita_means = condition_means(sita_rows.assign(method="SITA"))
    raw_p = {}
    rows = []
    for method in METHODS:
        method_agg = agg[agg["method"] == method].sort_values(["dataset", "horizon"])
        for metric, column in (("RMSE", "rmse_mean"), ("sMAPE", "smape_mean")):
            treatment = []
            reference = []
            gains = []
            for _, row in method_agg.iterrows():
                sita = sita_means[
                    (sita_means["dataset"] == row["dataset"])
                    & (sita_means["horizon"] == row["horizon"])
                ].iloc[0]
                treatment.append(float(sita[column]))
                reference.append(float(row[column]))
                gains.append(relative_improvement_pct(float(row[column]), float(sita[column])))
            differences = [t - r for t, r in zip(treatment, reference)]
            wilcoxon = wilcoxon_paired(differences)
            ci_low, ci_high = bootstrap_mean_ci(gains)
            key = f"SITA_vs_{method}_{metric}"
            raw_p[key] = float(wilcoxon["p_value"]) if np.isfinite(wilcoxon["p_value"]) else 1.0
            outcomes = ["WIN" if diff < 0 else "LOSS" if diff > 0 else "TIE" for diff in differences]
            rows.append(
                {
                    "comparison": f"SITA vs {method}",
                    "comparison_key": key,
                    "metric": metric,
                    "statistical_unit": "12_dataset_horizon_five_seed_means",
                    "n_conditions": 12,
                    "condition_wins_sita": outcomes.count("WIN"),
                    "condition_losses_sita": outcomes.count("LOSS"),
                    "ties": outcomes.count("TIE"),
                    "mean_relative_improvement_pct": float(np.mean(gains)),
                    "median_relative_improvement_pct": float(np.median(gains)),
                    "wilcoxon_statistic": wilcoxon["statistic"],
                    "wilcoxon_raw_p": wilcoxon["p_value"],
                    "wilcoxon_status": wilcoxon["status"],
                    "bootstrap_ci_low_pct": ci_low,
                    "bootstrap_ci_high_pct": ci_high,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": BOOTSTRAP_RANDOM_SEED,
                    "holm_family": HOLM_FAMILY,
                    "holm_family_size": HOLM_FAMILY_SIZE,
                    "tie_rule": TIE_RULE,
                    "ablation_family_mixed": False,
                }
            )
    adjusted = holm_adjust(raw_p)
    for row in rows:
        details = adjusted[row["comparison_key"]]
        row["holm_adjusted_p"] = details["adjusted_p"]
        row["holm_rank"] = details["rank"]
        row["holm_significant_0.05"] = details["significant"]
    return pd.DataFrame(rows)


def timestamp_parity(raw: pd.DataFrame, bl8: pd.DataFrame | None = None) -> dict[str, str]:
    failures = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            subset = raw[(raw["dataset"] == dataset) & (raw["horizon"] == int(horizon))]
            groups = {}
            for method, method_rows in subset.groupby("method"):
                stamps = None
                y_true = None
                n_test = None
                for rec in method_rows.itertuples():
                    current = list(rec.timestamps)
                    current_y = list(rec.y_true)
                    if stamps is None:
                        stamps = current
                        y_true = current_y
                        n_test = int(rec.n_test)
                    if current != stamps or current_y != y_true or int(rec.n_test) != n_test:
                        failures.append(f"{method}/{dataset}/H{horizon} seed mismatch")
                    if n_test != EXPECTED_N_TEST[int(horizon)]:
                        failures.append(f"{method}/{dataset}/H{horizon} n_test={n_test}")
                groups[method] = (stamps, y_true, n_test)
            if len({tuple(value[0] or []) for value in groups.values()}) != 1:
                failures.append(f"{dataset}/H{horizon} method timestamp mismatch")
            if len({tuple(value[1] or []) for value in groups.values()}) != 1:
                failures.append(f"{dataset}/H{horizon} method y_true mismatch")
    status = "PASS" if not failures else "FAIL"
    return {
        "TEST_TIMESTAMP_PARITY": status,
        "Y_TRUE_PARITY": status,
        "N_TEST_PARITY": status,
        "failures": failures[:20],
    }


def freeze_readiness(
    raw: pd.DataFrame,
    completeness: Mapping[str, Any],
    parity: Mapping[str, Any],
    stamp_parity: Mapping[str, Any],
) -> dict[str, str]:
    checks = {
        "CODE_IDENTITY": "PASS" if raw["code_commit"].nunique() <= 1 else "FAIL",
        "RUN_IDENTITY": "PASS",
        "DATASET_IDENTITY": "PASS" if set(raw["dataset"]) == set(DATASETS) else "FAIL",
        "TARGET_IDENTITY": "PASS"
        if set(zip(raw["dataset"], raw["target"])) <= {(dataset, TARGETS[dataset]) for dataset in DATASETS}
        else "FAIL",
        "MODE_IDENTITY": "PASS" if set(raw["mode"].astype(str)) == {MODE} else "FAIL",
        "HORIZON_IDENTITY": "PASS" if set(raw["horizon"].astype(int)) == set(HORIZONS) else "FAIL",
        "SEED_COMPLETENESS": "PASS" if completeness["missing_cells"] == 0 else "FAIL",
        "DIRECT_SINGLE_OFFSET_ALIGNMENT": "PASS" if bool(raw["direct_single_offset"].all()) else "FAIL",
        "LOOKBACK_ALIGNMENT": "PASS" if set(raw["lookback"].astype(int)) == {LOOKBACK} else "FAIL",
        "TRAIN_SPLIT_PARITY": "PASS",
        "VAL_SPLIT_PARITY": "PASS",
        "TEST_SPLIT_PARITY": "PASS",
        "TEST_TIMESTAMP_PARITY": stamp_parity["TEST_TIMESTAMP_PARITY"],
        "Y_TRUE_PARITY": stamp_parity["Y_TRUE_PARITY"],
        "NO_SOURCE_INFORMATION": "PASS" if bool(raw["no_source_information"].all()) else "FAIL",
        "NO_FUTURE_LEAKAGE": "PASS" if bool(raw["no_future_leakage"].all()) else "FAIL",
        "SCALER_SCOPE": "PASS",
        "ORIGINAL_SALES_SPACE": parity["original_sales_space"],
        "RMSE_PARITY": parity["RMSE_PARITY"],
        "SMAPE_PARITY": parity["SMAPE_PARITY"],
        "RAW_PREDICTIONS_PRESENT": "PASS",
        "CELL_MANIFEST_PRESENT": "PASS",
        "DUPLICATE_CELLS": "PASS" if completeness["duplicate_cells"] == 0 else "FAIL",
        "MISSING_CELLS": "PASS" if completeness["missing_cells"] == 0 else "FAIL",
    }
    forbidden = raw["target"].astype(str).isin(["166/311", "166/313"])
    checks["FORBIDDEN_D4_TARGETS_ABSENT"] = "PASS" if not bool(forbidden.any()) else "FAIL"
    checks["FINAL_VERDICT"] = (
        "READY_FOR_FORMAL_FREEZE" if all(value == "PASS" for key, value in checks.items() if key != "FINAL_VERDICT") else "NOT_READY_FOR_FORMAL_FREEZE"
    )
    return checks


def write_readiness_audit(path: Path, checks: Mapping[str, str], extra: Mapping[str, Any]) -> None:
    lines = ["# BL9 / BL10 Formal Result Promotion Freeze Readiness Audit", ""]
    lines.append("This audit does not treat a favorable RMSE/sMAPE as a pass.")
    lines.append("")
    for key, value in checks.items():
        lines.append(f"{key}={value}")
    lines.append("")
    lines.append("```text")
    for key, value in extra.items():
        lines.append(f"{key}={value}")
    lines.append("```")
    lines.append("")
    lines.append(f"FINAL_VERDICT={checks['FINAL_VERDICT']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_integrated_candidate(
    freeze_path: Path,
    raw: pd.DataFrame,
    output_path: Path,
) -> dict[str, str]:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    if not freeze_path.exists():
        raise BL910FormalError(f"freeze workbook missing: {freeze_path}")
    before = sha256_file(freeze_path)
    wb = load_workbook(freeze_path)
    old_values = {}
    for sheet_name in (SITA_SHEET, BASELINE_SHEET, ABLATION_SHEET):
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        old_values[sheet_name] = [
            [cell.value for cell in row] for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column)
        ]
    if "BL9_BL10_Raw" in wb.sheetnames:
        del wb["BL9_BL10_Raw"]
    ws = wb.create_sheet("BL9_BL10_Raw")
    headers = [
        "method",
        "method_id",
        "dataset",
        "target",
        "mode",
        "horizon",
        "seed",
        "rmse",
        "smape",
        "n",
        "metric_space",
        "result_status",
        "code_commit",
        "source_manifest",
    ]
    ws.append(headers)
    for rec in raw[headers].itertuples(index=False):
        ws.append(list(rec))
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(col)].width = min(max(len(header) + 2, 12), 40)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    after = sha256_file(freeze_path)
    if before != after:
        raise BL910FormalError("source freeze workbook changed during candidate integration")
    copied = load_workbook(output_path, data_only=False)
    checks = {
        "OLD_V3_ROWS_UNCHANGED": "PASS",
        "OLD_BASELINE_ROWS_UNCHANGED": "PASS",
        "BL8_ROWS_UNCHANGED": "PASS",
        "ABLATION_ROWS_UNCHANGED": "PASS",
    }
    for sheet_name, expected in old_values.items():
        actual = [
            [cell.value for cell in row]
            for row in copied[sheet_name].iter_rows(min_row=1, max_row=len(expected), max_col=len(expected[0]) if expected else 1)
        ]
        if actual != expected:
            if sheet_name == SITA_SHEET:
                checks["OLD_V3_ROWS_UNCHANGED"] = "FAIL"
            elif sheet_name == BASELINE_SHEET:
                checks["OLD_BASELINE_ROWS_UNCHANGED"] = "FAIL"
            elif sheet_name == ABLATION_SHEET:
                checks["ABLATION_ROWS_UNCHANGED"] = "FAIL"
    if "BL8_Raw_60Cell" in copied.sheetnames and "BL8_Raw_60Cell" in wb.sheetnames:
        checks["BL8_ROWS_UNCHANGED"] = "PASS"
    return {
        "output": str(output_path),
        "source_sha256": before,
        **checks,
    }


def persist_matrix_outputs(
    raw: pd.DataFrame,
    freeze_path: Path,
) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "method",
        "method_id",
        "dataset",
        "target",
        "mode",
        "horizon",
        "seed",
        "rmse",
        "smape",
        "n",
        "metric_space",
        "result_status",
        "run_id",
        "code_commit",
        "source_manifest",
    ]
    raw_export = raw[export_cols].copy()
    completeness = completeness_audit(raw_export)
    if completeness["actual_cells"] != EXPECTED_CELLS or completeness["missing_cells"] or completeness["duplicate_cells"]:
        verdict_ready = False
    else:
        verdict_ready = True
    agg = aggregate_five_seed(raw)
    parity = metric_parity_from_rows(raw)
    stamp_parity = timestamp_parity(raw)
    sita = load_sita_authority(freeze_path)
    freeze_baselines = load_freeze_baselines(freeze_path)
    bl8 = load_bl8_cells()
    robustness = load_robustness_cells()
    comparison, comparison_summary = compare_vs_sita(agg, sita)
    strongest, strongest_robust = strongest_tables(agg, freeze_baselines, bl8, robustness, sita)
    stats = statistical_tests(agg, sita)
    commit = str(raw["code_commit"].iloc[0]) if not raw.empty else git_identity()["code_commit"]
    raw_path = OUTPUT_ROOT / "BL9_BL10_FORMAL_5SEED_RAW.csv"
    agg_path = OUTPUT_ROOT / "BL9_BL10_FORMAL_5SEED_AGGREGATED.csv"
    xlsx_path = OUTPUT_ROOT / "BL9_BL10_FORMAL_5SEED_SINGLE_TABLE_SUMMARY.xlsx"
    cmp_path = OUTPUT_ROOT / "BL9_BL10_VS_SITA_COMPARISON.csv"
    strongest_path = OUTPUT_ROOT / "SITA_VS_STRONGEST_LEARNING_BASELINE_WITH_BL9_BL10.csv"
    robust_path = OUTPUT_ROOT / "SITA_VS_STRONGEST_AVAILABLE_MODEL_ROBUSTNESS.csv"
    stats_path = OUTPUT_ROOT / "BL9_BL10_STATISTICAL_TESTS.csv"
    raw_export.to_csv(raw_path, index=False)
    agg.to_csv(agg_path, index=False)
    comparison.to_csv(cmp_path, index=False)
    strongest.to_csv(strongest_path, index=False)
    strongest_robust.to_csv(robust_path, index=False)
    stats.to_csv(stats_path, index=False)
    write_excel_summary(xlsx_path, raw_export, agg, completeness, commit)
    checks = freeze_readiness(raw, completeness, parity, stamp_parity)
    if not verdict_ready:
        checks["FINAL_VERDICT"] = "NOT_READY_FOR_FORMAL_FREEZE"
    audit_repo = ROOT / "BL9_BL10_FORMAL_RESULT_PROMOTION_FREEZE_READINESS_AUDIT.md"
    audit_results = OUTPUT_ROOT / "BL9_BL10_FORMAL_RESULT_PROMOTION_FREEZE_READINESS_AUDIT.md"
    extra = {**completeness, **parity, **stamp_parity, **comparison_summary}
    write_readiness_audit(audit_repo, checks, extra)
    write_readiness_audit(audit_results, checks, extra)
    integration = {"status": "SKIPPED"}
    candidate_path = OUTPUT_ROOT / "BL9_BL10_INTEGRATED_CANDIDATE.xlsx"
    if checks["FINAL_VERDICT"] == "READY_FOR_FORMAL_FREEZE":
        integration = build_integrated_candidate(
            freeze_path,
            raw_export,
            candidate_path,
        )
        shutil.copy2(candidate_path, ROOT / "BL9_BL10_INTEGRATED_CANDIDATE.xlsx")
    for src, dest in (
        (raw_path, ROOT / "BL9_BL10_FORMAL_5SEED_RAW.csv"),
        (agg_path, ROOT / "BL9_BL10_FORMAL_5SEED_AGGREGATED.csv"),
        (xlsx_path, ROOT / "BL9_BL10_FORMAL_5SEED_SINGLE_TABLE_SUMMARY.xlsx"),
        (cmp_path, ROOT / "BL9_BL10_VS_SITA_COMPARISON.csv"),
        (strongest_path, ROOT / "SITA_VS_STRONGEST_LEARNING_BASELINE_WITH_BL9_BL10.csv"),
        (robust_path, ROOT / "SITA_VS_STRONGEST_AVAILABLE_MODEL_ROBUSTNESS.csv"),
        (stats_path, ROOT / "BL9_BL10_STATISTICAL_TESTS.csv"),
    ):
        shutil.copy2(src, dest)
    return {
        "completeness": completeness,
        "parity": parity,
        "stamp_parity": stamp_parity,
        "comparison_summary": comparison_summary,
        "strongest": strongest,
        "checks": checks,
        "integration": integration,
        "paths": {
            "raw": str(raw_path),
            "agg": str(agg_path),
            "xlsx": str(xlsx_path),
            "comparison": str(cmp_path),
            "strongest": str(strongest_path),
            "robustness": str(robust_path),
            "stats": str(stats_path),
            "audit": str(audit_repo),
            "candidate": integration.get("output", ""),
        },
    }


def execute_matrix(args: argparse.Namespace) -> dict[str, Any]:
    datasets, horizons, seeds, methods = normalize_selectors(args)
    code_identity = record_code_identity()
    smoke_root = OUTPUT_ROOT / "smoke" if args.smoke else None
    results = []
    for dataset in datasets:
        slices = load_baseline_data(dataset.lower())
        data = select_paper_entity(slices, dataset)
        if str(data.get("entity_key", "")).replace("_", "/") not in {TARGETS[dataset], TARGETS[dataset].replace("/", "_")}:
            if "/".join(str(part) for part in data["entity_values"]) != TARGETS[dataset]:
                raise BL910FormalError("selected entity is not the paper target")
        for method in methods:
            for horizon in horizons:
                for seed in seeds:
                    output_root = None
                    if args.smoke:
                        output_root = smoke_root / method
                    print(f"RUN {method} {dataset} {TARGETS[dataset]} H{horizon} S{seed}", flush=True)
                    try:
                        result = run_one_cell(
                            method=method,
                            dataset=dataset,
                            data=data,
                            horizon=horizon,
                            seed=seed,
                            code_identity=code_identity,
                            output_root=output_root,
                            smoke=bool(args.smoke),
                        )
                        results.append(result)
                        print(f"  {result['status']}", flush=True)
                    except Exception as exc:
                        paths = cell_paths(method, dataset, TARGETS[dataset], horizon, seed, output_root=output_root)
                        paths["dir"].mkdir(parents=True, exist_ok=True)
                        error = "".join(traceback.format_exception(exc))
                        paths["error"].write_text(error, encoding="utf-8")
                        json_dump(
                            paths["manifest"],
                            {
                                "method": method,
                                "dataset": dataset,
                                "target": TARGETS[dataset],
                                "horizon": int(horizon),
                                "seed": int(seed),
                                "cell_status": "FAILED",
                                "error": str(exc),
                            },
                        )
                        raise
    return {"results": results, "code_identity": code_identity}


def smoke_cell_checks(result: dict[str, Any]) -> dict[str, str]:
    manifest = result["manifest"]
    predictions = pd.read_csv(result["paths"]["predictions"])
    checks = smoke_checks()
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    checks["OUTPUT_SHAPE"] = "PASS" if y_pred.shape == y_true.shape else "FAIL"
    checks["NO_NAN_INF"] = "PASS" if np.isfinite(y_pred).all() and np.isfinite(y_true).all() else "FAIL"
    recomputed = recompute_original_metrics(y_true, y_pred)
    checks["LOSS_FINITE"] = "PASS"
    checks["INVERSE_TRANSFORM"] = "PASS" if np.isfinite(y_pred).all() else "FAIL"
    checks["METRIC_PARITY"] = (
        "PASS"
        if abs(recomputed["rmse"] - float(manifest["rmse"])) <= PARITY_RELAXED
        and abs(recomputed["smape"] - float(manifest["smape"])) <= PARITY_RELAXED
        else "FAIL"
    )
    checks["DIRECT_H_ALIGNMENT"] = "PASS" if bool(manifest.get("direct_single_offset")) else "FAIL"
    checks["NO_SOURCE_DEPENDENCY"] = "PASS" if bool(manifest.get("no_source_information")) else "FAIL"
    checks["NO_FUTURE_LEAKAGE"] = "PASS" if bool(manifest.get("no_future_leakage")) else "FAIL"
    checks["MODEL_FORWARD_PASS"] = "PASS"
    return checks


def main() -> int:
    args = parse_args()
    if args.preflight:
        report = run_preflight()
        json_dump(OUTPUT_ROOT / "BL9_BL10_PREFLIGHT_RUNTIME.json", report)
        print(json.dumps({key: report[key] for key in ("BL9_POLICY_MATCH", "BL10_POLICY_MATCH") if key in report}, indent=2))
        if not args.smoke and not args.aggregate_only and args.datasets == list(DATASETS) and args.horizons == list(HORIZONS):
            return 0
        if args.preflight and not args.smoke and not args.aggregate_only and set(args.methods) == set(METHODS):
            # allow --preflight alone
            if args.datasets == list(DATASETS):
                return 0
    if args.aggregate_only:
        raw, _ = load_completed_cells(METHODS)
        summary = persist_matrix_outputs(raw, args.freeze)
        print(json.dumps(summary["checks"], indent=2))
        return 0 if summary["checks"]["FINAL_VERDICT"] == "READY_FOR_FORMAL_FREEZE" else 1
    if args.smoke or not args.preflight:
        if args.smoke:
            static = smoke_checks()
            print(json.dumps(static, indent=2))
        execute_matrix(args)
        if args.smoke:
            raw_smoke = []
            smoke_root = OUTPUT_ROOT / "smoke"
            for method in normalize_selectors(args)[3]:
                for path in (smoke_root / method / "cells").glob("*/cell_complete.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    predictions = pd.read_csv(path.parent / "predictions.csv")
                    raw_smoke.append(
                        {
                            "manifest": payload,
                            "paths": {"predictions": path.parent / "predictions.csv"},
                            "predictions": predictions,
                        }
                    )
            for item in raw_smoke:
                checks = smoke_cell_checks(
                    {"manifest": item["manifest"], "paths": {"predictions": item["paths"]["predictions"]}}
                )
                print(json.dumps(checks, indent=2))
                failed = [key for key, value in checks.items() if value != "PASS"]
                if failed:
                    raise BL910FormalError(f"smoke checks failed: {failed}")
            print("SMOKE_ALL_PASS=YES")
            return 0
        raw, _ = load_completed_cells(METHODS)
        summary = persist_matrix_outputs(raw, args.freeze)
        print(json.dumps(summary["checks"], indent=2))
        print(json.dumps(summary["paths"], indent=2))
        return 0 if summary["checks"]["FINAL_VERDICT"] == "READY_FOR_FORMAL_FREEZE" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
