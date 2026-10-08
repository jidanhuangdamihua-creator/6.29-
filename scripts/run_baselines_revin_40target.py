#!/usr/bin/env python3
"""D4/D5/D6 40-target +RevIN robustness runner (180-day protocol).

Does not modify main-table result files, SITA, MSML-TL, or the no-RevIN
baseline scripts. Resume via DONE.json. Failures go to
results/revin_baseline_failures.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from additional_target_robustness.baselines import partition_sales
from additional_target_robustness.horizon import verify_direct_horizon
from additional_target_robustness.paths import SEALED_ROOT, ensure_ablation_path, ensure_baseline_path
from additional_target_robustness.protocol import (
    HORIZONS,
    PAPER_TARGETS,
    SEEDS,
    TEST_DAYS,
    ProtocolError,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
)
from additional_target_robustness.target_data import (
    _columns,
    _read_filtered,
    load_target_series,
    sales_by_date,
    split_target_frame,
)

ensure_ablation_path()
ensure_baseline_path()
from sita_ablation.metrics import evaluate_original_sales

from bl4_lstm import fit_bl4
from bl4_lstm_revin import fit_bl4_revin
from bl6_dlinear import fit_bl6
from bl6_dlinear_revin import fit_bl6_revin
from bl7_tcn import fit_bl7
from bl7_tcn_revin import fit_bl7_revin
from bl10_timesnet import fit_bl10
from bl10_timesnet_revin import fit_bl10_revin

OUT_ROOT = REPO_ROOT / "outputs" / "baselines_revin_40target"
MANIFEST = (
    REPO_ROOT
    / "实验结果"
    / "非实验要用的结果"
    / "sita_d456_fixed40_headline_paper180swap"
    / "fixed40_manifest.csv"
)
RESULTS_CSV = REPO_ROOT / "results" / "baselines_revin_5seed.csv"
FAILURES_CSV = REPO_ROOT / "results" / "revin_baseline_failures.csv"
LOG_PATH = REPO_ROOT / "logs" / "revin_baselines_40target.log"
SMOKE_DATASET = "D5"
SMOKE_TARGET = "48/261700"
SMOKE_H = 1
SMOKE_SEED = 42
DATASET_ORDER = ("D5", "D6", "D4")
METHODS = (
    "DLinear_RevIN",
    "LSTM_RevIN",
    "TCN_RevIN",
    "TimesNet_RevIN",
)
METHOD_SLUGS = {
    "DLinear_RevIN": "dlinear_revin",
    "LSTM_RevIN": "lstm_revin",
    "TCN_RevIN": "tcn_revin",
    "TimesNet_RevIN": "timesnet_revin",
}
PARENT = {
    "DLinear_RevIN": "DLinear",
    "LSTM_RevIN": "LSTM",
    "TCN_RevIN": "TCN",
    "TimesNet_RevIN": "TimesNet",
}
FITTERS = {
    "DLinear_RevIN": fit_bl6_revin,
    "LSTM_RevIN": fit_bl4_revin,
    "TCN_RevIN": fit_bl7_revin,
    "TimesNet_RevIN": fit_bl10_revin,
}
PARENT_FITTERS = {
    "DLinear": fit_bl6,
    "LSTM": fit_bl4,
    "TCN": fit_bl7,
    "TimesNet": fit_bl10,
}
EXPECTED_ROWS = 4 * 3 * 40 * 3 * 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{_utc_now()} {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def mae_original_sales(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return float(np.mean(np.abs(pred - true)))


def load_identities(dataset: str) -> list[str]:
    frame = pd.read_csv(MANIFEST)
    ids = [
        str(row.target_identity)
        for row in frame.itertuples(index=False)
        if str(row.dataset) == dataset
    ]
    if len(ids) != 40 or len(set(ids)) != 40:
        raise RuntimeError(f"{dataset} 40-target list is not 40 unique identities: {len(ids)}")
    return ids


def _paper_target_path(dataset: str) -> Path:
    folder = {"D4": "dataset4", "D5": "dataset5", "D6": "dataset6"}[dataset]
    path = SEALED_ROOT / folder / "target.parquet"
    if not path.is_file():
        raise ProtocolError(f"missing {path}")
    return path


def load_series(dataset: str, identity: str) -> pd.DataFrame:
    if identity == PAPER_TARGETS[dataset]:
        calendar = calendar_for(dataset)
        date_max = calendar.test_end if dataset != "D5" else "2017-08-14"
        frame = _read_filtered(
            _paper_target_path(dataset),
            key_fields=calendar.key_fields,
            identity=identity,
            date_min=calendar.train_start,
            date_max=date_max,
            columns=_columns(calendar),
        )
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        if dataset == "D5":
            cutoff = pd.Timestamp("2017-08-14")
            frame = frame.loc[frame["date"] <= cutoff].copy()
        if frame.empty:
            raise ProtocolError(f"{dataset} {identity} missing from target.parquet")
        split_target_frame(frame, calendar)
        return frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    return load_target_series(dataset, identity)


def prepare_target(dataset: str, identity: str) -> dict:
    calendar = calendar_for(dataset)
    target = load_series(dataset, identity)
    train, validation, test = split_target_frame(target, calendar)
    if len(train) != 15 or len(validation) != 15 or len(test) != TEST_DAYS:
        raise ProtocolError(f"{dataset} {identity} split lengths")
    if dataset == "D5":
        cutoff = pd.Timestamp("2017-08-14")
        if (target["date"] > cutoff).any():
            raise ProtocolError(f"{dataset} {identity} has dates after 2017-08-14")
    return {
        "target": target,
        "sales": sales_by_date(target),
        "train_sales": partition_sales(target, calendar.train_start, calendar.train_end),
        "val_sales": partition_sales(target, calendar.val_start, calendar.val_end),
        "calendar": calendar,
        "test": test,
    }


def _predict_indexed(predict, index, sales: dict[str, float]) -> np.ndarray:
    predictions = np.empty(len(index.rows), dtype=np.float64)
    for position, row in enumerate(index.rows):
        visible = np.asarray(
            [sales[pd.Timestamp(day).strftime("%Y-%m-%d")] for day in row.input_dates],
            dtype=np.float64,
        )
        if visible.size != 10:
            raise ProtocolError("evaluation input is not a 10-day window")
        predictions[position] = float(predict(visible))
    return predictions


def cell_dir(dataset: str, method: str, identity: str, horizon: int, seed: int) -> Path:
    safe = identity.replace("/", "_")
    return OUT_ROOT / dataset / METHOD_SLUGS[method] / "cells" / f"{safe}__h{horizon}__s{seed}"


def cell_key(dataset: str, method: str, identity: str, horizon: int, seed: int, protocol_sha: str) -> str:
    payload = {
        "dataset": dataset,
        "method": method,
        "target": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "n_test": TEST_DAYS,
        "protocol_sha256": protocol_sha,
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def inspect_cell(directory: Path, expected_key: str) -> str:
    done = directory / "DONE.json"
    pred = directory / "predictions.csv"
    if not done.is_file() or not pred.is_file():
        return "MISS"
    payload = json.loads(done.read_text(encoding="utf-8"))
    if payload.get("cell_key") != expected_key or payload.get("status") != "COMPLETED":
        return "HASH_MISMATCH"
    frame = pd.read_csv(pred)
    if len(frame) != TEST_DAYS:
        return "N_TEST_MISMATCH"
    return "HIT"


def write_cell(
    directory: Path,
    *,
    cell_key_value: str,
    dataset: str,
    method: str,
    identity: str,
    horizon: int,
    seed: int,
    dates: tuple[str, ...],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protocol_sha: str,
    extra: dict | None = None,
) -> dict:
    if y_true.shape != (TEST_DAYS,) or y_pred.shape != (TEST_DAYS,) or len(dates) != TEST_DAYS:
        raise ProtocolError(f"{dataset} {method} n_test is not 180")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ProtocolError(f"{dataset} {method} non-finite arrays")
    pair = evaluate_original_sales(y_true, y_pred)
    mae = mae_original_sales(y_true, y_pred)
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = directory.with_name(directory.name + ".partial")
    if tmp_dir.exists():
        for child in sorted(tmp_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "dataset": dataset,
            "target": identity,
            "method": method,
            "h": int(horizon),
            "seed": int(seed),
            "target_date": list(dates),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(tmp_dir / "predictions.csv", index=False)
    record = {
        "cell_key": cell_key_value,
        "status": "COMPLETED",
        "dataset": dataset,
        "target": identity,
        "method": method,
        "h": int(horizon),
        "seed": int(seed),
        "n_test": TEST_DAYS,
        "RMSE": pair.rmse,
        "sMAPE": pair.smape,
        "MAE": mae,
        "protocol_sha256": protocol_sha,
        "trained_at": _utc_now(),
        **(extra or {}),
    }
    (tmp_dir / "DONE.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp_dir.replace(directory)
    return record


def append_failure(row: dict) -> None:
    FAILURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    header = not FAILURES_CSV.is_file()
    frame.to_csv(FAILURES_CSV, mode="a", header=header, index=False)


def count_params(fitted) -> int | None:
    model = getattr(fitted, "model", None)
    if model is None:
        return None
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_one(
    dataset: str,
    method: str,
    identity: str,
    horizon: int,
    seed: int,
    prepared: dict,
    protocol_sha: str,
    *,
    fitter=None,
) -> dict:
    index = build_evaluation_index(dataset, int(horizon))
    dates = index.label_date_text()
    y_true = np.asarray([prepared["sales"][day] for day in dates], dtype=np.float64)
    fit = FITTERS[method] if fitter is None else fitter
    fitted = fit(
        prepared["train_sales"],
        prepared["val_sales"],
        horizon=int(horizon),
        seed=int(seed),
    )
    y_pred = _predict_indexed(fitted.predict, index, prepared["sales"])
    extra = {"n_params": count_params(fitted)}
    directory = cell_dir(dataset, method, identity, horizon, seed)
    key = cell_key(dataset, method, identity, horizon, seed, protocol_sha)
    return write_cell(
        directory,
        cell_key_value=key,
        dataset=dataset,
        method=method,
        identity=identity,
        horizon=horizon,
        seed=seed,
        dates=dates,
        y_true=y_true,
        y_pred=y_pred,
        protocol_sha=protocol_sha,
        extra=extra,
    )


def iter_cells(datasets: tuple[str, ...], methods: tuple[str, ...]):
    for dataset in datasets:
        identities = load_identities(dataset)
        for identity in identities:
            for method in methods:
                for horizon in HORIZONS:
                    for seed in SEEDS:
                        yield dataset, identity, method, int(horizon), int(seed)


def assemble_results(datasets: tuple[str, ...] = DATASET_ORDER) -> pd.DataFrame:
    rows = []
    protocol_sha = frozen_protocol_sha256()
    for dataset, identity, method, horizon, seed in iter_cells(datasets, METHODS):
        directory = cell_dir(dataset, method, identity, horizon, seed)
        key = cell_key(dataset, method, identity, horizon, seed, protocol_sha)
        if inspect_cell(directory, key) != "HIT":
            continue
        payload = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": dataset,
                "target": identity,
                "method": method,
                "h": int(horizon),
                "seed": int(seed),
                "RMSE": float(payload["RMSE"]),
                "sMAPE": float(payload["sMAPE"]),
                "MAE": float(payload["MAE"]),
            }
        )
    frame = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(columns=["dataset", "target", "method", "h", "seed", "RMSE", "sMAPE", "MAE"])
    frame.to_csv(RESULTS_CSV, index=False)
    return frame


def run_smoke(methods: tuple[str, ...] = METHODS) -> dict:
    verify_direct_horizon()
    protocol_sha = frozen_protocol_sha256()
    prepared = prepare_target(SMOKE_DATASET, SMOKE_TARGET)
    smoke_rows = []
    for method in methods:
        parent = PARENT[method]
        revin_rec = run_one(
            SMOKE_DATASET,
            method,
            SMOKE_TARGET,
            SMOKE_H,
            SMOKE_SEED,
            prepared,
            protocol_sha,
        )
        parent_fitted = PARENT_FITTERS[parent](
            prepared["train_sales"],
            prepared["val_sales"],
            horizon=SMOKE_H,
            seed=SMOKE_SEED,
        )
        index = build_evaluation_index(SMOKE_DATASET, SMOKE_H)
        dates = index.label_date_text()
        y_true = np.asarray([prepared["sales"][day] for day in dates], dtype=np.float64)
        y_parent = _predict_indexed(parent_fitted.predict, index, prepared["sales"])
        parent_pair = evaluate_original_sales(y_true, y_parent)
        pred_path = cell_dir(SMOKE_DATASET, method, SMOKE_TARGET, SMOKE_H, SMOKE_SEED) / "predictions.csv"
        preds = pd.read_csv(pred_path)
        y_hat = preds["y_pred"].to_numpy(dtype=np.float64)
        smoke_rows.append(
            {
                "method": method,
                "parent": parent,
                "n_params_revin": revin_rec.get("n_params"),
                "n_params_parent": count_params(parent_fitted),
                "rmse_revin": revin_rec["RMSE"],
                "smape_revin": revin_rec["sMAPE"],
                "mae_revin": revin_rec["MAE"],
                "rmse_parent": parent_pair.rmse,
                "smape_parent": parent_pair.smape,
                "mae_parent": mae_original_sales(y_true, y_parent),
                "pred_nan": int(np.isnan(y_hat).sum()),
                "pred_min": float(np.min(y_hat)),
                "pred_max": float(np.max(y_hat)),
                "true_min": float(np.min(y_true)),
                "true_max": float(np.max(y_true)),
                "identical_to_parent": bool(np.allclose(y_hat, y_parent, rtol=0, atol=0)),
            }
        )
        _log(f"SMOKE {method} RMSE={revin_rec['RMSE']:.6g} parent={parent_pair.rmse:.6g}")
    return {"rows": smoke_rows, "dataset": SMOKE_DATASET, "target": SMOKE_TARGET}


def run_all(datasets: tuple[str, ...], methods: tuple[str, ...], *, limit: int | None = None) -> dict:
    verify_direct_horizon()
    protocol_sha = frozen_protocol_sha256()
    hits = misses = failures = 0
    prepared_cache: dict[tuple[str, str], dict] = {}
    n_done = 0
    for dataset, identity, method, horizon, seed in iter_cells(datasets, methods):
        if limit is not None and n_done >= limit:
            break
        directory = cell_dir(dataset, method, identity, horizon, seed)
        key = cell_key(dataset, method, identity, horizon, seed, protocol_sha)
        status = inspect_cell(directory, key)
        if status == "HIT":
            hits += 1
            continue
        try:
            cache_key = (dataset, identity)
            if cache_key not in prepared_cache:
                prepared_cache[cache_key] = prepare_target(dataset, identity)
                if len(prepared_cache) > 2:
                    oldest = next(iter(prepared_cache))
                    if oldest != cache_key:
                        prepared_cache.pop(oldest, None)
            run_one(dataset, method, identity, horizon, seed, prepared_cache[cache_key], protocol_sha)
            misses += 1
            n_done += 1
            _log(f"OK {dataset} {identity} {method} h={horizon} s={seed}")
        except Exception as exc:  # noqa: BLE001 - record and continue
            failures += 1
            n_done += 1
            detail = f"{type(exc).__name__}: {exc}"
            _log(f"FAIL {dataset} {identity} {method} h={horizon} s={seed} {detail}")
            append_failure(
                {
                    "dataset": dataset,
                    "target": identity,
                    "method": method,
                    "h": horizon,
                    "seed": seed,
                    "error": detail,
                    "traceback": traceback.format_exc()[-4000:],
                    "written_at": _utc_now(),
                }
            )
    frame = assemble_results(datasets)
    missing = EXPECTED_ROWS - len(frame) if set(datasets) == set(DATASET_ORDER) and set(methods) == set(METHODS) else None
    return {
        "hits": hits,
        "ran": misses,
        "failures": failures,
        "assembled_rows": int(len(frame)),
        "expected_rows": EXPECTED_ROWS,
        "missing_rows": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--datasets", default="D5,D6,D4")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if args.smoke:
        payload = run_smoke(methods)
        out = REPO_ROOT / "outputs" / "baselines_revin_40target" / "smoke.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.assemble:
        frame = assemble_results(datasets)
        print(f"assembled {len(frame)} rows -> {RESULTS_CSV}")
        return 0
    payload = run_all(datasets, methods, limit=args.limit)
    print(json.dumps(payload, indent=2))
    return 0 if int(payload["failures"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
