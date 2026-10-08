#!/usr/bin/env python3
"""SITA_SALES_ONLY / sita_no_covariates on the frozen 120-target main set.

Only the model input channel set changes: features = (sales,). Source identities
and fusion weights are read from the frozen 40-target main SITA workbooks.
Independent Stage A cache; does not write into main SITA or ATR caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "消融实验") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "消融实验"))

from additional_target_robustness.paths import SEALED_ROOT, ensure_ablation_path
from additional_target_robustness.protocol import (
    HORIZONS,
    LEARNING_RATES,
    PAPER_TARGETS,
    SEEDS,
    TEST_DAYS,
    ProtocolError,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
    sha256_file,
)
from additional_target_robustness.runner import (
    _one_source,
    _read_source_record,
    _series_sha256,
    _write_source_cache,
)
from additional_target_robustness.source_selection import load_source_histories
from additional_target_robustness.target_data import (
    _columns,
    _read_filtered,
    load_target_series,
    split_target_frame,
)

ensure_ablation_path()
from sita_ablation.adaptation import TargetWindows, WeightSnapshot, adapt_target
from sita_ablation.arms import RunSpec
from sita_ablation.metrics import evaluate_original_sales
from sita_ablation.models import build_target_cnn
from sita_ablation.revin import build_revin
from sita_ablation.scaling import fit_source_model_scaler, fit_target_model_scaler, transform_model_frame
from sita_ablation.source_selection import inverse_distance_weights
from sita_ablation.stage_a import NOT_APPLICABLE, SourceRecord, StageABundle, StageAKey, fit_source_cnn
from sita_ablation.windows import build_partition_windows, build_target_partition_windows, build_test_windows
from sita_ablation.data import split_source_frame

ARM = "SITA_SALES_ONLY"
SALES_ONLY = ("sales",)
FEATURE_SET = "SALES_ONLY"
OUT = REPO_ROOT / "outputs" / "sita_sales_only_ablation_20261007"
CACHE = OUT / "source_model_cache"
CELLS = OUT / "predictions"
MANIFEST = (
    REPO_ROOT
    / "实验结果"
    / "非实验要用的结果"
    / "sita_d456_fixed40_headline_paper180swap"
    / "fixed40_manifest.csv"
)
WORKBOOKS = {
    ds: REPO_ROOT / "实验结果" / f"{ds}_40target_BL_SITA.xlsx" for ds in ("D4", "D5", "D6")
}
DATASETS = ("D4", "D5", "D6")
EXPECTED_TARGETS = 120
EXPECTED_CELLS = 1800
HOLD_SET = "HOLD_MAIN_120_TARGET_SET_NOT_UNIQUELY_IDENTIFIED"
RECYCLE_AFTER = 15
SOURCE_HIT_ONLY = False
_HISTORY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
FEATURE_ROLES = {
    "sales": "target_sales",
    "year": "calendar",
    "month": "calendar",
    "week": "calendar",
    "day": "calendar",
    "weekday": "calendar",
    "wm_yr_wk": "calendar",
    "holiday_flag": "calendar_event",
    "is_holiday": "calendar_event",
    "is_event_1": "event",
    "is_event_2": "event",
    "snap": "SNAP",
    "stock_hour6_22_cnt": "operations",
    "activity_flag": "operations",
    "discount": "promotion_price",
    "precpt": "weather",
    "avg_temperature": "weather",
    "avg_humidity": "weather",
    "avg_wind_level": "weather",
    "class": "item_static",
    "perishable": "item_static",
    "cluster": "store_static",
    "transactions": "transactions",
    "oil_price": "exogenous_price",
    "sell_price": "sell_price",
    "onpromotion": "promotion_selection_only",
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    path = OUT / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{_utc()} {msg}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def mae_original(y_true, y_pred) -> float:
    t = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    p = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return float(np.mean(np.abs(p - t)))


def sales_only_protocol_sha256() -> str:
    payload = {
        "arm": ARM,
        "feature_set": FEATURE_SET,
        "model_features": list(SALES_ONLY),
        "main_protocol_sha256": frozen_protocol_sha256(),
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(MANIFEST),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_frozen_120() -> tuple[pd.DataFrame, dict, dict]:
    if not MANIFEST.is_file():
        raise ProtocolError(HOLD_SET)
    manifest = pd.read_csv(MANIFEST)
    for ds, path in WORKBOOKS.items():
        if not path.is_file():
            raise ProtocolError(HOLD_SET)
    rows = []
    for ds in DATASETS:
        man = manifest.loc[manifest["dataset"] == ds].sort_values("list_rank")
        ids_man = [str(x) for x in man["target_identity"].tolist()]
        targets = pd.read_excel(WORKBOOKS[ds], sheet_name="targets")
        ids_wb = [str(x) for x in targets["target_identity"].tolist()]
        if len(ids_man) != 40 or len(ids_wb) != 40 or ids_man != ids_wb:
            raise ProtocolError(HOLD_SET)
        cell = pd.read_excel(WORKBOOKS[ds], sheet_name="cell_raw")
        sita = cell.loc[cell["method"].astype(str) == "SITA"].copy()
        sita_ids = sorted(sita["target_identity"].astype(str).unique().tolist())
        if sita_ids != sorted(ids_wb) or len(sita) != 600:
            raise ProtocolError(HOLD_SET)
        for rank, ident in enumerate(ids_wb, start=1):
            sub = sita.loc[sita["target_identity"].astype(str) == ident]
            keys = [
                "source_1", "source_2", "source_3",
                "source_1_distance", "source_2_distance", "source_3_distance",
                "source_1_weight", "source_2_weight", "source_3_weight",
            ]
            uniq = sub[keys].drop_duplicates()
            if len(uniq) != 1 or uniq.isna().any().any():
                raise ProtocolError(f"frozen Top-3 is not unique for {ds} {ident}")
            rec = uniq.iloc[0]
            distances = np.array(
                [float(rec["source_1_distance"]), float(rec["source_2_distance"]), float(rec["source_3_distance"])],
                dtype=np.float64,
            )
            weights = inverse_distance_weights(distances)
            stored = np.array(
                [float(rec["source_1_weight"]), float(rec["source_2_weight"]), float(rec["source_3_weight"])],
                dtype=np.float64,
            )
            if not np.allclose(weights, stored, rtol=1e-8, atol=1e-10):
                raise ProtocolError(f"fusion weights do not match inverse-distance for {ds} {ident}")
            pred_map = {}
            for _, row in sub.iterrows():
                pred_map[(int(row["horizon"]), int(row["seed"]))] = str(row["result_dir"])
            rows.append({
                "dataset": ds,
                "list_rank": rank,
                "target_identity": ident,
                "source_1": str(rec["source_1"]),
                "source_2": str(rec["source_2"]),
                "source_3": str(rec["source_3"]),
                "source_1_distance": float(rec["source_1_distance"]),
                "source_2_distance": float(rec["source_2_distance"]),
                "source_3_distance": float(rec["source_3_distance"]),
                "source_1_weight": float(stored[0]),
                "source_2_weight": float(stored[1]),
                "source_3_weight": float(stored[2]),
                "n_sita_cells": int(len(sub)),
                "workbook": str(WORKBOOKS[ds].relative_to(REPO_ROOT)),
                "workbook_sha256": sha256_file(WORKBOOKS[ds]),
                "result_dirs": pred_map,
            })
    if len(rows) != EXPECTED_TARGETS:
        raise ProtocolError(HOLD_SET)
    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "result_dirs"} for row in rows])
    lookup = {(row["dataset"], row["target_identity"]): row for row in rows}
    meta = {
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(MANIFEST),
        "n_targets": {ds: 40 for ds in DATASETS},
        "total": 120,
        "workbooks": {ds: {"path": str(WORKBOOKS[ds].relative_to(REPO_ROOT)), "sha256": sha256_file(WORKBOOKS[ds])} for ds in DATASETS},
        "identities": {ds: [r["target_identity"] for r in rows if r["dataset"] == ds] for ds in DATASETS},
    }
    return frame, lookup, meta


_FROZEN: tuple[pd.DataFrame, dict, dict] | None = None


def frozen() -> tuple[pd.DataFrame, dict, dict]:
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = load_frozen_120()
    return _FROZEN


def feature_audit_rows() -> list[dict]:
    rows = []
    for ds in DATASETS:
        calendar = calendar_for(ds)
        original = list(calendar.model_features)
        extra = [c for c in calendar.similarity_variables if c not in original]
        if original[0] != "sales" or "sales" not in original:
            raise ProtocolError(f"{ds} official model_features do not start from sales")
        if len(original) <= 1:
            raise ProtocolError(f"{ds} official F is not > 1")
        for col in original:
            rows.append({
                "dataset": ds,
                "original_feature": col,
                "original_role": FEATURE_ROLES.get(col, "other_model_input"),
                "kept_in_sales_only": col == "sales",
                "reason": "target_sales_channel" if col == "sales" else "exogenous_or_calendar_removed_from_model_input",
                "scope": "model_input",
                "original_n_model_features": len(original),
            })
        for col in extra:
            rows.append({
                "dataset": ds,
                "original_feature": col,
                "original_role": FEATURE_ROLES.get(col, "similarity_only"),
                "kept_in_sales_only": False,
                "reason": "similarity_variable_not_used_in_this_ablation_selection_is_frozen",
                "scope": "source_selection_only",
                "original_n_model_features": len(original),
            })
    return rows


def resolve_main_cell_dir(result_dir: str) -> Path:
    path = REPO_ROOT / result_dir
    if path.name == "DONE.json":
        path = path.parent
    return path


def load_main_predictions(result_dir: str) -> pd.DataFrame:
    directory = resolve_main_cell_dir(result_dir)
    csv_path = directory / "predictions.csv"
    parquet_path = directory / "predictions.parquet"
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
    elif parquet_path.is_file():
        frame = pd.read_parquet(parquet_path)
    else:
        raise ProtocolError(f"main SITA predictions missing under {directory}")
    date_col = next((c for c in ("target_date", "label_date", "date") if c in frame.columns), None)
    true_col = next((c for c in ("y_true", "y_true_original") if c in frame.columns), None)
    pred_col = next((c for c in ("y_pred", "y_pred_original") if c in frame.columns), None)
    if date_col is None or true_col is None:
        raise ProtocolError(f"main SITA prediction columns unexpected: {list(frame.columns)}")
    out = pd.DataFrame({
        "target_date": [pd.Timestamp(x).strftime("%Y-%m-%d") for x in frame[date_col]],
        "y_true": np.asarray(frame[true_col], dtype=np.float64),
    })
    if pred_col is not None:
        out["y_pred"] = np.asarray(frame[pred_col], dtype=np.float64)
    if len(out) != TEST_DAYS:
        raise ProtocolError(f"main SITA n_test is {len(out)} not 180")
    return out


PAPER_TARGET_PARQUET = {
    "D4": "dataset4/target.parquet",
    "D5": "dataset5/target.parquet",
    "D6": "dataset6/target.parquet",
}


def load_target_for_sales_only(dataset: str, identity: str) -> pd.DataFrame:
    """Paper † identities live on target.parquet, not the 70-target source universe."""
    if identity != PAPER_TARGETS[dataset]:
        return load_target_series(dataset, identity)
    calendar = calendar_for(dataset)
    path = SEALED_ROOT / PAPER_TARGET_PARQUET[dataset]
    if not path.is_file():
        raise ProtocolError(f"paper target parquet missing: {path}")
    date_max = calendar.test_end if dataset != "D5" else "2017-08-14"
    frame = _read_filtered(
        path,
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
        if (frame["date"] > cutoff).any():
            raise ProtocolError("D5 paper target still contains dates after 2017-08-14")
    if frame.empty:
        raise ProtocolError(f"paper target {identity} has no rows in {path.name}")
    for column in calendar.model_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolError(f"{dataset} {identity} has a non-finite {column}")
    train, validation, test = split_target_frame(frame, calendar)
    del train, validation, test
    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


def cell_dir(dataset: str, identity: str, horizon: int, seed: int) -> Path:
    return CELLS / dataset / identity.replace("/", "_") / f"h{horizon}" / f"s{seed}"


def inspect(directory: Path) -> str:
    done = directory / "DONE.json"
    pred = directory / "predictions.csv"
    if not done.is_file() or not pred.is_file():
        return "MISS"
    payload = json.loads(done.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETED":
        return "NOT_COMPLETED"
    needed = ("RMSE", "sMAPE", "MAE")
    if any(k not in payload or not np.isfinite(float(payload[k])) for k in needed):
        return "INCOMPLETE_METRICS"
    if int(payload.get("n_test", 0)) != TEST_DAYS:
        return "N_TEST"
    return "HIT"


def write_done(directory: Path, record: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "DONE.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(directory / "DONE.json")


def owned_cache(path: Path) -> Path:
    resolved = path.resolve()
    root = CACHE.resolve()
    if resolved != root and root not in resolved.parents:
        raise ProtocolError("sales-only source cache escaped its output directory")
    return path


def load_or_train_sales_only_source(histories, *, source_key, horizon, seed, dataset, key_fields):
    import tensorflow as tf

    series = _one_source(histories, source_key, key_fields)
    identity = "/".join(source_key)
    payload = {
        "dataset": dataset,
        "source_identity": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "architecture": "cnn",
        "feature_set": FEATURE_SET,
        "model_features": list(SALES_ONLY),
        "source_data_sha256": _series_sha256(series, SALES_ONLY),
        "protocol_sha256": sales_only_protocol_sha256(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    directory = owned_cache(CACHE / digest)
    done_path = directory / "DONE.json"
    if done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("cache_key_sha256") != digest or done.get("status") != "COMPLETED":
            raise ProtocolError("HOLD_HASH_MISMATCH")
        record = _read_source_record(directory, source_key)
        if tuple(json.loads((directory / "scaler.json").read_text())["columns"]) != SALES_ONLY:
            raise ProtocolError("sales-only cache scaler is not sales-only")
        return record, "CACHE_HIT"
    if SOURCE_HIT_ONLY:
        raise ProtocolError(f"HOLD_STAGE_A_CACHE_MISS {dataset} {identity} h={horizon} seed={seed}")
    if directory.exists():
        shutil.rmtree(directory)
    partial = directory.with_name(directory.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(int(seed))
    split = split_source_frame(series, SALES_ONLY)
    scaler = fit_source_model_scaler(split.train, SALES_ONLY)
    if scaler.columns != SALES_ONLY:
        raise ProtocolError("source scaler columns are not sales-only")
    transformed = transform_model_frame(scaler, split.train)
    windows = build_partition_windows(
        transformed, key_fields=key_fields, key=source_key,
        feature_columns=SALES_ONLY, horizon=int(horizon), partition="train",
    )
    x = np.asarray(windows.x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 10 or x.shape[2] != 1:
        raise ProtocolError(f"source input shape {x.shape} is not [*,10,1]")
    if int(x.shape[0]) == 0:
        raise ProtocolError(f"source windows empty for {identity} h={horizon} seed={seed}")
    weights = fit_source_cnn(x, np.asarray(windows.y_scaled, dtype=np.float32))
    _write_source_cache(partial, payload, digest, source_key, weights, scaler, int(windows.y_scaled.shape[0]))
    partial.rename(directory)
    tf.keras.backend.clear_session()
    return _read_source_record(directory, source_key), "TRAINED"


def make_spec(dataset: str, horizon: int, seed: int) -> RunSpec:
    return RunSpec(
        arm="F4",
        protocol_tag="main",
        dataset=dataset,
        horizon=int(horizon),
        seed=int(seed),
        backbone="cnn",
        revin=True,
        freeze_conv=False,
        selection_rule="top",
        requested_k=3,
        weight_mode="inverse_distance",
        comparison_days=30,
        target_learning_rate=float(LEARNING_RATES[dataset]),
    )


def run_cell(dataset: str, identity: str, horizon: int, seed: int) -> dict:
    import tensorflow as tf

    _, lookup, _meta = frozen()
    frozen_row = lookup[(dataset, identity)]
    calendar = calendar_for(dataset)
    features = SALES_ONLY
    target = load_target_for_sales_only(dataset, identity)
    train, validation, _test = split_target_frame(target, calendar)
    scaler = fit_target_model_scaler(train, features)
    if scaler.columns != SALES_ONLY:
        raise ProtocolError("target scaler is not sales-only")
    index = build_evaluation_index(dataset, horizon)
    dates = list(index.label_date_text())
    main_pred = load_main_predictions(frozen_row["result_dirs"][(int(horizon), int(seed))])
    if list(main_pred["target_date"]) != dates:
        raise ProtocolError("target_dates_sales_only != target_dates_main_sita")
    if set(dates) & set(scaler.fit_dates):
        raise ProtocolError("target scaler fit includes a test date")
    train_scaled = transform_model_frame(scaler, train)
    validation_scaled = transform_model_frame(scaler, validation)
    observed = pd.concat(
        [validation, target.loc[target["date"] >= pd.Timestamp(calendar.test_start)]],
        ignore_index=True,
    )
    observed_scaled = transform_model_frame(scaler, observed)
    key = tuple(identity.split("/"))
    train_windows = build_target_partition_windows(
        train_scaled, key_fields=calendar.key_fields, key=key,
        feature_columns=features, horizon=horizon, partition="train",
    )
    validation_windows = build_target_partition_windows(
        validation_scaled, key_fields=calendar.key_fields, key=key,
        feature_columns=features, horizon=horizon, partition="validation",
    )
    test_windows = build_test_windows(
        observed_scaled, test_start=calendar.test_start, test_end=calendar.test_end,
        feature_columns=features, horizon=horizon,
    )
    got = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in test_windows.label_date)
    if got != tuple(dates):
        raise ProtocolError("test windows do not match evaluation index")
    y_true_main = main_pred["y_true"].to_numpy(dtype=np.float64)
    y_true_local = np.asarray(test_windows.y_original, dtype=np.float64)
    if y_true_local.shape != (TEST_DAYS,) or not np.allclose(y_true_local, y_true_main, rtol=0.0, atol=1e-8):
        raise ProtocolError("y_true_sales_only != y_true_main_sita")
    if tuple(train_windows.x.shape[1:]) != (10, 1) or tuple(test_windows.x.shape[1:]) != (10, 1):
        raise ProtocolError(f"target input is not [*,10,1]: train={train_windows.x.shape} test={test_windows.x.shape}")
    selected = [frozen_row["source_1"], frozen_row["source_2"], frozen_row["source_3"]]
    distances = np.array(
        [frozen_row["source_1_distance"], frozen_row["source_2_distance"], frozen_row["source_3_distance"]],
        dtype=np.float64,
    )
    weights = inverse_distance_weights(distances)
    stored = np.array(
        [frozen_row["source_1_weight"], frozen_row["source_2_weight"], frozen_row["source_3_weight"]],
        dtype=np.float64,
    )
    if not np.allclose(weights, stored, rtol=1e-8, atol=1e-10):
        raise ProtocolError("sales-only fusion weights != main SITA fusion weights")
    histories = histories_for(dataset, identity)
    sources = []
    cache_status = []
    for rank, source_id in enumerate(selected):
        source_key = tuple(source_id.split("/"))
        trained, status = load_or_train_sales_only_source(
            histories, source_key=source_key, horizon=horizon, seed=seed,
            dataset=dataset, key_fields=calendar.key_fields,
        )
        cache_status.append(status)
        sources.append(
            SourceRecord(
                rank=rank, source_key=source_key, weights=trained.weights,
                scaler=trained.scaler, sample_count=trained.sample_count,
            )
        )
    spec = make_spec(dataset, horizon, seed)
    bundle = StageABundle(
        key=StageAKey(
            dataset=dataset, horizon=horizon, seed=seed, backbone="cnn",
            selection_rule="top", trained_k=3,
            comparison_start=calendar.comparison_start, comparison_end=calendar.comparison_end,
            source_history_start=calendar.history_start, source_history_end=calendar.history_end,
            model_features=SALES_ONLY, similarity_variables=calendar.similarity_variables,
            scaling_rule="per_similarity_variable_minmax", draw_rule=None,
            frozen_config_sha256=sales_only_protocol_sha256(),
        ),
        status="COMPLETED",
        distances=distances,
        selected=tuple(tuple(item.split("/")) for item in selected),
        sources=tuple(sources),
        source_validation_metric=NOT_APPLICABLE,
        source_validation_value_space=NOT_APPLICABLE,
        cache_dir=CACHE,
    )
    tf.keras.utils.set_random_seed(int(seed))
    model, _ = build_target_cnn((10, 1))
    tensors: dict[str, tuple[np.ndarray, ...]] = {}
    for layer in model.layers:
        arrays = layer.get_weights()
        if arrays:
            tensors[layer.name] = tuple(np.asarray(array, dtype=np.float32) for array in arrays)
    revin = build_revin(1)
    tensors["affine_weight"] = (np.asarray(revin.affine_weight.numpy(), dtype=np.float32),)
    tensors["affine_bias"] = (np.asarray(revin.affine_bias.numpy(), dtype=np.float32),)
    result = adapt_target(
        spec, TargetWindows(train_windows, validation_windows, test_windows),
        scaler, WeightSnapshot(tensors), bundle,
    )
    y_pred = np.asarray(result.test_predictions, dtype=np.float64)
    if y_pred.shape != (TEST_DAYS,) or not np.isfinite(y_pred).all():
        raise ProtocolError("sales-only predictions are not 180 finite values")
    pair = evaluate_original_sales(y_true_local, y_pred)
    directory = cell_dir(dataset, identity, horizon, seed)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target_date": dates, "y_true": y_true_local, "y_pred": y_pred}).to_csv(
        directory / "predictions.csv", index=False
    )
    tf.keras.backend.clear_session()
    record = {
        "status": "COMPLETED",
        "arm": ARM,
        "feature_set": FEATURE_SET,
        "dataset": dataset,
        "target": identity,
        "h": int(horizon),
        "seed": int(seed),
        "sMAPE": pair.smape,
        "RMSE": pair.rmse,
        "MAE": mae_original(y_true_local, y_pred),
        "n_test": TEST_DAYS,
        "n_input_features": 1,
        "input_features": list(SALES_ONLY),
        "source_input_shape": [None, 10, 1],
        "target_input_shape": [None, 10, 1],
        "selected_sources": selected,
        "source_weights": stored.tolist(),
        "cache_status": cache_status,
        "protocol_sha256": sales_only_protocol_sha256(),
        "manifest_sha256": sha256_file(MANIFEST),
        "trained_at": _utc(),
    }
    write_done(directory, record)
    return record


def write_audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame, lookup, meta = frozen()
    audit = pd.DataFrame(feature_audit_rows())
    _atomic_csv(OUT / "feature_audit.csv", audit)
    _atomic_csv(OUT / "covariate_ablation_feature_audit.csv", audit)
    keep = audit.loc[(audit.scope == "model_input") & (audit.kept_in_sales_only)]
    for ds in DATASETS:
        kept = keep.loc[keep.dataset == ds, "original_feature"].tolist()
        if kept != ["sales"]:
            raise ProtocolError(f"{ds} sales-only kept features are {kept}")
    target_audit = frame.copy()
    target_audit["n_input_features_sales_only"] = 1
    target_audit["input_features_sales_only"] = "sales"
    _atomic_csv(OUT / "target_audit.csv", target_audit)
    align_rows = []
    for row in lookup.values():
        align_rows.append({
            "dataset": row["dataset"],
            "target_identity": row["target_identity"],
            "main_top3": f"{row['source_1']},{row['source_2']},{row['source_3']}",
            "sales_only_top3": f"{row['source_1']},{row['source_2']},{row['source_3']}",
            "identities_identical": True,
            "weights_identical": True,
            "source_1_weight": row["source_1_weight"],
            "source_2_weight": row["source_2_weight"],
            "source_3_weight": row["source_3_weight"],
        })
    align = pd.DataFrame(align_rows)
    if len(align) != 120 or not bool(align["identities_identical"].all()):
        raise ProtocolError("Top-3 alignment is incomplete")
    _atomic_csv(OUT / "source_selection_alignment.csv", align)
    req_rows = []
    for ds in DATASETS:
        sources = sorted({
            lookup[(ds, ident)][k]
            for ident in meta["identities"][ds]
            for k in ("source_1", "source_2", "source_3")
        })
        for source in sources:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    req_rows.append({
                        "dataset": ds,
                        "source_identity": source,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "feature_set": FEATURE_SET,
                        "existing_sales_only_cache": "CACHE_MISS",
                        "multivariate_cache_counted_as_hit": False,
                    })
        n_src = len(sources)
        _log(f"AUDIT {ds} unique_required_sources={n_src} keys={n_src * 15}")
    req = pd.DataFrame(req_rows)
    _atomic_csv(OUT / "stage_a_sales_only_requirements.csv", req)
    req = refresh_requirements()
    n_features = {ds: len(calendar_for(ds).model_features) for ds in DATASETS}
    snapshot = {
        "arm": ARM,
        "feature_set": FEATURE_SET,
        "created_at": _utc(),
        "manifest": meta,
        "n_original_model_features": n_features,
        "n_sales_only_model_features": {ds: 1 for ds in DATASETS},
        "learning_rates": LEARNING_RATES,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "expected_cells": EXPECTED_CELLS,
        "sales_only_protocol_sha256": sales_only_protocol_sha256(),
        "main_protocol_sha256": frozen_protocol_sha256(),
        "cache_dir": str(CACHE.relative_to(REPO_ROOT)),
        "source_selection": "frozen_from_40target_workbooks_not_recomputed",
        "n_required_source_identities": int(req["source_identity"].nunique()),
        "n_required_stage_a_keys": int(len(req)),
        "status": "READY_FOR_SALES_ONLY_CANARY",
    }
    _atomic_json(OUT / "protocol_snapshot.json", snapshot)
    return snapshot


def refresh_requirements() -> pd.DataFrame:
    req_path = OUT / "stage_a_sales_only_requirements.csv"
    if not req_path.is_file():
        return pd.DataFrame()
    req = pd.read_csv(req_path)
    if not CACHE.is_dir():
        return req
    done = {p.parent.name for p in CACHE.glob("*/DONE.json")}
    # cannot map digest->identity without key.json
    hits = 0
    rows = []
    for p in CACHE.glob("*/key.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("feature_set") != FEATURE_SET:
            continue
        hits += 1
        rows.append((payload.get("dataset"), payload.get("source_identity"), int(payload.get("horizon")), int(payload.get("seed"))))
    hit_set = set(rows)
    status = []
    for _, row in req.iterrows():
        key = (row["dataset"], row["source_identity"], int(row["horizon"]), int(row["seed"]))
        status.append("CACHE_HIT" if key in hit_set else "CACHE_MISS")
    req = req.copy()
    req["existing_sales_only_cache"] = status
    _atomic_csv(req_path, req)
    return req


def canary_targets(meta: dict) -> list[tuple[str, str]]:
    return [(ds, meta["identities"][ds][0]) for ds in DATASETS]


def histories_for(dataset: str, identity: str) -> pd.DataFrame:
    store = identity.split("/")[0]
    key = (dataset, store)
    cached = _HISTORY_CACHE.get(key)
    if cached is not None:
        return cached
    frame = load_source_histories(dataset, target_identity=identity, target_frame=pd.DataFrame())
    _HISTORY_CACHE[key] = frame
    return frame


def recycle(phase: str, datasets: tuple[str, ...], newly: int, leftover: int) -> None:
    if newly < RECYCLE_AFTER or leftover <= 0:
        return
    _log(
        f"recycling TensorFlow after {newly} leftover={leftover} "
        f"phase={phase} datasets={','.join(datasets)}"
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--datasets",
            ",".join(datasets),
            "--skip-audit",
        ],
    )


def phase_stage_a(datasets: tuple[str, ...]) -> dict:
    req = refresh_requirements()
    if req.empty:
        raise ProtocolError("stage_a_sales_only_requirements.csv is missing")
    pending = req.loc[req["dataset"].isin(datasets) & (req["existing_sales_only_cache"] == "CACHE_MISS")]
    leftover = int(len(pending))
    newly = trained = hits = failures = 0
    _log(f"STAGE_A start datasets={','.join(datasets)} pending={leftover}")
    for _, row in pending.iterrows():
        ds = str(row["dataset"])
        ident = str(row["source_identity"])
        horizon = int(row["horizon"])
        seed = int(row["seed"])
        try:
            _record, status = load_or_train_sales_only_source(
                histories_for(ds, ident),
                source_key=tuple(ident.split("/")),
                horizon=horizon,
                seed=seed,
                dataset=ds,
                key_fields=calendar_for(ds).key_fields,
            )
            _log(f"{ds} stage-a {status} {ident} h={horizon} s={seed}")
            if status == "TRAINED":
                trained += 1
                newly += 1
            else:
                hits += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            detail = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            _log(f"FAIL stage-a {ds} {ident} h={horizon} s={seed} {detail}")
            append_failure({
                "dataset": ds,
                "target": ident,
                "h": horizon,
                "seed": seed,
                "error": detail[:500],
                "traceback": traceback.format_exc().replace("\n", " | ")[-4000:],
                "written_at": _utc(),
            })
        leftover -= 1
        recycle("stage-a", datasets, newly, leftover)
    req = refresh_requirements()
    sub = req.loc[req["dataset"].isin(datasets)] if len(req) else req
    n_miss = int((sub["existing_sales_only_cache"] == "CACHE_MISS").sum()) if len(sub) else 0
    summary = {
        "status": "STAGE_A_OK" if n_miss == 0 else "STAGE_A_INCOMPLETE",
        "datasets": list(datasets),
        "trained": trained,
        "hits": hits,
        "failures": failures,
        "n_miss": n_miss,
        "finished_at": _utc(),
    }
    _log(json.dumps(summary, sort_keys=True))
    if n_miss and newly > 0:
        recycle("stage-a", datasets, RECYCLE_AFTER, n_miss)
    if n_miss:
        raise ProtocolError(f"HOLD_STAGE_A_CACHE_MISS {n_miss} keys still missing")
    return summary


def phase_sita(datasets: tuple[str, ...]) -> dict:
    global SOURCE_HIT_ONLY
    SOURCE_HIT_ONLY = True
    req = refresh_requirements()
    sub = req.loc[req["dataset"].isin(datasets)] if len(req) else req
    n_miss = int((sub["existing_sales_only_cache"] == "CACHE_MISS").sum()) if len(sub) else 0
    if n_miss:
        raise ProtocolError(f"HOLD_STAGE_A_CACHE_MISS {n_miss} keys; run --phase stage-a first")
    pairs = [(ds, ident, h, s) for ds, ident, h, s in iter_cells() if ds in datasets]
    return run_selected(pairs, recycle_phase="sita", datasets=datasets)


def append_failure(row: dict) -> None:
    path = OUT / "failures.csv"
    cleaned = dict(row)
    for key in ("error", "traceback"):
        if key in cleaned and cleaned[key] is not None:
            cleaned[key] = str(cleaned[key]).replace("\n", " | ")
    frame = pd.DataFrame([cleaned])
    frame.to_csv(path, mode="a", header=not path.is_file(), index=False)


def append_record(row: dict) -> None:
    path = OUT / "run_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def iter_cells():
    _, _lookup, meta = frozen()
    for ds in DATASETS:
        for ident in meta["identities"][ds]:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    yield ds, ident, int(horizon), int(seed)


def assemble() -> pd.DataFrame:
    rows = []
    for ds, ident, horizon, seed in iter_cells():
        directory = cell_dir(ds, ident, horizon, seed)
        if inspect(directory) != "HIT":
            continue
        payload = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
        rows.append({
            "dataset": ds, "target": ident, "condition": ARM, "h": horizon, "seed": seed,
            "sMAPE": float(payload["sMAPE"]), "RMSE": float(payload["RMSE"]), "MAE": float(payload["MAE"]),
            "n_test": int(payload["n_test"]),
        })
    frame = pd.DataFrame(rows)
    _atomic_csv(OUT / "metrics_seed_level.csv", frame if not frame.empty else pd.DataFrame(
        columns=["dataset", "target", "condition", "h", "seed", "sMAPE", "RMSE", "MAE", "n_test"]
    ))
    return frame


def run_selected(
    pairs: list[tuple[str, str, int, int]],
    *,
    recycle_phase: str | None = None,
    datasets: tuple[str, ...] = DATASETS,
) -> dict:
    ran = hits = failures = 0
    leftover = sum(
        1 for ds, ident, horizon, seed in pairs
        if inspect(cell_dir(ds, ident, horizon, seed)) != "HIT"
    )
    newly = 0
    for ds, ident, horizon, seed in pairs:
        directory = cell_dir(ds, ident, horizon, seed)
        if inspect(directory) == "HIT":
            hits += 1
            continue
        try:
            rec = run_cell(ds, ident, horizon, seed)
            _log(f"OK {ds} {ident} h={horizon} s={seed} RMSE={rec['RMSE']:.6g}")
            append_record(rec)
            ran += 1
            newly += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            detail = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            _log(f"FAIL {ds} {ident} h={horizon} s={seed} {detail}")
            append_failure({
                "dataset": ds, "target": ident, "h": horizon, "seed": seed,
                "error": detail[:500],
                "traceback": traceback.format_exc().replace("\n", " | ")[-4000:],
                "written_at": _utc(),
            })
        leftover -= 1
        if recycle_phase:
            recycle(recycle_phase, datasets, newly, leftover)
    refresh_requirements()
    frame = assemble()
    return {"hits": hits, "ran": ran, "failures": failures, "assembled": int(len(frame))}


def write_final_audit(snapshot: dict) -> dict:
    frame = assemble()
    req = refresh_requirements()
    by = frame.groupby("dataset").size().to_dict() if not frame.empty else {}
    fail_n = 0
    fail_path = OUT / "failures.csv"
    if fail_path.is_file():
        fail_n = int(len(pd.read_csv(fail_path)))
    payload = {
        "status": "SALES_ONLY_ABLATION_COMPLETE" if len(frame) == EXPECTED_CELLS and fail_n == 0 else "SALES_ONLY_ABLATION_PARTIAL",
        "expected_cells": EXPECTED_CELLS,
        "completed": int(len(frame)),
        "failed": int(fail_n),
        "by_dataset": {ds: int(by.get(ds, 0)) for ds in DATASETS},
        "n_targets": 120,
        "top3_aligned": 120,
        "n_required_sources": int(req["source_identity"].nunique()) if len(req) else None,
        "n_required_keys": int(len(req)) if len(req) else None,
        "cache_hits": int((req["existing_sales_only_cache"] == "CACHE_HIT").sum()) if len(req) else None,
        "cache_miss": int((req["existing_sales_only_cache"] == "CACHE_MISS").sum()) if len(req) else None,
        "protocol_sha256": snapshot.get("sales_only_protocol_sha256"),
        "finished_at": _utc(),
    }
    _atomic_json(OUT / "final_audit.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        default="all",
        choices=("audit", "canary", "stage-a", "sita", "all", "assemble", "run"),
    )
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--datasets", default="D4,D5,D6")
    args = parser.parse_args()
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    if args.canary:
        phase = "canary"
    elif args.audit:
        phase = "audit"
    elif args.assemble:
        phase = "assemble"
    else:
        phase = args.phase
    if args.skip_audit and (OUT / "protocol_snapshot.json").is_file():
        snapshot = json.loads((OUT / "protocol_snapshot.json").read_text(encoding="utf-8"))
    else:
        snapshot = write_audit()
        _log(f"AUDIT {snapshot['status']} features={snapshot['n_original_model_features']}")
    if phase == "assemble":
        frame = assemble()
        print(f"assembled {len(frame)}")
        return 0
    if phase == "audit":
        print(json.dumps({
            "status": snapshot["status"],
            "n_original_model_features": snapshot["n_original_model_features"],
            "n_sales_only_model_features": snapshot["n_sales_only_model_features"],
            "n_required_source_identities": snapshot["n_required_source_identities"],
            "n_required_stage_a_keys": snapshot["n_required_stage_a_keys"],
            "manifest": snapshot["manifest"]["manifest"],
            "manifest_sha256": snapshot["manifest"]["manifest_sha256"],
            "workbooks": snapshot["manifest"]["workbooks"],
            "identities": snapshot["manifest"]["identities"],
            "total": snapshot["manifest"]["total"],
        }, indent=2))
        return 0
    if phase == "canary":
        _, _, meta = frozen()
        pairs = [(ds, ident, 1, 42) for ds, ident in canary_targets(meta)]
        payload = run_selected(pairs)
        ok = payload["failures"] == 0 and payload["assembled"] >= 3
        snapshot["status"] = "READY_FOR_SALES_ONLY_FULL_RUN" if ok else "BLOCKED_SALES_ONLY_PROTOCOL"
        snapshot["canary"] = payload
        _atomic_json(OUT / "protocol_snapshot.json", snapshot)
        print(json.dumps({"status": snapshot["status"], **payload}, indent=2))
        return 0 if ok else 1
    if phase == "stage-a":
        payload = phase_stage_a(datasets)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if phase == "sita":
        payload = phase_sita(datasets)
        final = write_final_audit(snapshot)
        print(json.dumps({"run": payload, "final": final}, indent=2, sort_keys=True))
        return 0 if int(payload["failures"]) == 0 else 1
    if phase == "run":
        pairs = [(ds, ident, h, s) for ds, ident, h, s in iter_cells() if ds in datasets]
        payload = run_selected(pairs)
        final = write_final_audit(snapshot)
        print(json.dumps({"run": payload, "final": final}, indent=2, sort_keys=True))
        return 0 if int(payload["failures"]) == 0 else 1
    if phase == "all":
        phase_stage_a(datasets)
        payload = phase_sita(datasets)
        final = write_final_audit(snapshot)
        print(json.dumps({"run": payload, "final": final}, indent=2, sort_keys=True))
        return 0 if int(payload["failures"]) == 0 else 1
    raise ProtocolError(f"unknown phase {phase}")


if __name__ == "__main__":
    raise SystemExit(main())
