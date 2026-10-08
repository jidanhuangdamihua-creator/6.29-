#!/usr/bin/env python3
"""D5/D6 60-target robustness baselines. Does not run SITA or touch Stage A cache."""

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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from additional_target_robustness.baselines import (
    fit_and_predict_dlinear,
    fit_and_predict_lightgbm,
    fit_and_predict_lstm,
    fit_and_predict_patchtst,
    fit_and_predict_tcn,
    fit_and_predict_timesnet,
    materialize_seasonal_seeds,
    partition_dates,
    partition_sales,
    predict_seasonal,
)
from additional_target_robustness.horizon import verify_direct_horizon
from additional_target_robustness.paths import ensure_ablation_path
from additional_target_robustness.protocol import (
    HORIZONS,
    SEEDS,
    TEST_DAYS,
    ProtocolError,
    ProtocolHold,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
    sha256_file,
)
from additional_target_robustness.source_selection import _component, load_source_histories
from additional_target_robustness.target_data import load_target_series, sales_by_date, split_target_frame

ensure_ablation_path()
from sita_ablation.metrics import evaluate_original_sales

DATASETS = {
    "D5": {
        "out": REPO_ROOT / "outputs" / "d5_60target_robustness_20261004",
        "manifest_name": "d5_60target_manifest.json",
        "manifest_sha256": "613f992d448a2717f6ab119bd10a3d2ebc7a708e72aca28f35020148f86c63b8",
        "preflight_target": "48/261700",
        "store": "48",
        "group_value": "GROCERY I",
    },
    "D6": {
        "out": REPO_ROOT / "outputs" / "d6_60target_robustness_20261004",
        "manifest_name": "d6_60target_manifest.json",
        "manifest_sha256": "6fb0b8045ec912a661fbc2d705ee38ae6429b3d2af3423020b3a09447d30b0f7",
        "preflight_target": "CA_1/FOODS_3_726",
        "store": "CA_1",
        "group_value": "FOODS_3",
    },
}
PREFLIGHT_HORIZON = 1
PREFLIGHT_SEED = 42
METHODS = (
    "Seasonal Naive",
    "LightGBM",
    "LSTM",
    "DLinear",
    "TCN",
    "PatchTST",
    "TimesNet",
    "MSML-TL",
)
CPU_METHODS = ("Seasonal Naive", "LightGBM")
METHOD_SLUGS = {
    "Seasonal Naive": "seasonal_naive",
    "LightGBM": "lightgbm",
    "LSTM": "lstm",
    "DLinear": "dlinear",
    "TCN": "tcn",
    "PatchTST": "patchtst",
    "TimesNet": "timesnet",
    "MSML-TL": "msml_tl",
}
EXPECTED_PER_METHOD = 900
EXPECTED_TOTAL = 7200
MSML_BLOCK = "BLOCKED_MSML_TL_ADAPTER"
MSML_SOURCE_EPOCHS = 50
MSML_TARGET_EPOCHS = 50
MSML_BATCH_SIZE = 16
MSML_LEARNING_RATE = 1e-4


class BaselineStop(RuntimeError):
    def __init__(self, token: str, detail: str = "") -> None:
        super().__init__(token if not detail else f"{token}: {detail}")
        self.token = token
        self.detail = detail


class _Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _cfg(dataset: str) -> dict:
    return DATASETS[dataset]


def _baseline_root(dataset: str) -> Path:
    return _cfg(dataset)["out"] / "baselines"


def _msml_cache(dataset: str) -> Path:
    return _baseline_root(dataset) / "msml_tl" / "source_cnn_cache"


def load_frozen_manifest(dataset: str) -> list[dict]:
    cfg = _cfg(dataset)
    path = cfg["out"] / cfg["manifest_name"]
    actual = sha256_file(path)
    if actual != cfg["manifest_sha256"]:
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} manifest sha256 {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload["records"])
    if len(records) != 60:
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} n_targets={len(records)}")
    identities = [str(item["target_identity"]) for item in records]
    if len(set(identities)) != 60:
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} duplicate targets")
    if cfg["preflight_target"] not in identities:
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} missing {cfg['preflight_target']}")
    return records


def _cell_dir(dataset: str, method: str, identity: str, horizon: int, seed: int) -> Path:
    slug = METHOD_SLUGS[method]
    safe = identity.replace("/", "_")
    return _baseline_root(dataset) / slug / "cells" / f"{safe}__h{horizon}__s{seed}"


def _cell_key(dataset: str, method: str, identity: str, horizon: int, seed: int, protocol_sha: str) -> str:
    payload = {
        "dataset": dataset,
        "method": method,
        "target": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "manifest_sha256": _cfg(dataset)["manifest_sha256"],
        "protocol_sha256": protocol_sha,
        "n_test": TEST_DAYS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _inspect_cell(directory: Path, expected_key: str) -> str:
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


def _write_cell(
    dataset: str,
    directory: Path,
    *,
    cell_key: str,
    method: str,
    identity: str,
    horizon: int,
    seed: int,
    dates: tuple[str, ...],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    deterministic_reuse: bool,
    protocol_sha: str,
    extra: dict | None = None,
) -> dict:
    if len(dates) != TEST_DAYS or y_true.shape != (TEST_DAYS,) or y_pred.shape != (TEST_DAYS,):
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} {method} n_test is not 180")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} {method} non-finite arrays")
    pair = evaluate_original_sales(y_true, y_pred)
    directory.mkdir(parents=True, exist_ok=True)
    pred_frame = pd.DataFrame(
        {
            "dataset": dataset,
            "target_identity": identity,
            "method": method,
            "horizon": int(horizon),
            "seed": int(seed),
            "target_date": list(dates),
            "y_true": y_true,
            "y_pred": y_pred,
            "deterministic_reuse": bool(deterministic_reuse),
        }
    )
    tmp_dir = directory.with_name(directory.name + ".partial")
    if tmp_dir.exists():
        for child in sorted(tmp_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=False)
    pred_frame.to_csv(tmp_dir / "predictions.csv", index=False)
    record = {
        "cell_key": cell_key,
        "status": "COMPLETED",
        "dataset": dataset,
        "method": method,
        "target_identity": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "n_test": TEST_DAYS,
        "rmse": pair.rmse,
        "smape": pair.smape,
        "prediction_finite_count": int(np.isfinite(y_pred).sum()),
        "target_finite_count": int(np.isfinite(y_true).sum()),
        "first_target_date": dates[0],
        "last_target_date": dates[-1],
        "deterministic_reuse": bool(deterministic_reuse),
        "manifest_sha256": _cfg(dataset)["manifest_sha256"],
        "protocol_sha256": protocol_sha,
        "trained_at": _utc_now(),
        **(extra or {}),
    }
    (tmp_dir / "DONE.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp_dir.replace(directory)
    return record


def _prepared_target(dataset: str, identity: str) -> dict:
    calendar = calendar_for(dataset)
    target = load_target_series(dataset, identity)
    train, validation, test = split_target_frame(target, calendar)
    if len(train) != 15 or len(validation) != 15 or len(test) != TEST_DAYS:
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} {identity} split lengths")
    if dataset == "D5":
        cutoff = pd.Timestamp("2017-08-14")
        if (target["date"] > cutoff).any() or (target["date"] == pd.Timestamp("2017-08-15")).any():
            raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} {identity} has dates after 2017-08-14")
    group_values = sorted({_component(value) for value in target[calendar.group_field].tolist() if _component(value)})
    if group_values != [_cfg(dataset)["group_value"]]:
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} {identity} group={group_values}")
    sales = sales_by_date(target)
    return {
        "target": target,
        "train": train,
        "validation": validation,
        "test": test,
        "sales": sales,
        "train_sales": partition_sales(target, calendar.train_start, calendar.train_end),
        "val_sales": partition_sales(target, calendar.val_start, calendar.val_end),
        "train_dates": partition_dates(target, calendar.train_start, calendar.train_end),
        "calendar": calendar,
    }


def _predict_method(
    dataset: str,
    method: str,
    prepared: dict,
    index,
    *,
    identity: str,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, bool, dict]:
    if method == "Seasonal Naive":
        base = predict_seasonal(dataset, identity, horizon, index, prepared["sales"])
        copied = materialize_seasonal_seeds(base)
        match = next(item for item in copied if int(item["seed"]) == int(seed))
        return np.asarray(match["y_pred"], dtype=np.float64), True, {}
    if method == "LightGBM":
        pred = fit_and_predict_lightgbm(
            prepared["train_sales"],
            prepared["train_dates"],
            index,
            prepared["sales"],
            horizon=horizon,
            seed=seed,
        )
        return pred, False, {}
    if method == "LSTM":
        return (
            fit_and_predict_lstm(
                prepared["train_sales"],
                prepared["val_sales"],
                index,
                prepared["sales"],
                horizon=horizon,
                seed=seed,
            ),
            False,
            {},
        )
    if method == "DLinear":
        return (
            fit_and_predict_dlinear(
                prepared["train_sales"],
                prepared["val_sales"],
                index,
                prepared["sales"],
                horizon=horizon,
                seed=seed,
            ),
            False,
            {},
        )
    if method == "TCN":
        return (
            fit_and_predict_tcn(
                prepared["train_sales"],
                prepared["val_sales"],
                index,
                prepared["sales"],
                horizon=horizon,
                seed=seed,
            ),
            False,
            {},
        )
    if method == "PatchTST":
        return (
            fit_and_predict_patchtst(
                prepared["train_sales"],
                prepared["val_sales"],
                index,
                prepared["sales"],
                horizon=horizon,
                seed=seed,
            ),
            False,
            {},
        )
    if method == "TimesNet":
        return (
            fit_and_predict_timesnet(
                prepared["train_sales"],
                prepared["val_sales"],
                index,
                prepared["sales"],
                horizon=horizon,
                seed=seed,
            ),
            False,
            {},
        )
    if method == "MSML-TL":
        pred, extra = fit_and_predict_msml_tl(
            dataset, prepared, index, identity=identity, horizon=horizon, seed=seed
        )
        return pred, False, extra
    raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"unknown method {method}")


def _msml_series_digest(frame: pd.DataFrame, feature_cols) -> str:
    digest = hashlib.sha256()
    ordered = frame.sort_values("date", kind="mergesort")
    for _, row in ordered.iterrows():
        digest.update(pd.Timestamp(row["date"]).strftime("%Y-%m-%d").encode("utf-8"))
        for column in feature_cols:
            value = row[column]
            token = "NA" if pd.isna(value) else repr(value)
            digest.update(b"\0")
            digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _msml_source_cache_key(
    dataset: str,
    *,
    source_identity: str,
    horizon: int,
    seed: int,
    feature_cols,
    series_digest: str,
) -> str:
    payload = {
        "dataset": dataset,
        "method": "MSML-TL",
        "source_identity": source_identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "architecture": "msml_base_cnn",
        "learning_rate": MSML_LEARNING_RATE,
        "source_epochs": MSML_SOURCE_EPOCHS,
        "batch_size": MSML_BATCH_SIZE,
        "window_size": 10,
        "feature_cols": list(feature_cols),
        "source_series_sha256": series_digest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_msml_source_cache(directory: Path, *, cache_key: str, model, input_shape, num_samples, source_key) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights = model.get_weights()
    np.savez_compressed(directory / "weights.npz", **{f"w{index}": array for index, array in enumerate(weights)})
    payload = {
        "status": "COMPLETED",
        "cache_key": cache_key,
        "input_shape": [int(dim) for dim in input_shape],
        "num_samples": int(num_samples),
        "source_key": [str(part) for part in source_key] if source_key is not None else [],
        "n_weight_tensors": len(weights),
    }
    (directory / "DONE.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_msml_source_cache(directory: Path, *, learning_rate: float):
    from src.models.cnn_model import build_base_cnn

    done = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
    stored = np.load(directory / "weights.npz")
    weights = [np.asarray(stored[f"w{index}"]) for index in range(int(done["n_weight_tensors"]))]
    model = build_base_cnn(tuple(int(dim) for dim in done["input_shape"]), learning_rate=learning_rate)
    model.set_weights(weights)
    return model, tuple(int(dim) for dim in done["input_shape"]), int(done["num_samples"])


def fit_and_predict_msml_tl(dataset: str, prepared: dict, index, *, identity: str, horizon: int, seed: int) -> tuple[np.ndarray, dict]:
    from src.data_processing.data_preprocessing import normalize_features, temporal_split_by_ratio_or_dates, to_cnn_tensor
    from src.evaluation.metrics import inverse_transform_sales
    from src.protocols.reproducibility import set_protocol_seed
    from src.transfer_methods import msml_tl as msml_mod
    from src.transfer_methods.msml_tl import run_msml_tl

    calendar = prepared["calendar"]
    store_field, item_field = calendar.key_fields
    store, item = identity.split("/", 1)
    source = load_source_histories(dataset, target_identity=identity, target_frame=prepared["target"]).copy()
    for field in calendar.key_fields:
        source[field] = source[field].map(_component)
    if calendar.group_field in source.columns:
        source = source.loc[source[calendar.group_field].map(_component) == _cfg(dataset)["group_value"]].copy()
    source = source.loc[~((source[store_field] == store) & (source[item_field] == item))].copy()
    if source.empty:
        raise BaselineStop(MSML_BLOCK, f"{dataset} {identity} has an empty same-group source pool")
    if (source[store_field] != store).any():
        raise BaselineStop(MSML_BLOCK, f"{dataset} {identity} source pool left the target store")
    source["entity_id"] = source[store_field]
    source["item_id"] = source[item_field]
    source.attrs["split_role"] = "source"
    source.attrs["split_mode"] = "ratio"

    target = prepared["target"].copy()
    for field in calendar.key_fields:
        target[field] = target[field].map(_component)
    target["entity_id"] = target[store_field]
    target["item_id"] = target[item_field]
    target.attrs["split_role"] = "target"
    target.attrs["split_mode"] = "paper_split_protocol"

    features = tuple(calendar.model_features)
    holder: dict = {"source_cache_hit": 0, "source_cache_miss": 0, "source_cache_keys": []}
    orig_split = temporal_split_by_ratio_or_dates
    orig_norm = normalize_features
    orig_eval = msml_mod.evaluate_msml_model
    orig_train = msml_mod.train_source_cnn_for_msml

    def split_wrap(frame):
        parts = orig_split(frame)
        if str(frame.attrs.get("split_role", "")).lower() == "target":
            holder["raw_parts"] = parts
        return parts

    def norm_wrap(train_df, val_df, test_df, feature_columns=None, **kwargs):
        result = orig_norm(train_df, val_df, test_df, feature_columns=feature_columns, **kwargs)
        holder["norm_parts"] = result[:3]
        holder["scaler"] = result[3]
        holder["feature_columns"] = result[4]
        return result

    def eval_wrap(target_model, target_test_df, feature_cols, horizon=1, window_size=10, **kwargs):
        del target_test_df, kwargs
        norm_train, norm_val, norm_test = holder["norm_parts"]
        full = pd.concat([norm_train, norm_val, norm_test], ignore_index=True)
        full["date"] = pd.to_datetime(full["date"]).dt.normalize()
        full = full.sort_values("date", kind="mergesort").reset_index(drop=True)
        by_date = {
            pd.Timestamp(day).normalize(): full.loc[full["date"] == pd.Timestamp(day).normalize()]
            for day in full["date"]
        }
        windows = []
        for row in index.rows:
            mats = []
            for day in row.input_dates:
                block = by_date.get(pd.Timestamp(day).normalize())
                if block is None or block.empty:
                    raise BaselineStop(MSML_BLOCK, f"missing MSML input date {day}")
                mats.append(block.loc[:, list(feature_cols)].to_numpy(dtype=np.float32)[0])
            windows.append(np.stack(mats, axis=0))
        tensor = to_cnn_tensor(np.stack(windows, axis=0))
        y_pred_norm = np.asarray(target_model.predict(tensor, verbose=0), dtype=np.float64).reshape(-1)
        y_pred = inverse_transform_sales(
            y_pred_norm,
            sales_scaler=holder["scaler"],
            feature_columns=holder["feature_columns"],
        )
        y_true = np.asarray([prepared["sales"][day] for day in index.label_date_text()], dtype=np.float64)
        if y_pred.shape != (TEST_DAYS,) or y_true.shape != (TEST_DAYS,):
            raise BaselineStop(MSML_BLOCK, f"MSML evaluation n_test={y_pred.shape}")
        pair = evaluate_original_sales(y_true, y_pred)
        holder["y_pred"] = y_pred
        holder["y_true"] = y_true
        return {
            "rmse": pair.rmse,
            "accuracy": float("nan"),
            "smape": pair.smape,
            "y_pred": y_pred,
            "y_true": y_true,
            "prediction_shape": tuple(y_pred.shape),
        }

    def train_wrap(
        source_sequence_df,
        feature_cols,
        horizon=1,
        window_size=10,
        learning_rate=0.001,
        source_epochs=3,
        batch_size=16,
        source_key=None,
        transformation_reuse_context=None,
    ):
        identity_key = "/".join(str(part) for part in source_key) if source_key is not None else ""
        series_digest = _msml_series_digest(source_sequence_df, feature_cols)
        cache_key = _msml_source_cache_key(
            dataset,
            source_identity=identity_key,
            horizon=int(horizon),
            seed=int(seed),
            feature_cols=feature_cols,
            series_digest=series_digest,
        )
        directory = _msml_cache(dataset) / cache_key
        if (directory / "DONE.json").is_file() and (directory / "weights.npz").is_file():
            model, input_shape, num_samples = _load_msml_source_cache(directory, learning_rate=float(learning_rate))
            holder["source_cache_hit"] += 1
            holder["source_cache_keys"].append({"source": identity_key, "cache_key": cache_key, "status": "CACHE_HIT"})
            return {
                "model": model,
                "input_shape": input_shape,
                "num_samples": num_samples,
                "source_key": source_key,
            }
        result = orig_train(
            source_sequence_df,
            feature_cols,
            horizon=horizon,
            window_size=window_size,
            learning_rate=learning_rate,
            source_epochs=source_epochs,
            batch_size=batch_size,
            source_key=source_key,
            transformation_reuse_context=transformation_reuse_context,
        )
        _write_msml_source_cache(
            directory,
            cache_key=cache_key,
            model=result["model"],
            input_shape=result["input_shape"],
            num_samples=result["num_samples"],
            source_key=source_key,
        )
        holder["source_cache_miss"] += 1
        holder["source_cache_keys"].append({"source": identity_key, "cache_key": cache_key, "status": "TRAINED"})
        return result

    set_protocol_seed(int(seed), include_frameworks=True)
    msml_mod.temporal_split_by_ratio_or_dates = split_wrap
    msml_mod.normalize_features = norm_wrap
    msml_mod.evaluate_msml_model = eval_wrap
    msml_mod.train_source_cnn_for_msml = train_wrap
    try:
        result = run_msml_tl(
            source_df=source,
            target_df=target,
            feature_cols=features,
            k=3,
            horizon=int(horizon),
            window_size=10,
            weight_mode="inverse_distance",
            learning_rate=MSML_LEARNING_RATE,
            source_epochs=MSML_SOURCE_EPOCHS,
            target_epochs=MSML_TARGET_EPOCHS,
            batch_size=MSML_BATCH_SIZE,
            group_cols=calendar.key_fields,
            formal_bl8=False,
            seed=int(seed),
        )
    except BaselineStop:
        raise
    except Exception as exc:
        raise BaselineStop(MSML_BLOCK, f"{type(exc).__name__}: {exc}") from exc
    finally:
        msml_mod.temporal_split_by_ratio_or_dates = orig_split
        msml_mod.normalize_features = orig_norm
        msml_mod.evaluate_msml_model = orig_eval
        msml_mod.train_source_cnn_for_msml = orig_train

    selected = []
    for item_meta in result.get("meta", {}).get("selected_sources") or []:
        key = item_meta.get("source_key")
        if isinstance(key, (list, tuple)):
            selected.append("/".join(str(part) for part in key))
        else:
            selected.append(str(key))
    if any(item_sel == identity for item_sel in selected):
        raise BaselineStop(MSML_BLOCK, "MSML-TL selected the target identity")
    if any(not item_sel.startswith(f"{store}/") for item_sel in selected if item_sel):
        raise BaselineStop(MSML_BLOCK, f"MSML-TL selected a cross-store source: {selected}")
    pred = np.asarray(holder["y_pred"], dtype=np.float64)
    extra = {
        "msml_selected_sources": selected,
        "msml_formal_bl8": False,
        "msml_used_sita_top3": False,
        "msml_source_cache_hit": int(holder.get("source_cache_hit", 0)),
        "msml_source_cache_miss": int(holder.get("source_cache_miss", 0)),
        "msml_used_stage_a_cache": False,
    }
    return pred, extra


def _y_true(index, sales: dict[str, float]) -> np.ndarray:
    return np.asarray([sales[day] for day in index.label_date_text()], dtype=np.float64)


def _progress_tables(dataset: str, records: list[dict]) -> None:
    root = _baseline_root(dataset)
    if not records:
        empty = pd.DataFrame()
        _atomic_csv(root / "baseline_metrics.csv", empty)
        _atomic_csv(root / "baseline_progress.csv", empty)
        _atomic_csv(root / "baseline_failures.csv", empty)
        return
    frame = pd.DataFrame(records)
    _atomic_csv(root / "baseline_progress.csv", frame)
    metrics_cols = [
        column
        for column in (
            "method",
            "target_identity",
            "horizon",
            "seed",
            "n_test",
            "rmse",
            "smape",
            "status",
            "deterministic_reuse",
        )
        if column in frame.columns
    ]
    _atomic_csv(root / "baseline_metrics.csv", frame.loc[:, metrics_cols])
    failures = frame.loc[~frame["status"].isin(["COMPLETED", "SKIP"])].copy() if "status" in frame.columns else frame
    _atomic_csv(root / "baseline_failures.csv", failures)


def _load_existing_progress(dataset: str) -> list[dict]:
    path = _baseline_root(dataset) / "baseline_progress.csv"
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def run_cells(
    dataset: str,
    *,
    identities: list[str],
    methods: tuple[str, ...],
    horizons: tuple[int, ...],
    seeds: tuple[int, ...],
    log: _Log,
    msml_blocked: bool,
) -> list[dict]:
    protocol_sha = frozen_protocol_sha256()
    contract = verify_direct_horizon()
    log.write(f"{dataset} horizon contract {json.dumps(contract, sort_keys=True)}")
    calendar = calendar_for(dataset)
    root = _baseline_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    for slug in METHOD_SLUGS.values():
        (root / slug).mkdir(parents=True, exist_ok=True)
    progress = _load_existing_progress(dataset)
    done = {
        (str(row.get("method")), str(row.get("target_identity")), int(row.get("horizon")), int(row.get("seed")))
        for row in progress
        if str(row.get("status")) in {"COMPLETED", "SKIP"}
    }
    indexes = {horizon: build_evaluation_index(dataset, horizon) for horizon in horizons}
    for horizon, index in indexes.items():
        dates = index.label_date_text()
        if len(dates) != TEST_DAYS or dates[0] != calendar.test_start or dates[-1] != calendar.test_end:
            raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} evaluation index h={horizon}")
    prepared_cache: dict[str, dict] = {}
    for method in methods:
        if method == "MSML-TL" and msml_blocked:
            continue
        for identity in identities:
            prepared = prepared_cache.get(identity)
            for horizon in horizons:
                index = indexes[horizon]
                dates = index.label_date_text()
                seasonal_cache = None
                aborted_msml = False
                for seed in seeds:
                    tuple_key = (method, identity, int(horizon), int(seed))
                    cell_key = _cell_key(dataset, method, identity, horizon, seed, protocol_sha)
                    directory = _cell_dir(dataset, method, identity, horizon, seed)
                    inspect = _inspect_cell(directory, cell_key)
                    if inspect == "HIT" or tuple_key in done:
                        if inspect == "HIT":
                            payload = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
                            payload["status"] = "SKIP"
                            if tuple_key not in done:
                                progress.append(payload)
                                done.add(tuple_key)
                        continue
                    if inspect == "HASH_MISMATCH":
                        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"hash mismatch {directory}")
                    if prepared is None:
                        prepared = _prepared_target(dataset, identity)
                        prepared_cache[identity] = prepared
                    y_true = _y_true(index, prepared["sales"])
                    try:
                        if method == "Seasonal Naive" and seasonal_cache is not None:
                            y_pred, reused, extra = seasonal_cache, True, {}
                        else:
                            y_pred, reused, extra = _predict_method(
                                dataset,
                                method,
                                prepared,
                                index,
                                identity=identity,
                                horizon=horizon,
                                seed=seed,
                            )
                            if method == "Seasonal Naive":
                                seasonal_cache = y_pred
                        record = _write_cell(
                            dataset,
                            directory,
                            cell_key=cell_key,
                            method=method,
                            identity=identity,
                            horizon=horizon,
                            seed=seed,
                            dates=dates,
                            y_true=y_true,
                            y_pred=np.asarray(y_pred, dtype=np.float64),
                            deterministic_reuse=reused,
                            protocol_sha=protocol_sha,
                            extra=extra,
                        )
                        progress.append(record)
                        done.add(tuple_key)
                        _append_jsonl(root / "baseline_run_records.jsonl", record)
                        _progress_tables(dataset, progress)
                        log.write(
                            f"{dataset} {method} {identity} h={horizon} seed={seed} n_test=180 "
                            f"rmse={record['rmse']:.6g} reuse={reused}"
                        )
                    except BaselineStop as exc:
                        failure = {
                            "method": method,
                            "target_identity": identity,
                            "horizon": int(horizon),
                            "seed": int(seed),
                            "status": exc.token,
                            "error": exc.detail,
                            "n_test": "",
                            "trained_at": _utc_now(),
                        }
                        progress.append(failure)
                        _append_jsonl(root / "baseline_run_records.jsonl", failure)
                        _progress_tables(dataset, progress)
                        if method == "MSML-TL" and exc.token == MSML_BLOCK:
                            log.write(f"{dataset} {MSML_BLOCK}: {exc.detail}")
                            msml_blocked = True
                            aborted_msml = True
                            break
                        raise
                    except Exception as exc:
                        failure = {
                            "method": method,
                            "target_identity": identity,
                            "horizon": int(horizon),
                            "seed": int(seed),
                            "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                            "n_test": "",
                            "trained_at": _utc_now(),
                        }
                        progress.append(failure)
                        _append_jsonl(root / "baseline_run_records.jsonl", failure)
                        _progress_tables(dataset, progress)
                        log.write(traceback.format_exc())
                        if method == "MSML-TL":
                            log.write(f"{dataset} {MSML_BLOCK}: unexpected failure")
                            msml_blocked = True
                            aborted_msml = True
                            break
                        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", str(exc)) from exc
                if aborted_msml:
                    break
            if msml_blocked and method == "MSML-TL":
                break
    return progress


def run_preflight(dataset: str, log: _Log) -> dict:
    root = _baseline_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    pre_dir = root / "preflight"
    pre_dir.mkdir(parents=True, exist_ok=True)
    target = _cfg(dataset)["preflight_target"]
    rows = []
    msml_blocked = False
    for method in METHODS:
        log.write(f"{dataset} preflight {method} {target} h={PREFLIGHT_HORIZON} seed={PREFLIGHT_SEED}")
        try:
            progress = run_cells(
                dataset,
                identities=[target],
                methods=(method,),
                horizons=(PREFLIGHT_HORIZON,),
                seeds=(PREFLIGHT_SEED,),
                log=log,
                msml_blocked=False,
            )
            latest = progress[-1]
            status = str(latest.get("status", "FAILED"))
            n_test = latest.get("n_test", "")
            if method == "MSML-TL" and status == MSML_BLOCK:
                msml_blocked = True
            rows.append(
                {
                    "method": method,
                    "target": target,
                    "h": PREFLIGHT_HORIZON,
                    "seed": PREFLIGHT_SEED,
                    "n_test": n_test,
                    "status": status,
                    "error": latest.get("error", ""),
                }
            )
        except BaselineStop as exc:
            status = exc.token
            if method == "MSML-TL":
                msml_blocked = True
                status = MSML_BLOCK if exc.token == MSML_BLOCK else exc.token
            rows.append(
                {
                    "method": method,
                    "target": target,
                    "h": PREFLIGHT_HORIZON,
                    "seed": PREFLIGHT_SEED,
                    "n_test": "",
                    "status": status,
                    "error": exc.detail,
                }
            )
            if method != "MSML-TL":
                raise
    frame = pd.DataFrame(rows)
    _atomic_csv(pre_dir / "preflight_results.csv", frame)
    ordinary = frame.loc[frame["method"] != "MSML-TL"]
    ordinary_ok = bool(
        (ordinary["status"].isin(["COMPLETED", "SKIP"])).all()
        and (ordinary["n_test"].astype(str) == str(TEST_DAYS)).all()
    )
    payload = {
        "dataset": dataset,
        "ordinary_ok": ordinary_ok,
        "msml_blocked": msml_blocked,
        "rows": rows,
        "finished_at": _utc_now(),
    }
    _atomic_json(pre_dir / "DONE.json", payload)
    if not ordinary_ok:
        raise BaselineStop("BLOCKED_BASELINE_PROTOCOL", f"{dataset} ordinary baseline preflight failed")
    return payload


def final_audit(dataset: str, log: _Log, msml_blocked: bool) -> dict:
    root = _baseline_root(dataset)
    methods = [method for method in METHODS if not (method == "MSML-TL" and msml_blocked)]
    summary = []
    all_n_test = True
    completed_total = 0
    failed_total = 0
    for method in METHODS:
        slug = METHOD_SLUGS[method]
        done_files = list((root / slug / "cells").glob("*/DONE.json")) if (root / slug / "cells").exists() else []
        n_completed = 0
        n_failed = 0
        for path in done_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "COMPLETED" and int(payload.get("n_test", 0)) == TEST_DAYS:
                n_completed += 1
            else:
                n_failed += 1
                all_n_test = False
        if method == "MSML-TL" and msml_blocked:
            summary.append(
                {"method": method, "expected": EXPECTED_PER_METHOD, "completed": n_completed, "failed": MSML_BLOCK}
            )
            continue
        summary.append({"method": method, "expected": EXPECTED_PER_METHOD, "completed": n_completed, "failed": n_failed})
        completed_total += n_completed
        failed_total += n_failed
    expected_now = EXPECTED_PER_METHOD * len(methods)
    if completed_total == expected_now and failed_total == 0 and all_n_test:
        status = "BASELINE_FULL_RUN_COMPLETE" if not msml_blocked else "BASELINE_PARTIAL_COMPLETE"
    elif completed_total > 0:
        status = "BASELINE_PARTIAL_COMPLETE"
    else:
        status = "BLOCKED_BASELINE_PROTOCOL"
    payload = {
        "status": status,
        "dataset": dataset,
        "methods": summary,
        "completed_total": completed_total,
        "failed_total": failed_total,
        "expected_total": expected_now,
        "all_successful_cells_n_test_180": all_n_test,
        "msml_blocked": msml_blocked,
        "sita_ran": False,
        "audited_at": _utc_now(),
    }
    _atomic_json(root / "baseline_final_audit.json", payload)
    log.write(json.dumps({"dataset": dataset, "status": status, "completed": completed_total, "failed": failed_total}))
    return payload


def run_dataset(dataset: str, *, phase: str, selected: tuple[str, ...], log: _Log) -> dict:
    records = load_frozen_manifest(dataset)
    identities = [str(item["target_identity"]) for item in records]
    root = _baseline_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    msml_blocked = False
    preflight_path = root / "preflight" / "DONE.json"
    if phase in {"preflight", "all"}:
        payload = run_preflight(dataset, log)
        msml_blocked = bool(payload["msml_blocked"])
    elif preflight_path.is_file():
        msml_blocked = bool(json.loads(preflight_path.read_text(encoding="utf-8")).get("msml_blocked", False))
    if phase in {"run", "all"}:
        run_cells(
            dataset,
            identities=identities,
            methods=selected,
            horizons=HORIZONS,
            seeds=SEEDS,
            log=log,
            msml_blocked=msml_blocked or "MSML-TL" not in selected,
        )
    if phase in {"audit", "all", "run"}:
        return final_audit(dataset, log, msml_blocked)
    return {"status": "PREFLIGHT_ONLY", "dataset": dataset, "msml_blocked": msml_blocked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "run", "audit", "all"), default="all")
    parser.add_argument("--dataset", choices=("D5", "D6", "both"), default="both")
    parser.add_argument("--methods", default="", help="comma-separated method names")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    datasets = ("D5", "D6") if args.dataset == "both" else (args.dataset,)
    selected = tuple(item.strip() for item in args.methods.split(",") if item.strip()) or METHODS
    if args.cpu_only:
        selected = tuple(method for method in selected if method in CPU_METHODS)
    unknown = [method for method in selected if method not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}")
    log = _Log(_cfg(datasets[0])["out"] / "baselines" / "baseline_run_log.txt")
    try:
        audits = []
        for dataset in datasets:
            dataset_log = _Log(_baseline_root(dataset) / "baseline_run_log.txt")
            audit = run_dataset(dataset, phase=args.phase, selected=selected, log=dataset_log)
            audits.append(audit)
            print(f"{dataset} {audit['status']}", flush=True)
        if args.dataset == "both" and args.phase in {"audit", "all", "run"}:
            statuses = [item["status"] for item in audits]
            if all(item == "BASELINE_FULL_RUN_COMPLETE" for item in statuses):
                combined = "BASELINE_FULL_RUN_COMPLETE"
            elif any(item in {"BASELINE_FULL_RUN_COMPLETE", "BASELINE_PARTIAL_COMPLETE"} for item in statuses):
                combined = "BASELINE_PARTIAL_COMPLETE"
            else:
                combined = "BLOCKED_BASELINE_PROTOCOL"
            payload = {"status": combined, "datasets": audits, "sita_ran": False, "audited_at": _utc_now()}
            for dataset in datasets:
                _atomic_json(_baseline_root(dataset) / "baseline_combined_audit.json", payload)
            print(combined, flush=True)
            return 0 if combined in {"BASELINE_FULL_RUN_COMPLETE", "BASELINE_PARTIAL_COMPLETE"} else 1
        return 0
    except (BaselineStop, ProtocolHold, ProtocolError) as exc:
        token = getattr(exc, "token", type(exc).__name__)
        detail = getattr(exc, "detail", str(exc))
        log.write(f"{token}: {detail}")
        for dataset in datasets:
            _atomic_json(_baseline_root(dataset) / "HOLD.json", {"status": token, "detail": detail, "written_at": _utc_now()})
        print(token, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
