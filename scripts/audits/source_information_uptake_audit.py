"""Read-only audit for source information uptake in transfer-learning runs.

The audit writes only under ``outputs/audits`` by default. It reuses the
project data loading, split, KNN source selection, CNN construction, training,
prediction, and RMSE helpers without changing the main experiment logic.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from cnn_model import build_base_cnn
from data_preprocessing import (
    build_source_target_split,
    build_tabular_sequence,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from environment import setup_reproducibility
from msml_tl import fuse_source_models_layerwise, load_fused_params_into_target_model
from source_selector import SourceSelector
from src.evaluation.metrics import compute_metrics_with_protocol
from src.utils.runtime_control import keras_verbose


OUT_DIR = ROOT / "outputs" / "audits"
DETAILS_CSV = OUT_DIR / "source_information_uptake_details.csv"
SUMMARY_CSV = OUT_DIR / "source_information_uptake_summary.csv"
REPORT_MD = OUT_DIR / "source_information_uptake.md"

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
DATASET_BY_ID = {str(v): k for k, v in DATASET_ID.items()}

METHOD_VARIANTS = [
    "No-TL",
    "Random-init-target-finetune",
    "KNN-source-pretrain-finetune",
    "Random-source-pretrain-finetune",
    "Shuffled-source-pretrain-finetune",
    "KNN-source-pretrain-frozen-backbone",
    "Source-only-prediction",
]

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "method_variant",
    "horizon",
    "random_seed",
    "selected_sources",
    "random_sources",
    "source_count",
    "target_window_days",
    "target_train_size",
    "target_val_size",
    "target_test_size",
    "source_train_size",
    "pretrain_train_loss_first_epoch",
    "pretrain_train_loss_final_epoch",
    "pretrain_val_rmse",
    "before_finetune_val_rmse",
    "before_finetune_test_rmse",
    "after_finetune_val_rmse",
    "after_finetune_test_rmse",
    "target_train_loss_first_epoch",
    "target_train_loss_final_epoch",
    "target_val_rmse_first_epoch",
    "target_val_rmse_final_epoch",
    "frozen_backbone",
    "shuffled_source",
    "shuffle_type",
    "random_source",
    "source_only",
    "rmse_delta_vs_notl",
    "rmse_pct_delta_vs_notl",
    "rmse_delta_vs_random_init",
    "rmse_pct_delta_vs_random_init",
    "rmse_delta_vs_knn_source",
    "rmse_pct_delta_vs_knn_source",
    "source_uptake_status",
    "forgetting_status",
    "notes",
    "run_time_seconds",
    "error_message",
]

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "horizon",
    "method_variant",
    "n_runs",
    "mean_test_rmse",
    "std_test_rmse",
    "mean_delta_vs_notl",
    "mean_pct_delta_vs_notl",
    "mean_delta_vs_random_init",
    "mean_pct_delta_vs_random_init",
    "win_rate_vs_notl",
    "win_rate_vs_random_init",
    "win_rate_vs_random_source",
    "source_uptake_status",
    "interpretation",
]


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    target_train: pd.DataFrame
    target_val: pd.DataFrame
    target_test: pd.DataFrame
    target_train_scaled: pd.DataFrame
    target_val_scaled: pd.DataFrame
    target_test_scaled: pd.DataFrame
    target_scaler: object
    feature_columns: List[str]


def _load_config() -> Dict[str, Any]:
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    cfg["paper_reproduction"].setdefault("metric_protocol", {})["strict_paper_metrics"] = False
    return cfg


def _metric_protocol(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))


def _as_float(value: object) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_finite(values: Sequence[object]) -> float:
    vals = [_as_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_delta(value: float, baseline: float) -> object:
    return value - baseline if np.isfinite(value) and np.isfinite(baseline) else ""


def _safe_pct_delta(value: float, baseline: float) -> object:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
        return ""
    return (value - baseline) / baseline


def _pct_better(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return (baseline - candidate) / baseline


def _close_within_one_pct(a: float, b: float) -> bool:
    return np.isfinite(a) and np.isfinite(b) and b != 0 and abs(a - b) / abs(b) < 0.01


def _key_to_text(key: object) -> str:
    if isinstance(key, (tuple, list)):
        return "|".join(str(x) for x in key)
    return str(key)


def _sources_to_text(sources: Sequence[Dict[str, Any]]) -> str:
    return ";".join(_key_to_text(row.get("source_key", "")) for row in sources)


def _limit_rows_by_group(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        out = df.copy()
        out.attrs = df.attrs.copy()
        return out
    parts = []
    group_cols = ["entity_id", "item_id"]
    group_count = max(1, df.groupby(group_cols).ngroups) if all(c in df.columns for c in group_cols) else 1
    per_group = max(8, int(np.ceil(max_rows / group_count)))
    grouped = df.sort_values("date").groupby(group_cols, sort=False) if all(c in df.columns for c in group_cols) else [(None, df)]
    for _, group in grouped:
        parts.append(group.tail(per_group))
    sort_cols = [col for col in ["date", "entity_id", "item_id"] if col in df.columns]
    limited = pd.concat(parts, axis=0, ignore_index=True)
    if sort_cols:
        limited = limited.sort_values(sort_cols)
    out = limited.tail(max_rows).copy() if len(limited) > max_rows else limited.copy()
    out.attrs = df.attrs.copy()
    return out


def _prepare_dataset(dataset_name: str, cfg: Dict[str, Any], max_target_rows: int | None = None) -> DatasetBundle:
    data_path = ROOT / str(cfg["dataset_paths"][dataset_name])
    raw_df = load_dataset(dataset_name=dataset_name, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    local_cfg = dict(cfg)
    local_cfg["dataset_name"] = dataset_name
    source_df, target_df = build_source_target_split(processed_df, local_cfg)
    target_df = _limit_rows_by_group(target_df, max_target_rows)
    if max_target_rows is not None and max_target_rows > 0:
        target_df.attrs["split_role"] = "target"
        target_df.attrs["split_mode"] = "ratio"
        target_df.attrs["split_config"] = {"train_ratio": 0.25, "val_ratio": 0.25, "test_ratio": 0.5}
    target_train, target_val, target_test = temporal_split_by_ratio_or_dates(target_df.copy())
    train_scaled, val_scaled, test_scaled, scaler, feature_columns = normalize_features(target_train, target_val, target_test)
    return DatasetBundle(
        dataset_name=dataset_name,
        source_df=source_df,
        target_df=target_df,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        target_train_scaled=train_scaled,
        target_val_scaled=val_scaled,
        target_test_scaled=test_scaled,
        target_scaler=scaler,
        feature_columns=feature_columns,
    )


def _make_sequences(df: pd.DataFrame, horizon: int, window_size: int, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x, y = build_tabular_sequence(df, horizon=horizon, window_size=window_size, feature_cols=feature_cols)
    return to_cnn_tensor(x), y


def _evaluate(
    model: object,
    df: pd.DataFrame,
    horizon: int,
    window_size: int,
    feature_cols: Sequence[str],
    metric_protocol: Dict[str, Any],
    scaler: object,
) -> float:
    x, y = _make_sequences(df, horizon=horizon, window_size=window_size, feature_cols=feature_cols)
    if len(y) == 0:
        return float("nan")
    y_pred = model.predict(x, verbose=0)
    result = compute_metrics_with_protocol(
        y_true=y,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=scaler,
        feature_columns=feature_cols,
    )
    return float(result["rmse"])


def _history_loss(history: object, key: str, index: int) -> float:
    values = getattr(history, "history", {}).get(key, [])
    if not values:
        return float("nan")
    return float(values[index])


def _history_rmse(history: object, key: str, index: int) -> float:
    loss = _history_loss(history, key, index)
    return float(np.sqrt(loss)) if np.isfinite(loss) else float("nan")


def _fit_target(
    model: object,
    bundle: DatasetBundle,
    horizon: int,
    window_size: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
):
    import tensorflow as tf

    x_train, y_train = _make_sequences(bundle.target_train_scaled, horizon, window_size, bundle.feature_columns)
    x_val, y_val = _make_sequences(bundle.target_val_scaled, horizon, window_size, bundle.feature_columns)
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("Target train/val split produced zero windows; reduce window_size/horizon.")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=keras_verbose(),
    )


def _source_split(source_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_for_split = source_rows.copy()
    try:
        return temporal_split_by_ratio_or_dates(source_for_split)
    except ValueError as exc:
        if "days split requires" not in str(exc):
            raise
        source_for_split = source_rows.copy()
        source_for_split.attrs["split_role"] = "source"
        source_for_split.attrs["split_mode"] = "ratio"
        source_for_split.attrs["split_config"] = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
        return temporal_split_by_ratio_or_dates(source_for_split)


def _fit_source_model(
    source_rows: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int,
    window_size: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    random_seed: int,
    shuffle_source: bool = False,
) -> Dict[str, Any]:
    train_df, val_df, test_df = _source_split(source_rows)
    train_scaled, val_scaled, test_scaled, scaler, source_features = normalize_features(
        train_df,
        val_df,
        test_df,
        feature_cols=feature_cols,
    )
    x_train, y_train = _make_sequences(train_scaled, horizon, window_size, source_features)
    x_val, y_val = _make_sequences(val_scaled, horizon, window_size, source_features)
    if len(y_train) == 0:
        raise ValueError("Source split produced zero training windows; reduce window_size/horizon.")
    if shuffle_source:
        rng = np.random.default_rng(random_seed)
        y_train = rng.permutation(y_train)
    model = build_base_cnn(x_train.shape[1:], learning_rate=learning_rate)
    fit_kwargs = {"epochs": epochs, "batch_size": batch_size, "verbose": keras_verbose()}
    if len(y_val):
        fit_kwargs["validation_data"] = (x_val, y_val)
    history = model.fit(x_train, y_train, **fit_kwargs)
    pretrain_val_rmse = float("nan")
    if len(y_val):
        pred = model.predict(x_val, verbose=0)
        pretrain_val_rmse = float(
            compute_metrics_with_protocol(y_val, pred, sales_scaler=scaler, feature_columns=source_features)["rmse"]
        )
    return {
        "model": model,
        "history": history,
        "input_shape": x_train.shape[1:],
        "source_train_size": int(len(y_train)),
        "pretrain_train_loss_first_epoch": _history_loss(history, "loss", 0),
        "pretrain_train_loss_final_epoch": _history_loss(history, "loss", -1),
        "pretrain_val_rmse": pretrain_val_rmse,
    }


def _source_rows_for_selection(source_df: pd.DataFrame, sources: Sequence[Dict[str, Any]], max_source_rows: int | None) -> pd.DataFrame:
    parts = []
    for selected in sources:
        key = selected.get("source_key")
        key_tuple = tuple(key) if isinstance(key, (tuple, list)) else (key,)
        if len(key_tuple) < 2:
            continue
        entity_id, item_id = key_tuple[0], key_tuple[1]
        part = source_df[(source_df["entity_id"] == entity_id) & (source_df["item_id"] == item_id)].copy()
        limited = _limit_rows_by_group(part, max_source_rows)
        if max_source_rows is not None and max_source_rows > 0:
            limited.attrs["split_role"] = "source"
            limited.attrs["split_mode"] = "ratio"
            limited.attrs["split_config"] = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
        parts.append(limited)
    if not parts:
        raise ValueError("No source rows matched selected sources.")
    out = pd.concat(parts, axis=0, ignore_index=True)
    if max_source_rows is not None and max_source_rows > 0:
        out.attrs["split_role"] = "source"
        out.attrs["split_mode"] = "ratio"
        out.attrs["split_config"] = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
    return out


def _select_knn_sources(bundle: DatasetBundle, k: int, knn_representation: str | None) -> List[Dict[str, Any]]:
    selector = SourceSelector()
    result = selector.select_top_k_sources(
        target_df=bundle.target_df,
        source_df=bundle.source_df,
        feature_cols=bundle.feature_columns,
        k=k,
        weight_mode="inverse_distance",
        include_sales_in_knn=True,
        knn_representation=knn_representation,
    )
    return list(result.get("sources", []))


def _select_random_sources(source_df: pd.DataFrame, exclude_sources: Sequence[Dict[str, Any]], k: int, random_seed: int) -> List[Dict[str, Any]]:
    excluded = {tuple(row.get("source_key", ())) for row in exclude_sources}
    keys = []
    for key, _ in source_df.groupby(["entity_id", "item_id"], sort=False):
        key_tuple = tuple(key) if isinstance(key, tuple) else (key,)
        if key_tuple not in excluded:
            keys.append(key_tuple)
    if len(keys) < k:
        keys = [tuple(key) if isinstance(key, tuple) else (key,) for key, _ in source_df.groupby(["entity_id", "item_id"], sort=False)]
    rng = random.Random(random_seed)
    chosen = rng.sample(keys, k=min(k, len(keys)))
    weight = 1.0 / max(1, len(chosen))
    return [{"source_key": key, "distance": float("nan"), "weight": weight} for key in chosen]


def _row_base(bundle: DatasetBundle, method_variant: str, horizon: int, random_seed: int) -> Dict[str, Any]:
    return {
        column: ""
        for column in DETAIL_COLUMNS
    } | {
        "dataset_id": DATASET_ID[bundle.dataset_name],
        "dataset_name": bundle.dataset_name,
        "method_variant": method_variant,
        "horizon": int(horizon),
        "random_seed": int(random_seed),
        "target_window_days": int(bundle.target_df["date"].nunique()) if "date" in bundle.target_df.columns else len(bundle.target_df),
        "target_train_size": int(len(bundle.target_train)),
        "target_val_size": int(len(bundle.target_val)),
        "target_test_size": int(len(bundle.target_test)),
        "frozen_backbone": False,
        "shuffled_source": False,
        "shuffle_type": "",
        "random_source": False,
        "source_only": False,
    }


def _error_row(
    bundle: DatasetBundle,
    method_variant: str,
    horizon: int,
    random_seed: int,
    error: BaseException,
    notes: str = "",
    elapsed: float = 0.0,
) -> Dict[str, Any]:
    row = _row_base(bundle, method_variant, horizon, random_seed)
    row.update(
        {
            "source_uptake_status": "ERROR",
            "forgetting_status": "NOT_EVALUATED",
            "notes": notes,
            "run_time_seconds": round(float(elapsed), 3),
            "error_message": f"{type(error).__name__}: {error}",
        }
    )
    return row


def _fit_source_models_for_sources(
    bundle: DatasetBundle,
    sources: Sequence[Dict[str, Any]],
    horizon: int,
    window_size: int,
    learning_rate: float,
    source_epochs: int,
    batch_size: int,
    random_seed: int,
    max_source_rows: int | None,
    shuffle_source: bool = False,
) -> Dict[str, Any]:
    pretrains = []
    source_train_size = 0
    for offset, source in enumerate(sources):
        source_rows = _source_rows_for_selection(bundle.source_df, [source], max_source_rows)
        pretrain = _fit_source_model(
            source_rows=source_rows,
            feature_cols=bundle.feature_columns,
            horizon=horizon,
            window_size=window_size,
            learning_rate=learning_rate,
            epochs=source_epochs,
            batch_size=batch_size,
            random_seed=random_seed + offset,
            shuffle_source=shuffle_source,
        )
        source_train_size += int(pretrain.get("source_train_size", 0))
        pretrains.append(pretrain)
    return {
        "pretrains": pretrains,
        "source_train_size": source_train_size,
        "pretrain_train_loss_first_epoch": _mean_finite([p.get("pretrain_train_loss_first_epoch") for p in pretrains]),
        "pretrain_train_loss_final_epoch": _mean_finite([p.get("pretrain_train_loss_final_epoch") for p in pretrains]),
        "pretrain_val_rmse": _mean_finite([p.get("pretrain_val_rmse") for p in pretrains]),
    }


def _fused_pretrained_model(input_shape: Tuple[int, ...], pretrains: Sequence[Dict[str, Any]], weights: Sequence[float], learning_rate: float):
    if not pretrains:
        raise ValueError("No source pretrain models available.")
    if len(pretrains) == 1:
        model = build_base_cnn(input_shape, learning_rate=learning_rate)
        model.set_weights(pretrains[0]["model"].get_weights())
        return model
    source_models = [p["model"] for p in pretrains]
    weights_arr = np.asarray(list(weights), dtype=np.float64)
    if weights_arr.size != len(source_models) or not np.isfinite(weights_arr).all() or weights_arr.sum() <= 0:
        weights_arr = np.full(len(source_models), 1.0 / len(source_models))
    else:
        weights_arr = weights_arr / weights_arr.sum()
    fused = fuse_source_models_layerwise(source_models, list(weights_arr), ["conv1", "conv2"])
    model = build_base_cnn(input_shape, learning_rate=learning_rate)
    return load_fused_params_into_target_model(model, fused)


def _apply_frozen_backbone(model: object) -> None:
    for layer in model.layers:
        cls_name = layer.__class__.__name__.lower()
        name = layer.name.lower()
        if "conv" in cls_name or "conv" in name or "pool" in name or "flatten" in name:
            layer.trainable = False
        else:
            layer.trainable = True


def _target_initialized_row(
    bundle: DatasetBundle,
    method_variant: str,
    model: object,
    horizon: int,
    window_size: int,
    learning_rate: float,
    target_epochs: int,
    batch_size: int,
    metric_protocol: Dict[str, Any],
    random_seed: int,
    selected_sources: Sequence[Dict[str, Any]] = (),
    random_sources: Sequence[Dict[str, Any]] = (),
    pretrain_meta: Dict[str, Any] | None = None,
    frozen_backbone: bool = False,
    shuffled_source: bool = False,
    shuffle_type: str = "",
    random_source: bool = False,
    source_only: bool = False,
    notes: str = "",
) -> Dict[str, Any]:
    start = time.monotonic()
    row = _row_base(bundle, method_variant, horizon, random_seed)
    source_count = len(selected_sources) or len(random_sources)
    row.update(
        {
            "selected_sources": _sources_to_text(selected_sources),
            "random_sources": _sources_to_text(random_sources),
            "source_count": int(source_count),
            "source_train_size": "" if pretrain_meta is None else pretrain_meta.get("source_train_size", ""),
            "pretrain_train_loss_first_epoch": "" if pretrain_meta is None else pretrain_meta.get("pretrain_train_loss_first_epoch", ""),
            "pretrain_train_loss_final_epoch": "" if pretrain_meta is None else pretrain_meta.get("pretrain_train_loss_final_epoch", ""),
            "pretrain_val_rmse": "" if pretrain_meta is None else pretrain_meta.get("pretrain_val_rmse", ""),
            "before_finetune_val_rmse": _evaluate(
                model, bundle.target_val_scaled, horizon, window_size, bundle.feature_columns, metric_protocol, bundle.target_scaler
            ),
            "before_finetune_test_rmse": _evaluate(
                model, bundle.target_test_scaled, horizon, window_size, bundle.feature_columns, metric_protocol, bundle.target_scaler
            ),
            "frozen_backbone": bool(frozen_backbone),
            "shuffled_source": bool(shuffled_source),
            "shuffle_type": shuffle_type,
            "random_source": bool(random_source),
            "source_only": bool(source_only),
            "notes": notes,
        }
    )
    if source_only:
        row.update(
            {
                "after_finetune_val_rmse": row["before_finetune_val_rmse"],
                "after_finetune_test_rmse": row["before_finetune_test_rmse"],
                "run_time_seconds": round(time.monotonic() - start, 3),
            }
        )
        return row

    history = _fit_target(model, bundle, horizon, window_size, learning_rate, target_epochs, batch_size)
    row.update(
        {
            "after_finetune_val_rmse": _evaluate(
                model, bundle.target_val_scaled, horizon, window_size, bundle.feature_columns, metric_protocol, bundle.target_scaler
            ),
            "after_finetune_test_rmse": _evaluate(
                model, bundle.target_test_scaled, horizon, window_size, bundle.feature_columns, metric_protocol, bundle.target_scaler
            ),
            "target_train_loss_first_epoch": _history_loss(history, "loss", 0),
            "target_train_loss_final_epoch": _history_loss(history, "loss", -1),
            "target_val_rmse_first_epoch": _history_rmse(history, "val_loss", 0),
            "target_val_rmse_final_epoch": _history_rmse(history, "val_loss", -1),
            "run_time_seconds": round(time.monotonic() - start, 3),
        }
    )
    return row


def _run_variant(callable_obj, bundle: DatasetBundle, method_variant: str, horizon: int, random_seed: int, notes: str = "") -> Dict[str, Any]:
    start = time.monotonic()
    try:
        return callable_obj()
    except Exception as exc:  # audit rows must record failures instead of hiding them
        return _error_row(bundle, method_variant, horizon, random_seed, exc, notes=notes, elapsed=time.monotonic() - start)


def _audit_dataset_horizon(
    bundle: DatasetBundle,
    cfg: Dict[str, Any],
    horizon: int,
    random_seed: int,
    source_epochs: int,
    target_epochs: int,
    batch_size: int,
    window_size: int,
    max_sources: int,
    max_source_rows: int | None,
    knn_representation: str | None,
) -> List[Dict[str, Any]]:
    setup_reproducibility(random_seed)
    metric_protocol = _metric_protocol(cfg)
    learning_rate = float(cfg.get("single_experiment", {}).get("learning_rate", 1e-4))
    x_train, _ = _make_sequences(bundle.target_train_scaled, horizon, window_size, bundle.feature_columns)
    if len(x_train) == 0:
        raise ValueError("Target training split produced zero windows; reduce window_size/horizon.")
    input_shape = x_train.shape[1:]
    source_count = int(max_sources or cfg.get("k", 3))
    knn_sources = _select_knn_sources(bundle, source_count, knn_representation)
    random_sources = _select_random_sources(bundle.source_df, knn_sources, source_count, random_seed)

    rows: List[Dict[str, Any]] = []

    rows.append(
        _run_variant(
            lambda: _target_initialized_row(
                bundle=bundle,
                method_variant="No-TL",
                model=build_base_cnn(input_shape, learning_rate=learning_rate),
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                target_epochs=target_epochs,
                batch_size=batch_size,
                metric_protocol=metric_protocol,
                random_seed=random_seed,
                notes="Target-only baseline using current No-TL target training settings.",
            ),
            bundle,
            "No-TL",
            horizon,
            random_seed,
        )
    )

    rows.append(
        _run_variant(
            lambda: _target_initialized_row(
                bundle=bundle,
                method_variant="Random-init-target-finetune",
                model=build_base_cnn(input_shape, learning_rate=learning_rate),
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                target_epochs=target_epochs,
                batch_size=batch_size,
                metric_protocol=metric_protocol,
                random_seed=random_seed,
                notes="Same target fine-tune settings as TL, with random initialized weights and no source pretraining.",
            ),
            bundle,
            "Random-init-target-finetune",
            horizon,
            random_seed,
        )
    )

    def pretrained_finetune_row(method_variant: str, sources: Sequence[Dict[str, Any]], random_source_flag: bool, shuffled: bool, frozen: bool):
        pretrain_meta = _fit_source_models_for_sources(
            bundle=bundle,
            sources=sources,
            horizon=horizon,
            window_size=window_size,
            learning_rate=learning_rate,
            source_epochs=source_epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            max_source_rows=max_source_rows,
            shuffle_source=shuffled,
        )
        weights = [float(src.get("weight", 1.0 / max(1, len(sources)))) for src in sources]
        model = _fused_pretrained_model(input_shape, pretrain_meta["pretrains"], weights, learning_rate)
        if frozen:
            _apply_frozen_backbone(model)
        notes_parts = ["Source models are trained per selected source and fused with existing layerwise fusion."]
        if shuffled:
            notes_parts.append("Audit perturbation uses label shuffle before source pretraining.")
        if frozen:
            notes_parts.append("Conv/pool/flatten backbone layers are frozen during target fine-tune; dense output remains trainable.")
        return _target_initialized_row(
            bundle=bundle,
            method_variant=method_variant,
            model=model,
            horizon=horizon,
            window_size=window_size,
            learning_rate=learning_rate,
            target_epochs=target_epochs,
            batch_size=batch_size,
            metric_protocol=metric_protocol,
            random_seed=random_seed,
            selected_sources=() if random_source_flag else sources,
            random_sources=sources if random_source_flag else (),
            pretrain_meta=pretrain_meta,
            frozen_backbone=frozen,
            shuffled_source=shuffled,
            shuffle_type="label_shuffle" if shuffled else "",
            random_source=random_source_flag,
            notes=" ".join(notes_parts),
        )

    for method_variant, sources, random_source_flag, shuffled, frozen in [
        ("KNN-source-pretrain-finetune", knn_sources, False, False, False),
        ("Random-source-pretrain-finetune", random_sources, True, False, False),
        ("Shuffled-source-pretrain-finetune", knn_sources, False, True, False),
        ("KNN-source-pretrain-frozen-backbone", knn_sources, False, False, True),
    ]:
        rows.append(
            _run_variant(
                lambda mv=method_variant, src=sources, rs=random_source_flag, sh=shuffled, fr=frozen: pretrained_finetune_row(
                    mv, src, rs, sh, fr
                ),
                bundle,
                method_variant,
                horizon,
                random_seed,
            )
        )

    for source in knn_sources:
        def source_only_row(src=source):
            pretrain_meta = _fit_source_models_for_sources(
                bundle=bundle,
                sources=[src],
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                source_epochs=source_epochs,
                batch_size=batch_size,
                random_seed=random_seed,
                max_source_rows=max_source_rows,
                shuffle_source=False,
            )
            model = _fused_pretrained_model(input_shape, pretrain_meta["pretrains"], [1.0], learning_rate)
            return _target_initialized_row(
                bundle=bundle,
                method_variant="Source-only-prediction",
                model=model,
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                target_epochs=target_epochs,
                batch_size=batch_size,
                metric_protocol=metric_protocol,
                random_seed=random_seed,
                selected_sources=[src],
                pretrain_meta=pretrain_meta,
                source_only=True,
                notes="No target fine-tune; each KNN source model directly predicts the target validation/test split.",
            )

        rows.append(_run_variant(source_only_row, bundle, "Source-only-prediction", horizon, random_seed))

    _add_delta_and_status(rows)
    return rows


def _rmse_for(rows: Sequence[Dict[str, Any]], variant: str, reducer: str = "first") -> float:
    vals = [_as_float(row.get("after_finetune_test_rmse")) for row in rows if row.get("method_variant") == variant]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals)) if reducer == "mean" else vals[0]


def _append_status(statuses: Dict[int, List[str]], row_index: int, status: str) -> None:
    statuses.setdefault(row_index, []).append(status)


def _add_delta_and_status(rows: List[Dict[str, Any]]) -> None:
    notl = _rmse_for(rows, "No-TL")
    random_init = _rmse_for(rows, "Random-init-target-finetune")
    knn = _rmse_for(rows, "KNN-source-pretrain-finetune")
    random_source = _rmse_for(rows, "Random-source-pretrain-finetune")
    shuffled = _rmse_for(rows, "Shuffled-source-pretrain-finetune")
    frozen = _rmse_for(rows, "KNN-source-pretrain-frozen-backbone")
    source_only = _rmse_for(rows, "Source-only-prediction", reducer="mean")

    statuses: Dict[int, List[str]] = {}
    forgetting: Dict[int, str] = {}
    variant_to_indices: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        variant_to_indices.setdefault(str(row.get("method_variant")), []).append(idx)

    for idx in variant_to_indices.get("KNN-source-pretrain-finetune", []):
        if _close_within_one_pct(knn, random_init):
            _append_status(statuses, idx, "NO_EVIDENCE_OF_SOURCE_UPTAKE")
        if _pct_better(knn, random_init) > 0.03:
            _append_status(statuses, idx, "SOURCE_PRETRAINING_USEFUL")
        if _pct_better(knn, random_source) > 0.03:
            _append_status(statuses, idx, "KNN_SOURCE_USEFUL")
        if _close_within_one_pct(knn, random_source):
            _append_status(statuses, idx, "KNN_NOT_BETTER_THAN_RANDOM_SOURCE")

    for idx in variant_to_indices.get("Shuffled-source-pretrain-finetune", []):
        if _close_within_one_pct(shuffled, knn):
            _append_status(statuses, idx, "SOURCE_LABEL_OR_TEMPORAL_INFO_NOT_USED")

    for idx in variant_to_indices.get("KNN-source-pretrain-frozen-backbone", []):
        if _pct_better(frozen, knn) > 0.03:
            _append_status(statuses, idx, "SOURCE_FEATURES_USEFUL_BUT_FORGOTTEN")

    for idx in variant_to_indices.get("Source-only-prediction", []):
        this_rmse = _as_float(rows[idx].get("after_finetune_test_rmse"))
        if _pct_better(this_rmse, notl) > 0.03:
            _append_status(statuses, idx, "SOURCE_MODEL_HAS_TRANSFER_SIGNAL")

    for idx, row in enumerate(rows):
        before = _as_float(row.get("before_finetune_test_rmse"))
        after = _as_float(row.get("after_finetune_test_rmse"))
        if np.isfinite(before) and np.isfinite(after) and after > before * 1.03:
            forgetting[idx] = "CATASTROPHIC_FORGETTING_RISK"
        else:
            forgetting[idx] = "NO_CLEAR_FORGETTING_RISK" if np.isfinite(before) and np.isfinite(after) else "NOT_EVALUATED"

    for idx, row in enumerate(rows):
        if row.get("error_message"):
            row["source_uptake_status"] = "ERROR"
            row["forgetting_status"] = "NOT_EVALUATED"
            continue
        test_rmse = _as_float(row.get("after_finetune_test_rmse"))
        row["rmse_delta_vs_notl"] = _safe_delta(test_rmse, notl)
        row["rmse_pct_delta_vs_notl"] = _safe_pct_delta(test_rmse, notl)
        row["rmse_delta_vs_random_init"] = _safe_delta(test_rmse, random_init)
        row["rmse_pct_delta_vs_random_init"] = _safe_pct_delta(test_rmse, random_init)
        row["rmse_delta_vs_knn_source"] = _safe_delta(test_rmse, knn)
        row["rmse_pct_delta_vs_knn_source"] = _safe_pct_delta(test_rmse, knn)
        row["source_uptake_status"] = "|".join(statuses.get(idx, [])) or "REFERENCE_OR_INCONCLUSIVE"
        row["forgetting_status"] = forgetting.get(idx, "NOT_EVALUATED")


def _interpret_status(status: str) -> str:
    if not status:
        return "证据不足。"
    if "ERROR" in status:
        return "该变体运行失败，需查看 details 的 error_message。"
    mapping = {
        "NO_EVIDENCE_OF_SOURCE_UPTAKE": "KNN 源预训练与随机初始化差异小于 1%，没有源信息吸收证据。",
        "SOURCE_PRETRAINING_USEFUL": "KNN 源预训练相对随机初始化降低 RMSE 超过 3%。",
        "KNN_SOURCE_USEFUL": "KNN 源相对随机源降低 RMSE 超过 3%。",
        "KNN_NOT_BETTER_THAN_RANDOM_SOURCE": "KNN 源与随机源差异小于 1%。",
        "SOURCE_LABEL_OR_TEMPORAL_INFO_NOT_USED": "打乱源标签后的结果接近真实源，源标签/时序信息可能未被利用。",
        "SOURCE_FEATURES_USEFUL_BUT_FORGOTTEN": "冻结 backbone 优于 full fine-tune，源特征可能被覆盖。",
        "SOURCE_MODEL_HAS_TRANSFER_SIGNAL": "source-only 直接预测优于 No-TL，单源模型含可迁移信号。",
        "REFERENCE_OR_INCONCLUSIVE": "参考行或当前比较未触发明确判定。",
    }
    return "；".join(mapping.get(part, part) for part in str(status).split("|") if part)


def build_summary(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    details = pd.DataFrame(list(rows), columns=DETAIL_COLUMNS)
    if details.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    numeric_cols = [
        "after_finetune_test_rmse",
        "rmse_delta_vs_notl",
        "rmse_pct_delta_vs_notl",
        "rmse_delta_vs_random_init",
        "rmse_pct_delta_vs_random_init",
    ]
    for col in numeric_cols:
        details[col] = pd.to_numeric(details[col], errors="coerce")

    out_rows: List[Dict[str, Any]] = []
    grouped = details.groupby(["dataset_id", "dataset_name", "horizon", "method_variant"], dropna=False)
    for key, group in grouped:
        dataset_id, dataset_name, horizon, method_variant = key
        statuses = sorted(
            {
                part
                for value in group["source_uptake_status"].dropna().astype(str)
                for part in value.split("|")
                if part
            }
        )
        random_source_rmse = details[
            (details["dataset_id"] == dataset_id)
            & (details["horizon"] == horizon)
            & (details["method_variant"] == "Random-source-pretrain-finetune")
        ]["after_finetune_test_rmse"].mean()
        rmse = group["after_finetune_test_rmse"]
        status_text = "|".join(statuses)
        out_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "horizon": horizon,
                "method_variant": method_variant,
                "n_runs": int(len(group)),
                "mean_test_rmse": float(rmse.mean()) if not rmse.dropna().empty else float("nan"),
                "std_test_rmse": float(rmse.std(ddof=0)) if not rmse.dropna().empty else float("nan"),
                "mean_delta_vs_notl": float(group["rmse_delta_vs_notl"].mean()),
                "mean_pct_delta_vs_notl": float(group["rmse_pct_delta_vs_notl"].mean()),
                "mean_delta_vs_random_init": float(group["rmse_delta_vs_random_init"].mean()),
                "mean_pct_delta_vs_random_init": float(group["rmse_pct_delta_vs_random_init"].mean()),
                "win_rate_vs_notl": float((group["rmse_delta_vs_notl"] < 0).mean()),
                "win_rate_vs_random_init": float((group["rmse_delta_vs_random_init"] < 0).mean()),
                "win_rate_vs_random_source": float((rmse < random_source_rmse).mean()) if np.isfinite(random_source_rmse) else float("nan"),
                "source_uptake_status": status_text,
                "interpretation": _interpret_status(status_text),
            }
        )
    return pd.DataFrame(out_rows, columns=SUMMARY_COLUMNS)


def _summary_rmse(summary: pd.DataFrame, dataset_id: int, variant: str) -> float:
    hit = summary[(summary["dataset_id"] == dataset_id) & (summary["method_variant"] == variant)]
    if hit.empty:
        return float("nan")
    return float(hit["mean_test_rmse"].iloc[0])


def _better_text(candidate: float, baseline: float, threshold: float = 0.0) -> str:
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        return "无足够数据"
    improvement = _pct_better(candidate, baseline)
    if improvement > threshold:
        return "是"
    if improvement < -threshold:
        return "否"
    return "接近"


def _dataset_table(summary: pd.DataFrame, dataset_id: int) -> str:
    cols = ["method_variant", "n_runs", "mean_test_rmse", "mean_delta_vs_notl", "source_uptake_status"]
    table = summary[summary["dataset_id"] == dataset_id][cols].copy()
    if table.empty:
        return "无结果。"
    for col in ["mean_test_rmse", "mean_delta_vs_notl"]:
        table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def build_markdown_report(rows: Sequence[Dict[str, Any]], summary: pd.DataFrame) -> str:
    details = pd.DataFrame(list(rows), columns=DETAIL_COLUMNS)
    lines = [
        "# source_information_uptake_audit",
        "",
        "## 1. 审计目的",
        "",
        "检查 TL / MSWA / MSSB / MSML-TL-RFE 等迁移思路表现不佳时，当前 CNN 迁移流程是否真实吸收了源域信息。",
        "",
        "## 2. 实验设置",
        "",
        "- 默认 seed 为 42。",
        "- 默认 horizon 使用主配置 horizon=1。",
        "- 默认只写入 outputs/audits，不覆盖主实验 CSV。",
        "- RMSE 使用工程现有 compute_metrics_with_protocol 口径。",
        "",
        "## 3. method_variant 含义",
        "",
        "- No-TL：只用目标域训练。",
        "- Random-init-target-finetune：随机初始化后使用 TL fine-tune 阶段的目标训练设置。",
        "- KNN-source-pretrain-finetune：KNN 选源，源预训练后目标域 full fine-tune。",
        "- Random-source-pretrain-finetune：随机选同数量源，源预训练后目标域 full fine-tune。",
        "- Shuffled-source-pretrain-finetune：KNN 源标签打乱后预训练，再目标域 fine-tune。",
        "- KNN-source-pretrain-frozen-backbone：KNN 源预训练后冻结 Conv/pool/flatten backbone，仅训练 dense 输出层。",
        "- Source-only-prediction：每个 KNN 源单独训练，不做目标域 fine-tune，直接预测目标 test set。",
        "",
        "## 4. Dataset1 / Dataset2 / Dataset3 的结果表",
        "",
    ]
    for dataset_id in [1, 2, 3]:
        name = f"Dataset{dataset_id}"
        if not details.empty and dataset_id in set(pd.to_numeric(details["dataset_id"], errors="coerce").dropna().astype(int)):
            name = str(details.loc[pd.to_numeric(details["dataset_id"], errors="coerce") == dataset_id, "dataset_name"].iloc[0])
        lines.extend([f"### {name}", "", _dataset_table(summary, dataset_id), ""])

    lines.extend(["## 5. 源信息吸收判断", ""])
    all_transfer_wins = []
    for dataset_id in sorted(pd.to_numeric(details.get("dataset_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique()):
        name = str(details.loc[pd.to_numeric(details["dataset_id"], errors="coerce") == dataset_id, "dataset_name"].iloc[0])
        notl = _summary_rmse(summary, dataset_id, "No-TL")
        transfer_variants = [
            "KNN-source-pretrain-finetune",
            "Random-source-pretrain-finetune",
            "Shuffled-source-pretrain-finetune",
            "KNN-source-pretrain-frozen-backbone",
        ]
        wins = [_pct_better(_summary_rmse(summary, dataset_id, v), notl) > 0.03 for v in transfer_variants]
        all_transfer_wins.extend(wins)
        status = "|".join(
            sorted(
                {
                    part
                    for value in details.loc[pd.to_numeric(details["dataset_id"], errors="coerce") == dataset_id, "source_uptake_status"].astype(str)
                    for part in value.split("|")
                    if part
                }
            )
        )
        lines.append(f"- {name}：{status or 'INCONCLUSIVE'}。")
    if all_transfer_wins and not any(all_transfer_wins):
        lines.append("- 当前没有足够证据表明迁移方法真实吸收了源域信息。")
    lines.append("")

    comparison_sections = [
        ("## 6. KNN 选源是否优于随机源", "KNN-source-pretrain-finetune", "Random-source-pretrain-finetune", 0.03),
        ("## 7. 源预训练是否优于随机初始化", "KNN-source-pretrain-finetune", "Random-init-target-finetune", 0.03),
        ("## 8. Shuffled-source 是否接近真实源", "Shuffled-source-pretrain-finetune", "KNN-source-pretrain-finetune", 0.01),
        ("## 9. Frozen backbone 是否优于 full fine-tune", "KNN-source-pretrain-frozen-backbone", "KNN-source-pretrain-finetune", 0.03),
        ("## 10. Source-only 是否显示可迁移信号", "Source-only-prediction", "No-TL", 0.03),
    ]
    for title, candidate_variant, baseline_variant, threshold in comparison_sections:
        lines.extend([title, ""])
        for dataset_id in sorted(pd.to_numeric(details.get("dataset_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique()):
            name = str(details.loc[pd.to_numeric(details["dataset_id"], errors="coerce") == dataset_id, "dataset_name"].iloc[0])
            candidate = _summary_rmse(summary, dataset_id, candidate_variant)
            baseline = _summary_rmse(summary, dataset_id, baseline_variant)
            if "Shuffled" in title:
                verdict = "接近" if _close_within_one_pct(candidate, baseline) else "不接近或无足够数据"
            else:
                verdict = _better_text(candidate, baseline, threshold)
            lines.append(f"- {name}：{verdict}。")
        lines.append("")

    lines.extend(["## 11. 是否存在 catastrophic forgetting 风险", ""])
    for dataset_id in sorted(pd.to_numeric(details.get("dataset_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique()):
        name = str(details.loc[pd.to_numeric(details["dataset_id"], errors="coerce") == dataset_id, "dataset_name"].iloc[0])
        flags = details[
            (pd.to_numeric(details["dataset_id"], errors="coerce") == dataset_id)
            & (details["forgetting_status"].astype(str) == "CATASTROPHIC_FORGETTING_RISK")
        ]["method_variant"].astype(str).tolist()
        lines.append(f"- {name}：{'存在，涉及 ' + ', '.join(flags) if flags else '未见明确证据'}。")
    lines.extend(["", "## 12. 总结结论", ""])
    lines.append("不同 dataset 的结论按各自结果解释；若结果不一致，不强行合并为单一结论。")
    if all_transfer_wins and not any(all_transfer_wins):
        lines.append("当前没有足够证据表明迁移方法真实吸收了源域信息。")
    lines.extend(["", "## 13. 局限性说明", ""])
    lines.append("- 该脚本是审计包装，不修改主训练逻辑、CNN 结构、KNN、RFE、split 或 RMSE 公式。")
    lines.append("- 为控制成本，默认每个 dataset 只跑 horizon=1；quick 模式进一步缩小 epoch、source 数和样本行数。")
    lines.append("- 源扰动采用 label shuffle；如果某个变体失败，details CSV 的 error_message 记录原因。")
    return "\n".join(lines)


def _resolve_output_paths(output_dir: str | Path | None) -> Tuple[Path, Path, Path]:
    out_dir = Path(output_dir) if output_dir is not None else OUT_DIR
    return (
        out_dir / "source_information_uptake_details.csv",
        out_dir / "source_information_uptake_summary.csv",
        out_dir / "source_information_uptake.md",
    )


def run_audit(
    datasets: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    random_seed: int = 42,
    source_epochs: int | None = None,
    target_epochs: int | None = None,
    batch_size: int | None = None,
    window_size: int | None = None,
    max_sources: int | None = None,
    max_source_rows: int | None = None,
    max_target_rows: int | None = None,
    knn_representation: str | None = None,
    write_outputs: bool = True,
    output_dir: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    cfg = _load_config()
    selected_datasets = list(datasets or DATASETS)
    selected_horizons = [int(h) for h in (horizons or [int(cfg.get("horizon", 1))])]
    exp = cfg.get("single_experiment", {})
    source_epochs = int(source_epochs if source_epochs is not None else cfg.get("source_epochs", exp.get("source_epochs", 2)))
    target_epochs = int(target_epochs if target_epochs is not None else cfg.get("target_epochs", exp.get("target_epochs", 2)))
    batch_size = int(batch_size if batch_size is not None else cfg.get("batch_size", exp.get("batch_size", 16)))
    window_size = int(window_size if window_size is not None else exp.get("window_size", 10))
    max_sources = int(max_sources if max_sources is not None else cfg.get("k", 3))
    if knn_representation is None:
        knn_representation = (
            cfg.get("paper_reproduction", {})
            .get("strict_source_selection", {})
            .get("knn_representation")
        )

    all_rows: List[Dict[str, Any]] = []
    for dataset_name in selected_datasets:
        try:
            bundle = _prepare_dataset(dataset_name, cfg, max_target_rows=max_target_rows)
            for horizon in selected_horizons:
                try:
                    all_rows.extend(
                        _audit_dataset_horizon(
                            bundle=bundle,
                            cfg=cfg,
                            horizon=horizon,
                            random_seed=random_seed,
                            source_epochs=source_epochs,
                            target_epochs=target_epochs,
                            batch_size=batch_size,
                            window_size=window_size,
                            max_sources=max_sources,
                            max_source_rows=max_source_rows,
                            knn_representation=knn_representation,
                        )
                    )
                except Exception as exc:
                    for variant in METHOD_VARIANTS:
                        all_rows.append(_error_row(bundle, variant, horizon, random_seed, exc, notes="Dataset/horizon audit failed."))
        except Exception as exc:
            fake_bundle = DatasetBundle(
                dataset_name=dataset_name,
                source_df=pd.DataFrame(),
                target_df=pd.DataFrame(),
                target_train=pd.DataFrame(),
                target_val=pd.DataFrame(),
                target_test=pd.DataFrame(),
                target_train_scaled=pd.DataFrame(),
                target_val_scaled=pd.DataFrame(),
                target_test_scaled=pd.DataFrame(),
                target_scaler=None,
                feature_columns=[],
            )
            for horizon in selected_horizons:
                for variant in METHOD_VARIANTS:
                    all_rows.append(_error_row(fake_bundle, variant, horizon, random_seed, exc, notes="Dataset preparation failed."))

    details = pd.DataFrame(all_rows, columns=DETAIL_COLUMNS)
    summary = build_summary(all_rows)
    report = build_markdown_report(all_rows, summary)
    if write_outputs:
        details_path, summary_path, report_path = _resolve_output_paths(output_dir)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(details_path, index=False)
        summary.to_csv(summary_path, index=False)
        report_path.write_text(report, encoding="utf-8")
    return {"details": details, "summary": summary}


def _parse_csv_arg(value: str, cast=str) -> List[Any]:
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def _datasets_from_arg(dataset_id: str) -> List[str]:
    value = str(dataset_id or "all").strip().lower()
    if value == "all":
        return list(DATASETS)
    if value in DATASET_BY_ID:
        return [DATASET_BY_ID[value]]
    raise ValueError("--dataset-id must be one of 1, 2, 3, all")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="all", help="Dataset id: 1, 2, 3, or all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run Dataset1 with minimal epochs/source rows for tests.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--datasets", default=None, help="Backward-compatible comma-separated dataset names.")
    parser.add_argument("--horizons", default="1", help="Comma-separated horizons.")
    parser.add_argument("--random-seed", type=int, default=None, help="Backward-compatible alias for --seed.")
    parser.add_argument("--source-epochs", type=int, default=None)
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--max-target-rows", type=int, default=None)
    parser.add_argument("--knn-representation", default=None)
    args = parser.parse_args(argv)

    seed = int(args.random_seed if args.random_seed is not None else args.seed)
    datasets = _parse_csv_arg(args.datasets, str) if args.datasets else _datasets_from_arg(args.dataset_id)
    horizons = _parse_csv_arg(args.horizons, int)
    source_epochs = args.source_epochs
    target_epochs = args.target_epochs
    batch_size = args.batch_size
    window_size = args.window_size
    max_sources = args.max_sources
    max_source_rows = args.max_source_rows
    max_target_rows = args.max_target_rows

    if args.quick:
        datasets = ["Dataset1"]
        horizons = [1]
        source_epochs = 50 if source_epochs is None else source_epochs
        target_epochs = 50 if target_epochs is None else target_epochs
        batch_size = 4 if batch_size is None else batch_size
        window_size = 8 if window_size is None else window_size
        max_sources = 1 if max_sources is None else max_sources
        max_source_rows = 40 if max_source_rows is None else max_source_rows
        max_target_rows = 70 if max_target_rows is None else max_target_rows

    run_audit(
        datasets=datasets,
        horizons=horizons,
        random_seed=seed,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
        window_size=window_size,
        max_sources=max_sources,
        max_source_rows=max_source_rows,
        max_target_rows=max_target_rows,
        knn_representation=args.knn_representation,
        write_outputs=True,
        output_dir=args.output_dir,
    )
    details_path, summary_path, report_path = _resolve_output_paths(args.output_dir)
    print(f"Wrote {details_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
