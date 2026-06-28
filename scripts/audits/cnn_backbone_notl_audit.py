"""Read-only audit of the No-TL CNN backbone for small-sample settings.

This script is isolated from the main experiment runner. It reuses the current
No-TL data construction path and cnn_model.py, but does not modify training,
KNN, RFE, cleaning, split, or metric code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from data_preprocessing import (
    build_source_target_split,
    build_tabular_sequence,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from environment import setup_logging, setup_reproducibility
from src.evaluation.metrics import compute_metrics_with_protocol
from src.models.no_tl_model import build_no_tl_cnn_model


OUT_DIR = ROOT / "outputs" / "audits"
SHAPE_AUDIT_CSV = OUT_DIR / "cnn_backbone_shape_audit.csv"
ABLATION_CSV = OUT_DIR / "cnn_backbone_notl_ablation.csv"
REPORT_MD = OUT_DIR / "cnn_backbone_notl_ablation.md"

RANDOM_SEED = 42
DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
HORIZON = 1

SHAPE_AUDIT_COLUMNS = [
    "dataset_id",
    "dataset",
    "model_name",
    "random_seed",
    "input_shape",
    "conv1d_layer_count",
    "filters",
    "kernel_size",
    "padding",
    "pooling",
    "flatten_dense_params",
    "output_activation",
    "total_params",
    "trainable_params",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_windows",
    "val_windows",
    "test_windows",
    "window_size",
    "feature_dim",
    "layer_output_shapes",
    "minimum_time_dim",
    "time_dimension_risk",
    "risk_notes",
]

ABLATION_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "model_name",
    "random_seed",
    "horizon",
    "target_epochs",
    "batch_size",
    "window_size",
    "feature_dim",
    "train_windows",
    "val_windows",
    "test_windows",
    "rmse",
    "normalized_rmse",
    "original_scale_rmse",
    "accuracy",
    "normalized_accuracy",
    "original_scale_accuracy",
    "mae",
    "normalized_mae",
    "original_scale_mae",
    "metric_space",
    "prediction_shape",
    "run_time_seconds",
    "status",
    "error_message",
    "notes",
]


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    factory: Callable[[Tuple[int, int], float], Any] | None
    notes: str


@dataclass(frozen=True)
class SplitBundle:
    dataset: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_scaled: pd.DataFrame
    val_scaled: pd.DataFrame
    test_scaled: pd.DataFrame
    scaler: Any
    feature_columns: List[str]


def _build_conv1_gap_dense(input_shape: Tuple[int, int], learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=16, kernel_size=3, padding="valid", activation="relu", name="conv1")(inputs)
    x = GlobalAveragePooling1D(name="gap")(x)
    outputs = Dense(1, name="dense_out")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def _build_conv1_flatten_dense(input_shape: Tuple[int, int], learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Conv1D, Dense, Flatten
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=16, kernel_size=3, padding="valid", activation="relu", name="conv1")(inputs)
    x = Flatten(name="flatten")(x)
    outputs = Dense(1, name="dense_out")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def _build_dense_only_mlp(input_shape: Tuple[int, int], learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = Flatten(name="flatten")(inputs)
    x = Dense(16, activation="relu", name="dense_hidden")(x)
    outputs = Dense(1, name="dense_out")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


BACKBONE_SPECS = [
    BackboneSpec(
        name="current_3layer_cnn",
        factory=lambda input_shape, learning_rate: build_no_tl_cnn_model(input_shape, learning_rate),
        notes="Current No-TL backbone from src/models/cnn_model.py.",
    ),
    BackboneSpec(
        name="conv1_gap_dense",
        factory=_build_conv1_gap_dense,
        notes="Audit-only 1-layer Conv1D + GlobalAveragePooling1D + Dense.",
    ),
    BackboneSpec(
        name="conv1_flatten_dense",
        factory=_build_conv1_flatten_dense,
        notes="Audit-only 1-layer Conv1D + Flatten + Dense.",
    ),
    BackboneSpec(
        name="dense_only_mlp",
        factory=_build_dense_only_mlp,
        notes="Audit-only Dense MLP without convolution.",
    ),
    BackboneSpec(
        name="naive_persistence",
        factory=None,
        notes="No-training baseline: predict last observed normalized sales in the input window.",
    ),
]


def _load_config() -> Dict[str, Any]:
    return json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))


def _prepare_dataset(dataset: str, cfg: Dict[str, Any]) -> SplitBundle:
    data_path = ROOT / str(cfg["dataset_paths"][dataset])
    raw_df = load_dataset(dataset_name=dataset, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    _, target_df = build_source_target_split(processed_df, cfg)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df.copy())
    train_scaled, val_scaled, test_scaled, scaler, feature_columns = normalize_features(train_df, val_df, test_df)
    return SplitBundle(
        dataset=dataset,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        scaler=scaler,
        feature_columns=feature_columns,
    )


def _build_sequences(bundle: SplitBundle, window_size: int) -> Dict[str, np.ndarray]:
    x_train, y_train = build_tabular_sequence(bundle.train_scaled, horizon=HORIZON, window_size=window_size)
    x_val, y_val = build_tabular_sequence(bundle.val_scaled, horizon=HORIZON, window_size=window_size)
    x_test, y_test = build_tabular_sequence(bundle.test_scaled, horizon=HORIZON, window_size=window_size)
    return {
        "x_train": to_cnn_tensor(x_train),
        "y_train": y_train,
        "x_val": to_cnn_tensor(x_val),
        "y_val": y_val,
        "x_test": to_cnn_tensor(x_test),
        "y_test": y_test,
    }


def _shape_tuple(value: Any) -> Tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError:
        return tuple(value.as_list())


def _layer_shapes(model: Any) -> List[str]:
    shapes: List[str] = []
    for layer in model.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        try:
            output_shape = _shape_tuple(layer.output.shape)
        except Exception:
            output_shape = ("unknown",)
        shapes.append(f"{layer.name}:{output_shape}")
    return shapes


def _activation_name(layer: Any) -> str:
    activation = getattr(layer, "activation", None)
    return getattr(activation, "__name__", "linear" if activation is None else str(activation))


def _summarize_model(model: Any, bundle: SplitBundle, sequences: Dict[str, np.ndarray], spec: BackboneSpec, window_size: int) -> Dict[str, Any]:
    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == "Conv1D"]
    pool_layers = [
        layer
        for layer in model.layers
        if layer.__class__.__name__ in {"MaxPooling1D", "AveragePooling1D", "GlobalAveragePooling1D"}
    ]
    flatten_dense_params = int(
        sum(layer.count_params() for layer in model.layers if layer.__class__.__name__ in {"Flatten", "Dense"})
    )

    time_dims: List[int] = []
    for layer in model.layers:
        try:
            shape = _shape_tuple(layer.output.shape)
        except Exception:
            continue
        if len(shape) >= 3 and isinstance(shape[1], int):
            time_dims.append(int(shape[1]))
    minimum_time_dim = min(time_dims) if time_dims else np.nan
    risk_parts: List[str] = []
    if len(sequences["y_train"]) < 10:
        risk_parts.append("very_few_train_windows")
    if time_dims and min(time_dims) <= 1:
        risk_parts.append("time_dim_collapsed_to_1")
    if window_size <= 3 and conv_layers:
        risk_parts.append("window_near_conv_kernel_size")
    if spec.name == "current_3layer_cnn" and len(sequences["y_train"]) <= flatten_dense_params:
        risk_parts.append("train_windows_not_greater_than_flatten_dense_params")
    if spec.name == "current_3layer_cnn" and pool_layers and window_size < 8:
        risk_parts.append("short_window_with_two_pooling_layers")

    output_layer = model.layers[-1]
    return {
        "dataset_id": DATASET_ID[bundle.dataset],
        "dataset": bundle.dataset,
        "model_name": spec.name,
        "random_seed": RANDOM_SEED,
        "input_shape": tuple(sequences["x_train"].shape[1:]),
        "conv1d_layer_count": int(len(conv_layers)),
        "filters": "|".join(str(layer.filters) for layer in conv_layers) or "NONE",
        "kernel_size": "|".join(str(tuple(layer.kernel_size)) for layer in conv_layers) or "NONE",
        "padding": "|".join(str(layer.padding) for layer in conv_layers) or "NONE",
        "pooling": "|".join(f"{layer.name}:{layer.__class__.__name__}" for layer in pool_layers) or "NONE",
        "flatten_dense_params": flatten_dense_params,
        "output_activation": _activation_name(output_layer),
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(np.prod(weight.shape) for weight in model.trainable_weights)),
        "train_rows": int(len(bundle.train_df)),
        "val_rows": int(len(bundle.val_df)),
        "test_rows": int(len(bundle.test_df)),
        "train_windows": int(len(sequences["y_train"])),
        "val_windows": int(len(sequences["y_val"])),
        "test_windows": int(len(sequences["y_test"])),
        "window_size": int(window_size),
        "feature_dim": int(sequences["x_train"].shape[-1]),
        "layer_output_shapes": " | ".join(_layer_shapes(model)),
        "minimum_time_dim": minimum_time_dim,
        "time_dimension_risk": bool(risk_parts),
        "risk_notes": "|".join(risk_parts) if risk_parts else "none",
    }


def _persistence_shape_row(bundle: SplitBundle, sequences: Dict[str, np.ndarray], window_size: int) -> Dict[str, Any]:
    return {
        "dataset_id": DATASET_ID[bundle.dataset],
        "dataset": bundle.dataset,
        "model_name": "naive_persistence",
        "random_seed": RANDOM_SEED,
        "input_shape": tuple(sequences["x_train"].shape[1:]),
        "conv1d_layer_count": 0,
        "filters": "NONE",
        "kernel_size": "NONE",
        "padding": "NONE",
        "pooling": "NONE",
        "flatten_dense_params": 0,
        "output_activation": "NONE",
        "total_params": 0,
        "trainable_params": 0,
        "train_rows": int(len(bundle.train_df)),
        "val_rows": int(len(bundle.val_df)),
        "test_rows": int(len(bundle.test_df)),
        "train_windows": int(len(sequences["y_train"])),
        "val_windows": int(len(sequences["y_val"])),
        "test_windows": int(len(sequences["y_test"])),
        "window_size": int(window_size),
        "feature_dim": int(sequences["x_train"].shape[-1]),
        "layer_output_shapes": "naive:last_observed_sales",
        "minimum_time_dim": int(window_size),
        "time_dimension_risk": len(sequences["y_train"]) < 10,
        "risk_notes": "very_few_train_windows" if len(sequences["y_train"]) < 10 else "none",
    }


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray, metric_protocol: Dict[str, Any], bundle: SplitBundle) -> Dict[str, Any]:
    return compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=bundle.scaler,
        feature_columns=bundle.feature_columns,
    )


def _empty_ablation_row(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    spec: BackboneSpec,
    window_size: int,
    target_epochs: int,
    batch_size: int,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "dataset_id": DATASET_ID[bundle.dataset],
        "dataset": bundle.dataset,
        "method": "No-TL",
        "model_name": spec.name,
        "random_seed": RANDOM_SEED,
        "horizon": HORIZON,
        "target_epochs": int(target_epochs),
        "batch_size": int(batch_size),
        "window_size": int(window_size),
        "feature_dim": int(sequences["x_train"].shape[-1]),
        "train_windows": int(len(sequences["y_train"])),
        "val_windows": int(len(sequences["y_val"])),
        "test_windows": int(len(sequences["y_test"])),
        "rmse": np.nan,
        "normalized_rmse": np.nan,
        "original_scale_rmse": np.nan,
        "accuracy": np.nan,
        "normalized_accuracy": np.nan,
        "original_scale_accuracy": np.nan,
        "mae": np.nan,
        "normalized_mae": np.nan,
        "original_scale_mae": np.nan,
        "metric_space": "normalized_minmax_space",
        "prediction_shape": "(skipped)",
        "run_time_seconds": 0.0,
        "status": status,
        "error_message": error_message,
        "notes": spec.notes,
    }


def _run_persistence(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    metric_protocol: Dict[str, Any],
    spec: BackboneSpec,
    window_size: int,
    target_epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    if len(sequences["y_test"]) == 0:
        return _empty_ablation_row(bundle, sequences, spec, window_size, target_epochs, batch_size, "SKIPPED", "empty test windows")
    if "sales" not in bundle.feature_columns:
        return _empty_ablation_row(bundle, sequences, spec, window_size, target_epochs, batch_size, "ERROR", "sales not in feature columns")

    start = time.perf_counter()
    sales_idx = bundle.feature_columns.index("sales")
    y_pred = sequences["x_test"][:, -1, sales_idx].reshape(-1, 1)
    metric = _metric_dict(sequences["y_test"], y_pred, metric_protocol, bundle)
    return _ablation_metric_row(
        bundle=bundle,
        sequences=sequences,
        spec=spec,
        metric=metric,
        y_pred=y_pred,
        window_size=window_size,
        target_epochs=target_epochs,
        batch_size=batch_size,
        run_time_seconds=time.perf_counter() - start,
    )


def _ablation_metric_row(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    spec: BackboneSpec,
    metric: Dict[str, Any],
    y_pred: np.ndarray,
    window_size: int,
    target_epochs: int,
    batch_size: int,
    run_time_seconds: float,
) -> Dict[str, Any]:
    return {
        "dataset_id": DATASET_ID[bundle.dataset],
        "dataset": bundle.dataset,
        "method": "No-TL",
        "model_name": spec.name,
        "random_seed": RANDOM_SEED,
        "horizon": HORIZON,
        "target_epochs": int(target_epochs),
        "batch_size": int(batch_size),
        "window_size": int(window_size),
        "feature_dim": int(sequences["x_train"].shape[-1]),
        "train_windows": int(len(sequences["y_train"])),
        "val_windows": int(len(sequences["y_val"])),
        "test_windows": int(len(sequences["y_test"])),
        "rmse": float(metric["rmse"]),
        "normalized_rmse": float(metric["normalized_rmse"]),
        "original_scale_rmse": metric.get("original_scale_rmse"),
        "accuracy": float(metric["accuracy"]),
        "normalized_accuracy": float(metric["normalized_accuracy"]),
        "original_scale_accuracy": metric.get("original_scale_accuracy"),
        "mae": float(metric["mae"]),
        "normalized_mae": float(metric["normalized_mae"]),
        "original_scale_mae": metric.get("original_scale_mae"),
        "metric_space": str(metric["metric_space"]),
        "prediction_shape": tuple(y_pred.shape),
        "run_time_seconds": float(run_time_seconds),
        "status": "OK",
        "error_message": "",
        "notes": spec.notes,
    }


def _run_keras_model(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    metric_protocol: Dict[str, Any],
    spec: BackboneSpec,
    window_size: int,
    target_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Dict[str, Any]:
    if len(sequences["y_train"]) == 0 or len(sequences["y_test"]) == 0:
        return _empty_ablation_row(bundle, sequences, spec, window_size, target_epochs, batch_size, "SKIPPED", "empty train/test windows")

    import tensorflow as tf

    setup_reproducibility(RANDOM_SEED)
    tf.keras.backend.clear_session()
    setup_reproducibility(RANDOM_SEED)

    start = time.perf_counter()
    try:
        model = spec.factory(sequences["x_train"].shape[1:], learning_rate)  # type: ignore[misc]
        fit_kwargs: Dict[str, Any] = {"epochs": int(target_epochs), "batch_size": int(batch_size), "verbose": 0}
        if len(sequences["y_val"]) > 0:
            fit_kwargs["validation_data"] = (sequences["x_val"], sequences["y_val"])
        model.fit(sequences["x_train"], sequences["y_train"], **fit_kwargs)
        y_pred = model.predict(sequences["x_test"], verbose=0)
        metric = _metric_dict(sequences["y_test"], y_pred, metric_protocol, bundle)
    except Exception as exc:
        return _empty_ablation_row(
            bundle,
            sequences,
            spec,
            window_size,
            target_epochs,
            batch_size,
            "ERROR",
            f"{type(exc).__name__}: {exc}",
        )

    return _ablation_metric_row(
        bundle=bundle,
        sequences=sequences,
        spec=spec,
        metric=metric,
        y_pred=y_pred,
        window_size=window_size,
        target_epochs=target_epochs,
        batch_size=batch_size,
        run_time_seconds=time.perf_counter() - start,
    )


def _format_value(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return " ".join(str(value).replace("|", "/").replace("\n", " ").split())


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _write_report(shape_df: pd.DataFrame, ablation_df: pd.DataFrame, window_size: int, target_epochs: int, batch_size: int) -> None:
    current_shape = shape_df[shape_df["model_name"].eq("current_3layer_cnn")].copy()
    ok = ablation_df[ablation_df["status"].eq("OK")].copy()
    pivot = ok.pivot_table(index="dataset", columns="model_name", values="normalized_rmse", aggfunc="first").reset_index()

    comparisons: List[Dict[str, Any]] = []
    for baseline in ["conv1_gap_dense", "conv1_flatten_dense", "dense_only_mlp", "naive_persistence"]:
        paired = pivot.dropna(subset=["current_3layer_cnn", baseline]).copy() if baseline in pivot.columns else pd.DataFrame()
        comparisons.append(
            {
                "baseline": baseline,
                "paired_datasets": int(len(paired)),
                "baseline_better_count": int((paired[baseline] < paired["current_3layer_cnn"]).sum()) if not paired.empty else 0,
                "mean_current_minus_baseline": float((paired["current_3layer_cnn"] - paired[baseline]).mean()) if not paired.empty else np.nan,
            }
        )
    compare_df = pd.DataFrame(comparisons)

    risk_count = int(current_shape["time_dimension_risk"].astype(bool).sum())
    current_worse_count = int(compare_df["baseline_better_count"].max()) if not compare_df.empty else 0
    structural_mismatch = risk_count > 0
    backbone_possible = current_worse_count > 0 or structural_mismatch

    lines = [
        "# CNN Backbone No-TL Shape Audit And Ablation",
        "",
        f"Scope: No-TL only; Dataset1/2/3; seed={RANDOM_SEED}; horizon={HORIZON}; window_size={window_size}; target_epochs={target_epochs}; batch_size={batch_size}. The script reuses the current No-TL data construction and `src/models/cnn_model.py`; it does not modify main training, KNN, RFE, data cleaning, split, or RMSE code.",
        "",
        "## Output Files",
        "",
        f"- Shape audit: `{SHAPE_AUDIT_CSV.relative_to(ROOT)}`",
        f"- No-TL ablation: `{ABLATION_CSV.relative_to(ROOT)}`",
        f"- Report: `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Current CNN Structure",
        "",
        _markdown_table(
            current_shape,
            [
                "dataset",
                "input_shape",
                "conv1d_layer_count",
                "filters",
                "kernel_size",
                "padding",
                "pooling",
                "flatten_dense_params",
                "output_activation",
                "trainable_params",
                "layer_output_shapes",
                "risk_notes",
            ],
        ),
        "",
        "Current CNN is Conv1D(32, k=3, same) + MaxPool(2) + Conv1D(64, k=3, same) + MaxPool(2) + Conv1D(128, k=3, same) + Flatten + Dense(1 linear). With window_size=10, the time axis becomes 10 -> 5 -> 5 -> 2 -> 2 before flattening.",
        "",
        "## No-TL Window Counts And Shape Risk",
        "",
        _markdown_table(
            current_shape,
            ["dataset", "train_windows", "val_windows", "test_windows", "window_size", "feature_dim", "minimum_time_dim", "time_dimension_risk", "risk_notes"],
        ),
        "",
        "Answer: 当前 CNN 对 paper-aligned No-TL 小样本存在结构性不适配风险。主要原因不是 kernel 直接报错，而是 No-TL 的训练窗很少，两个 pooling 后时间维只剩 2，再接 Flatten/Dense，会让参数量和样本量关系偏紧，验证窗也可能很少或为空。这个风险在 shape audit 中以 `risk_notes` 记录。",
        "",
        "## Ablation Results",
        "",
        _markdown_table(
            ok.sort_values(["dataset", "normalized_rmse"]),
            ["dataset", "model_name", "train_windows", "val_windows", "test_windows", "normalized_rmse", "original_scale_rmse", "status"],
        ),
        "",
        "Current CNN paired comparisons:",
        "",
        _markdown_table(compare_df, ["baseline", "paired_datasets", "baseline_better_count", "mean_current_minus_baseline"]),
        "",
        "## Paper-Alignment Interpretation",
        "",
        "合理审计: 在独立脚本中记录 shape/参数量/窗口数，加入更小的 Conv1D+GAP、Conv1D+Flatten、Dense-only、persistence 作为 sensitivity baselines，并保持 No-TL 数据、split、metric 口径不变。这些不会覆盖主结果。",
        "",
        "会改变复现设定: 把主实验 CNN 改成新 backbone、调 window_size/epochs/batch_size 后替换论文复现表、改变 RMSE 计算空间、重做 split 或清洗、把消融模型用于 KNN/RFE/迁移路径。这些都应作为扩展实验而非主复现。",
        "",
        f"当前 No-TL 与原论文差距是否可能由 CNN backbone 造成: {'可能是贡献因素之一' if backbone_possible else '本次审计没有给出强证据'}。但它不能单独解释所有差距，因为 split 证据、metric space、horizon/窗口构造、样本量和随机训练噪声也同时影响结果。",
        "",
        "建议: 先保留主实验 CNN 和主复现结果，不直接改 CNN。把本脚本结果作为 sensitivity analysis 报告；只有当多个 seed、相同 split/metric 下持续显示更小 backbone 显著优于当前 CNN，才考虑新增一个明确标注的 lightweight-CNN 扩展实验。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level="WARNING", log_file=None)
    setup_reproducibility(RANDOM_SEED)

    cfg = _load_config()
    single_cfg = dict(cfg.get("single_experiment", {}))
    window_size = int(single_cfg.get("window_size", 10))
    target_epochs = int(single_cfg.get("target_epochs", cfg.get("target_epochs", 2)))
    batch_size = int(single_cfg.get("batch_size", cfg.get("batch_size", 16)))
    learning_rate = float(single_cfg.get("learning_rate", 1e-4))
    metric_protocol = dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))

    shape_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        bundle = _prepare_dataset(dataset, cfg)
        sequences = _build_sequences(bundle, window_size)
        for spec in BACKBONE_SPECS:
            if spec.factory is None:
                shape_rows.append(_persistence_shape_row(bundle, sequences, window_size))
                ablation_rows.append(_run_persistence(bundle, sequences, metric_protocol, spec, window_size, target_epochs, batch_size))
                continue

            import tensorflow as tf

            tf.keras.backend.clear_session()
            model = spec.factory(sequences["x_train"].shape[1:], learning_rate)
            shape_rows.append(_summarize_model(model, bundle, sequences, spec, window_size))
            ablation_rows.append(
                _run_keras_model(
                    bundle=bundle,
                    sequences=sequences,
                    metric_protocol=metric_protocol,
                    spec=spec,
                    window_size=window_size,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            )

    shape_df = pd.DataFrame(shape_rows).sort_values(["dataset_id", "model_name"])
    ablation_df = pd.DataFrame(ablation_rows).sort_values(["dataset_id", "model_name"])
    shape_df[SHAPE_AUDIT_COLUMNS].to_csv(SHAPE_AUDIT_CSV, index=False, encoding="utf-8")
    ablation_df[ABLATION_COLUMNS].to_csv(ABLATION_CSV, index=False, encoding="utf-8")
    _write_report(shape_df, ablation_df, window_size, target_epochs, batch_size)

    print(f"Wrote {SHAPE_AUDIT_CSV}")
    print(f"Wrote {ABLATION_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
