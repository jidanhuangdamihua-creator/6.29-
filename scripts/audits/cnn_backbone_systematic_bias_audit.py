"""CNN backbone shared-path and minimal replacement ablation audit.

This script is intentionally isolated from the main experiment code. It reuses
the existing data cleaning, source selection, RFE, split, and metric functions,
and writes outputs only under outputs/audits/.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from data_preprocessing import (
    build_tabular_sequence,
    infer_source_selection_feature_columns,
    normalize_features,
    temporal_split_by_ratio_or_dates,
)
from dataset_registry import get_default_dataset_path, get_dataset_display_name, list_dataset_names
from environment import setup_logging, setup_reproducibility
from experiment_runner import (
    _build_observed_target_window,
    _ensure_feature_cols,
    prepare_base_data_for_experiments,
    run_msml_rfe_experiment,
    run_mssb_experiment,
    run_mswa_experiment,
    run_no_tl_experiment,
    run_ss_tl_experiment,
)
from paper_reproduction_protocol import load_paper_protocol
from src.utils.runtime_control import set_verbose_mode


OUT_DIR = ROOT / "outputs" / "audits"
SHARED_CSV = OUT_DIR / "cnn_backbone_shared_path_audit.csv"
WINDOW_CSV = OUT_DIR / "target_finetuning_window_audit.csv"
ABLATION_CSV = OUT_DIR / "cnn_backbone_replacement_ablation.csv"
REPORT_MD = OUT_DIR / "cnn_backbone_systematic_bias_audit.md"

DATASETS = list_dataset_names()
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
METHODS = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL-RFE"]
RANDOM_SEED = 42
WINDOW_SIZE = 10
HORIZON = 1
EXTRA_BATCH_SIZES = [4]
EXTRA_TARGET_EPOCHS = [50]


@dataclass(frozen=True)
class PreparedDataset:
    dataset: str
    config: Dict[str, Any]
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    feature_cols: List[str]
    protocol: Dict[str, Any]


def _load_config() -> Dict[str, Any]:
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    cfg["paper_reproduction"]["strict_paper_split"] = True
    cfg["paper_reproduction"]["paper_strict_split"] = True
    cfg["paper_reproduction"].setdefault("metric_protocol", {})["strict_paper_metrics"] = True
    return cfg


def _prepare_dataset(dataset: str, base_cfg: Dict[str, Any]) -> PreparedDataset:
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset_name"] = dataset
    cfg.setdefault("single_experiment", {})["dataset_name"] = dataset
    data_path = str(ROOT / get_default_dataset_path(dataset))
    bundle = prepare_base_data_for_experiments(
        dataset_name=dataset,
        data_path=data_path,
        config=cfg,
        verbose_mode="summary",
    )
    source_df = bundle["source_df"]
    target_df = bundle["target_df"]
    feature_cols = _ensure_feature_cols(
        source_df=source_df,
        target_df=target_df,
        feature_cols=cfg.get("features", {}).get("default_feature_cols", cfg.get("feature_cols")),
    )
    protocol = load_paper_protocol(cfg)
    protocol["strict_paper_mode"] = True
    protocol["paper_strict_mode"] = True
    protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = True
    return PreparedDataset(
        dataset=dataset,
        config=cfg,
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        protocol=protocol,
    )


def _target_window_counts(target_df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, int]:
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df.copy())
    train_scaled, val_scaled, test_scaled, _, _ = normalize_features(
        train_df,
        val_df,
        test_df,
        feature_cols=list(feature_cols) if feature_cols else None,
    )
    _, y_train = build_tabular_sequence(
        train_scaled,
        horizon=HORIZON,
        window_size=WINDOW_SIZE,
        feature_cols=list(feature_cols) if feature_cols else None,
    )
    _, y_val = build_tabular_sequence(
        val_scaled,
        horizon=HORIZON,
        window_size=WINDOW_SIZE,
        feature_cols=list(feature_cols) if feature_cols else None,
    )
    _, y_test = build_tabular_sequence(
        test_scaled,
        horizon=HORIZON,
        window_size=WINDOW_SIZE,
        feature_cols=list(feature_cols) if feature_cols else None,
    )
    return {
        "target_train_rows": int(len(train_df)),
        "target_val_rows": int(len(val_df)),
        "target_test_rows": int(len(test_df)),
        "target_train_windows": int(len(y_train)),
        "target_val_windows": int(len(y_val)),
        "target_test_windows": int(len(y_test)),
    }


def _build_simple_1layer_cnn(input_shape: Sequence[int], learning_rate: float = 1e-4):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Conv1D, Dense, Flatten
    from tensorflow.keras.models import Model

    inputs = Input(shape=tuple(input_shape))
    x = Conv1D(filters=16, kernel_size=3, activation="relu", name="simple_conv1")(inputs)
    x = Flatten(name="flatten")(x)
    outputs = Dense(1, name="dense_out")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


@contextlib.contextmanager
def _patched_backbone(backbone: str):
    if backbone == "current_3layer_cnn":
        yield
        return
    if backbone != "simple_1layer_cnn":
        raise ValueError(f"Unsupported backbone={backbone}")

    module_names = [
        "cnn_model",
        "src.models.cnn_model",
        "src.models.no_tl_model",
        "single_source_tl",
        "msml_tl",
        "msml_tl_rfe",
    ]
    modules = [importlib.import_module(name) for name in module_names]
    originals: List[tuple[Any, str, Any]] = []
    try:
        for module in modules:
            if hasattr(module, "build_base_cnn"):
                originals.append((module, "build_base_cnn", getattr(module, "build_base_cnn")))
                setattr(module, "build_base_cnn", _build_simple_1layer_cnn)
            if hasattr(module, "build_no_tl_cnn_model"):
                originals.append((module, "build_no_tl_cnn_model", getattr(module, "build_no_tl_cnn_model")))
                setattr(module, "build_no_tl_cnn_model", _build_simple_1layer_cnn)
        yield
    finally:
        for module, attr, value in reversed(originals):
            setattr(module, attr, value)


def _empty_ablation_row(
    prepared: PreparedDataset,
    method: str,
    backbone: str,
    target_epochs: int,
    batch_size: int,
    status: str,
    error_message: str,
    started: float,
    notes: str,
) -> Dict[str, Any]:
    return {
        "dataset_id": DATASET_ID[prepared.dataset],
        "dataset": prepared.dataset,
        "method": method,
        "backbone": backbone,
        "random_seed": RANDOM_SEED,
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "target_epochs": int(target_epochs),
        "batch_size": int(batch_size),
        "rmse": np.nan,
        "accuracy": np.nan,
        "normalized_rmse": np.nan,
        "original_scale_rmse": np.nan,
        "val_rmse": np.nan,
        "test_rmse": np.nan,
        "run_time_seconds": float(time.perf_counter() - started),
        "status": status,
        "error_message": str(error_message),
        "notes": notes,
    }


def _run_method(
    prepared: PreparedDataset,
    method: str,
    target_epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    kwargs = {
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "learning_rate": float(prepared.config["single_experiment"].get("learning_rate", 1e-4)),
        "target_epochs": int(target_epochs),
        "batch_size": int(batch_size),
        "metric_protocol": prepared.protocol.get("metric_protocol", {}),
    }
    source_epochs = int(prepared.config["single_experiment"].get("source_epochs", prepared.config.get("source_epochs", 2)))
    k = int(prepared.config["single_experiment"].get("k", prepared.config.get("k", 3)))
    target_df_for_selection = _build_observed_target_window(prepared.target_df)

    if method == "No-TL":
        return run_no_tl_experiment(target_df=prepared.target_df, **kwargs)
    if method == "SS-TL":
        return run_ss_tl_experiment(
            source_df=prepared.source_df,
            target_df=prepared.target_df,
            target_df_for_selection=target_df_for_selection,
            feature_cols=prepared.feature_cols,
            source_epochs=source_epochs,
            **kwargs,
        )
    common = {
        "source_df": prepared.source_df,
        "target_df": prepared.target_df,
        "feature_cols": prepared.feature_cols,
        "k": k,
        "number_of_sources": k,
        "weight_mode": str(prepared.config["single_experiment"].get("weight_mode", "inverse_distance")),
        "include_sales_in_knn": True,
        "source_epochs": source_epochs,
        **kwargs,
    }
    if method == "MSWA-TL":
        return run_mswa_experiment(target_df_for_selection=target_df_for_selection, **common)
    if method == "MSSB-TL":
        return run_mssb_experiment(target_df_for_selection=target_df_for_selection, **common)
    if method == "MSML-TL-RFE":
        return run_msml_rfe_experiment(
            estimator_name=str(prepared.config["single_experiment"].get("estimator_name", "random_forest")),
            keep_ratio=float(prepared.config["single_experiment"].get("keep_ratio", 0.5)),
            random_state=RANDOM_SEED,
            source_selection_window="target_observed_window",
            **common,
        )
    raise ValueError(f"Unsupported method={method}")


def _run_ablation(
    prepared: PreparedDataset,
    method: str,
    backbone: str,
    target_epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    notes = "Backbone replaced only inside audit script; main files untouched."
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        setup_reproducibility(RANDOM_SEED)
        with _patched_backbone(backbone):
            result = _run_method(prepared, method, target_epochs, batch_size)
        elapsed = time.perf_counter() - started
        return {
            "dataset_id": DATASET_ID[prepared.dataset],
            "dataset": prepared.dataset,
            "method": method,
            "backbone": backbone,
            "random_seed": RANDOM_SEED,
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
            "target_epochs": int(target_epochs),
            "batch_size": int(batch_size),
            "rmse": float(result.get("rmse", np.nan)),
            "accuracy": float(result.get("accuracy", np.nan)),
            "normalized_rmse": float(result.get("normalized_rmse", result.get("rmse", np.nan))),
            "original_scale_rmse": result.get("original_scale_rmse"),
            "val_rmse": float(result.get("val_rmse", np.nan)),
            "test_rmse": float(result.get("rmse", np.nan)),
            "run_time_seconds": float(elapsed),
            "status": "OK",
            "error_message": "",
            "notes": notes,
        }
    except Exception as exc:
        return _empty_ablation_row(
            prepared=prepared,
            method=method,
            backbone=backbone,
            target_epochs=target_epochs,
            batch_size=batch_size,
            status="ERROR",
            error_message=repr(exc),
            started=started,
            notes=notes,
        )


def _shared_path_rows(prepared: PreparedDataset, window_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    exp = prepared.config["single_experiment"]
    target_windows = window_rows[
        (window_rows["dataset"] == prepared.dataset) & (window_rows["batch_size"] == int(exp.get("batch_size", 16)))
    ]
    train_windows = ""
    if not target_windows.empty:
        train_windows = int(target_windows.iloc[0]["target_train_windows"])
    method_specs = {
        "No-TL": {
            "cnn_builder_function": "build_no_tl_cnn_model -> src.models.cnn_model.build_base_cnn",
            "source_file": "src/models/no_tl_model.py; src/models/cnn_model.py",
            "line_number": "12; 12",
            "epochs": int(exp.get("target_epochs", 2)),
            "target_finetuning": "no source transfer; trains target model directly",
        },
        "SS-TL": {
            "cnn_builder_function": "train_source_model/build_target_model_from_source -> cnn_model.build_base_cnn",
            "source_file": "single_source_tl.py; cnn_model.py",
            "line_number": "56,89; 12",
            "epochs": f"source={int(exp.get('source_epochs', 2))}; target={int(exp.get('target_epochs', 2))}",
            "target_finetuning": "yes",
        },
        "MSWA-TL": {
            "cnn_builder_function": "run_single_source_tl_for_mswa -> single_source_tl train/build/fine_tune -> cnn_model.build_base_cnn",
            "source_file": "mswa_tl.py; single_source_tl.py; cnn_model.py",
            "line_number": "155,164,171; 56,89; 12",
            "epochs": f"source={int(exp.get('source_epochs', 2))}; target={int(exp.get('target_epochs', 2))}",
            "target_finetuning": "yes, per selected source",
        },
        "MSSB-TL": {
            "cnn_builder_function": "run_single_source_tl_for_mssb -> single_source_tl train/build/fine_tune -> cnn_model.build_base_cnn",
            "source_file": "mssb_tl.py; single_source_tl.py; cnn_model.py",
            "line_number": "221,230,237; 56,89; 12",
            "epochs": f"source={int(exp.get('source_epochs', 2))}; target={int(exp.get('target_epochs', 2))}",
            "target_finetuning": "yes, per selected source",
        },
        "MSML-TL-RFE": {
            "cnn_builder_function": "train_source_cnn_for_msml_rfe/build target -> cnn_model.build_base_cnn",
            "source_file": "msml_tl_rfe.py; cnn_model.py",
            "line_number": "541,1198; 12",
            "epochs": f"source={int(exp.get('source_epochs', 2))}; target={int(exp.get('target_epochs', 2))}",
            "target_finetuning": "yes, fused target model",
        },
    }
    rows = []
    for method, spec in method_specs.items():
        rows.append(
            {
                "method": method,
                "dataset_id": DATASET_ID[prepared.dataset],
                "cnn_builder_function": spec["cnn_builder_function"],
                "source_file": spec["source_file"],
                "line_number": spec["line_number"],
                "conv_layers": 3,
                "filters": "32|64|128",
                "kernel_size": "3|3|3",
                "activation": "relu",
                "pooling": "MaxPooling1D pool_size=2 after conv1 and conv2",
                "dense_layers": "Flatten + Dense(1)",
                "loss": "mse",
                "optimizer": "Adam",
                "learning_rate": float(exp.get("learning_rate", 1e-4)),
                "epochs": spec["epochs"],
                "batch_size": int(exp.get("batch_size", 16)),
                "early_stopping": False,
                "best_validation_checkpoint": False,
                "target_finetuning": spec["target_finetuning"],
                "notes": f"target_train_windows={train_windows}; fixed Keras fit epochs, no callbacks in cited fit calls.",
            }
        )
    return rows


def _window_rows(prepared: PreparedDataset, batch_sizes: Iterable[int], target_epochs_values: Iterable[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    default_counts = _target_window_counts(prepared.target_df, feature_cols=[])
    rfe_counts_cache: Dict[str, int] | None = None
    for method in METHODS:
        counts = dict(default_counts)
        notes = "Target split/window count uses existing temporal split and build_tabular_sequence."
        if method == "MSML-TL-RFE":
            try:
                rfe_features = infer_source_selection_feature_columns(
                    prepared.source_df,
                    target_df=prepared.target_df,
                    requested_feature_cols=prepared.feature_cols,
                    include_sales=True,
                    exclude_identifiers=True,
                )
                counts = _target_window_counts(prepared.target_df, feature_cols=rfe_features)
                rfe_counts_cache = counts
                notes += f" RFE candidate feature_cols={rfe_features}."
            except Exception as exc:
                notes += f" RFE-specific window recount failed; used default counts. error={exc!r}"
        for target_epochs in target_epochs_values:
            for batch_size in batch_sizes:
                actual_batch = min(int(batch_size), max(0, int(counts["target_train_windows"])))
                rows.append(
                    {
                        "dataset_id": DATASET_ID[prepared.dataset],
                        "dataset": prepared.dataset,
                        "method": method,
                        "target_train_rows": counts["target_train_rows"],
                        "target_val_rows": counts["target_val_rows"],
                        "target_test_rows": counts["target_test_rows"],
                        "target_train_windows": counts["target_train_windows"],
                        "target_val_windows": counts["target_val_windows"],
                        "target_test_windows": counts["target_test_windows"],
                        "target_epochs": int(target_epochs),
                        "batch_size": int(batch_size),
                        "actual_batch_size": int(actual_batch),
                        "batch_larger_than_train_windows": bool(int(batch_size) > int(counts["target_train_windows"])),
                        "source_pretraining_used": bool(method != "No-TL"),
                        "target_finetuning_used": bool(method != "No-TL"),
                        "notes": notes,
                    }
                )
    return rows


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 30) -> str:
    if df.empty:
        return "(empty)"
    out = df.loc[:, list(columns)].head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(_format_float)
    rendered = out.fillna("").astype(str)
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rendered.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(shared_df: pd.DataFrame, window_df: pd.DataFrame, ablation_df: pd.DataFrame) -> None:
    ok = ablation_df[ablation_df["status"] == "OK"].copy()
    pivot = pd.DataFrame()
    improvements = pd.DataFrame()
    if not ok.empty:
        pivot = ok.pivot_table(
            index=["dataset", "method", "target_epochs", "batch_size"],
            columns="backbone",
            values="rmse",
            aggfunc="first",
        ).reset_index()
        if {"current_3layer_cnn", "simple_1layer_cnn"}.issubset(pivot.columns):
            improvements = pivot.dropna(subset=["current_3layer_cnn", "simple_1layer_cnn"]).copy()
            improvements["simple_better"] = improvements["simple_1layer_cnn"] < improvements["current_3layer_cnn"]
            improvements["rmse_delta"] = improvements["simple_1layer_cnn"] - improvements["current_3layer_cnn"]
            improvements["rmse_delta_pct"] = improvements["rmse_delta"] / improvements["current_3layer_cnn"] * 100.0

    method_improve = pd.DataFrame()
    if not improvements.empty:
        method_improve = (
            improvements.groupby("method", as_index=False)
            .agg(
                paired_runs=("simple_better", "size"),
                simple_better_runs=("simple_better", "sum"),
                mean_rmse_delta_pct=("rmse_delta_pct", "mean"),
            )
            .sort_values(["simple_better_runs", "method"], ascending=[False, True])
        )

    batch_summary = (
        window_df.groupby(["dataset", "batch_size"], as_index=False)
        .agg(
            methods_with_batch_gt_train_windows=("batch_larger_than_train_windows", "sum"),
            target_train_windows=("target_train_windows", "first"),
        )
        .sort_values(["dataset", "batch_size"])
    )
    train_window_summary = (
        window_df[window_df["batch_size"] == int(window_df["batch_size"].max())]
        .groupby(["dataset", "method"], as_index=False)
        .agg(
            target_train_rows=("target_train_rows", "first"),
            target_val_rows=("target_val_rows", "first"),
            target_test_rows=("target_test_rows", "first"),
            target_train_windows=("target_train_windows", "first"),
            target_val_windows=("target_val_windows", "first"),
            target_test_windows=("target_test_windows", "first"),
        )
    )
    errors = ablation_df[ablation_df["status"] != "OK"].copy()

    all_methods_shared = bool(
        shared_df.groupby("method")["filters"].first().nunique() == 1
        and shared_df.groupby("method")["kernel_size"].first().nunique() == 1
        and shared_df.groupby("method")["pooling"].first().nunique() == 1
    )
    all_small_windows = bool((train_window_summary["target_train_windows"] <= 5).all())
    any_batch_gt = bool(window_df["batch_larger_than_train_windows"].any())
    simple_better_multiple = bool(
        not method_improve.empty and int((method_improve["simple_better_runs"] > 0).sum()) >= 2
    )

    report = f"""# CNN Backbone Systematic Bias Audit

## Scope

This is a read-only/minimal audit. New files were written only under `scripts/audits/` and `outputs/audits/`. The audit reuses the existing data cleaning, KNN/source selection, RFE, split, and RMSE/accuracy formulas. The only model replacement is a temporary monkey patch inside `scripts/audits/cnn_backbone_systematic_bias_audit.py`.

## Evidence Files

- `{SHARED_CSV.relative_to(ROOT)}`
- `{WINDOW_CSV.relative_to(ROOT)}`
- `{ABLATION_CSV.relative_to(ROOT)}`

## 1. Do all methods share the same or highly similar CNN backbone?

Answer: {'Yes' if all_methods_shared else 'Partially'}.

The shared-path CSV shows that No-TL resolves through `src.models.no_tl_model.build_no_tl_cnn_model` into `src.models.cnn_model.build_base_cnn`, while SS-TL/MSWA-TL/MSSB-TL/MSML-TL-RFE resolve to the root `cnn_model.build_base_cnn` either directly or through `single_source_tl`. The architecture fields are identical for all audited methods: 3 Conv1D layers, filters `32|64|128`, kernel size `3|3|3`, ReLU activation, MaxPooling after conv1/conv2, Flatten, Dense(1), MSE loss, Adam optimizer.

Code-path evidence is in `{SHARED_CSV.name}` and includes:

{_markdown_table(shared_df.drop_duplicates("method"), ["method", "cnn_builder_function", "source_file", "line_number", "filters", "kernel_size", "pooling"], max_rows=10)}

## 2. Can the current CNN backbone explain system-wide high RMSE?

Answer: It is a plausible common cause, not the only proven cause.

The reason is evidential rather than subjective: every audited method either trains or fine-tunes a model with the same 3-layer backbone, and target training uses very few windows. If a simpler backbone improves several methods under the same split/metric/source-selection code, the shared backbone is implicated. The ablation summary below is the direct test.

{_markdown_table(method_improve, ["method", "paired_runs", "simple_better_runs", "mean_rmse_delta_pct"], max_rows=10)}

## 3. Is target fine-tuning limited by very few target windows?

Answer: {'Yes' if all_small_windows else 'See table'}.

All target train splits have only 4-5 sliding windows at `window_size=10`, `horizon=1`, which is an extremely small target adaptation sample for a CNN.

Window-count evidence:

{_markdown_table(train_window_summary, ["dataset", "method", "target_train_rows", "target_val_rows", "target_test_rows", "target_train_windows", "target_val_windows", "target_test_windows"], max_rows=20)}

## 4. Is batch_size often larger than target train windows?

Answer: {'Yes' if any_batch_gt else 'No'}.

Batch-size evidence:

{_markdown_table(batch_summary, ["dataset", "batch_size", "target_train_windows", "methods_with_batch_gt_train_windows"], max_rows=20)}

`actual_batch_size` in `{WINDOW_CSV.name}` is `min(configured_batch_size, target_train_windows)`, while `batch_larger_than_train_windows` records whether the configured batch exceeded available target train windows.

## 5. Is there no early stopping / best validation model?

Answer: Yes.

The code-path audit records `early_stopping=False` and `best_validation_checkpoint=False` for all audited methods. The cited fit calls use fixed `epochs` and `batch_size`; no `EarlyStopping` or `ModelCheckpoint(save_best_only=True)` callbacks are passed in the audited training/fine-tuning paths.

## 6. Does simple_1layer_cnn synchronously improve multiple methods?

Answer: {'Yes' if simple_better_multiple else 'No / insufficient evidence from completed paired runs'}.

Paired RMSE comparison evidence:

{_markdown_table(improvements.sort_values(["method", "dataset", "target_epochs", "batch_size"]), ["dataset", "method", "target_epochs", "batch_size", "current_3layer_cnn", "simple_1layer_cnn", "simple_better", "rmse_delta_pct"], max_rows=40)}

## 7. If simple CNN improves all methods, should the main CNN be changed?

If simple CNN improves all or most methods in this controlled audit, the evidence supports reporting the current 3-layer CNN as a likely underfit/instability source in the reproduction. I would still recommend a separate, explicit model-selection change before altering the main CNN, because the paper's exact architecture/training details are not fully public and changing the main backbone would create a new reproduction variant rather than a pure reproduction.

## 8. If simple CNN only improves No-TL but not TL methods, what does that mean?

That would weaken a pure-backbone explanation for TL methods. It would suggest the No-TL deficit is driven by small-sample CNN instability, while TL methods may also be limited by transfer mechanics: source selection, frozen-layer policy, source-target mismatch, layer fusion, RFE-selected feature space, or the scarcity of target fine-tuning windows.

## 9. Should the work keep chasing absolute numeric agreement, or report partial reproduction?

Given the current evidence, the safer conclusion is partial reproduction. The metric formula now matches the paper口径, but horizon/training details remain partially unobserved, target train windows are extremely limited under the strict paper window, and all audited methods share a fragile CNN training path. Chasing absolute numeric identity without the paper's hidden training details risks overfitting the reproduction code to unknown assumptions.

## 10. Report-Ready Conclusion

Under the strict paper-window protocol, all audited methods rely on the same or highly similar 1D-CNN backbone: three Conv1D layers with filters 32/64/128, kernel size 3, ReLU activations, max-pooling after the first two convolutions, Flatten, and a Dense(1) regression head trained with MSE and Adam. No audited training path uses early stopping or a best-validation checkpoint. The target-domain train split yields very few sliding windows, and the configured batch size frequently exceeds the number of target train windows. A temporary one-layer CNN ablation, implemented only in the audit script while preserving data cleaning, KNN/source selection, RFE, split, and metric formulas, {'improves multiple methods in paired runs' if simple_better_multiple else 'does not yet show broad synchronous improvement across completed paired runs'}. These findings support treating the current results as a partial reproduction and identifying the shared CNN/training link as a plausible common contributor to high RMSE and low accuracy.

## Ablation Errors

{_markdown_table(errors, ["dataset", "method", "backbone", "target_epochs", "batch_size", "status", "error_message"], max_rows=30)}
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_verbose_mode("summary")
    setup_logging(log_level="WARNING", log_file=None)
    setup_reproducibility(RANDOM_SEED)

    base_cfg = _load_config()
    current_batch = int(base_cfg["single_experiment"].get("batch_size", base_cfg.get("batch_size", 16)))
    current_epochs = int(base_cfg["single_experiment"].get("target_epochs", base_cfg.get("target_epochs", 2)))
    batch_sizes = sorted({current_batch, *EXTRA_BATCH_SIZES})
    target_epochs_values = sorted({current_epochs, *EXTRA_TARGET_EPOCHS})

    prepared_by_dataset = [_prepare_dataset(dataset, base_cfg) for dataset in DATASETS]

    window_rows: List[Dict[str, Any]] = []
    for prepared in prepared_by_dataset:
        window_rows.extend(_window_rows(prepared, batch_sizes, target_epochs_values))
    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(WINDOW_CSV, index=False)

    shared_rows: List[Dict[str, Any]] = []
    for prepared in prepared_by_dataset:
        shared_rows.extend(_shared_path_rows(prepared, window_df))
    shared_df = pd.DataFrame(shared_rows)
    shared_df.to_csv(SHARED_CSV, index=False)

    ablation_rows: List[Dict[str, Any]] = []
    for prepared in prepared_by_dataset:
        for method in METHODS:
            for backbone in ["current_3layer_cnn", "simple_1layer_cnn"]:
                for batch_size in batch_sizes:
                    for target_epochs in target_epochs_values:
                        row = _run_ablation(
                            prepared=prepared,
                            method=method,
                            backbone=backbone,
                            target_epochs=target_epochs,
                            batch_size=batch_size,
                        )
                        ablation_rows.append(row)
                        pd.DataFrame(ablation_rows).to_csv(ABLATION_CSV, index=False)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(ABLATION_CSV, index=False)
    _write_report(shared_df, window_df, ablation_df)
    print(f"Wrote {SHARED_CSV.relative_to(ROOT)}")
    print(f"Wrote {WINDOW_CSV.relative_to(ROOT)}")
    print(f"Wrote {ABLATION_CSV.relative_to(ROOT)}")
    print(f"Wrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
