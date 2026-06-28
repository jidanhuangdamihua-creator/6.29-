"""Small-sample target training protocol ablation audit.

This script is intentionally isolated from the main experiment code. It reuses
the existing data cleaning, KNN/source selection, RFE, split, CNN structure, and
metric functions. Outputs are written only under outputs/audits/.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

import experiment_runner as experiment_runner_module
from data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
)
from dataset_registry import get_dataset_display_name, get_default_dataset_path, list_dataset_names
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
DETAIL_CSV = OUT_DIR / "small_sample_training_protocol_ablation.csv"
SUMMARY_CSV = OUT_DIR / "small_sample_training_protocol_summary.csv"
REPORT_MD = OUT_DIR / "small_sample_training_protocol_audit.md"

DATASETS = list_dataset_names()
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
METHODS = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL-RFE"]
RANDOM_SEED = 42
WINDOW_SIZE = 10
HORIZON = 1
CNN_STRUCTURE = "paper_aligned_current_3layer_cnn_conv1d_32_64_128"
CURRENT_PROTOCOL = "current_protocol"
SMALL_SAMPLE_PROTOCOL = "small_sample_training_protocol"

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "protocol",
    "random_seed",
    "window_size",
    "horizon",
    "cnn_structure",
    "batch_size",
    "target_epochs",
    "early_stopping",
    "restore_best_weights",
    "actual_epochs_run",
    "target_train_windows",
    "target_val_windows",
    "target_test_windows",
    "rmse",
    "accuracy",
    "normalized_rmse",
    "original_scale_rmse",
    "val_rmse",
    "test_rmse",
    "run_time_seconds",
    "status",
    "error_message",
    "notes",
]

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "current_protocol_rmse",
    "small_sample_protocol_rmse",
    "rmse_diff",
    "rmse_percent_change",
    "current_protocol_accuracy",
    "small_sample_protocol_accuracy",
    "accuracy_diff",
    "accuracy_percent_change",
    "improved",
    "notes",
]


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    batch_size: int
    target_epochs: int
    early_stopping: bool
    restore_best_weights: bool


@dataclass
class FitRecorder:
    target_epoch_runs: List[int] = field(default_factory=list)

    @property
    def actual_epochs_run(self) -> int:
        if not self.target_epoch_runs:
            return 0
        return int(max(self.target_epoch_runs))

    @property
    def notes(self) -> str:
        if not self.target_epoch_runs:
            return "No target fit with validation_data was recorded."
        return "target_fit_epochs_run=" + "|".join(str(int(v)) for v in self.target_epoch_runs)


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
    original_save_split_summary = experiment_runner_module._save_split_protocol_summary
    try:
        experiment_runner_module._save_split_protocol_summary = lambda *args, **kwargs: None
        bundle = prepare_base_data_for_experiments(
            dataset_name=dataset,
            data_path=data_path,
            config=cfg,
            verbose_mode="summary",
        )
    finally:
        experiment_runner_module._save_split_protocol_summary = original_save_split_summary
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


def _protocol_specs(config: Dict[str, Any]) -> List[ProtocolSpec]:
    exp = config.get("single_experiment", {})
    return [
        ProtocolSpec(
            name=CURRENT_PROTOCOL,
            batch_size=int(exp.get("batch_size", config.get("batch_size", 16))),
            target_epochs=int(exp.get("target_epochs", config.get("target_epochs", 2))),
            early_stopping=False,
            restore_best_weights=False,
        ),
        ProtocolSpec(
            name=SMALL_SAMPLE_PROTOCOL,
            batch_size=4,
            target_epochs=50,
            early_stopping=True,
            restore_best_weights=True,
        ),
    ]


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
        "target_train_windows": int(len(y_train)),
        "target_val_windows": int(len(y_val)),
        "target_test_windows": int(len(y_test)),
    }


@contextlib.contextmanager
def _patched_target_fit(protocol: ProtocolSpec, recorder: FitRecorder):
    import tensorflow as tf

    original_fit = tf.keras.Model.fit

    def fit_with_protocol(model, x=None, y=None, *args, **kwargs):
        is_target_fit = "validation_data" in kwargs and kwargs.get("validation_data") is not None
        if is_target_fit:
            if protocol.early_stopping:
                callbacks = list(kwargs.get("callbacks") or [])
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=protocol.restore_best_weights,
                    )
                )
                kwargs["callbacks"] = callbacks
            kwargs["epochs"] = int(protocol.target_epochs)
            kwargs["batch_size"] = int(protocol.batch_size)
        history = original_fit(model, x, y, *args, **kwargs)
        if is_target_fit:
            recorder.target_epoch_runs.append(len(history.history.get("loss", [])))
        return history

    try:
        tf.keras.Model.fit = fit_with_protocol
        yield
    finally:
        tf.keras.Model.fit = original_fit


def _run_method(prepared: PreparedDataset, method: str, protocol: ProtocolSpec) -> Dict[str, Any]:
    exp = prepared.config.get("single_experiment", {})
    source_epochs = int(exp.get("source_epochs", prepared.config.get("source_epochs", 2)))
    current_batch_size = int(exp.get("batch_size", prepared.config.get("batch_size", 16)))
    k = int(exp.get("k", prepared.config.get("k", 3)))
    common_kwargs = {
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "learning_rate": float(exp.get("learning_rate", 1e-4)),
        "target_epochs": int(protocol.target_epochs),
        "batch_size": int(current_batch_size),
        "metric_protocol": prepared.protocol.get("metric_protocol", {}),
    }
    target_df_for_selection = _build_observed_target_window(prepared.target_df)

    if method == "No-TL":
        return run_no_tl_experiment(
            target_df=prepared.target_df,
            batch_size=int(protocol.batch_size),
            **{k: v for k, v in common_kwargs.items() if k != "batch_size"},
        )
    if method == "SS-TL":
        return run_ss_tl_experiment(
            source_df=prepared.source_df,
            target_df=prepared.target_df,
            target_df_for_selection=target_df_for_selection,
            feature_cols=prepared.feature_cols,
            source_epochs=source_epochs,
            **common_kwargs,
        )

    transfer_common = {
        "source_df": prepared.source_df,
        "target_df": prepared.target_df,
        "target_df_for_selection": target_df_for_selection,
        "feature_cols": prepared.feature_cols,
        "k": k,
        "number_of_sources": k,
        "weight_mode": str(exp.get("weight_mode", "inverse_distance")),
        "include_sales_in_knn": True,
        "source_epochs": source_epochs,
        **common_kwargs,
    }
    if method == "MSWA-TL":
        return run_mswa_experiment(**transfer_common)
    if method == "MSSB-TL":
        return run_mssb_experiment(**transfer_common)
    if method == "MSML-TL-RFE":
        transfer_common.pop("target_df_for_selection", None)
        return run_msml_rfe_experiment(
            estimator_name=str(exp.get("estimator_name", "random_forest")),
            keep_ratio=float(exp.get("keep_ratio", 0.5)),
            random_state=RANDOM_SEED,
            source_selection_window="target_observed_window",
            **transfer_common,
        )
    raise ValueError(f"Unsupported method={method}")


def _empty_detail_row(
    prepared: PreparedDataset,
    method: str,
    protocol: ProtocolSpec,
    counts: Dict[str, int],
    started: float,
    status: str,
    error_message: str,
    notes: str,
) -> Dict[str, Any]:
    return {
        "dataset_id": DATASET_ID[prepared.dataset],
        "dataset": prepared.dataset,
        "method": method,
        "protocol": protocol.name,
        "random_seed": RANDOM_SEED,
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "cnn_structure": CNN_STRUCTURE,
        "batch_size": int(protocol.batch_size),
        "target_epochs": int(protocol.target_epochs),
        "early_stopping": bool(protocol.early_stopping),
        "restore_best_weights": bool(protocol.restore_best_weights),
        "actual_epochs_run": 0,
        "target_train_windows": int(counts["target_train_windows"]),
        "target_val_windows": int(counts["target_val_windows"]),
        "target_test_windows": int(counts["target_test_windows"]),
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


def _run_one(prepared: PreparedDataset, method: str, protocol: ProtocolSpec) -> Dict[str, Any]:
    started = time.perf_counter()
    counts = _target_window_counts(prepared.target_df, prepared.feature_cols)
    recorder = FitRecorder()
    notes = (
        "Main program untouched; data cleaning, split, KNN/source selection, RFE, "
        "metric formulas, and current 3-layer CNN are reused. Source pretraining "
        "keeps current batch_size; patched protocol applies only to target fits "
        "that already provide validation_data."
    )
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        setup_reproducibility(RANDOM_SEED)
        with _patched_target_fit(protocol, recorder):
            result = _run_method(prepared, method, protocol)
        elapsed = time.perf_counter() - started
        return {
            "dataset_id": DATASET_ID[prepared.dataset],
            "dataset": prepared.dataset,
            "method": method,
            "protocol": protocol.name,
            "random_seed": RANDOM_SEED,
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
            "cnn_structure": CNN_STRUCTURE,
            "batch_size": int(protocol.batch_size),
            "target_epochs": int(protocol.target_epochs),
            "early_stopping": bool(protocol.early_stopping),
            "restore_best_weights": bool(protocol.restore_best_weights),
            "actual_epochs_run": int(recorder.actual_epochs_run),
            "target_train_windows": int(counts["target_train_windows"]),
            "target_val_windows": int(counts["target_val_windows"]),
            "target_test_windows": int(counts["target_test_windows"]),
            "rmse": float(result.get("rmse", np.nan)),
            "accuracy": float(result.get("accuracy", np.nan)),
            "normalized_rmse": float(result.get("normalized_rmse", result.get("rmse", np.nan))),
            "original_scale_rmse": result.get("original_scale_rmse"),
            "val_rmse": float(result.get("val_rmse", np.nan)),
            "test_rmse": float(result.get("test_rmse", result.get("rmse", np.nan))),
            "run_time_seconds": float(elapsed),
            "status": "OK",
            "error_message": "",
            "notes": f"{notes} {recorder.notes}",
        }
    except Exception as exc:
        return _empty_detail_row(
            prepared=prepared,
            method=method,
            protocol=protocol,
            counts=counts,
            started=started,
            status="ERROR",
            error_message=repr(exc),
            notes=notes,
        )


def _percent_change(new_value: float, old_value: float) -> float:
    try:
        old = float(old_value)
        new = float(new_value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(old) or np.isnan(new) or np.isclose(old, 0.0):
        return float("nan")
    return float((new - old) / old * 100.0)


def _build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if detail_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    grouped = detail_df.groupby(["dataset_id", "dataset", "method"], dropna=False)
    for (dataset_id, dataset, method), group in grouped:
        current = group[group["protocol"] == CURRENT_PROTOCOL]
        small = group[group["protocol"] == SMALL_SAMPLE_PROTOCOL]
        current_ok = not current.empty and str(current.iloc[0].get("status", "")) == "OK"
        small_ok = not small.empty and str(small.iloc[0].get("status", "")) == "OK"
        current_rmse = float(current.iloc[0]["rmse"]) if current_ok else np.nan
        small_rmse = float(small.iloc[0]["rmse"]) if small_ok else np.nan
        current_acc = float(current.iloc[0]["accuracy"]) if current_ok else np.nan
        small_acc = float(small.iloc[0]["accuracy"]) if small_ok else np.nan
        rmse_diff = small_rmse - current_rmse if current_ok and small_ok else np.nan
        acc_diff = small_acc - current_acc if current_ok and small_ok else np.nan
        improved = bool(current_ok and small_ok and small_rmse < current_rmse and small_acc > current_acc)
        notes = []
        if not current_ok:
            notes.append("current_protocol missing or failed")
        if not small_ok:
            notes.append("small_sample_training_protocol missing or failed")
        if current_ok and small_ok:
            notes.append("paired comparison")
        rows.append(
            {
                "dataset_id": int(dataset_id),
                "dataset": str(dataset),
                "method": str(method),
                "current_protocol_rmse": current_rmse,
                "small_sample_protocol_rmse": small_rmse,
                "rmse_diff": rmse_diff,
                "rmse_percent_change": _percent_change(small_rmse, current_rmse),
                "current_protocol_accuracy": current_acc,
                "small_sample_protocol_accuracy": small_acc,
                "accuracy_diff": acc_diff,
                "accuracy_percent_change": _percent_change(small_acc, current_acc),
                "improved": improved,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(["dataset_id", "method"]).reset_index(drop=True)


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


def _write_report(detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    ok_summary = summary_df.dropna(subset=["current_protocol_rmse", "small_sample_protocol_rmse"]).copy()
    improved = ok_summary[ok_summary["improved"] == True].copy()
    not_improved = ok_summary[ok_summary["improved"] != True].copy()
    errors = detail_df[detail_df["status"] != "OK"].copy()

    method_sync = pd.DataFrame()
    dataset_sync = pd.DataFrame()
    if not ok_summary.empty:
        method_sync = (
            ok_summary.groupby("method", as_index=False)
            .agg(
                paired_runs=("improved", "size"),
                improved_runs=("improved", "sum"),
                mean_rmse_percent_change=("rmse_percent_change", "mean"),
                mean_accuracy_percent_change=("accuracy_percent_change", "mean"),
            )
            .sort_values(["improved_runs", "method"], ascending=[False, True])
        )
        dataset_sync = (
            ok_summary.groupby("dataset", as_index=False)
            .agg(
                paired_runs=("improved", "size"),
                improved_runs=("improved", "sum"),
                mean_rmse_percent_change=("rmse_percent_change", "mean"),
                mean_accuracy_percent_change=("accuracy_percent_change", "mean"),
            )
            .sort_values(["improved_runs", "dataset"], ascending=[False, True])
        )

    best = ok_summary.sort_values("rmse_percent_change", ascending=True).head(10)
    multiple_methods_improved = False
    if not method_sync.empty:
        multiple_methods_improved = int((method_sync["improved_runs"] > 0).sum()) >= 2
    majority_improved = False
    if not ok_summary.empty:
        majority_improved = int(ok_summary["improved"].sum()) >= max(1, int(np.ceil(len(ok_summary) / 2)))

    if majority_improved and multiple_methods_improved:
        cause_answer = (
            "The CSV supports small-sample target training protocol as an important shared cause of weak metrics, "
            "because the controlled protocol change improved a majority of paired runs across multiple methods. "
            "It does not prove this is the only cause."
        )
        reproduction_label = "reproduction with protocol limitation"
        main_change_answer = (
            "No. The main paper-aligned reproduction should not be changed, because the main run must keep the "
            "paper-aligned current protocol. The improved protocol should be reported as a supplemental ablation."
        )
    elif multiple_methods_improved:
        cause_answer = (
            "The CSV supports small-sample target training protocol as a plausible contributing cause, but the "
            "evidence is mixed because improvements are not a majority of paired runs."
        )
        reproduction_label = "partial reproduction with protocol limitation"
        main_change_answer = (
            "No. Keep the main paper-aligned run unchanged and report this as a supplemental improvement experiment."
        )
    else:
        cause_answer = (
            "The CSV does not support small-sample target training protocol as the dominant shared cause on its own."
        )
        reproduction_label = "partial reproduction"
        main_change_answer = (
            "No. There is not enough controlled evidence to modify the main reproduction protocol."
        )

    conclusion = (
        f"Under the unchanged current 3-layer CNN, unchanged split/source-selection/RFE/data-cleaning/metric pipeline, "
        f"the small-sample target training protocol changed only the target fit schedule to batch_size=4, "
        f"target_epochs=50, EarlyStopping(monitor='val_loss', patience=10), and restore_best_weights=True. "
        f"Across {len(ok_summary)} paired comparisons, {int(ok_summary['improved'].sum()) if not ok_summary.empty else 0} "
        f"showed lower RMSE and higher accuracy than the current protocol. Therefore, the current results should be "
        f"reported as {reproduction_label}; the protocol ablation is supplemental evidence rather than a replacement "
        f"for the paper-aligned main experiment."
    )

    report = f"""# Small-Sample Training Protocol Ablation Audit

## Scope

This is a minimal supplemental experiment. It does not modify the main program and does not overwrite main experiment results. It reuses the existing data cleaning, split, KNN/source selection, RFE, RMSE/accuracy formulas, selected features, and the paper-aligned current 3-layer CNN structure.

## Evidence Files

- `{DETAIL_CSV.relative_to(ROOT)}`
- `{SUMMARY_CSV.relative_to(ROOT)}`
- `{REPORT_MD.relative_to(ROOT)}`

## Protocols

- `current_protocol`: batch_size=16, target_epochs=current config, no EarlyStopping, no best validation checkpoint.
- `small_sample_training_protocol`: target batch_size=4, target_epochs=50, EarlyStopping monitor=`val_loss`, patience=10, restore_best_weights=True. Source pretraining is kept at the current batch setting to isolate the target small-sample training protocol.

## 1. Did batch_size=4 + early stopping + restore_best_weights synchronously improve multiple methods?

Answer: {'Yes' if multiple_methods_improved else 'No'}.

Method-level evidence:

{_markdown_table(method_sync, ["method", "paired_runs", "improved_runs", "mean_rmse_percent_change", "mean_accuracy_percent_change"], max_rows=10)}

## 2. Which datasets/methods improved the most?

Largest RMSE reductions:

{_markdown_table(best, ["dataset", "method", "current_protocol_rmse", "small_sample_protocol_rmse", "rmse_diff", "rmse_percent_change", "current_protocol_accuracy", "small_sample_protocol_accuracy"], max_rows=10)}

Dataset-level evidence:

{_markdown_table(dataset_sync, ["dataset", "paired_runs", "improved_runs", "mean_rmse_percent_change", "mean_accuracy_percent_change"], max_rows=10)}

## 3. Which methods did not improve?

{_markdown_table(not_improved, ["dataset", "method", "current_protocol_rmse", "small_sample_protocol_rmse", "rmse_diff", "rmse_percent_change", "current_protocol_accuracy", "small_sample_protocol_accuracy"], max_rows=30)}

## 4. Does this show that weak metrics mainly come from the small-sample training protocol?

{cause_answer}

## 5. If improvement is clear, should the main reproduction be modified?

{main_change_answer}

## 6. If the main experiment is not modified, should this be reported as a supplemental improvement experiment?

Yes. This ablation is exactly a supplemental improvement experiment because it keeps the paper-aligned CNN and all non-training-protocol components unchanged while changing only the target small-sample training schedule.

## 7. How should the current reproduction be characterized?

`{reproduction_label}`.

## 8. Report-ready conclusion

{conclusion}

## Paired Summary

{_markdown_table(summary_df, ["dataset", "method", "current_protocol_rmse", "small_sample_protocol_rmse", "rmse_percent_change", "current_protocol_accuracy", "small_sample_protocol_accuracy", "accuracy_percent_change", "improved"], max_rows=50)}

## Errors

{_markdown_table(errors, ["dataset", "method", "protocol", "status", "error_message"], max_rows=30)}
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def run_audit(methods: Iterable[str] = METHODS, datasets: Iterable[str] = DATASETS) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_verbose_mode("summary")
    setup_logging(log_level="WARNING", log_file=None)
    base_cfg = _load_config()

    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        prepared = _prepare_dataset(dataset, base_cfg)
        print(f"[audit_dataset] dataset={dataset} display_name={get_dataset_display_name(dataset)}")
        for method in methods:
            for spec in _protocol_specs(prepared.config):
                print(f"[audit_run] dataset={dataset} method={method} protocol={spec.name}")
                row = _run_one(prepared, method, spec)
                rows.append(row)
                print(
                    "[audit_result] "
                    f"dataset={dataset} method={method} protocol={spec.name} "
                    f"status={row['status']} rmse={row['rmse']} accuracy={row['accuracy']} "
                    f"actual_epochs_run={row['actual_epochs_run']}"
                )

    detail_df = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    summary_df = _build_summary(detail_df)
    detail_df.to_csv(DETAIL_CSV, index=False, encoding="utf-8")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    _write_report(detail_df, summary_df)
    print(f"[audit_saved] detail={DETAIL_CSV.relative_to(ROOT)}")
    print(f"[audit_saved] summary={SUMMARY_CSV.relative_to(ROOT)}")
    print(f"[audit_saved] report={REPORT_MD.relative_to(ROOT)}")


def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()
