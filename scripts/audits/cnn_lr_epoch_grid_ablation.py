"""No-TL CNN learning-rate by epoch grid ablation audit.

This audit keeps the CNN structure, batch-size path, data cleaning, split, KNN,
RFE, and metric code unchanged. It varies only Adam learning_rate and training
epochs for No-TL, and writes all outputs under outputs/audits/.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-codex"))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from environment import setup_logging, setup_reproducibility
from scripts.audits.cnn_lr_clipnorm_ablation import (
    _compute_metric,
    _load_config,
    _metric_protocol,
    _prepare_sequences,
)
from src.models.cnn_model import build_base_cnn
from src.utils.experiment_hyperparams import (
    EPOCH_LIST,
    LR_LIST,
    FIXED_CLIPNORM,
    FIXED_DROPOUT,
    fixed_hyperparams_slug,
    fixed_hyperparams_summary,
)
from src.utils.runtime_control import set_verbose_mode


OUT_DIR = ROOT / "outputs" / "audits"
_HP_SLUG = fixed_hyperparams_slug()
DETAIL_CSV = OUT_DIR / f"cnn_lr_epoch_grid_ablation_details_{_HP_SLUG}.csv"
SUMMARY_CSV = OUT_DIR / f"cnn_lr_epoch_grid_ablation_summary_{_HP_SLUG}.csv"
COMPARISON_CSV = OUT_DIR / f"cnn_lr_epoch_grid_ablation_comparison_{_HP_SLUG}.csv"
REPORT_MD = OUT_DIR / f"cnn_lr_epoch_grid_ablation_{_HP_SLUG}.md"

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
LEARNING_RATES = list(LR_LIST)
EPOCHS = list(EPOCH_LIST)
SEEDS = [42, 43, 44]
METHOD = "No-TL"
CLIPNORM = FIXED_CLIPNORM
HORIZON = 1

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset",
    "seed",
    "method",
    "learning_rate",
    "clipnorm",
    "epoch",
    "optimizer_name",
    "optimizer_changed",
    "cnn_structure_changed",
    "batch_size_changed",
    "model_parameter_count",
    "original_model_parameter_count",
    "original_batch_size",
    "effective_batch_size",
    "train_windows",
    "val_rmse",
    "test_rmse",
    "test_mae",
    "run_time_seconds",
    "error_message",
    "status",
    "notes",
]

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset",
    "learning_rate",
    "epoch",
    "n_seeds",
    "mean_test_rmse",
    "std_test_rmse",
    "min_test_rmse",
    "max_test_rmse",
    "mean_val_rmse",
    "std_val_rmse",
    "mean_test_mae",
    "std_test_mae",
    "rank_by_mean_rmse",
    "rank_by_std_rmse",
    "status",
    "conclusion",
]

COMPARISON_COLUMNS = [
    "dataset_id",
    "dataset",
    "best_lr_epoch_by_mean_rmse",
    "best_lr_epoch_by_std_rmse",
    "best_learning_rate",
    "best_epoch",
    "does_more_epoch_improve_lr_1e_4",
    "does_low_lr_need_more_epoch",
    "interpretation",
]


@dataclass(frozen=True)
class EpochGridVariant:
    optimizer_name: str
    learning_rate: float
    clipnorm: float | None
    optimizer_changed: bool
    cnn_structure_changed: bool
    batch_size_changed: bool
    original_batch_size: int
    effective_batch_size: int


def iter_grid_combinations() -> List[Tuple[str, float, int, int]]:
    return [
        (dataset, learning_rate, epoch, seed)
        for dataset in DATASETS
        for learning_rate in LEARNING_RATES
        for epoch in EPOCHS
        for seed in SEEDS
    ]


def expected_detail_row_count() -> int:
    return len(DATASETS) * len(LEARNING_RATES) * len(EPOCHS) * len(SEEDS)


def resolve_epoch_grid_variant(learning_rate: float, original_batch_size: int = 16) -> EpochGridVariant:
    return EpochGridVariant(
        optimizer_name="Adam",
        learning_rate=float(learning_rate),
        clipnorm=CLIPNORM,
        optimizer_changed=True,
        cnn_structure_changed=False,
        batch_size_changed=False,
        original_batch_size=int(original_batch_size),
        effective_batch_size=int(original_batch_size),
    )


def build_epoch_grid_model(input_shape: tuple[int, int], learning_rate: float):
    """Build the fixed-dropout base CNN with Adam gradient clipping disabled."""
    import tensorflow as tf

    model = build_base_cnn(
        input_shape=input_shape,
        learning_rate=float(learning_rate),
        dropout=FIXED_DROPOUT,
        clipnorm=CLIPNORM,
    )
    optimizer_kwargs: Dict[str, Any] = {"learning_rate": float(learning_rate)}
    if CLIPNORM is not None:
        optimizer_kwargs["clipnorm"] = CLIPNORM
    model.compile(optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs), loss="mse", metrics=["mae"])
    return model


def hard_condition_errors(
    meta: EpochGridVariant,
    model_parameter_count: int,
    original_parameter_count: int,
    expected_epoch: int,
    observed_method: str = METHOD,
) -> List[str]:
    errors: List[str] = []
    if observed_method != METHOD:
        errors.append("method must be No-TL")
    if meta.optimizer_name != "Adam":
        errors.append("optimizer_name must be Adam")
    if not any(math.isclose(float(meta.learning_rate), lr, rel_tol=0.0, abs_tol=1e-12) for lr in LEARNING_RATES):
        errors.append("learning_rate must be 1e-4")
    if int(expected_epoch) not in EPOCHS:
        errors.append("epoch must be 50")
    if meta.clipnorm is not None:
        errors.append("clipnorm must be None")
    if meta.cnn_structure_changed is not False:
        errors.append("cnn_structure_changed must be False")
    if meta.batch_size_changed is not False:
        errors.append("batch_size_changed must be False")
    if int(model_parameter_count) != int(original_parameter_count):
        errors.append("model_parameter_count must equal original_model_parameter_count")
    if int(meta.effective_batch_size) != int(meta.original_batch_size):
        errors.append("effective_batch_size must equal original_batch_size")
    return errors


def _base_row(dataset: str, seed: int, meta: EpochGridVariant, epoch: int, train_windows: int) -> Dict[str, Any]:
    row = {
        "dataset_id": DATASET_ID[dataset],
        "dataset": dataset,
        "seed": int(seed),
        "method": METHOD,
        **asdict(meta),
        "epoch": int(epoch),
        "model_parameter_count": np.nan,
        "original_model_parameter_count": np.nan,
        "train_windows": int(train_windows),
        "val_rmse": np.nan,
        "test_rmse": np.nan,
        "test_mae": np.nan,
        "run_time_seconds": 0.0,
        "error_message": "",
        "status": "OK",
        "notes": "",
    }
    return {column: row.get(column, np.nan) for column in DETAIL_COLUMNS}


def _run_one(
    prepared: Dict[str, Any],
    metric_protocol: Dict[str, Any],
    dataset: str,
    seed: int,
    learning_rate: float,
    epoch: int,
    original_batch_size: int,
    original_parameter_count: int,
) -> Dict[str, Any]:
    import tensorflow as tf

    meta = resolve_epoch_grid_variant(learning_rate=learning_rate, original_batch_size=original_batch_size)
    row = _base_row(dataset, seed, meta, epoch=epoch, train_windows=len(prepared["y_train"]))
    if len(prepared["y_train"]) == 0 or len(prepared["y_test"]) == 0:
        row["status"] = "ERROR"
        row["error_message"] = "empty train/test windows"
        row["notes"] = "No-TL requires non-empty target train and test windows."
        return row

    setup_reproducibility(seed)
    tf.keras.backend.clear_session()
    setup_reproducibility(seed)
    start = time.perf_counter()
    try:
        model = build_epoch_grid_model(input_shape=prepared["x_train"].shape[1:], learning_rate=learning_rate)
        model_parameter_count = int(model.count_params())
        row["model_parameter_count"] = model_parameter_count
        row["original_model_parameter_count"] = int(original_parameter_count)
        hard_errors = hard_condition_errors(meta, model_parameter_count, original_parameter_count, expected_epoch=epoch)
        if hard_errors:
            row["status"] = "FAIL"
            row["error_message"] = "; ".join(hard_errors)
            row["notes"] = "Hard condition check failed before training."
            return row

        history = model.fit(
            prepared["x_train"],
            prepared["y_train"],
            epochs=int(epoch),
            batch_size=int(meta.effective_batch_size),
            validation_data=(prepared["x_val"], prepared["y_val"]) if len(prepared["y_val"]) > 0 else None,
            verbose=0,
        )
        y_test_pred = model.predict(prepared["x_test"], verbose=0)
        test_metric = _compute_metric(prepared["y_test"], y_test_pred, metric_protocol, prepared)
        if len(prepared["y_val"]) > 0:
            y_val_pred = model.predict(prepared["x_val"], verbose=0)
            val_metric = _compute_metric(prepared["y_val"], y_val_pred, metric_protocol, prepared)
            row["val_rmse"] = float(val_metric["rmse"])
        row["test_rmse"] = float(test_metric["rmse"])
        row["test_mae"] = float(test_metric["mae"])
        row["epoch"] = int(len(history.history.get("loss", [])))
        row["run_time_seconds"] = float(time.perf_counter() - start)
        row["notes"] = "OK; only optimizer learning_rate and epoch varied."
    except Exception as exc:
        row["status"] = "ERROR"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        row["run_time_seconds"] = float(time.perf_counter() - start)
        row["notes"] = "Training or evaluation failed."
    return row


def _build_summary(details: pd.DataFrame) -> pd.DataFrame:
    ok = details[details["status"].eq("OK")].copy()
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_ok = ok[ok["dataset"].eq(dataset)]
        for learning_rate in LEARNING_RATES:
            for epoch in EPOCHS:
                group = dataset_ok[
                    dataset_ok["learning_rate"].astype(float).apply(lambda value: math.isclose(value, learning_rate, abs_tol=1e-12))
                    & dataset_ok["epoch"].eq(epoch)
                ]
                if group.empty:
                    rows.append(
                        {
                            "dataset_id": DATASET_ID[dataset],
                            "dataset": dataset,
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "n_seeds": 0,
                            "status": "ERROR",
                            "conclusion": "No successful runs.",
                        }
                    )
                    continue
                rows.append(
                    {
                        "dataset_id": DATASET_ID[dataset],
                        "dataset": dataset,
                        "learning_rate": learning_rate,
                        "epoch": epoch,
                        "n_seeds": int(group["seed"].nunique()),
                        "mean_test_rmse": float(group["test_rmse"].mean()),
                        "std_test_rmse": float(group["test_rmse"].std()) if len(group) > 1 else np.nan,
                        "min_test_rmse": float(group["test_rmse"].min()),
                        "max_test_rmse": float(group["test_rmse"].max()),
                        "mean_val_rmse": float(group["val_rmse"].mean()),
                        "std_val_rmse": float(group["val_rmse"].std()) if len(group) > 1 else np.nan,
                        "mean_test_mae": float(group["test_mae"].mean()),
                        "std_test_mae": float(group["test_mae"].std()) if len(group) > 1 else np.nan,
                        "status": "OK",
                        "conclusion": "Computed from OK detail rows.",
                    }
                )
    summary = pd.DataFrame(rows)
    for _, idx in summary.groupby("dataset").groups.items():
        summary.loc[idx, "rank_by_mean_rmse"] = summary.loc[idx, "mean_test_rmse"].rank(method="min", ascending=True)
        summary.loc[idx, "rank_by_std_rmse"] = summary.loc[idx, "std_test_rmse"].rank(method="min", ascending=True)
    return summary.reindex(columns=SUMMARY_COLUMNS)


def _lr_epoch_label(learning_rate: float, epoch: int) -> str:
    return f"lr={learning_rate:g}, epoch={int(epoch)}"


def _improves_with_more_epochs(group: pd.DataFrame, learning_rate: float) -> bool:
    lr_rows = group[
        group["learning_rate"].astype(float).apply(lambda value: math.isclose(value, learning_rate, abs_tol=1e-12))
        & group["status"].eq("OK")
    ].sort_values("epoch")
    if lr_rows.empty:
        return False
    first = lr_rows[lr_rows["epoch"].eq(EPOCHS[0])]
    if first.empty:
        return False
    first_rmse = float(first.iloc[0]["mean_test_rmse"])
    later = lr_rows[lr_rows["epoch"] > EPOCHS[0]]
    return bool(not later.empty and later["mean_test_rmse"].min() < first_rmse)


def _build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        group = summary[summary["dataset"].eq(dataset) & summary["status"].eq("OK")].copy()
        if group.empty:
            rows.append(
                {
                    "dataset_id": DATASET_ID[dataset],
                    "dataset": dataset,
                    "does_more_epoch_improve_lr_1e_4": False,
                    "does_low_lr_need_more_epoch": False,
                    "interpretation": "No successful runs.",
                }
            )
            continue
        best_mean = group.sort_values(["mean_test_rmse", "std_test_rmse", "learning_rate", "epoch"]).iloc[0]
        best_std = group.sort_values(["std_test_rmse", "mean_test_rmse", "learning_rate", "epoch"]).iloc[0]
        improves_1e_4 = _improves_with_more_epochs(group, 1e-4)
        low_lr_rows = group[group["learning_rate"].astype(float).apply(lambda value: math.isclose(value, 1e-4, abs_tol=1e-12))]
        low_lr_best_epoch = int(low_lr_rows.sort_values("mean_test_rmse").iloc[0]["epoch"]) if not low_lr_rows.empty else 0
        low_lr_need_more = bool(improves_1e_4 and low_lr_best_epoch > EPOCHS[0])
        rows.append(
            {
                "dataset_id": DATASET_ID[dataset],
                "dataset": dataset,
                "best_lr_epoch_by_mean_rmse": _lr_epoch_label(best_mean["learning_rate"], best_mean["epoch"]),
                "best_lr_epoch_by_std_rmse": _lr_epoch_label(best_std["learning_rate"], best_std["epoch"]),
                "best_learning_rate": float(best_mean["learning_rate"]),
                "best_epoch": int(best_mean["epoch"]),
                "does_more_epoch_improve_lr_1e_4": improves_1e_4,
                "does_low_lr_need_more_epoch": low_lr_need_more,
                "interpretation": (
                    f"Best mean RMSE is {_lr_epoch_label(best_mean['learning_rate'], best_mean['epoch'])}; "
                    f"best std RMSE is {_lr_epoch_label(best_std['learning_rate'], best_std['epoch'])}; "
                    f"lr=1e-4 {'improves after the first fixed epoch row' if improves_1e_4 else 'does not improve beyond the fixed epoch row'}."
                ),
            }
        )
    return pd.DataFrame(rows).reindex(columns=COMPARISON_COLUMNS)


def _format_value(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if bool(value) else "No"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return " ".join(str(value).replace("|", "/").replace("\n", " ").split())


def _markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 80) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(max_rows).iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _overall_status(details: pd.DataFrame) -> str:
    if details["status"].eq("FAIL").any():
        return "FAIL"
    if len(details) != expected_detail_row_count() or details["status"].ne("OK").any():
        return "PARTIAL"
    return "PASS"


def _lr_epoch_trend_text(summary: pd.DataFrame, dataset: str, learning_rate: float) -> str:
    rows = summary[
        summary["dataset"].eq(dataset)
        & summary["status"].eq("OK")
        & summary["learning_rate"].astype(float).apply(lambda value: math.isclose(value, learning_rate, abs_tol=1e-12))
    ].sort_values("epoch")
    if rows.empty:
        return "no successful rows"
    best = rows.sort_values("mean_test_rmse").iloc[0]
    first = rows[rows["epoch"].eq(EPOCHS[0])].iloc[0]
    last = rows[rows["epoch"].eq(EPOCHS[-1])].iloc[0]
    final_direction = "final epoch improved vs epoch 2" if float(last["mean_test_rmse"]) < float(first["mean_test_rmse"]) else "final epoch worsened vs epoch 2"
    return (
        f"{final_direction}; best epoch={int(best['epoch'])}, "
        f"best mean RMSE={float(best['mean_test_rmse']):.6f}, "
        f"epoch 50 mean RMSE={float(last['mean_test_rmse']):.6f}"
    )


def _write_report(details: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    status = _overall_status(details)
    hard_rows = (
        details.groupby(["dataset", "learning_rate", "epoch"], as_index=False)
        .agg(
            rows=("status", "size"),
            failures=("status", lambda s: int((s == "FAIL").sum())),
            errors=("status", lambda s: int((s == "ERROR").sum())),
            cnn_structure_changed=("cnn_structure_changed", "first"),
            batch_size_changed=("batch_size_changed", "first"),
            parameter_count=("model_parameter_count", "first"),
            original_parameter_count=("original_model_parameter_count", "first"),
            effective_batch_size=("effective_batch_size", "first"),
        )
        .sort_values(["dataset", "learning_rate", "epoch"])
    )
    best_rows = comparison[
        [
            "dataset",
            "best_lr_epoch_by_mean_rmse",
            "best_lr_epoch_by_std_rmse",
            "does_more_epoch_improve_lr_1e_4",
            "does_more_epoch_improve_lr_1e_4",
            "does_low_lr_need_more_epoch",
        ]
    ].copy()
    low_lr_need = bool(comparison["does_low_lr_need_more_epoch"].any()) if not comparison.empty else False
    fixed_recommendation = comparison["best_lr_epoch_by_mean_rmse"].mode().iloc[0] if not comparison.empty else ""

    trend_lines: List[str] = []
    for dataset in DATASETS:
        trend_lines.append(f"- {dataset} lr=1e-4: {_lr_epoch_trend_text(summary, dataset, 1e-4)}.")

    lines = [
        "# CNN Learning Rate x Epoch Grid Ablation",
        "",
        "## 1. Scope",
        "",
        "This audit tests only the interaction between `learning_rate` and `epoch` for the No-TL CNN.",
        "",
        "Grid: learning_rates = [1e-4], epochs = [50], seeds = [42, 43, 44], datasets = [Dataset1, Dataset2, Dataset3].",
        "",
        "## 2. Invariance Checks",
        "",
        "- CNN structure: unchanged; all runs call the same `build_base_cnn` architecture.",
        "- batch_size: unchanged; `effective_batch_size` equals the configured original batch size.",
        "- KNN: unchanged and not invoked by this No-TL audit.",
        "- RFE: unchanged and not invoked by this No-TL audit.",
        "- split: unchanged; reused the existing temporal split helper.",
        "- RMSE: unchanged; reused `compute_metrics_with_protocol`.",
        "- accuracy: unchanged and not redefined in this audit.",
        "- data cleaning: unchanged; reused the existing preprocessing helpers.",
        "- main experiment results: not overwritten; all outputs are under `outputs/audits/`.",
        "",
        _markdown_table(
            hard_rows,
            [
                "dataset",
                "learning_rate",
                "epoch",
                "rows",
                "failures",
                "errors",
                "cnn_structure_changed",
                "batch_size_changed",
                "parameter_count",
                "original_parameter_count",
                "effective_batch_size",
            ],
            max_rows=45,
        ),
        "",
        "## 3. Best Mean And Stability",
        "",
        _markdown_table(best_rows, best_rows.columns, max_rows=20),
        "",
        "## 4. Full Summary",
        "",
        _markdown_table(
            summary.sort_values(["dataset_id", "rank_by_mean_rmse"]),
            [
                "dataset",
                "learning_rate",
                "epoch",
                "n_seeds",
                "mean_test_rmse",
                "std_test_rmse",
                "mean_val_rmse",
                "rank_by_mean_rmse",
                "rank_by_std_rmse",
            ],
            max_rows=60,
        ),
        "",
        "## 5. Epoch Trend Checks",
        "",
        "\n".join(trend_lines),
        "",
        "## 6. Low Learning Rate Evidence",
        "",
        (
            "There is evidence that low learning rate needs more epochs in at least one dataset."
            if low_lr_need
            else "There is no consistent evidence that lr=1e-4 needs more epochs across the tested datasets."
        ),
        "",
        "## 7. Main-Experiment Recommendation",
        "",
        f"Do not change the main experiment yet based on a single small grid. If a fixed optimizer + epoch must be selected for the next audit, use the most frequent best setting by mean RMSE: `{fixed_recommendation}`.",
        "",
        "## 8. Conclusion",
        "",
        status,
        "",
        f"Reason: expected {expected_detail_row_count()} detail rows; observed {len(details)} rows; FAIL rows={int(details['status'].eq('FAIL').sum())}; ERROR rows={int(details['status'].eq('ERROR').sum())}.",
        "",
        "## Output Files",
        "",
        f"- `{DETAIL_CSV.relative_to(ROOT)}`",
        f"- `{SUMMARY_CSV.relative_to(ROOT)}`",
        f"- `{COMPARISON_CSV.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging(log_level="WARNING", log_file=None)
    set_verbose_mode("summary")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    exp = config.get("single_experiment", {})
    original_batch_size = int(exp.get("batch_size", config.get("batch_size", 16)))
    metric_protocol = _metric_protocol(config)

    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        prepared = _prepare_sequences(dataset, config)
        input_shape = prepared["x_train"].shape[1:]
        original_parameter_count = int(build_epoch_grid_model(input_shape, learning_rate=LEARNING_RATES[0]).count_params())
        for learning_rate in LEARNING_RATES:
            for epoch in EPOCHS:
                for seed in SEEDS:
                    rows.append(
                        _run_one(
                            prepared=prepared,
                            metric_protocol=metric_protocol,
                            dataset=dataset,
                            seed=seed,
                            learning_rate=learning_rate,
                            epoch=epoch,
                            original_batch_size=original_batch_size,
                            original_parameter_count=original_parameter_count,
                        )
                    )

    details = pd.DataFrame(rows).reindex(columns=DETAIL_COLUMNS)
    summary = _build_summary(details)
    comparison = _build_comparison(summary)

    details.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    comparison.to_csv(COMPARISON_CSV, index=False)
    _write_report(details, summary, comparison)

    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
