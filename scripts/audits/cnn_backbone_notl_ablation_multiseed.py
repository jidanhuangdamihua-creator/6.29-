"""Multi-seed No-TL CNN backbone ablation audit.

This script is read-only with respect to the main experiment path. It reuses
the current No-TL data construction, split, window size, metrics, epochs, and
batch size, then writes audit-only outputs under outputs/audits/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from environment import setup_logging, setup_reproducibility
from scripts.audits.cnn_backbone_notl_audit import (
    ABLATION_COLUMNS,
    BACKBONE_SPECS,
    DATASET_ID,
    DATASETS,
    HORIZON,
    OUT_DIR,
    BackboneSpec,
    SplitBundle,
    _ablation_metric_row,
    _build_sequences,
    _empty_ablation_row,
    _format_value,
    _load_config,
    _metric_dict,
    _prepare_dataset,
)


SEEDS = [42, 43, 44, 45, 46]
MULTISEED_CSV = OUT_DIR / "cnn_backbone_notl_ablation_multiseed.csv"
MULTISEED_REPORT_MD = OUT_DIR / "cnn_backbone_notl_ablation_multiseed.md"
MODEL_NAMES = ["current_3layer_cnn", "conv1_gap_dense", "naive_persistence"]
MULTISEED_BACKBONE_SPECS = [spec for spec in BACKBONE_SPECS if spec.name in MODEL_NAMES]

MULTISEED_COLUMNS = list(ABLATION_COLUMNS)


def _with_seed(row: Dict[str, Any], seed: int) -> Dict[str, Any]:
    row = dict(row)
    row["random_seed"] = int(seed)
    return row


def _run_persistence_for_seed(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    metric_protocol: Dict[str, Any],
    spec: BackboneSpec,
    window_size: int,
    target_epochs: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    if len(sequences["y_test"]) == 0:
        return _with_seed(
            _empty_ablation_row(bundle, sequences, spec, window_size, target_epochs, batch_size, "SKIPPED", "empty test windows"),
            seed,
        )
    if "sales" not in bundle.feature_columns:
        return _with_seed(
            _empty_ablation_row(bundle, sequences, spec, window_size, target_epochs, batch_size, "ERROR", "sales not in feature columns"),
            seed,
        )

    start = time.perf_counter()
    sales_idx = bundle.feature_columns.index("sales")
    y_pred = sequences["x_test"][:, -1, sales_idx].reshape(-1, 1)
    metric = _metric_dict(sequences["y_test"], y_pred, metric_protocol, bundle)
    row = _ablation_metric_row(
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
    return _with_seed(row, seed)


def _run_keras_model_for_seed(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    metric_protocol: Dict[str, Any],
    spec: BackboneSpec,
    window_size: int,
    target_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, Any]:
    if len(sequences["y_train"]) == 0 or len(sequences["y_test"]) == 0:
        return _with_seed(
            _empty_ablation_row(bundle, sequences, spec, window_size, target_epochs, batch_size, "SKIPPED", "empty train/test windows"),
            seed,
        )

    import tensorflow as tf

    setup_reproducibility(seed)
    tf.keras.backend.clear_session()
    setup_reproducibility(seed)

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
        return _with_seed(
            _empty_ablation_row(
                bundle,
                sequences,
                spec,
                window_size,
                target_epochs,
                batch_size,
                "ERROR",
                f"{type(exc).__name__}: {exc}",
            ),
            seed,
        )

    row = _ablation_metric_row(
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
    return _with_seed(row, seed)


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _summary_by_dataset_model(ablation_df: pd.DataFrame) -> pd.DataFrame:
    ok = ablation_df[ablation_df["status"].eq("OK")].copy()
    return (
        ok.groupby(["dataset", "model_name"], as_index=False)
        .agg(
            seeds_run=("random_seed", "nunique"),
            mean_normalized_rmse=("normalized_rmse", "mean"),
            std_normalized_rmse=("normalized_rmse", "std"),
            min_normalized_rmse=("normalized_rmse", "min"),
            max_normalized_rmse=("normalized_rmse", "max"),
        )
        .sort_values(["dataset", "mean_normalized_rmse", "model_name"])
        .reset_index(drop=True)
    )


def _majority_comparison(ablation_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    ok = ablation_df[ablation_df["status"].eq("OK")].copy()
    pivot = ok.pivot_table(
        index=["dataset", "random_seed"],
        columns="model_name",
        values="normalized_rmse",
        aggfunc="first",
    ).reset_index()
    rows: List[Dict[str, Any]] = []
    for dataset, group in pivot.groupby("dataset", sort=True):
        paired = group.dropna(subset=["current_3layer_cnn", baseline]).copy()
        better = paired[baseline] < paired["current_3layer_cnn"] if not paired.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "paired_seeds": int(len(paired)),
                "baseline_better_seed_count": int(better.sum()) if not paired.empty else 0,
                "baseline_better_majority": bool(int(better.sum()) > len(paired) / 2) if not paired.empty else False,
                "mean_current_minus_baseline": float((paired["current_3layer_cnn"] - paired[baseline]).mean()) if not paired.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_report(ablation_df: pd.DataFrame, window_size: int, target_epochs: int, batch_size: int) -> None:
    summary = _summary_by_dataset_model(ablation_df)
    gap_compare = _majority_comparison(ablation_df, "conv1_gap_dense")
    persistence_compare = _majority_comparison(ablation_df, "naive_persistence")
    compare = pd.concat([gap_compare, persistence_compare], ignore_index=True)

    gap_majority_datasets = int(gap_compare["baseline_better_majority"].sum()) if not gap_compare.empty else 0
    persistence_majority_datasets = int(persistence_compare["baseline_better_majority"].sum()) if not persistence_compare.empty else 0
    stable_disadvantage = gap_majority_datasets >= 2 or persistence_majority_datasets >= 2
    stable_text = (
        "存在稳定结构性劣势信号"
        if stable_disadvantage
        else "未形成稳定结构性劣势结论"
    )

    error_rows = ablation_df[ablation_df["status"].ne("OK")][
        ["dataset", "model_name", "random_seed", "status", "error_message"]
    ].copy()

    lines = [
        "# CNN Backbone No-TL Multi-Seed Ablation",
        "",
        f"Scope: No-TL only; Dataset1/2/3; seeds={SEEDS}; horizon={HORIZON}; window_size={window_size}; target_epochs={target_epochs}; batch_size={batch_size}. Current split, metric protocol, window size, target epochs, and batch size are read from the existing configuration and are not changed. No main experiment result files are read or overwritten.",
        "",
        "## Output Files",
        "",
        f"- Multi-seed ablation CSV: `{MULTISEED_CSV.relative_to(ROOT)}`",
        f"- Multi-seed report: `{MULTISEED_REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Mean/Std Normalized RMSE",
        "",
        _markdown_table(
            summary,
            [
                "dataset",
                "model_name",
                "seeds_run",
                "mean_normalized_rmse",
                "std_normalized_rmse",
                "min_normalized_rmse",
                "max_normalized_rmse",
            ],
        ),
        "",
        "## Majority-Seed Comparisons Against Current CNN",
        "",
        _markdown_table(
            compare,
            [
                "dataset",
                "baseline",
                "paired_seeds",
                "baseline_better_seed_count",
                "baseline_better_majority",
                "mean_current_minus_baseline",
            ],
        ),
        "",
        f"conv1_gap_dense majority result: better than current_3layer_cnn in {gap_majority_datasets}/3 datasets by majority seed count.",
        f"naive_persistence majority result: better than current_3layer_cnn in {persistence_majority_datasets}/3 datasets by majority seed count.",
        "",
        "## Stable Structural Disadvantage Judgment",
        "",
        f"当前 CNN 是否存在稳定结构性劣势: {stable_text}. The judgment is based only on paired seeds under the unchanged No-TL data/split/window/metric/training protocol. A positive `mean_current_minus_baseline` means current_3layer_cnn has higher normalized RMSE than the baseline.",
        "",
        "Interpretation: If conv1_gap_dense or naive_persistence wins on most seeds for most datasets, the result supports treating the current 3-layer CNN as sensitivity-risky for the paper-aligned No-TL small-sample setting. This remains an audit finding, not a replacement for the main reproduction backbone.",
        "",
        "## Non-OK Rows",
        "",
        _markdown_table(error_rows, ["dataset", "model_name", "random_seed", "status", "error_message"]),
    ]
    MULTISEED_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level="WARNING", log_file=None)

    cfg = _load_config()
    single_cfg = dict(cfg.get("single_experiment", {}))
    window_size = int(single_cfg.get("window_size", 10))
    target_epochs = int(single_cfg.get("target_epochs", cfg.get("target_epochs", 2)))
    batch_size = int(single_cfg.get("batch_size", cfg.get("batch_size", 16)))
    learning_rate = float(single_cfg.get("learning_rate", 1e-4))
    metric_protocol = dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))

    bundles = {dataset: _prepare_dataset(dataset, cfg) for dataset in DATASETS}
    sequences_by_dataset = {dataset: _build_sequences(bundle, window_size) for dataset, bundle in bundles.items()}

    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        bundle = bundles[dataset]
        sequences = sequences_by_dataset[dataset]
        for seed in SEEDS:
            for spec in MULTISEED_BACKBONE_SPECS:
                if spec.factory is None:
                    rows.append(
                        _run_persistence_for_seed(
                            bundle=bundle,
                            sequences=sequences,
                            metric_protocol=metric_protocol,
                            spec=spec,
                            window_size=window_size,
                            target_epochs=target_epochs,
                            batch_size=batch_size,
                            seed=seed,
                        )
                    )
                else:
                    rows.append(
                        _run_keras_model_for_seed(
                            bundle=bundle,
                            sequences=sequences,
                            metric_protocol=metric_protocol,
                            spec=spec,
                            window_size=window_size,
                            target_epochs=target_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            seed=seed,
                        )
                    )

    ablation_df = pd.DataFrame(rows).sort_values(["dataset_id", "random_seed", "model_name"])
    ablation_df[MULTISEED_COLUMNS].to_csv(MULTISEED_CSV, index=False, encoding="utf-8")
    _write_report(ablation_df, window_size=window_size, target_epochs=target_epochs, batch_size=batch_size)

    print(f"Wrote {MULTISEED_CSV}")
    print(f"Wrote {MULTISEED_REPORT_MD}")


if __name__ == "__main__":
    main()
