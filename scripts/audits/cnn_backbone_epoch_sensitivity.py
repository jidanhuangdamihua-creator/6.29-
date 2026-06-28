"""No-TL epoch sensitivity audit for CNN backbones.

This script does not modify the main experiment path. It keeps the current
paper-aligned split, window size, batch size, and metric protocol, then varies
only target_epochs for audit-only No-TL model comparisons.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from environment import setup_logging
from scripts.audits.cnn_backbone_notl_audit import (
    ABLATION_COLUMNS,
    BACKBONE_SPECS,
    DATASETS,
    HORIZON,
    OUT_DIR,
    _build_sequences,
    _format_value,
    _load_config,
    _prepare_dataset,
)
from scripts.audits.cnn_backbone_notl_ablation_multiseed import (
    MODEL_NAMES,
    _run_keras_model_for_seed,
    _run_persistence_for_seed,
)


SEED = 42
TARGET_EPOCHS = [50]
EPOCH_SENSITIVITY_CSV = OUT_DIR / "cnn_backbone_epoch_sensitivity.csv"
EPOCH_SENSITIVITY_REPORT_MD = OUT_DIR / "cnn_backbone_epoch_sensitivity.md"
EPOCH_BACKBONE_SPECS = [spec for spec in BACKBONE_SPECS if spec.name in MODEL_NAMES]
EPOCH_SENSITIVITY_COLUMNS = list(ABLATION_COLUMNS)


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _best_by_dataset_model(ablation_df: pd.DataFrame) -> pd.DataFrame:
    ok = ablation_df[ablation_df["status"].eq("OK")].copy()
    rows: List[Dict[str, Any]] = []
    for (dataset, model_name), group in ok.groupby(["dataset", "model_name"], sort=True):
        best = group.sort_values("normalized_rmse").iloc[0]
        ep2 = group[group["target_epochs"].eq(2)]
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "epoch2_normalized_rmse": float(ep2.iloc[0]["normalized_rmse"]) if not ep2.empty else np.nan,
                "best_target_epochs": int(best["target_epochs"]),
                "best_normalized_rmse": float(best["normalized_rmse"]),
                "best_minus_epoch2": (
                    float(best["normalized_rmse"] - ep2.iloc[0]["normalized_rmse"])
                    if not ep2.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "best_normalized_rmse", "model_name"]).reset_index(drop=True)


def _paired_epoch_comparison(ablation_df: pd.DataFrame) -> pd.DataFrame:
    ok = ablation_df[ablation_df["status"].eq("OK")].copy()
    pivot = ok.pivot_table(
        index=["dataset", "target_epochs"],
        columns="model_name",
        values="normalized_rmse",
        aggfunc="first",
    ).reset_index()

    rows: List[Dict[str, Any]] = []
    for dataset, group in pivot.groupby("dataset", sort=True):
        for baseline in ["conv1_gap_dense", "naive_persistence"]:
            paired = group.dropna(subset=["current_3layer_cnn", baseline]).copy()
            better = paired[baseline] < paired["current_3layer_cnn"] if not paired.empty else pd.Series(dtype=bool)
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "paired_epochs": int(len(paired)),
                    "baseline_better_epoch_count": int(better.sum()) if not paired.empty else 0,
                    "baseline_better_majority": bool(int(better.sum()) > len(paired) / 2) if not paired.empty else False,
                    "current_best_beats_baseline_any_epoch": (
                        bool((paired["current_3layer_cnn"] < paired[baseline]).any()) if not paired.empty else False
                    ),
                    "mean_current_minus_baseline": (
                        float((paired["current_3layer_cnn"] - paired[baseline]).mean()) if not paired.empty else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _current_vs_persistence_best(ablation_df: pd.DataFrame) -> pd.DataFrame:
    ok = ablation_df[ablation_df["status"].eq("OK")].copy()
    pivot = ok.pivot_table(
        index=["dataset", "target_epochs"],
        columns="model_name",
        values="normalized_rmse",
        aggfunc="first",
    ).reset_index()
    rows: List[Dict[str, Any]] = []
    for dataset, group in pivot.groupby("dataset", sort=True):
        current = group.dropna(subset=["current_3layer_cnn"])
        persistence = group.dropna(subset=["naive_persistence"])
        best_current = float(current["current_3layer_cnn"].min()) if not current.empty else np.nan
        best_current_epoch = int(current.sort_values("current_3layer_cnn").iloc[0]["target_epochs"]) if not current.empty else np.nan
        persistence_rmse = float(persistence["naive_persistence"].iloc[0]) if not persistence.empty else np.nan
        rows.append(
            {
                "dataset": dataset,
                "best_current_epoch": best_current_epoch,
                "best_current_normalized_rmse": best_current,
                "naive_persistence_normalized_rmse": persistence_rmse,
                "best_current_beats_persistence": bool(best_current < persistence_rmse) if not pd.isna(best_current) and not pd.isna(persistence_rmse) else False,
                "best_current_minus_persistence": best_current - persistence_rmse if not pd.isna(best_current) and not pd.isna(persistence_rmse) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_report(ablation_df: pd.DataFrame, window_size: int, batch_size: int) -> None:
    ok = ablation_df[ablation_df["status"].eq("OK")].copy()
    best = _best_by_dataset_model(ablation_df)
    compare = _paired_epoch_comparison(ablation_df)
    current_vs_persistence = _current_vs_persistence_best(ablation_df)
    error_rows = ablation_df[ablation_df["status"].ne("OK")][
        ["dataset", "model_name", "target_epochs", "status", "error_message"]
    ].copy()

    current_best = best[best["model_name"].eq("current_3layer_cnn")].copy()
    current_epoch2 = current_best["epoch2_normalized_rmse"]
    current_improved = bool((current_best["best_minus_epoch2"] < 0).any()) if not current_best.empty else False
    current_large_improved = bool((current_best["best_minus_epoch2"] < -0.05).any()) if not current_best.empty else False
    current_beats_persistence_count = int(current_vs_persistence["best_current_beats_persistence"].sum()) if not current_vs_persistence.empty else 0

    gap_compare = compare[compare["baseline"].eq("conv1_gap_dense")]
    gap_majority_datasets = int(gap_compare["baseline_better_majority"].sum()) if not gap_compare.empty else 0
    gap_stable = gap_majority_datasets >= 2

    if current_beats_persistence_count == 0:
        persistence_answer = "不能。即使取 current_3layer_cnn 在 epoch grid 上的最佳结果，也没有任何 dataset 超过 naive_persistence。"
    elif current_beats_persistence_count < len(current_vs_persistence):
        persistence_answer = f"只能部分超过。current_3layer_cnn 的最佳 epoch 在 {current_beats_persistence_count}/{len(current_vs_persistence)} 个 dataset 超过 naive_persistence。"
    else:
        persistence_answer = "可以。current_3layer_cnn 的最佳 epoch 在所有 dataset 上超过 naive_persistence。"

    if gap_stable:
        gap_answer = f"是。conv1_gap_dense 在 {gap_majority_datasets}/3 个 dataset 的多数 epoch 下优于 current_3layer_cnn。"
    else:
        gap_answer = f"不稳定。conv1_gap_dense 只在 {gap_majority_datasets}/3 个 dataset 的多数 epoch 下优于 current_3layer_cnn。"

    if current_large_improved:
        training_vs_structure = (
            "epoch 增加确实改善了部分 current_3layer_cnn 结果，因此 epoch=2 是训练设置风险；"
            "但若改善后仍常被 persistence 或更小 CNN 超过，则仍保留 CNN 结构/容量与小样本不匹配的证据。"
        )
    elif current_improved:
        training_vs_structure = (
            "epoch 增加带来轻微改善，更像训练设置和随机优化噪声共同作用；"
            "当前证据不足以把问题完全归因于 epoch=2。"
        )
    else:
        training_vs_structure = (
            "epoch 增加没有稳定改善 current_3layer_cnn，因此不能把差距简单归因于 epoch=2 训练不够；"
            "小样本下的 CNN 结构/容量风险更值得作为 sensitivity finding 保留。"
        )

    lines = [
        "# CNN Backbone Epoch Sensitivity Audit",
        "",
        f"Scope: No-TL only; Dataset1/2/3; seed={SEED}; target_epochs={TARGET_EPOCHS}; horizon={HORIZON}; window_size={window_size}; batch_size={batch_size}. Current split, window size, batch size, and metric protocol are reused from the existing paper-aligned configuration. No main experiment code or main result files are modified.",
        "",
        "## Output Files",
        "",
        f"- Epoch sensitivity CSV: `{EPOCH_SENSITIVITY_CSV.relative_to(ROOT)}`",
        f"- Epoch sensitivity report: `{EPOCH_SENSITIVITY_REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Per-Epoch Normalized RMSE",
        "",
        _markdown_table(
            ok.sort_values(["dataset", "target_epochs", "model_name"]),
            ["dataset", "target_epochs", "model_name", "train_windows", "val_windows", "test_windows", "normalized_rmse", "original_scale_rmse", "status"],
        ),
        "",
        "## Best Epoch By Dataset And Model",
        "",
        _markdown_table(
            best,
            ["dataset", "model_name", "epoch2_normalized_rmse", "best_target_epochs", "best_normalized_rmse", "best_minus_epoch2"],
        ),
        "",
        "## Paired Comparisons Against Current CNN",
        "",
        _markdown_table(
            compare,
            [
                "dataset",
                "baseline",
                "paired_epochs",
                "baseline_better_epoch_count",
                "baseline_better_majority",
                "current_best_beats_baseline_any_epoch",
                "mean_current_minus_baseline",
            ],
        ),
        "",
        "## Current CNN Best Epoch Versus Persistence",
        "",
        _markdown_table(
            current_vs_persistence,
            [
                "dataset",
                "best_current_epoch",
                "best_current_normalized_rmse",
                "naive_persistence_normalized_rmse",
                "best_current_beats_persistence",
                "best_current_minus_persistence",
            ],
        ),
        "",
        "## Required Answers",
        "",
        f"1. 当前 3-layer CNN 是否只是因为 epoch=2 没训练够: {'不是只能归因于 epoch=2。' if not current_large_improved else 'epoch=2 训练不足是贡献因素之一，但不是唯一解释。'} {training_vs_structure}",
        "",
        f"2. epoch 增加后 current_3layer_cnn 是否能超过 naive_persistence: {persistence_answer}",
        "",
        f"3. conv1_gap_dense 是否仍然稳定优于 current_3layer_cnn: {gap_answer}",
        "",
        f"4. 如果 epoch 增加改善结果，这属于训练设置问题还是 CNN 结构问题: {training_vs_structure}",
        "",
        "## Non-OK Rows",
        "",
        _markdown_table(error_rows, ["dataset", "model_name", "target_epochs", "status", "error_message"]),
    ]
    EPOCH_SENSITIVITY_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level="WARNING", log_file=None)

    cfg = _load_config()
    single_cfg = dict(cfg.get("single_experiment", {}))
    window_size = int(single_cfg.get("window_size", 10))
    batch_size = int(single_cfg.get("batch_size", cfg.get("batch_size", 16)))
    learning_rate = float(single_cfg.get("learning_rate", 1e-4))
    metric_protocol = dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))

    bundles = {dataset: _prepare_dataset(dataset, cfg) for dataset in DATASETS}
    sequences_by_dataset = {dataset: _build_sequences(bundle, window_size) for dataset, bundle in bundles.items()}

    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        bundle = bundles[dataset]
        sequences = sequences_by_dataset[dataset]
        for target_epochs in TARGET_EPOCHS:
            for spec in EPOCH_BACKBONE_SPECS:
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
                            seed=SEED,
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
                            seed=SEED,
                        )
                    )

    ablation_df = pd.DataFrame(rows).sort_values(["dataset_id", "target_epochs", "model_name"])
    ablation_df[EPOCH_SENSITIVITY_COLUMNS].to_csv(EPOCH_SENSITIVITY_CSV, index=False, encoding="utf-8")
    _write_report(ablation_df, window_size=window_size, batch_size=batch_size)

    print(f"Wrote {EPOCH_SENSITIVITY_CSV}")
    print(f"Wrote {EPOCH_SENSITIVITY_REPORT_MD}")


if __name__ == "__main__":
    main()
