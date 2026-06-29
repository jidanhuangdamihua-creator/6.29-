"""Paper alignment smoke test runner (Dataset1 only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment.experiment_runner import (
    prepare_base_data_for_experiments,
    run_msml_experiment,
    run_msml_rfe_experiment,
)
from paper_reproduction_protocol import (
    build_alignment_fields,
    ensure_paper_track_allowed,
    get_extended_source_counts,
    get_paper_source_counts,
    load_paper_protocol,
    resolve_strict_paper_mode,
    validate_paper_protocol_config,
)


DATASET_NAMES = ["Dataset1"]
METHODS = ["MSML-TL", "MSML-TL-RFE"]
INFORMATION_SHARING_MODES = [
    "without_information_sharing",
    "with_information_sharing",
]

HORIZON = 1
WEIGHT_MODE = "inverse_distance"
KEEP_RATIO = 0.5
SOURCE_EPOCHS = 2
TARGET_EPOCHS = 2
BATCH_SIZE = 16

OUTPUT_DIR = ROOT / "outputs" / "paper_alignment_smoke_test"
RESULTS_CSV = OUTPUT_DIR / "dataset1_alignment_smoke_results.csv"
FORMATTED_CSV = OUTPUT_DIR / "dataset1_alignment_smoke_results_formatted.csv"
RMSE_PNG = OUTPUT_DIR / "dataset1_alignment_smoke_rmse.png"


def _load_config() -> Dict[str, Any]:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small auditable paper-alignment smoke test.")
    parser.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help="Restrict smoke runs to paper-track-valid source counts only.",
    )
    parser.add_argument(
        "--strict-paper-split",
        action="store_true",
        help="Force strict paper split protocol (observed/forecast windows) without fallback.",
    )
    return parser.parse_args()


def _scenario_to_bool(mode: str) -> bool:
    if mode == "with_information_sharing":
        return True
    if mode == "without_information_sharing":
        return False
    raise ValueError(f"Unsupported information_sharing_mode: {mode}")


def _apply_information_sharing_filter(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    use_information_sharing = _scenario_to_bool(mode)
    if use_information_sharing:
        return source_df

    target_entities = set(target_df["entity_id"].dropna().unique().tolist())
    filtered = source_df[source_df["entity_id"].isin(target_entities)].copy()
    if filtered.empty:
        raise ValueError(
            "No source rows left under without_information_sharing; "
            "target/source entity_id overlap is empty."
        )
    return filtered


def _count_available_source_candidates(source_df: pd.DataFrame) -> int:
    if source_df.empty:
        return 0
    return int(len(source_df[["entity_id", "item_id"]].drop_duplicates()))


def _run_one_experiment(
    dataset_name: str,
    method_name: str,
    k: int,
    information_sharing_mode: str,
    cfg: Dict[str, Any],
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
) -> Dict[str, Any]:
    ds_paths = cfg["dataset_paths"]
    feature_cols = cfg["features"]["default_feature_cols"]
    exp_cfg = cfg["single_experiment"]

    base = prepare_base_data_for_experiments(
        dataset_name=dataset_name,
        data_path=ds_paths[dataset_name],
        config=cfg,
    )
    source_df = _apply_information_sharing_filter(
        source_df=base["source_df"],
        target_df=base["target_df"],
        mode=information_sharing_mode,
    )
    target_df = base["target_df"]
    actual_k = _count_available_source_candidates(source_df)

    if actual_k < int(k):
        raise ValueError("insufficient source candidates under current information_sharing_mode")

    ensure_paper_track_allowed(
        method_name=method_name,
        requested_source_count=int(k),
        protocol=protocol,
        strict_paper_mode=strict_paper_mode,
    )

    common_kwargs: Dict[str, Any] = {
        "source_df": source_df,
        "target_df": target_df,
        "feature_cols": feature_cols,
        "k": int(k),
        "horizon": int(HORIZON),
        "window_size": int(exp_cfg.get("window_size", 10)),
        "weight_mode": str(WEIGHT_MODE),
        "learning_rate": float(exp_cfg.get("learning_rate", 0.001)),
        "source_epochs": int(SOURCE_EPOCHS),
        "target_epochs": int(TARGET_EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "metric_protocol": protocol.get("metric_protocol", {}),
    }

    if method_name == "MSML-TL":
        raw = run_msml_experiment(**common_kwargs)
    elif method_name == "MSML-TL-RFE":
        raw = run_msml_rfe_experiment(
            **common_kwargs,
            estimator_name=str(exp_cfg.get("estimator_name", "random_forest")),
            keep_ratio=float(KEEP_RATIO),
        )
    else:
        raise ValueError(f"Unsupported method in smoke test: {method_name}")

    alignment = build_alignment_fields(
        method_name=str(raw["method"]),
        requested_source_count=int(k),
        method_meta=raw.get("meta", {}),
        base_data=base,
        protocol=protocol,
    )

    return {
        "dataset_name": dataset_name,
        "method": str(raw["method"]),
        "k": int(k),
        "requested_k": int(k),
        "actual_k": int(actual_k),
        "source_count": int(k),
        "experiment_scope": alignment["experiment_scope"],
        "information_sharing_mode": information_sharing_mode,
        "experiment_track": alignment["experiment_track"],
        "source_protocol_aligned": bool(alignment.get("source_protocol_aligned", False)),
        "strict_paper_mode": bool(strict_paper_mode),
        "alignment_status": alignment["alignment_status"],
        "metric_alignment_status": alignment["metric_alignment_status"],
        "split_alignment_status": alignment["split_alignment_status"],
        "source_pretrained_alignment_status": alignment["source_pretrained_alignment_status"],
        "paper_metric_space": alignment["paper_metric_space"],
        "metric_space_current": str(raw.get("metric_space_current", alignment["current_metric_space"])),
        "metric_space_paper": str(raw.get("metric_space_paper", alignment["paper_metric_space"])),
        "paper_metric_aligned": bool(raw.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(raw.get("inverse_transform_applied", False)),
        "metric_notes": str(raw.get("metric_notes", "")),
        "paper_split_reference": alignment["paper_split_reference"],
        "target_window_days": alignment["target_window_days"],
        "target_window_expected_days": alignment["target_window_expected_days"],
        "target_window_range_days": alignment["target_window_range_days"],
        "target_window_unique_days": alignment["target_window_unique_days"],
        "target_strict_paper_mode": alignment["target_strict_paper_mode"],
        "weight_mode": WEIGHT_MODE,
        "keep_ratio": float(KEEP_RATIO),
        "rmse": float(raw["rmse"]),
        "accuracy": float(raw["accuracy"]),
        "prediction_shape": str(raw["prediction_shape"]),
        "pretrained_model_count": alignment["actual_pretrained_model_count"],
        "actual_pretrained_model_count": alignment["actual_pretrained_model_count"],
        "alignment_notes": alignment["alignment_notes"],
        "status": "success",
        "error_message": "",
    }


def _build_failed_row(
    dataset_name: str,
    method_name: str,
    k: int,
    information_sharing_mode: str,
    actual_k: int,
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
    exc: Exception,
) -> Dict[str, Any]:
    alignment = build_alignment_fields(
        method_name=method_name,
        requested_source_count=int(k),
        method_meta={},
        base_data=None,
        protocol=protocol,
    )
    return {
        "dataset_name": dataset_name,
        "method": method_name,
        "k": int(k),
        "requested_k": int(k),
        "actual_k": int(actual_k),
        "source_count": int(k),
        "experiment_scope": alignment["experiment_scope"],
        "information_sharing_mode": information_sharing_mode,
        "experiment_track": alignment["experiment_track"],
        "source_protocol_aligned": bool(alignment.get("source_protocol_aligned", False)),
        "strict_paper_mode": bool(strict_paper_mode),
        "alignment_status": alignment["alignment_status"],
        "metric_alignment_status": alignment["metric_alignment_status"],
        "split_alignment_status": alignment["split_alignment_status"],
        "source_pretrained_alignment_status": alignment["source_pretrained_alignment_status"],
        "paper_metric_space": alignment["paper_metric_space"],
        "metric_space_current": alignment["current_metric_space"],
        "metric_space_paper": alignment["paper_metric_space"],
        "paper_metric_aligned": False,
        "inverse_transform_applied": False,
        "metric_notes": "",
        "paper_split_reference": alignment["paper_split_reference"],
        "target_window_days": alignment["target_window_days"],
        "target_window_expected_days": alignment["target_window_expected_days"],
        "target_window_range_days": alignment["target_window_range_days"],
        "target_window_unique_days": alignment["target_window_unique_days"],
        "target_strict_paper_mode": alignment["target_strict_paper_mode"],
        "weight_mode": WEIGHT_MODE,
        "keep_ratio": float(KEEP_RATIO),
        "rmse": np.nan,
        "accuracy": np.nan,
        "prediction_shape": "N/A",
        "pretrained_model_count": alignment["actual_pretrained_model_count"],
        "actual_pretrained_model_count": alignment["actual_pretrained_model_count"],
        "alignment_notes": alignment["alignment_notes"],
        "status": "failed",
        "error_message": f"{type(exc).__name__}: {exc}",
    }


def _plot_rmse(df_success: pd.DataFrame, output_path: Path) -> None:
    plot_df = df_success.copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title("Dataset1 Alignment Smoke Test RMSE (no successful runs)")
        ax.set_xlabel("method | k | information_sharing_mode")
        ax.set_ylabel("rmse")
        ax.text(0.5, 0.5, "No successful experiments", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    plot_df["label"] = (
        plot_df["method"].astype(str)
        + "|k="
        + plot_df["k"].astype(str)
        + "|"
        + plot_df["information_sharing_mode"].astype(str)
    )
    plot_df = plot_df.sort_values(by="rmse", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = np.where(
        plot_df["information_sharing_mode"].eq("with_information_sharing"),
        "#1f77b4",
        "#ff7f0e",
    )
    ax.bar(plot_df["label"], plot_df["rmse"], color=colors)
    ax.set_title("Dataset1 Alignment Smoke Test RMSE")
    ax.set_xlabel("method | k | information_sharing_mode")
    ax.set_ylabel("rmse")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="with_information_sharing"),
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", label="without_information_sharing"),
    ]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cfg = _load_config()
    protocol = load_paper_protocol(cfg)
    strict_paper_mode = resolve_strict_paper_mode(cfg, explicit=bool(args.strict_paper_mode))
    strict_paper_split = bool(
        args.strict_paper_split
        or strict_paper_mode
        or cfg.get("paper_reproduction", {}).get("strict_paper_split", False)
    )
    protocol["strict_paper_mode"] = strict_paper_mode
    protocol["paper_strict_mode"] = strict_paper_mode
    protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = bool(strict_paper_mode)
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = strict_paper_mode
    cfg["paper_reproduction"]["paper_strict_mode"] = strict_paper_mode
    cfg["paper_reproduction"]["strict_paper_split"] = strict_paper_split
    cfg["paper_reproduction"]["paper_strict_split"] = strict_paper_split
    cfg["paper_reproduction"].setdefault("metric_protocol", {})["strict_paper_metrics"] = bool(strict_paper_mode)
    validation = validate_paper_protocol_config(protocol=protocol, strict_paper_mode=strict_paper_mode)
    print(
        "[paper_protocol_validation] "
        f"status={validation['status']} strict_paper_mode={strict_paper_mode} "
        f"strict_paper_split={strict_paper_split} "
        f"failures={len(validation['failures'])} warnings={len(validation['warnings'])}"
    )
    for warning in validation["warnings"]:
        print(f"[paper_protocol_todo] {warning}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    source_counts = get_paper_source_counts(protocol)
    if not strict_paper_mode:
        extended_counts = get_extended_source_counts(protocol)
        if extended_counts:
            source_counts.append(int(extended_counts[0]))

    for dataset_name in DATASET_NAMES:
        for method_name in METHODS:
            for k in source_counts:
                for information_sharing_mode in INFORMATION_SHARING_MODES:
                    print(
                        "Running: "
                        f"{dataset_name} | Method={method_name} | k={k} | "
                        f"information_sharing_mode={information_sharing_mode} | "
                        f"strict_paper_mode={strict_paper_mode}"
                    )
                    actual_k = 0
                    try:
                        base = prepare_base_data_for_experiments(
                            dataset_name=dataset_name,
                            data_path=cfg["dataset_paths"][dataset_name],
                            config=cfg,
                        )
                        filtered_source_df = _apply_information_sharing_filter(
                            source_df=base["source_df"],
                            target_df=base["target_df"],
                            mode=information_sharing_mode,
                        )
                        actual_k = _count_available_source_candidates(filtered_source_df)

                        if actual_k < int(k):
                            raise ValueError(
                                "insufficient source candidates under current information_sharing_mode"
                            )

                        row = _run_one_experiment(
                            dataset_name=dataset_name,
                            method_name=method_name,
                            k=k,
                            information_sharing_mode=information_sharing_mode,
                            cfg=cfg,
                            protocol=protocol,
                            strict_paper_mode=strict_paper_mode,
                        )
                        records.append(row)
                        print(f"Finished: RMSE={row['rmse']:.6f} | Accuracy={row['accuracy']:.6f}")
                    except Exception as exc:
                        failed = _build_failed_row(
                            dataset_name=dataset_name,
                            method_name=method_name,
                            k=k,
                            information_sharing_mode=information_sharing_mode,
                            actual_k=actual_k,
                            protocol=protocol,
                            strict_paper_mode=strict_paper_mode,
                            exc=exc,
                        )
                        records.append(failed)
                        print("Finished: RMSE=nan | Accuracy=nan")
                        print(f"Error: {failed['error_message']}")

    results_df = pd.DataFrame(
        records,
        columns=[
            "dataset_name",
            "method",
            "k",
            "requested_k",
            "actual_k",
            "source_count",
            "information_sharing_mode",
            "experiment_track",
            "strict_paper_mode",
            "alignment_status",
            "metric_alignment_status",
            "split_alignment_status",
            "source_pretrained_alignment_status",
            "paper_metric_space",
            "metric_space_current",
            "metric_space_paper",
            "paper_metric_aligned",
            "inverse_transform_applied",
            "metric_notes",
            "paper_split_reference",
            "target_window_days",
            "target_window_expected_days",
            "target_window_range_days",
            "target_window_unique_days",
            "target_strict_paper_mode",
            "weight_mode",
            "keep_ratio",
            "rmse",
            "accuracy",
            "prediction_shape",
            "actual_pretrained_model_count",
            "alignment_notes",
            "status",
            "error_message",
        ],
    )
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

    success_df = results_df[
        (results_df["status"] == "success") & results_df["rmse"].notna()
    ].copy()
    formatted_df = success_df.sort_values(by="rmse", ascending=True).reset_index(drop=True)
    if not formatted_df.empty:
        formatted_df.insert(0, "rank", range(1, len(formatted_df) + 1))
    formatted_df.to_csv(FORMATTED_CSV, index=False, encoding="utf-8")

    _plot_rmse(success_df, RMSE_PNG)

    total_experiments = len(DATASET_NAMES) * len(METHODS) * len(source_counts) * len(INFORMATION_SHARING_MODES)
    success_count = int((results_df["status"] == "success").sum())
    failure_count = int((results_df["status"] == "failed").sum())

    if not formatted_df.empty:
        best = formatted_df.iloc[0]
        best_line_1 = f"method={best['method']}"
        best_line_2 = f"k={int(best['k'])}"
        best_line_3 = f"information_sharing_mode={best['information_sharing_mode']}"
        best_line_4 = f"rmse={float(best['rmse']):.6f}"
    else:
        best_line_1 = "method=N/A"
        best_line_2 = "k=N/A"
        best_line_3 = "information_sharing_mode=N/A"
        best_line_4 = "rmse=N/A"

    print("Paper Alignment Smoke Test Completed Successfully")
    print()
    print("Total Experiments:")
    print(total_experiments)
    print()
    print("Success Count:")
    print(success_count)
    print()
    print("Failure Count:")
    print(failure_count)
    print()
    print("Best Setting:")
    print(best_line_1)
    print(best_line_2)
    print(best_line_3)
    print(best_line_4)
    print()
    print("Output Files:")
    print("- outputs/paper_alignment_smoke_test/dataset1_alignment_smoke_results.csv")
    print("- outputs/paper_alignment_smoke_test/dataset1_alignment_smoke_results_formatted.csv")
    print("- outputs/paper_alignment_smoke_test/dataset1_alignment_smoke_rmse.png")


if __name__ == "__main__":
    main()
