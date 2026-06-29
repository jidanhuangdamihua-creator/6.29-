"""单实验统一入口脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

from dataset_registry import normalize_dataset_name
from src.experiment.experiment_runner import results_to_dataframe, run_all_experiments, save_results_to_csv
from src.utils.environment import setup_logging
from paper_reproduction_protocol import (
    get_results_output_paths,
    load_paper_protocol,
    resolve_strict_paper_mode,
    validate_paper_protocol_config,
)
from src.utils.runtime_control import set_verbose_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment with summary/full console mode.")
    parser.add_argument(
        "--verbose-mode",
        choices=["summary", "full"],
        default="summary",
        help="Console output mode: summary or full.",
    )
    parser.add_argument(
        "--include-sales-in-knn",
        dest="include_sales_in_knn",
        action="store_true",
        default=True,
        help="Include 'sales' in KNN similarity features for source selection (default: on).",
    )
    parser.add_argument(
        "--exclude-sales-in-knn",
        dest="include_sales_in_knn",
        action="store_false",
        help="Exclude 'sales' from KNN similarity features for source selection.",
    )
    parser.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help="Force paper-track-only configuration and emit auditable alignment fields.",
    )
    parser.add_argument(
        "--strict-paper-split",
        action="store_true",
        help="Force strict paper split protocol (observed/forecast windows) without fallback.",
    )
    args = parser.parse_args()

    set_verbose_mode(args.verbose_mode)
    setup_logging(log_level="WARNING" if args.verbose_mode == "summary" else "INFO", log_file=None)

    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
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
    output_paths = get_results_output_paths(ROOT, protocol)
    print(
        "[paper_protocol_validation] "
        f"status={validation['status']} strict_paper_mode={strict_paper_mode} "
        f"strict_paper_split={strict_paper_split} "
        f"failures={len(validation['failures'])} warnings={len(validation['warnings'])}"
    )

    ds_paths = cfg["dataset_paths"]
    feature_cols = cfg["features"]["default_feature_cols"]
    exp_cfg = cfg["single_experiment"]

    result = run_all_experiments(
        dataset_name=exp_cfg["dataset_name"],
        data_path=ds_paths[exp_cfg["dataset_name"]],
        config=cfg,
        feature_cols=feature_cols,
        k=exp_cfg["k"],
        number_of_sources=exp_cfg["k"],
        horizon=exp_cfg["horizon"],
        window_size=exp_cfg["window_size"],
        weight_mode=exp_cfg["weight_mode"],
        estimator_name=exp_cfg["estimator_name"],
        keep_ratio=exp_cfg["keep_ratio"],
        include_sales_in_knn=bool(args.include_sales_in_knn),
        learning_rate=exp_cfg["learning_rate"],
        source_epochs=exp_cfg["source_epochs"],
        target_epochs=exp_cfg["target_epochs"],
        batch_size=exp_cfg["batch_size"],
        enabled_methods=cfg["methods"]["all_methods"],
        verbose_mode=args.verbose_mode,
        strict_paper_mode=strict_paper_mode,
    )

    df = results_to_dataframe(result)
    dataset_slug = normalize_dataset_name(exp_cfg["dataset_name"]).lower()
    out_path = output_paths["results_dir"] / f"{dataset_slug}_results.csv"
    save_results_to_csv(df, str(out_path))

    print("Main Experiment Completed")
    print("Results Path:")
    print(str(out_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
