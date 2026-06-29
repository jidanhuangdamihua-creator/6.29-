"""完整实验矩阵入口脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

from src.experiment.experiment_matrix_runner import run_experiment_matrix
from src.utils.environment import setup_logging
from paper_reproduction_protocol import (
    load_paper_protocol,
    resolve_strict_paper_mode,
    validate_paper_protocol_config,
)
from src.utils.runtime_control import set_verbose_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment matrix with summary/full mode.")
    parser.add_argument(
        "--verbose-mode",
        choices=["summary", "full"],
        default="summary",
        help="Console output mode: summary or full.",
    )
    parser.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help="Filter matrix runs to paper-track-valid settings only.",
    )
    args = parser.parse_args()

    set_verbose_mode(args.verbose_mode)
    setup_logging(log_level="WARNING" if args.verbose_mode == "summary" else "INFO", log_file=None)

    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    protocol = load_paper_protocol(cfg)
    strict_paper_mode = resolve_strict_paper_mode(cfg, explicit=bool(args.strict_paper_mode))
    protocol["strict_paper_mode"] = strict_paper_mode
    protocol["paper_strict_mode"] = strict_paper_mode
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = strict_paper_mode
    cfg["paper_reproduction"]["paper_strict_mode"] = strict_paper_mode
    validation = validate_paper_protocol_config(protocol=protocol, strict_paper_mode=strict_paper_mode)
    print(
        "[paper_protocol_validation] "
        f"status={validation['status']} strict_paper_mode={strict_paper_mode} "
        f"failures={len(validation['failures'])} warnings={len(validation['warnings'])}"
    )

    matrix = cfg["matrix"]

    result = run_experiment_matrix(
        data_path_map=cfg["dataset_paths"],
        config=cfg,
        feature_cols=cfg["features"]["default_feature_cols"],
        dataset_names=matrix["dataset_names"],
        horizons=matrix["horizons"],
        source_counts=matrix["source_counts"],
        weight_modes=matrix["weight_modes"],
        keep_ratios=matrix["keep_ratios"],
        enabled_methods_options=matrix["enabled_methods_options"],
        verbose_mode=args.verbose_mode,
        strict_paper_mode=strict_paper_mode,
    )

    print("Full Matrix Completed")
    print("Master Results Path:")
    print(result["master_results_path"])
    print("Snapshot Path:")
    print(result["snapshot_path"])


if __name__ == "__main__":
    main()
