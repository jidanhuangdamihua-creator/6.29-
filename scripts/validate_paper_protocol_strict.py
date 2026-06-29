"""Unified validator for strict paper protocol alignment.

Checks three dimensions:
1. Metric protocol declaration and current metric-space implementation.
2. Split window reconstruction consistency per dataset.
3. Source/pretrained-model cap and source-count policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.experiment_runner import prepare_base_data_for_experiments
from paper_reproduction_protocol import (
    MULTI_SOURCE_TL_METHODS,
    assess_metric_alignment,
    assess_source_pretrained_alignment,
    assess_split_alignment,
    get_extended_source_counts,
    get_paper_source_counts,
    get_results_output_paths,
    load_paper_protocol,
    resolve_experiment_track,
    resolve_strict_paper_mode,
    validate_paper_protocol_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate strict paper protocol alignment.")
    parser.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help="Enable strict failure mode. If any protocol check fails, script exits non-zero.",
    )
    return parser.parse_args()


def _load_config() -> Dict[str, Any]:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_metric(protocol: Dict[str, Any]) -> Dict[str, Any]:
    metric = assess_metric_alignment(protocol)
    return {
        "check": "metric_protocol",
        "status": metric["metric_alignment_status"],
        "details": json.dumps(
            {
                "paper_metric_space": metric["paper_metric_space"],
                "current_metric_space": metric["current_metric_space"],
                "notes": metric["metric_alignment_notes"],
            },
            ensure_ascii=False,
        ),
    }


def _check_split(cfg: Dict[str, Any], protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset_name, dataset_path in cfg["dataset_paths"].items():
        base = prepare_base_data_for_experiments(
            dataset_name=dataset_name,
            data_path=dataset_path,
            config=cfg,
            verbose_mode="summary",
        )
        split = assess_split_alignment(protocol=protocol, base_data=base)
        rows.append(
            {
                "check": f"split_protocol::{dataset_name}",
                "status": split["split_alignment_status"],
                "details": json.dumps(
                    {
                        "paper_split_reference": split["paper_split_reference"],
                        "target_window_days": split["target_window_days"],
                        "target_window_expected_days": split.get("target_window_expected_days", -1),
                        "target_window_range_days": split.get("target_window_range_days", -1),
                        "target_window_unique_days": split.get("target_window_unique_days", -1),
                        "split_runtime_matches_config": split["split_runtime_matches_config"],
                        "notes": split["split_alignment_notes"],
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def _check_source_cap(protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counts = get_paper_source_counts(protocol) + get_extended_source_counts(protocol)
    methods = ["No-TL", "SS-TL", *sorted(MULTI_SOURCE_TL_METHODS)]

    for method_name in methods:
        if method_name == "No-TL":
            requested_counts = [0]
        elif method_name == "SS-TL":
            requested_counts = [1]
        else:
            requested_counts = counts

        for requested_count in requested_counts:
            track = resolve_experiment_track(method_name, int(requested_count), protocol)
            actual_count = 0 if method_name == "No-TL" else (1 if method_name == "SS-TL" else int(requested_count))
            source = assess_source_pretrained_alignment(
                method_name=method_name,
                requested_source_count=int(requested_count),
                actual_pretrained_model_count=actual_count,
                protocol=protocol,
                experiment_track=track,
            )
            rows.append(
                {
                    "check": f"source_pretrained::{method_name}::k={requested_count}",
                    "status": source["source_pretrained_alignment_status"],
                    "details": json.dumps(
                        {
                            "experiment_track": track,
                            "paper_cap": source["paper_pretrained_model_cap"],
                            "requested_source_count": source["requested_source_count"],
                            "actual_pretrained_model_count": source["actual_pretrained_model_count"],
                            "notes": source["source_pretrained_alignment_notes"],
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    return rows


def main() -> None:
    args = _parse_args()
    cfg = _load_config()

    strict = resolve_strict_paper_mode(cfg, explicit=bool(args.strict_paper_mode))
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = strict
    cfg["paper_reproduction"]["paper_strict_mode"] = strict

    protocol = load_paper_protocol(cfg)
    protocol["strict_paper_mode"] = strict
    protocol["paper_strict_mode"] = strict

    config_check = validate_paper_protocol_config(protocol=protocol, strict_paper_mode=strict)

    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "check": "protocol_config",
            "status": config_check["status"],
            "details": json.dumps(config_check, ensure_ascii=False),
        }
    )
    rows.append(_check_metric(protocol))
    rows.extend(_check_split(cfg, protocol))
    rows.extend(_check_source_cap(protocol))

    results_df = pd.DataFrame(rows, columns=["check", "status", "details"])

    output_paths = get_results_output_paths(ROOT, protocol)
    output_paths["alignment_dir"].mkdir(parents=True, exist_ok=True)

    csv_path = output_paths["alignment_dir"] / "paper_protocol_strict_validation.csv"
    json_path = output_paths["alignment_dir"] / "paper_protocol_strict_validation.json"

    results_df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "strict_paper_mode": strict,
        "num_checks": int(len(results_df)),
        "status_counts": {
            k: int(v)
            for k, v in results_df["status"].value_counts(dropna=False).to_dict().items()
        },
        "checks": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Paper protocol strict validation completed")
    print(csv_path)
    print(json_path)
    print(results_df.to_string(index=False))

    if strict:
        fail_like = {"FAIL", "PARTIAL", "TODO"}
        if any(str(status).upper() in fail_like for status in results_df["status"].tolist()):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
