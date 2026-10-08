#!/usr/bin/env python3
"""Run paper-scope DLinear_RevIN and TCN_RevIN robustness cells.

Reuses the formal baseline runner, data loader, rolling-origin support, and
original-sales metrics. Does not overwrite frozen SITA/DLinear/TCN results.

Usage:
    python scripts/run_revin_strengthened_baselines.py --preflight
    python scripts/run_revin_strengthened_baselines.py --smoke
    python scripts/run_revin_strengthened_baselines.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "scripts" / "baselines"
for import_path in (PROJECT_ROOT, BASELINE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import pandas as pd

from baseline_data_loader import load_baseline_data
from revin_strengthened_protocol import (
    DEFAULT_OUTPUT_DIR,
    EXPECTED_CELLS,
    PAPER_DATASETS,
    PAPER_HORIZONS,
    PAPER_METHODS,
    PAPER_MODE,
    PAPER_SEEDS,
    PAPER_TARGETS,
    ROBUSTNESS_METHOD_IDS,
    aggregate_five_seed,
    annotate_robustness_rows,
    assert_identity_isolation,
    assert_isolated_output_dir,
    cell_completeness,
    filter_paper_entities,
    git_head,
    sha256_file,
)
from scripts.baselines.run_baselines_multiseed import (
    METHODS,
    ROBUSTNESS_METHODS,
    build_run_id,
    evaluate_entity_protocol,
)
from tools.promote_bl6_bl7_formal_results import FORMAL_TARGETS


def _preflight() -> dict[str, object]:
    if METHODS[-2:] != ("BL6_DLinear", "BL7_TCN"):
        raise RuntimeError("formal BL6/BL7 registry drifted")
    if ROBUSTNESS_METHODS != ("BL6_DLinear_RevIN", "BL7_TCN_RevIN"):
        raise RuntimeError("robustness method identities drifted")
    if set(METHODS).intersection(ROBUSTNESS_METHODS):
        raise RuntimeError("robustness methods leaked into the formal METHODS tuple")
    if PAPER_TARGETS != FORMAL_TARGETS:
        raise RuntimeError("paper target matrix drifted from the BL6/BL7 promotion authority")
    assert_identity_isolation(PAPER_METHODS)
    assert_identity_isolation(ROBUSTNESS_METHOD_IDS)
    payload = {
        "expected_cells": EXPECTED_CELLS,
        "datasets": list(PAPER_DATASETS),
        "targets": dict(PAPER_TARGETS),
        "horizons": list(PAPER_HORIZONS),
        "seeds": list(PAPER_SEEDS),
        "mode": PAPER_MODE,
        "methods": list(PAPER_METHODS),
        "method_ids": list(ROBUSTNESS_METHOD_IDS),
        "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        "formal_methods_unchanged": list(METHODS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _run_dataset(
    dataset_id: str,
    *,
    methods: tuple[str, ...],
    horizons: tuple[int, ...],
    seeds: tuple[int, ...],
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slices = filter_paper_entities(load_baseline_data(dataset_id))
    summaries = []
    raws = []
    for data in slices:
        summary, raw = evaluate_entity_protocol(
            data,
            methods=methods,
            horizons=horizons,
            seeds=seeds,
            return_raw=True,
            run_id=run_id,
        )
        summaries.append(summary)
        raws.append(raw)
    return pd.concat(summaries, ignore_index=True), pd.concat(raws, ignore_index=True)


def run_robustness(
    *,
    output_dir: Path,
    datasets: tuple[str, ...] = PAPER_DATASETS,
    methods: tuple[str, ...] = ROBUSTNESS_METHOD_IDS,
    horizons: tuple[int, ...] = PAPER_HORIZONS,
    seeds: tuple[int, ...] = PAPER_SEEDS,
    smoke: bool = False,
) -> dict[str, object]:
    destination = assert_isolated_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    head = git_head()
    run_id = build_run_id(
        "revin_robustness_smoke" if smoke else "revin_robustness",
        "-".join(dataset.lower() for dataset in datasets),
        tuple(PAPER_TARGETS[dataset] for dataset in datasets),
        methods,
        horizons,
        seeds,
    )
    summaries = []
    raws = []
    for dataset in datasets:
        summary, raw = _run_dataset(
            dataset.lower(),
            methods=methods,
            horizons=horizons,
            seeds=seeds,
            run_id=run_id,
        )
        summaries.append(summary)
        raws.append(raw)
    summary = annotate_robustness_rows(
        pd.concat(summaries, ignore_index=True),
        git_head=head,
        timestamp=timestamp,
    )
    raw = pd.concat(raws, ignore_index=True)
    raw["method_id"] = raw["method"].map(
        {
            "BL6_DLinear_RevIN": "BL6_DLinear_RevIN",
            "BL7_TCN_RevIN": "BL7_TCN_RevIN",
        }
    )
    raw["method"] = raw["method"].map(
        {
            "BL6_DLinear_RevIN": "DLinear_RevIN",
            "BL7_TCN_RevIN": "TCN_RevIN",
        }
    )
    raw["parent_method"] = raw["method"].map({"DLinear_RevIN": "DLinear", "TCN_RevIN": "TCN"})
    raw["robustness_variant"] = "RevIN"
    raw_path = destination / "REVIN_STRENGTHENED_BASELINES_PREDICTIONS.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    summary["prediction_file_path"] = str(raw_path)
    dlinear = summary.loc[summary["method"] == "DLinear_RevIN"].copy()
    tcn = summary.loc[summary["method"] == "TCN_RevIN"].copy()
    dlinear_path = destination / "DLINEAR_REVIN_FORMAL_5SEED_RAW.csv"
    tcn_path = destination / "TCN_REVIN_FORMAL_5SEED_RAW.csv"
    combined_path = destination / "REVIN_STRENGTHENED_BASELINES_5SEED_RAW.csv"
    dlinear.to_csv(dlinear_path, index=False, encoding="utf-8")
    tcn.to_csv(tcn_path, index=False, encoding="utf-8")
    summary.to_csv(combined_path, index=False, encoding="utf-8")
    completeness = {
        "EXPECTED_CELLS": int(len(datasets) * len(horizons) * len(seeds) * len(methods)),
        "ACTUAL_CELLS": int(len(summary)),
        "DUPLICATE_CELLS": 0,
        "MISSING_CELLS": 0,
        "smoke": bool(smoke),
    }
    if not smoke:
        completeness = cell_completeness(summary)
        if not completeness["complete"]:
            raise RuntimeError(f"RevIN robustness cell completeness failed: {completeness}")
        aggregated = aggregate_five_seed(summary)
        aggregated_path = destination / "REVIN_STRENGTHENED_BASELINES_5SEED_AGGREGATED.csv"
        aggregated.to_csv(aggregated_path, index=False, encoding="utf-8")
    else:
        aggregated_path = None
        aggregated = pd.DataFrame()
    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "code_commit": head,
        "output_dir": str(destination),
        "dlinear_raw": str(dlinear_path),
        "tcn_raw": str(tcn_path),
        "combined_raw": str(combined_path),
        "predictions": str(raw_path),
        "aggregated": None if aggregated_path is None else str(aggregated_path),
        "dlinear_sha256": sha256_file(dlinear_path) if dlinear_path.exists() else None,
        "tcn_sha256": sha256_file(tcn_path) if tcn_path.exists() else None,
        "prediction_sha256": sha256_file(raw_path),
        "methods": list(PAPER_METHODS),
        "method_ids": list(methods),
        "datasets": list(datasets),
        "targets": {dataset: PAPER_TARGETS[dataset] for dataset in datasets},
        "horizons": list(horizons),
        "seeds": list(seeds),
        "mode": PAPER_MODE,
        "revin": {
            "enabled": True,
            "affine": True,
            "eps": 1e-5,
            "target_channel": 0,
        },
        "parent_methods": {"DLinear_RevIN": "DLinear", "TCN_RevIN": "TCN"},
        "formal_METHODS_untouched": list(METHODS),
        "metric_space": "original_sales_space",
        "completeness": {key: completeness[key] for key in (
            "EXPECTED_CELLS", "ACTUAL_CELLS", "DUPLICATE_CELLS", "MISSING_CELLS"
        ) if key in completeness},
        "n_aggregated_rows": int(len(aggregated)),
        "smoke": bool(smoke),
    }
    manifest_path = destination / "REVIN_STRENGTHENED_BASELINES_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["completeness"], indent=2, sort_keys=True))
    print(f"Wrote robustness artifacts under {destination}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=",".join(PAPER_DATASETS),
        help="Comma-separated paper datasets. Default: D1,D4,D5,D6",
    )
    parser.add_argument(
        "--methods",
        default=",".join(ROBUSTNESS_METHOD_IDS),
        help="Comma-separated method ids. Default: BL6_DLinear_RevIN,BL7_TCN_RevIN",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.preflight:
        _preflight()
        return
    datasets = tuple(item.strip().upper() for item in str(args.datasets).split(",") if item.strip())
    methods = tuple(item.strip() for item in str(args.methods).split(",") if item.strip())
    unknown_datasets = [item for item in datasets if item not in PAPER_TARGETS]
    if unknown_datasets:
        raise ValueError(f"datasets must be a subset of {PAPER_DATASETS}, got {unknown_datasets}")
    unknown_methods = [item for item in methods if item not in ROBUSTNESS_METHOD_IDS]
    if unknown_methods:
        raise ValueError(f"methods must be a subset of {ROBUSTNESS_METHOD_IDS}, got {unknown_methods}")
    if args.smoke:
        run_robustness(
            output_dir=Path(args.output_dir),
            datasets=(datasets[0],),
            methods=methods[:1],
            horizons=(1,),
            seeds=(42,),
            smoke=True,
        )
        return
    run_robustness(
        output_dir=Path(args.output_dir),
        datasets=datasets,
        methods=methods,
        horizons=PAPER_HORIZONS,
        seeds=PAPER_SEEDS,
        smoke=False,
    )


if __name__ == "__main__":
    main()
