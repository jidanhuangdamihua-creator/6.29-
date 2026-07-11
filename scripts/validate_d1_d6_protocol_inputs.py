#!/usr/bin/env python3
"""Read-only D1-D6 protocol preflight; it never trains or rewrites datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


PARQUET_DIR = ROOT / "数据集" / "固化数据"
DATASET_CONFIG = {
    1: {"group_cols": ("store_id", "item_id"), "observed_start": "2017-06-05"},
    2: {"group_cols": ("brand_id", "item_id"), "observed_start": "2018-06-05"},
    3: {"group_cols": ("store_id",), "observed_start": "2015-01-03"},
    4: {
        "group_cols": ("store_id", "product_id"),
        "grouping_col": "second_category_id",
        "observed_start": "2024-12-16",
    },
    5: {
        "group_cols": ("store_nbr", "item_nbr"),
        "grouping_col": "family",
        "observed_start": "2017-01-17",
    },
    6: {
        "group_cols": ("store_id", "item_id"),
        "grouping_col": "dept_id",
        "observed_start": "2015-10-26",
    },
}


def validate_protocol_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    dataset_id: object,
    scenario: str,
    group_cols: Sequence[str],
    observed_start: object,
    k: int,
    grouping_col: str | None = None,
) -> dict[str, Any]:
    try:
        source, target = configure_protocol_frames(
            source_df,
            target_df,
            dataset_id=dataset_id,
            scenario=scenario,
            group_cols=group_cols,
            grouping_col=grouping_col,
            observed_start=observed_start,
        )
        selection = SourceSelector().select_top_k_sources(
            target,
            source,
            feature_cols=("sales",),
            k=int(k),
            group_cols=tuple(group_cols),
            weight_mode="inverse_distance",
        )
        meta = selection["meta"]
        ordered_top_k = [
            {
                "source_rank": item["source_rank"],
                "source_key": item["source_key"],
                "distance": item["distance"],
                "weight": item["weight"],
                "tie_group": item["tie_group"],
            }
            for item in selection["sources"]
        ]
        return {
            "status": "passed",
            "dataset_id": target.attrs["protocol_dataset_id"],
            "scenario": target.attrs["protocol_scenario"],
            "target_key": target.attrs["protocol_target_key"],
            "candidate_count": len(target.attrs["protocol_candidate_keys"]),
            "candidate_pool_digest": meta["candidate_pool_digest"],
            "candidate_pool_digest_input": meta["candidate_pool_digest_input"],
            "selection_result_digest": meta["selection_result_digest"],
            "candidate_count_total": meta["candidate_source_count"],
            "candidate_count_valid": meta["valid_source_count"],
            "candidate_exclusions": meta["source_skip_diagnostics"],
            "requested_k": meta["requested_k"],
            "effective_k": meta["effective_k"],
            "ordered_top_k": ordered_top_k,
            "cnn_provenance_validated": meta["cnn_provenance_validated"],
            "knn_observed_start": target.attrs["knn_observed_start"],
            "knn_observed_end": target.attrs["knn_observed_end"],
            "error": "",
        }
    except Exception as exc:  # preflight must return an auditable failure report
        return {
            "status": "failed",
            "dataset_id": str(dataset_id),
            "scenario": str(scenario),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _target_groups(target: pd.DataFrame, group_cols: Sequence[str]):
    if len(group_cols) == 1:
        for key, group in target.groupby(group_cols[0], sort=False):
            yield (key,), group
        return
    for key, group in target.groupby(list(group_cols), sort=False):
        yield tuple(key), group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[f"d{i}" for i in range(1, 7)])
    parser.add_argument("--scenario", choices=("without", "with"), required=True)
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    dataset_id = int(args.dataset[1:])
    cfg = DATASET_CONFIG[dataset_id]
    source = pd.read_parquet(args.parquet_dir / f"dataset{dataset_id}-source.parquet")
    target = pd.read_parquet(args.parquet_dir / f"dataset{dataset_id}-target.parquet")
    reports = []
    for target_key, target_group in _target_groups(target, cfg["group_cols"]):
        report = validate_protocol_frames(
            source,
            target_group,
            dataset_id=dataset_id,
            scenario=args.scenario,
            group_cols=cfg["group_cols"],
            grouping_col=cfg.get("grouping_col"),
            observed_start=cfg["observed_start"],
            k=args.k,
        )
        report["target_key_from_file"] = target_key
        reports.append(report)
    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    if not reports or any(report["status"] != "passed" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
