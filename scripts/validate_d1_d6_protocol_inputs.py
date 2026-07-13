#!/usr/bin/env python3
"""Read-only D1-D6 protocol preflight; it never trains or rewrites datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.candidate_pool import (
    InsufficientCandidatePoolError,
    PreparedDailySequencePool,
    prepare_daily_sequence_pool,
)
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


PARQUET_DIR = ROOT / "数据集" / "固化数据"
D1D2_PROTOCOL_PARQUET_DIR = (
    ROOT / "数据集" / "派生数据" / "d1d2_protocol_v1"
)


def resolve_parquet_dir(
    dataset_id: int,
    explicit_dir: Optional[Path] = None,
) -> Path:
    """Resolve the authoritative parquet directory for protocol preflight."""
    if explicit_dir is not None:
        return Path(explicit_dir)
    if dataset_id in (1, 2):
        return D1D2_PROTOCOL_PARQUET_DIR
    return PARQUET_DIR


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
    prepared_pool: PreparedDailySequencePool | None = None,
    exclusion_sample_limit: int = 20,
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
            prepared_pool=prepared_pool,
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
        exclusion_summary = summarize_candidate_exclusions(
            meta["source_skip_diagnostics"],
            sample_limit=exclusion_sample_limit,
        )
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
            "candidate_pool_digest_input_summary": summarize_candidate_digest_input(
                meta["candidate_pool_digest_input"],
                sample_limit=exclusion_sample_limit,
            ),
            "selection_result_digest": meta["selection_result_digest"],
            "candidate_count_total": meta["candidate_source_count"],
            "candidate_count_valid": meta["valid_source_count"],
            **exclusion_summary,
            "requested_k": meta["requested_k"],
            "effective_k": meta["effective_k"],
            "ordered_top_k": ordered_top_k,
            "cnn_provenance_validated": meta["cnn_provenance_validated"],
            "knn_observed_start": target.attrs["knn_observed_start"],
            "knn_observed_end": target.attrs["knn_observed_end"],
            "error": "",
        }
    except Exception as exc:  # preflight must return an auditable failure report
        exclusions = (
            exc.exclusions
            if isinstance(exc, InsufficientCandidatePoolError)
            else ()
        )
        return {
            "status": "failed",
            "dataset_id": str(dataset_id),
            "scenario": str(scenario),
            **summarize_candidate_exclusions(
                exclusions,
                sample_limit=exclusion_sample_limit,
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize_candidate_exclusions(
    exclusions: Sequence[dict[str, Any] | Any],
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    if sample_limit < 0:
        raise ValueError("candidate exclusion sample limit must be non-negative")
    normalized = [dict(item) for item in exclusions]
    reason_counts = Counter(str(item.get("reason", "unknown")) for item in normalized)
    return {
        "candidate_exclusion_count": len(normalized),
        "candidate_exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_exclusion_samples": normalized[:sample_limit],
        "candidate_exclusions_truncated": len(normalized) > sample_limit,
    }


def summarize_candidate_digest_input(
    digest_input: dict[str, Any] | Any,
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Keep default console output bounded without changing production digest input."""
    payload = dict(digest_input)
    candidate_keys = list(payload.pop("candidate_keys", []))
    return {
        **payload,
        "candidate_keys_count": len(candidate_keys),
        "candidate_keys_sample": candidate_keys[:sample_limit],
        "candidate_keys_truncated": len(candidate_keys) > sample_limit,
    }


def _target_groups(target: pd.DataFrame, group_cols: Sequence[str]):
    if len(group_cols) == 1:
        for key, group in target.groupby(group_cols[0], sort=False):
            yield (key,), group
        return
    for key, group in target.groupby(list(group_cols), sort=False):
        yield tuple(key), group


def build_preflight_reports(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    dataset_id: object,
    scenario: str,
    group_cols: Sequence[str],
    observed_start: object,
    k: int,
    grouping_col: str | None = None,
    exclusion_sample_limit: int = 20,
    pool_factory: Callable[..., PreparedDailySequencePool] = prepare_daily_sequence_pool,
) -> list[dict[str, Any]]:
    """Prepare source observations once and validate every target against that pool."""
    metadata_cols = (str(grouping_col),) if grouping_col else ()
    prepared_pool = pool_factory(
        source,
        group_cols=tuple(group_cols),
        observed_start=observed_start,
        metadata_cols=metadata_cols,
    )
    stub_columns = list(dict.fromkeys([*group_cols, "date", "sales", *metadata_cols]))
    source_stub = pd.DataFrame(columns=stub_columns)
    reports = []
    for target_key, target_group in _target_groups(target, group_cols):
        report = validate_protocol_frames(
            source_stub,
            target_group,
            dataset_id=dataset_id,
            scenario=scenario,
            group_cols=group_cols,
            grouping_col=grouping_col,
            observed_start=observed_start,
            k=k,
            prepared_pool=prepared_pool,
            exclusion_sample_limit=exclusion_sample_limit,
        )
        report["target_key_from_file"] = target_key
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[f"d{i}" for i in range(1, 7)])
    parser.add_argument("--scenario", choices=("without", "with"), required=True)
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--exclusion-sample-limit", type=int, default=20)
    args = parser.parse_args()
    dataset_id = int(args.dataset[1:])
    parquet_dir = resolve_parquet_dir(dataset_id, args.parquet_dir)
    cfg = DATASET_CONFIG[dataset_id]
    source_columns = list(
        dict.fromkeys(
            [*cfg["group_cols"], "date", "sales", *([cfg["grouping_col"]] if cfg.get("grouping_col") else [])]
        )
    )
    target_columns = list(source_columns)
    source = pd.read_parquet(
        parquet_dir / f"dataset{dataset_id}-source.parquet",
        columns=source_columns,
    )
    target = pd.read_parquet(
        parquet_dir / f"dataset{dataset_id}-target.parquet",
        columns=target_columns,
    )
    reports = build_preflight_reports(
        source,
        target,
        dataset_id=dataset_id,
        scenario=args.scenario,
        group_cols=cfg["group_cols"],
        grouping_col=cfg.get("grouping_col"),
        observed_start=cfg["observed_start"],
        k=args.k,
        exclusion_sample_limit=args.exclusion_sample_limit,
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    if not reports or any(report["status"] != "passed" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
