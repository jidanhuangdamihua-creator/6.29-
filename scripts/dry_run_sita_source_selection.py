#!/usr/bin/env python3
"""Select formal SITA sources and exit. This script does not train models."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocols.candidate_pool import (  # noqa: E402
    SOURCE_SELECTION_MODE_ALIGNED_30D,
    SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D,
)
from src.source_selection.source_selector import SourceSelector  # noqa: E402
from src.transfer_methods.ms_katl_v3_frame_assembly import assemble_v3_cell_frames  # noqa: E402


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _record(mode: str, cell: dict, payload: dict) -> dict:
    meta = payload.get("meta") or {}
    sources = payload.get("sources") or []
    return {
        "source_selection_mode": mode,
        "dataset": "D4",
        "scenario": cell.get("scenario"),
        "target_key": list(cell.get("protocol_target_key") or ()),
        "target_observed_start": meta.get("target_observed_start"),
        "target_observed_end": meta.get("target_observed_end"),
        "source_observation_cutoff": meta.get("source_observation_cutoff"),
        "source_history_start": meta.get("source_history_start"),
        "source_history_end": meta.get("source_history_end"),
        "scaler_min": meta.get("scaler_min"),
        "scaler_max": meta.get("scaler_max"),
        "scaler_fit_scope": meta.get("scaler_fit_scope"),
        "scaling": meta.get("scaling"),
        "scaler_range_is_zero": meta.get("scaler_range_is_zero", False),
        "scaled_outside_unit_interval": meta.get("scaled_outside_unit_interval", False),
        "candidate_search_records": meta.get("candidate_search_records", []),
        "excluded_candidates": meta.get("source_skip_diagnostics", []),
        "top3": sources,
        "training_called": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run formal SITA source selection and exit before training."
    )
    parser.add_argument("--dataset", default="D4")
    parser.add_argument("--target", default="166/258")
    parser.add_argument("--scenario", default="without")
    parser.add_argument(
        "--modes",
        default=f"{SOURCE_SELECTION_MODE_ALIGNED_30D},{SOURCE_SELECTION_MODE_HISTORICAL_WINDOW_30D}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "sita_v2" / "d4_166-258_without_source_selection.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dataset != "D4" or args.target != "166/258" or args.scenario != "without":
        print("This dry-run entry is limited to D4 target 166/258 without.", file=sys.stderr)
        return 2
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    cell = assemble_v3_cell_frames(
        args.dataset,
        args.target,
        horizon=1,
        seed=42,
        project_root=PROJECT_ROOT,
        mode=args.scenario,
    )
    records = []
    for mode in modes:
        payload = SourceSelector().select_top_k_sources(
            target_df=cell["target_model_df"],
            source_df=cell["source_model_df"],
            feature_cols=cell["model_feature_cols"],
            k=3,
            group_cols=tuple(cell["group_cols"]),
            weight_mode="inverse_distance",
            source_selection_mode=mode,
        )
        records.append(_record(mode, cell, payload))
    output = args.output
    if output.exists():
        output = output.with_name(f"{output.stem}_new{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"records": _jsonable(records), "training_called": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SOURCE_SELECTION_DRY_RUN={output}")
    print("TRAINING_CALLED=NO")
    for record in records:
        print(
            record["source_selection_mode"],
            [
                (
                    item.get("source_key"),
                    item.get("distance"),
                    item.get("weight"),
                    item.get("match_window_start"),
                    item.get("match_window_end"),
                )
                for item in record["top3"]
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
