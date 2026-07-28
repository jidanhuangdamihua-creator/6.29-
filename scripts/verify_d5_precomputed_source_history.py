"""Compare D5 precomputed and forced-recompute source frames before policy."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import SOLIDIFIED_KNN_ROOT, SOURCE_HISTORY_DAYS
from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.utils.d5_calendar_reconstruction import load_d5_authorities
from src.utils.parquet_data_loader import (
    expected_target_dates_from_windows,
    load_parquet_source_target_with_diagnostics,
    read_dataset_windows,
)


@contextmanager
def _force_recompute(enabled: bool):
    previous = os.environ.get("D5_FORCE_RECOMPUTE")
    if enabled:
        os.environ["D5_FORCE_RECOMPUTE"] = "1"
    else:
        os.environ.pop("D5_FORCE_RECOMPUTE", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("D5_FORCE_RECOMPUTE", None)
        else:
            os.environ["D5_FORCE_RECOMPUTE"] = previous


def _load_source_frame(
    *,
    repository_root: Path,
    mode: str,
    horizon: int,
    seed: int,
    force_recompute: bool,
) -> pd.DataFrame:
    if mode not in {"with", "without"}:
        raise ValueError("mode must be 'with' or 'without'")
    if horizon not in {1, 2, 3, 4, 5}:
        raise ValueError("horizon must be in 1..5")
    if seed not in {42, 43, 44, 45, 46}:
        raise ValueError("seed must be one of 42..46")

    formal_paths = resolve_formal_dataset_paths(5, repository_root=repository_root)
    knn_json_dir = SOLIDIFIED_KNN_ROOT / "Dataset5"
    windows = read_dataset_windows(5, knn_json_dir, info_sharing=mode)
    expected_dates = expected_target_dates_from_windows(windows)
    authorities = load_d5_authorities(
        repository_root / "数据集/原始数据/Dataset 5Favorita",
        use_holidays=True,
    )
    with _force_recompute(force_recompute):
        loaded = load_parquet_source_target_with_diagnostics(
            dataset_id=5,
            source_path=formal_paths.source_path,
            target_path=formal_paths.target_path,
            windows=windows,
            source_history_days=SOURCE_HISTORY_DAYS,
            expected_dates=expected_dates,
            d5_authorities=authorities,
        )
    return loaded.source_df


def compare_paths(
    *,
    repository_root: Path,
    mode: str,
    horizon: int,
    seed: int,
) -> None:
    precomputed = _load_source_frame(
        repository_root=repository_root,
        mode=mode,
        horizon=horizon,
        seed=seed,
        force_recompute=False,
    )
    recomputed = _load_source_frame(
        repository_root=repository_root,
        mode=mode,
        horizon=horizon,
        seed=seed,
        force_recompute=True,
    )
    columns = list(precomputed.columns)
    sort_columns = [column for column in ("store_nbr", "item_nbr", "date") if column in columns]
    if sort_columns:
        precomputed = precomputed.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
        recomputed = recomputed.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    assert_frame_equal(precomputed, recomputed, check_dtype=True, check_exact=True)
    print("D5 precomputed source verification: PASS")
    print(f"  mode={mode} horizon={horizon} seed={seed}")
    print(f"  precomputed_rows={len(precomputed)} recomputed_rows={len(recomputed)}")
    print(f"  columns={columns}")
    print("  source_frame_equal=True")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare D5 precomputed and forced-recompute source frames."
    )
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mode", choices=["with", "without"], default="without")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44, 45, 46], default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    compare_paths(
        repository_root=args.repository_root.resolve(),
        mode=args.mode,
        horizon=args.horizon,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
