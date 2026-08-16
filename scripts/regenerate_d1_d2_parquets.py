#!/usr/bin/env python3
"""Regenerate complete D1/D2 protocol parquets; never infer missing domains."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.experiment_protocol import PROTOCOL_VERSION, ProtocolViolation
from src.protocols.d2_source_calendarization import canonical_d2_entity_id


DEFAULT_D1_INPUT = ROOT / "数据集" / "原始数据" / "Dataset 1" / "train.csv"
DEFAULT_D2_INPUT = ROOT / "数据集" / "原始数据" / "Dataset 2" / "hierarchical_sales_data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "数据集" / "派生数据" / "d1d2_protocol_v1"


def _rename_required(frame: pd.DataFrame, aliases: dict[str, Sequence[str]]) -> pd.DataFrame:
    renamed = frame.copy()
    for canonical, options in aliases.items():
        if canonical in renamed.columns:
            continue
        match = next((name for name in options if name in renamed.columns), None)
        if match is None:
            raise ProtocolViolation(
                f"raw regeneration input is missing {canonical!r}; aliases={tuple(options)!r}"
            )
        renamed = renamed.rename(columns={match: canonical})
    return renamed


def _finalize_daily_frame(
    frame: pd.DataFrame,
    *,
    domain_col: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result[domain_col] = pd.to_numeric(result[domain_col], errors="raise").astype(int)
    result["item_id"] = pd.to_numeric(result["item_id"], errors="raise").astype(int)
    result["sales"] = pd.to_numeric(result["sales"], errors="raise")
    if result["date"].isna().any():
        raise ProtocolViolation("raw regeneration input contains invalid dates")
    key_cols = [domain_col, "item_id", "date"]
    if result.duplicated(key_cols).any():
        raise ProtocolViolation(f"raw regeneration input contains duplicate {key_cols}")
    if domain_col == "brand_id":
        result["entity_id"] = result[domain_col].map(canonical_d2_entity_id)
    else:
        result["entity_id"] = result[domain_col].astype(str)
    result["year"] = result["date"].dt.year.astype(int)
    result["month"] = result["date"].dt.month.astype(int)
    result["week"] = result["date"].dt.isocalendar().week.astype(int)
    result["day"] = result["date"].dt.day.astype(int)
    result = result.sort_values([domain_col, "item_id", "date"]).reset_index(drop=True)
    result.attrs.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "zero_demand_calendarization_declared": False,
            "generation_contract": "complete_explicit_source_pool",
        }
    )
    return result


def _assert_exact_entities(
    frame: pd.DataFrame,
    *,
    domain_col: str,
    expected: set[tuple[int, int]],
    role: str,
) -> None:
    actual = set(
        frame[[domain_col, "item_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ProtocolViolation(
            f"{role} entity contract mismatch: missing={missing!r} extra={extra!r}"
        )


def build_d1_protocol_frames(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normalized = _rename_required(
        raw,
        {
            "date": ("Date",),
            "store_id": ("store", "Store"),
            "item_id": ("item", "Item"),
            "sales": ("Sales",),
        },
    )
    normalized = _finalize_daily_frame(normalized, domain_col="store_id")
    source_expected = {(store, item) for store in range(1, 4) for item in range(1, 10)}
    source = normalized[
        normalized[["store_id", "item_id"]]
        .apply(tuple, axis=1)
        .isin(source_expected)
    ].copy()
    target = normalized[
        (normalized["store_id"] == 1) & (normalized["item_id"] == 10)
    ].copy()
    _assert_exact_entities(source, domain_col="store_id", expected=source_expected, role="D1 source")
    _assert_exact_entities(target, domain_col="store_id", expected={(1, 10)}, role="D1 target")
    return source, target


def build_d2_protocol_frames(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "DATE" in raw.columns and any(str(column).startswith("QTY_B") for column in raw.columns):
        rows = []
        for brand in range(1, 4):
            for item in range(1, 11):
                demand_col = f"QTY_B{brand}_{item}"
                if demand_col not in raw.columns:
                    raise ProtocolViolation(f"raw D2 wide input is missing required column {demand_col!r}")
                promo_col = f"PROMO_B{brand}_{item}"
                rows.append(
                    pd.DataFrame(
                        {
                            "date": raw["DATE"],
                            "brand_id": brand,
                            "item_id": item,
                            "sales": raw[demand_col],
                            "promo": raw[promo_col] if promo_col in raw.columns else 0,
                        }
                    )
                )
        raw = pd.concat(rows, ignore_index=True)
    normalized = _rename_required(
        raw,
        {
            "date": ("Date",),
            "brand_id": ("brand", "Brand"),
            "item_id": ("item", "Item"),
            "sales": ("Sales",),
        },
    )
    if "promo" not in normalized.columns:
        normalized["promo"] = 0
    normalized = _finalize_daily_frame(normalized, domain_col="brand_id")
    source_expected = {(brand, item) for brand in range(1, 4) for item in range(1, 10)}
    source = normalized[
        normalized[["brand_id", "item_id"]]
        .apply(tuple, axis=1)
        .isin(source_expected)
    ].copy()
    target = normalized[
        (normalized["brand_id"] == 1) & (normalized["item_id"] == 10)
    ].copy()
    _assert_exact_entities(source, domain_col="brand_id", expected=source_expected, role="D2 source")
    _assert_exact_entities(target, domain_col="brand_id", expected={(1, 10)}, role="D2 target")
    return source, target


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_pair(dataset_id: int, source: pd.DataFrame, target: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"dataset{dataset_id}-source.parquet"
    target_path = output_dir / f"dataset{dataset_id}-target.parquet"
    existing = [path for path in (source_path, target_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing protocol parquet(s): {existing!r}")
    source.to_parquet(source_path, index=False)
    target.to_parquet(target_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("d1", "d2", "all"), default="all")
    parser.add_argument("--d1-input", type=Path, default=DEFAULT_D1_INPUT)
    parser.add_argument(
        "--d2-input",
        type=Path,
        default=DEFAULT_D2_INPUT,
        help="D2 raw long table or DATE + QTY_B<brand>_<item> wide table.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.dataset in {"d1", "all"}:
        _write_pair(1, *build_d1_protocol_frames(_read_table(args.d1_input)), args.output_dir)
    if args.dataset in {"d2", "all"}:
        if args.d2_input is None:
            parser.error("--d2-input is required for D2; incomplete solidified parquet is not a raw source")
        _write_pair(2, *build_d2_protocol_frames(_read_table(args.d2_input)), args.output_dir)


if __name__ == "__main__":
    main()
