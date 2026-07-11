#!/usr/bin/env python3
"""Protocol-safe D1/D3 preprocessing primitives for offline regeneration."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from scripts.regenerate_d1_d2_parquets import build_d1_protocol_frames
from src.protocols.experiment_protocol import PROTOCOL_VERSION, ProtocolViolation


def build_d3_protocol_frames(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rename = {}
    for canonical, aliases in {
        "date": ("Date",),
        "store_id": ("Store", "store"),
        "sales": ("Sales",),
    }.items():
        if canonical not in raw.columns:
            match = next((name for name in aliases if name in raw.columns), None)
            if match is None:
                raise ProtocolViolation(f"D3 raw data is missing {canonical!r}")
            rename[match] = canonical
    normalized = raw.rename(columns=rename).copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized["store_id"] = pd.to_numeric(normalized["store_id"], errors="raise").astype(int)
    normalized["sales"] = pd.to_numeric(normalized["sales"], errors="raise")
    if normalized["date"].isna().any():
        raise ProtocolViolation("D3 raw data contains invalid dates")
    if normalized.duplicated(["store_id", "date"]).any():
        raise ProtocolViolation("D3 raw data contains duplicate store/date rows")
    normalized = normalized[normalized["store_id"].between(1, 30)].copy()
    actual_stores = set(normalized["store_id"].unique())
    expected_stores = set(range(1, 31))
    if actual_stores != expected_stores:
        raise ProtocolViolation(
            f"D3 raw data does not cover Store1-30: missing={sorted(expected_stores - actual_stores)}"
        )
    normalized["entity_id"] = normalized["store_id"].astype(str)
    normalized["year"] = normalized["date"].dt.year.astype(int)
    normalized["month"] = normalized["date"].dt.month.astype(int)
    normalized["week"] = normalized["date"].dt.isocalendar().week.astype(int)
    normalized["day"] = normalized["date"].dt.day.astype(int)
    normalized.attrs.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "zero_demand_calendarization_declared": False,
            "generation_contract": "D3_Store1_30",
        }
    )
    source = normalized[normalized["store_id"] != 10].sort_values(
        ["store_id", "date"]
    ).reset_index(drop=True)
    target = normalized[normalized["store_id"] == 10].sort_values("date").reset_index(drop=True)
    source.attrs = normalized.attrs.copy()
    target.attrs = normalized.attrs.copy()
    return source, target


__all__ = ["build_d1_protocol_frames", "build_d3_protocol_frames"]
