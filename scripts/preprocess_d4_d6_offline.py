#!/usr/bin/env python3
"""Explicit calendarization primitive for D4-D6 preprocessing."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.protocols.experiment_protocol import PROTOCOL_VERSION, ProtocolViolation


def calendarize_declared_zero_demand(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    zero_demand_semantics: bool,
) -> pd.DataFrame:
    if not zero_demand_semantics:
        raise ProtocolViolation(
            "calendarization requires an explicit declaration that missing dates have zero-demand semantics"
        )
    required = [*group_cols, "date", "sales"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"calendarization input is missing columns: {missing}")
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    if prepared["date"].isna().any() or prepared.duplicated([*group_cols, "date"]).any():
        raise ProtocolViolation("calendarization input contains invalid or duplicate dates")

    completed = []
    for raw_key, group in prepared.groupby(list(group_cols), sort=False, dropna=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        calendar = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        indexed = group.set_index("date").reindex(calendar)
        indexed.index.name = "date"
        for column, value in zip(group_cols, key):
            indexed[column] = value
        indexed["sales"] = pd.to_numeric(indexed["sales"], errors="coerce").fillna(0.0)
        for column in prepared.columns:
            if column in {*group_cols, "date", "sales"}:
                continue
            indexed[column] = indexed[column].ffill().bfill()
        completed.append(indexed.reset_index())
    result = pd.concat(completed, ignore_index=True).loc[:, prepared.columns]
    result = result.sort_values([*group_cols, "date"]).reset_index(drop=True)
    result.attrs.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "zero_demand_calendarization_declared": True,
            "zero_demand_calendarization_rule": "missing_calendar_day_sales_equals_zero",
        }
    )
    return result
