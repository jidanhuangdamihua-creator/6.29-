"""Small, explicit helpers for keeping large protocol metadata out of pandas ops."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import pandas as pd


HEAVY_SOURCE_HISTORY_ATTRS = frozenset(
    {
        "source_history_eligibility",
        "source_history_eligible_keys",
        "source_history_incomplete_keys",
        "source_history_duplicate_keys",
    }
)


def lightweight_frame_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return protocol attrs safe to carry through frequent DataFrame operations.

    Exact eligibility remains available in the explicit source-history result and
    in the prepared pool.  It is intentionally not carried by every derived
    pandas object.
    """

    return {
        key: value
        for key, value in attrs.items()
        if key not in HEAVY_SOURCE_HISTORY_ATTRS
    }


@contextmanager
def temporarily_detached_attrs(frame: pd.DataFrame) -> Iterator[None]:
    """Run a pandas operation without allowing it to deepcopy frame attrs."""

    original = frame.attrs
    frame.attrs = {}
    try:
        yield
    finally:
        frame.attrs = original


def copy_frame_with_lightweight_attrs(
    frame: pd.DataFrame,
    *,
    deep: bool = True,
) -> pd.DataFrame:
    """Copy frame data while explicitly allowlisting lightweight attrs."""

    attrs = lightweight_frame_attrs(frame.attrs)
    with temporarily_detached_attrs(frame):
        copied = frame.copy(deep=deep)
    copied.attrs = attrs
    return copied


def select_rows_with_lightweight_attrs(
    frame: pd.DataFrame,
    row_selector: Any,
    *,
    deep: bool = True,
) -> pd.DataFrame:
    """Select rows without pandas inheriting a large source attrs mapping."""

    attrs = lightweight_frame_attrs(frame.attrs)
    with temporarily_detached_attrs(frame):
        selected = frame.loc[row_selector].copy(deep=deep)
    selected.attrs = attrs
    return selected


__all__ = [
    "HEAVY_SOURCE_HISTORY_ATTRS",
    "copy_frame_with_lightweight_attrs",
    "lightweight_frame_attrs",
    "select_rows_with_lightweight_attrs",
    "temporarily_detached_attrs",
]
