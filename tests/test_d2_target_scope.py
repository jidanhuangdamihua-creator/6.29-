from __future__ import annotations

import pandas as pd
import pytest

from src.protocols.experiment_protocol import ProtocolViolation


def _target_with_history(*, missing_formal: str | None = None) -> pd.DataFrame:
    dates = pd.date_range("2014-01-02", "2018-12-31", freq="D")
    if missing_formal is not None:
        dates = dates.difference(pd.DatetimeIndex([pd.Timestamp(missing_formal)]))
    frame = pd.DataFrame(
        {
            "date": dates,
            "brand_id": 1,
            "item_id": 10,
            "sales": 1.0,
            "promo": 0.0,
        }
    )
    frame.attrs["split_role"] = "target"
    return frame


def test_formal_target_scope_uses_authority_window_not_tail_row_count() -> None:
    from src.protocols.formal_target_scope import (
        resolve_formal_target_window,
        scope_target_to_formal_window,
    )

    window = resolve_formal_target_window("D2")
    scoped = scope_target_to_formal_window(_target_with_history(), dataset_id="D2")

    assert (window.start, window.end, window.expected_days) == (
        pd.Timestamp("2018-06-01"),
        pd.Timestamp("2018-12-27"),
        210,
    )
    assert len(scoped) == 210
    assert scoped["date"].min() == pd.Timestamp("2018-06-01")
    assert scoped["date"].max() == pd.Timestamp("2018-12-27")
    assert scoped["date"].nunique() == 210


def test_formal_target_scope_fails_closed_on_missing_formal_date() -> None:
    from src.protocols.formal_target_scope import scope_target_to_formal_window

    with pytest.raises(ProtocolViolation, match="missing formal target dates"):
        scope_target_to_formal_window(
            _target_with_history(missing_formal="2018-11-01"),
            dataset_id="D2",
        )
