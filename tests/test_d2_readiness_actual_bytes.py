from __future__ import annotations

import pandas as pd


def test_readiness_calendar_report_uses_actual_dates_not_hypothetical_repairs() -> None:
    from tools.operations.gate1x_real_input_readiness import (
        evaluate_formal_target_calendar,
    )

    dates = pd.date_range("2018-06-01", "2018-12-27", freq="D").difference(
        pd.DatetimeIndex([pd.Timestamp("2018-08-15")])
    )
    frame = pd.DataFrame({"date": dates, "brand_id": 1, "item_id": 10, "sales": 1.0})
    report = evaluate_formal_target_calendar(frame, dataset_id="D2")

    assert report["actual"] == 209
    assert report["expected"] == 210
    assert report["missing_exact_keys"] == [
        {"key": ["1", "10"], "date": "2018-08-15"}
    ]
    assert report["ready"] is False
