from __future__ import annotations

import numpy as np
import pandas as pd

from src.protocols.candidate_pool import prepare_daily_sequence_pool


DATES = pd.date_range("2020-01-01", periods=180, freq="D")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": "A",
            "date": DATES,
            "sales": np.arange(180, dtype=np.float64),
            "required": 1.0,
            "unused_audit": np.nan,
        }
    )


def test_source_sales_repairs_are_shared_by_training_and_knn() -> None:
    frame = _frame().drop(index=[0]).reset_index(drop=True)
    frame.loc[frame["date"] == DATES[10], "sales"] = np.nan
    frame.loc[frame["date"] == DATES[20], "sales"] = -3.0

    pool = prepare_daily_sequence_pool(
        frame,
        group_cols=("entity",),
        observed_start=DATES[-30],
        observed_end=DATES[-1],
        pretrain_start=DATES[0],
        pretrain_end=DATES[-1],
        knn_feature_cols=("sales",),
        required_feature_cols=("sales",),
    )

    audit = pool.repair_audit_for(("A",))
    assert audit["repair_reason_counts"] == {
        "original_nan": 1,
        "original_negative": 1,
        "calendar_row_missing": 1,
    }
    full = pool.selected_source_frame((("A",),))
    assert full.loc[full["date"].isin((DATES[0], DATES[10], DATES[20])), "sales"].tolist() == [
        0.0,
        0.0,
        0.0,
    ]
    assert full["unused_audit"].isna().all()


def test_infinity_and_unresolved_used_fields_make_source_ineligible() -> None:
    infinite = _frame()
    infinite.loc[0, "sales"] = np.inf
    pool = prepare_daily_sequence_pool(
        infinite,
        group_cols=("entity",),
        observed_start=DATES[-30],
        pretrain_start=DATES[0],
        knn_feature_cols=("sales",),
        required_feature_cols=("sales", "required"),
    )
    assert pool.ineligible_reasons_for(("A",)) == ("source_sales_infinity",)

    unresolved = _frame()
    unresolved.loc[0, "required"] = np.nan
    pool = prepare_daily_sequence_pool(
        unresolved,
        group_cols=("entity",),
        observed_start=DATES[-30],
        pretrain_start=DATES[0],
        knn_feature_cols=("sales",),
        required_feature_cols=("sales", "required"),
    )
    assert pool.ineligible_reasons_for(("A",)) == ("unresolved_required_feature",)
