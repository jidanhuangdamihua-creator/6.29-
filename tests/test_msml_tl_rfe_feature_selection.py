from __future__ import annotations

import pandas as pd

from src.transfer_methods.msml_tl_rfe import (
    build_joint_rfe_training_dataframe,
    run_rfe_feature_selection,
)


def test_rfe_feature_selection_excludes_duplicate_target_column() -> None:
    df = pd.DataFrame(
        {
            "sales": [10.0, 12.0, 13.0, 15.0, 17.0, 18.0],
            "year": [2020, 2020, 2020, 2020, 2020, 2020],
            "month": [1, 2, 3, 4, 5, 6],
            "week": [1, 5, 9, 13, 17, 21],
            "day": [1, 2, 3, 4, 5, 6],
        }
    )

    joint = build_joint_rfe_training_dataframe(
        target_train_df=df,
        selected_source_dfs=[df],
        feature_cols=["sales", "year", "month", "week", "day"],
        target_col="sales",
    )
    result = run_rfe_feature_selection(
        train_df=joint,
        feature_cols=["sales", "year", "month", "week", "day"],
        target_col="sales",
        estimator_name="random_forest",
        keep_ratio=0.5,
        random_state=42,
    )

    assert joint.columns.tolist().count("sales") == 1
    assert result["num_original_features"] == 4
    assert "sales" in result["selected_feature_cols"]
    assert set(result["selected_feature_cols"]).issubset({"sales", "year", "month", "week", "day"})
