def test_notl_paper_aligned_split_horizon_contract():
    from scripts.audits import notl_paper_aligned_split_horizon_reproduction as audit

    assert audit.RESULT_CSV.name == "notl_paper_aligned_split_horizon_reproduction.csv"
    assert audit.REPORT_MD.name == "notl_paper_aligned_split_horizon_reproduction.md"
    assert audit.RESULT_CSV.parent.name == "audits"
    assert audit.RESULT_CSV.parent.parent.name == "outputs"
    assert audit.HORIZONS == [1, 2, 3, 4, 5]
    assert audit.PAPER_TABLE3_SPLITS == {
        "Dataset1": {"train": 15, "val": 15, "test": 185},
        "Dataset2": {"train": 14, "val": 15, "test": 179},
        "Dataset3": {"train": 16, "val": 15, "test": 181},
    }

    required_columns = {
        "row_type",
        "dataset",
        "horizon",
        "normalized_rmse",
        "accuracy",
        "train_windows",
        "val_windows",
        "test_windows",
        "y_true_shape",
        "y_pred_shape",
        "paper_train_rows",
        "paper_val_rows",
        "paper_test_rows",
        "test_rows_match_paper_table3",
        "csv_evidence",
    }
    assert required_columns.issubset(set(audit.RESULT_COLUMNS))
