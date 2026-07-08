# source domain 历史窗口天数，论文固定值
SOURCE_HISTORY_DAYS = 300

# D1–D6 统一结果 schema，与 run_full_paper_experiments._result_columns() 保持一致
RESULT_SCHEMA_COLUMNS = [
    "dataset",
    "target_entity_id",
    "target_store_id",
    "target_item_id",
    "method",
    "information_sharing",
    "source_count",
    "experiment_scope",
    "experiment_track",
    "source_protocol_aligned",
    "strict_paper_mode",
    "alignment_status",
    "metric_alignment_status",
    "split_alignment_status",
    "source_pretrained_alignment_status",
    "paper_metric_space",
    "metric_space_current",
    "metric_space_paper",
    "metric_space_used",
    "paper_metric_aligned",
    "inverse_transform_applied",
    "inverse_transform_available",
    "metric_notes",
    "paper_split_reference",
    "target_start_date",
    "target_end_date",
    "target_window_days",
    "target_window_expected_days",
    "target_window_range_days",
    "target_window_unique_days",
    "target_strict_paper_mode",
    "target_split_mode",
    "source_split_mode",
    "paper_pretrained_model_cap",
    "pretrained_model_count",
    "requested_source_count",
    "actual_pretrained_model_count",
    "requested_k",
    "effective_k",
    "valid_source_count",
    "skipped_source_count",
    "failed_source_count",
    "failed_source_keys",
    "skipped_nonfinite_source_count",
    "failed_sources",
    "date_alignment_mode",
    "learning_rate",
    "source_epochs",
    "target_epochs",
    "epochs",
    "clipnorm",
    "dropout",
    "rmse",
    "accuracy",
    "training_time",
    "mae",
    "mape",
    "smape",
    "rmse_current",
    "accuracy_current",
    "mae_current",
    "mape_current",
    "smape_current",
    "rmse_paper",
    "accuracy_paper",
    "mae_paper",
    "mape_paper",
    "smape_paper",
    "normalized_rmse",
    "normalized_accuracy",
    "normalized_mae",
    "normalized_mape",
    "normalized_smape",
    "original_scale_rmse",
    "original_scale_accuracy",
    "original_scale_mae",
    "original_scale_mape",
    "original_scale_smape",
    "prediction_shape",
    "y_pred_nan_count",
    "y_pred_inf_count",
    "y_true_nan_count",
    "y_true_inf_count",
    "X_test_nan_count",
    "X_test_inf_count",
    "model_weight_nan_count",
    "model_weight_inf_count",
    "feature_source",
    "knn_feature_mode",
    "source_selection_feature_cols",
    "model_feature_cols",
    "feature_consistency_status",
    "json_only_features",
    "runtime_only_features",
    "source_numeric_na_repaired",
    "repaired_columns",
    "alignment_notes",
    "error",
]


from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOLIDIFIED_KNN_ROOT: Path = _PROJECT_ROOT / "configs" / "solidified" / "knn"

D3_WITHOUT_INFO_SHARING_DOMAIN_FILTER: dict[str, object] = {
    "column": "region",
    "value": 1,
}

D1_TARGET_TRAIN_WINDOW: dict[str, str] = {
    "start": "2017-06-05",
    "end": "2017-06-19",
}

D2_TARGET_TRAIN_WINDOW: dict[str, str] = {
    "start": "2018-06-05",
    "end": "2018-06-19",
}

SOLIDIFIED_TARGET_WINDOWS: dict[int, dict[str, str]] = {
    4: {
        "train_start": "2024-12-16",
        "test_end": "2025-07-13",
    },
    5: {
        "train_start": "2017-01-17",
        "test_end": "2017-08-15",
    },
    6: {
        "train_start": "2015-10-26",
        "test_end": "2016-05-22",
    },
}
