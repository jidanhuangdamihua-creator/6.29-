# source domain 历史窗口天数，论文固定值
import json

SOURCE_HISTORY_DAYS = 300
D4_D6_RUNTIME_KNN_PROTOCOL_VERSION = "runtime_knn_windowed_stats_v1"
MIXED_METRIC_SPACE = "mixed_metric_space"
MIXED_METRIC_PROTOCOL_NOTE = (
    "non-strict protocol uses normalized RMSE and original-scale sMAPE when inverse transform is available"
)
RESULT_CONTRACT_VERSION = "d1_d6_superset_v1"
SCHEMA_FAMILY_D1_D3 = "d1_d3_single_target_runtime_knn"
SCHEMA_FAMILY_D4_D6 = "d4_d6_entity_solidified_knn"
NOT_APPLICABLE = "not_applicable"
UNKNOWN = "unknown"
NO_PAPER_REFERENCE = "no_paper_reference"

# D1–D6 统一结果 schema，与 run_full_paper_experiments._result_columns() 保持一致
RESULT_SCHEMA_COLUMNS = [
    "result_contract_version",
    "schema_family",
    "result_status",
    "failure_type",
    "dataset",
    "dataset_id",
    "scenario",
    "target_entity_key",
    "target_entity_id",
    "target_store_id",
    "target_item_id",
    "source_identifier",
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
    "current_metric_space",
    "metric_space_current",
    "metric_space_paper",
    "metric_space_used",
    "metric_space",
    "rmse_metric_space",
    "smape_metric_space",
    "paper_metric_aligned",
    "inverse_transform_applied",
    "inverse_transform_available",
    "metric_protocol",
    "metric_protocol_note",
    "metric_protocol_error",
    "metric_notes",
    "knn_json_domain_filter",
    "source_domain_filter",
    "source_domain_filter_applied",
    "source_domain_filter_reason",
    "source_pool_size_before_filter",
    "source_pool_size_after_filter",
    "source_domain_filter_error",
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
    "train_days",
    "val_days",
    "test_days",
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
    "selected_sources",
    "selection_authority",
    "protocol_version",
    "target_observed_start",
    "target_observed_end",
    "source_history_start",
    "source_history_end",
    "target_test_excluded",
    "source_future_excluded",
    "source_alignment_mode",
    "feature_cols",
    "representation",
    "scaling",
    "scaler_fit_scope",
    "selected_sources_runtime",
    "candidate_pool_digest",
    "selection_result_digest",
    "source_skip_diagnostics",
    "selected_source_count",
    "source_failure_messages",
    "source_domain_filter_name",
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


def preferred_columns_with_extras(columns, preferred_columns=None):
    """Return preferred schema columns first, preserving every extra column."""
    preferred_source = RESULT_SCHEMA_COLUMNS if preferred_columns is None else list(preferred_columns)
    seen = set()
    ordered = []
    input_columns = list(columns)
    input_set = set(input_columns)
    for column in preferred_source:
        if column in input_set and column not in seen:
            ordered.append(column)
            seen.add(column)
    for column in input_columns:
        if column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def stable_json_cell(value):
    """Serialize list/dict CSV cells with the D1-D6 contract JSON policy."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


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
