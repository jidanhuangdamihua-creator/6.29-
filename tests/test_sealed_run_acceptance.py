from __future__ import annotations

from datetime import timedelta

from src.protocols.sealing_protocol import get_target_window
from src.utils.prediction_artifacts import derive_formal_result_row
from src.utils.result_validation import validate_result_row_against_evaluated_trace


def _evaluated_h1_trace() -> list[dict[str, object]]:
    window = get_target_window(1)
    rows = []
    for index in range(180):
        forecast_origin = window.validation_end + timedelta(days=index)
        label_date = forecast_origin + timedelta(days=1)
        token = f"{index:064x}"
        rows.append({
            "run_id": "run-1", "cell_id": "D1/without/No-TL/42", "attempt_id": "attempt-1",
            "dataset_id": "D1", "scenario": "without", "target_entity_key": "1_10",
            "method": "No-TL", "seed": 42, "rollout_stream_key": "a" * 64,
            "forecast_origin": forecast_origin, "label_date": label_date, "horizon": 1,
            "truth_key": token, "sample_key": f"{index + 1000:064x}",
            "prediction_row_key": f"{index + 2000:064x}", "y_pred_raw": float(index),
            "y_pred_clipped": float(index), "was_clipped": False,
            "history_snapshot_digest": "b" * 64, "history_after_h1_commit_digest": "c" * 64,
            "input_digest": "d" * 64, "prediction_policy_id": "clipped_h1_v1",
            "predictor_feature_schema_digest": "1" * 64, "feature_mask_digest": "2" * 64,
            "y_true": float(index + 1), "is_synthetic_date": False,
            "evaluator_join_status": "matched",
        })
    return rows


def _identity() -> dict[str, object]:
    return {
        "artifact_path": "traces/evaluated.csv.gz", "artifact_sha256": "3" * 64,
        "canonical_content_sha256": "4" * 64, "semantic_prediction_digest": "5" * 64,
        "source_selection_identity": "6" * 64, "predictor_feature_schema_digest": "1" * 64,
        "feature_mask_digest": "2" * 64, "protocol_identity": "7" * 64,
        "input_identity": "8" * 64, "code_identity": "9" * 64,
    }


def test_result_metrics_are_recomputed_from_evaluated_trace() -> None:
    trace = _evaluated_h1_trace()
    row = derive_formal_result_row(trace, trace_identity=_identity())

    assert validate_result_row_against_evaluated_trace(
        row, trace, trace_identity=_identity()
    ) == ()

    tampered = {**row, "rmse": float(row["rmse"]) + 0.1}
    assert "rmse_recompute_mismatch" in validate_result_row_against_evaluated_trace(
        tampered, trace, trace_identity=_identity()
    )


def test_exact_horizon_calendar_is_required() -> None:
    trace = _evaluated_h1_trace()[:-1]
    row = derive_formal_result_row(trace, trace_identity=_identity())

    reasons = validate_result_row_against_evaluated_trace(
        row, trace, trace_identity=_identity()
    )
    assert "horizon_sample_count_mismatch" in reasons
    assert "forecast_origin_end_mismatch" in reasons
