from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

from src.data_processing.data_preprocessing import _is_identifier_like_column, infer_source_selection_feature_columns
from src.protocols.experiment_protocol import get_experiment_protocol


FEATURE_STATUS_ALIGNED = "aligned"
FEATURE_STATUS_JSON_AUTHORITY_RUNTIME_DIFF = "json_authority_runtime_diff"
FEATURE_STATUS_FALLBACK_RUNTIME_INFER = "fallback_runtime_infer"
FEATURE_STATUS_INVALID_JSON_FEATURES = "invalid_json_features"
FEATURE_STATUS_RFE_EXPECTED_SUBSET = "rfe_expected_subset"

FEATURE_CONSISTENCY_STATUSES = {
    FEATURE_STATUS_ALIGNED,
    FEATURE_STATUS_JSON_AUTHORITY_RUNTIME_DIFF,
    FEATURE_STATUS_FALLBACK_RUNTIME_INFER,
    FEATURE_STATUS_INVALID_JSON_FEATURES,
    FEATURE_STATUS_RFE_EXPECTED_SUBSET,
}


def _scenario_file_token(information_sharing: str) -> str:
    scenario = str(information_sharing).strip().lower()
    if scenario in {"without", "without_information_sharing"}:
        return "without"
    if scenario in {"with", "with_information_sharing"}:
        return "with"
    raise ValueError("information_sharing must be 'with' or 'without'")


def load_solidified_knn_selected_features(
    dataset_id: int,
    information_sharing: str,
    knn_root: str | Path,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load authoritative selected features from a solidified KNN JSON file."""
    root = Path(knn_root)
    path = root / f"Dataset{int(dataset_id)}" / f"knn_{_scenario_file_token(information_sharing)}_info_sharing.json"
    if payload is None and not path.exists():
        return {
            "selected_features": [],
            "knn_feature_mode": "",
            "source": "missing_solidified_json",
            "json_path": str(path),
        }
    if payload is None:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = dict(payload)
    feature_info = payload.get("feature_info", {}) if isinstance(payload.get("feature_info"), dict) else {}
    selected = feature_info.get("selected_features")
    if not selected:
        selected = payload.get("feature_cols", [])
    return {
        "selected_features": [str(col) for col in (selected or [])],
        "knn_feature_mode": str(feature_info.get("knn_feature_mode", "")),
        "source": "solidified_json",
        "json_path": str(payload.get("_path", path)),
        "payload": payload,
    }


def _is_numeric_or_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return True
    converted = pd.to_numeric(non_null, errors="coerce")
    return bool(converted.notna().all())


def _coerce_numeric_like_in_place(df: pd.DataFrame, col: str) -> None:
    if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
        return
    df[col] = pd.to_numeric(df[col], errors="coerce")


def validate_solidified_knn_features(
    feature_info: Dict[str, Any],
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    dataset_id: int,
) -> Dict[str, Any]:
    """Validate solidified JSON feature columns before using them as runtime authority."""
    selected = [str(col) for col in feature_info.get("selected_features", [])]
    missing_in_source = [col for col in selected if col not in source_df.columns]
    missing_in_target = [col for col in selected if col not in target_df.columns]
    identifier_columns = [
        col
        for col in selected
        if str(col).strip().lower() == "date" or _is_identifier_like_column(col)
    ]
    non_numeric = [
        col
        for col in selected
        if col in source_df.columns
        and col in target_df.columns
        and (
            not _is_numeric_or_numeric_like(source_df[col])
            or not _is_numeric_or_numeric_like(target_df[col])
        )
    ]
    errors = {
        "empty": not bool(selected),
        "missing_sales": "sales" not in selected,
        "missing_in_source": missing_in_source,
        "missing_in_target": missing_in_target,
        "identifier_columns": identifier_columns,
        "non_numeric_columns": non_numeric,
    }
    if any(
        [
            errors["empty"],
            errors["missing_sales"],
            missing_in_source,
            missing_in_target,
            identifier_columns,
            non_numeric,
        ]
    ):
        raise ValueError(
            "invalid_json_features: "
            f"dataset_id={int(dataset_id)} json_path={feature_info.get('json_path', '')} errors={errors}"
        )
    for col in selected:
        _coerce_numeric_like_in_place(source_df, col)
        _coerce_numeric_like_in_place(target_df, col)
    return {
        **feature_info,
        "selected_features": selected,
        "feature_source": "solidified_knn_json",
        "feature_consistency_status": FEATURE_STATUS_ALIGNED,
    }


def compare_json_and_runtime_features(
    json_features: Sequence[str],
    runtime_features: Sequence[str],
) -> Dict[str, Any]:
    """Compare JSON authority features with runtime-inferred diagnostics."""
    json_list = [str(col) for col in json_features]
    runtime_list = [str(col) for col in runtime_features]
    json_set = set(json_list)
    runtime_set = set(runtime_list)
    json_only = [col for col in json_list if col not in runtime_set]
    runtime_only = [col for col in runtime_list if col not in json_set]
    status = (
        FEATURE_STATUS_ALIGNED
        if not json_only and not runtime_only
        else FEATURE_STATUS_JSON_AUTHORITY_RUNTIME_DIFF
    )
    return {
        "feature_consistency_status": status,
        "json_only_features": json_only,
        "runtime_only_features": runtime_only,
    }


def resolve_knn_feature_columns(
    *,
    dataset_id: int,
    information_sharing: str,
    knn_root: str | Path,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    knn_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve D4-D6 feature columns with solidified JSON as authority."""
    protocol_features = list(get_experiment_protocol(dataset_id).knn_feature_columns)
    json_info = load_solidified_knn_selected_features(
        dataset_id=dataset_id,
        information_sharing=information_sharing,
        knn_root=knn_root,
        payload=knn_payload,
    )
    if int(dataset_id) == 5:
        payload = json_info.get("payload")
        declared = payload.get("knn_feature_columns") if isinstance(payload, dict) else None
        authority_path = json_info.get("json_path", "")
        if declared != protocol_features:
            raise ValueError(
                "D5 KNN contract mismatch: "
                f"expected features={protocol_features!r} actual features={declared!r} "
                f"authority_path={authority_path}"
            )
        if not json_info.get("selected_features"):
            raise ValueError(
                "D5 KNN authority has no model feature columns; "
                f"expected KNN features={protocol_features!r} authority_path={authority_path}"
            )
    runtime_info = infer_source_selection_feature_columns(source_df, target_df)
    runtime_features = [str(col) for col in runtime_info.get("selected_features", [])]
    # D5 keeps the model's historical 11-column candidate list separate from
    # the three-column KNN contract.  ``onpromotion`` is a KNN-only field in
    # that split, so it must not make an otherwise aligned model-feature
    # comparison look divergent.
    if int(dataset_id) == 5 and json_info.get("selected_features"):
        json_feature_set = set(json_info["selected_features"])
        runtime_features = [
            column
            for column in runtime_features
            if column in json_feature_set or column == "sales"
        ]

    if json_info.get("selected_features"):
        try:
            validated = validate_solidified_knn_features(
                json_info,
                source_df=source_df,
                target_df=target_df,
                dataset_id=dataset_id,
            )
        except ValueError:
            json_info["feature_consistency_status"] = FEATURE_STATUS_INVALID_JSON_FEATURES
            raise
        comparison = compare_json_and_runtime_features(
            validated["selected_features"],
            runtime_features,
        )
        return {
            "selected_features": list(validated["selected_features"]),
            "feature_source": "solidified_knn_json",
            "knn_feature_mode": validated.get("knn_feature_mode", ""),
            "source_selection_feature_cols": list(validated["selected_features"]),
            "runtime_inferred_feature_cols": runtime_features,
            **comparison,
        }

    if int(dataset_id) == 5:
        raise ValueError(
            "D5 KNN authority is missing selected model feature columns; "
            f"expected KNN features={protocol_features!r} authority_path={json_info.get('json_path', '')}"
        )
    return {
        "selected_features": runtime_features,
        "feature_source": "runtime_infer",
        "knn_feature_mode": runtime_info.get("knn_feature_mode", ""),
        "source_selection_feature_cols": runtime_features,
        "runtime_inferred_feature_cols": runtime_features,
        "feature_consistency_status": FEATURE_STATUS_FALLBACK_RUNTIME_INFER,
        "json_only_features": [],
        "runtime_only_features": runtime_features,
    }
