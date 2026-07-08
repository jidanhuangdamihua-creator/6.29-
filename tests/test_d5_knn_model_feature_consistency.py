from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.knn_feature_loader import (
    FEATURE_STATUS_ALIGNED,
    FEATURE_STATUS_INVALID_JSON_FEATURES,
    compare_json_and_runtime_features,
    load_solidified_knn_selected_features,
    validate_solidified_knn_features,
)


def _write_knn_json(root: Path, selected_features: list[str]) -> Path:
    dataset_dir = root / "Dataset5"
    dataset_dir.mkdir(parents=True)
    path = dataset_dir / "knn_without_info_sharing.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": 5,
                "info_sharing": "without",
                "feature_info": {
                    "selected_features": selected_features,
                    "knn_feature_mode": "paper_available_features_no_ids",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = pd.DataFrame({"sales": [1.0], "promo": [0.0], "store_nbr": [1]})
    target_df = pd.DataFrame({"sales": [2.0], "promo": [1.0], "store_nbr": [1]})
    return source_df, target_df


def test_load_solidified_knn_selected_features_prefers_feature_info(tmp_path: Path) -> None:
    json_path = _write_knn_json(tmp_path, ["sales", "promo"])

    info = load_solidified_knn_selected_features(
        dataset_id=5,
        information_sharing="without",
        knn_root=tmp_path,
    )

    assert info["selected_features"] == ["sales", "promo"]
    assert info["source"] == "solidified_json"
    assert info["json_path"] == str(json_path)
    assert info["knn_feature_mode"] == "paper_available_features_no_ids"


def test_validate_solidified_knn_features_rejects_identifier_columns(tmp_path: Path) -> None:
    _write_knn_json(tmp_path, ["sales", "store_nbr"])
    source_df, target_df = _frames()
    info = load_solidified_knn_selected_features(5, "without", tmp_path)

    with pytest.raises(ValueError, match="invalid_json_features"):
        validate_solidified_knn_features(
            info,
            source_df=source_df,
            target_df=target_df,
            dataset_id=5,
        )

    assert compare_json_and_runtime_features(
        ["sales", "promo"], ["sales", "promo"]
    )["feature_consistency_status"] == FEATURE_STATUS_ALIGNED
    assert FEATURE_STATUS_INVALID_JSON_FEATURES == "invalid_json_features"
