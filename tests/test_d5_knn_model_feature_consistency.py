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
    resolve_knn_feature_columns,
    validate_solidified_knn_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLIDIFIED_KNN_ROOT = PROJECT_ROOT / "configs" / "solidified" / "knn"
D5_SOLIDIFIED_FEATURES = [
    "sales",
    "year",
    "month",
    "week",
    "day",
    "class",
    "perishable",
    "cluster",
    "transactions",
    "oil_price",
    "is_holiday",
]


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


def _solidified_feature_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = pd.DataFrame(
        {
            feature: [float(index + 1), float(index + 2)]
            for index, feature in enumerate(D5_SOLIDIFIED_FEATURES)
        }
    )
    target_df = pd.DataFrame(
        {
            feature: [float(index + 3), float(index + 4)]
            for index, feature in enumerate(D5_SOLIDIFIED_FEATURES)
        }
    )
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


@pytest.mark.parametrize("information_sharing", ["without", "with"])
def test_d5_solidified_knn_json_features_parse_stably(
    information_sharing: str,
) -> None:
    json_path = (
        SOLIDIFIED_KNN_ROOT
        / "Dataset5"
        / f"knn_{information_sharing}_info_sharing.json"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    info = load_solidified_knn_selected_features(
        dataset_id=5,
        information_sharing=information_sharing,
        knn_root=SOLIDIFIED_KNN_ROOT,
    )

    assert payload["feature_cols"] == D5_SOLIDIFIED_FEATURES
    assert payload["feature_info"]["selected_features"] == D5_SOLIDIFIED_FEATURES
    assert info["selected_features"] == D5_SOLIDIFIED_FEATURES
    assert info["payload"]["feature_cols"] == D5_SOLIDIFIED_FEATURES
    assert info["source"] == "solidified_json"


def test_d5_resolved_knn_feature_columns_are_stable_across_scenarios() -> None:
    source_df, target_df = _solidified_feature_frames()
    resolved = {
        information_sharing: resolve_knn_feature_columns(
            dataset_id=5,
            information_sharing=information_sharing,
            knn_root=SOLIDIFIED_KNN_ROOT,
            source_df=source_df.copy(),
            target_df=target_df.copy(),
        )
        for information_sharing in ("without", "with")
    }

    assert resolved["without"] == resolved["with"]
    for info in resolved.values():
        assert info["selected_features"] == D5_SOLIDIFIED_FEATURES
        assert info["source_selection_feature_cols"] == D5_SOLIDIFIED_FEATURES
        assert info["feature_source"] == "solidified_knn_json"
        assert info["feature_consistency_status"] == FEATURE_STATUS_ALIGNED
        assert info["json_only_features"] == []
        assert info["runtime_only_features"] == []
