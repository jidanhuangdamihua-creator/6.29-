from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_unified_d1_d6 import validate_formal_sequence_feasibility
from src.data_processing.data_preprocessing import (
    STRICT_DATASET_PROTOCOL,
    build_tabular_sequence,
)
from src.protocols.formal_deployment_manifest import DeploymentManifestError


ROOT = Path(__file__).resolve().parents[1]


def _one_d2_target_group(days: int) -> pd.DataFrame:
    dates = pd.date_range("2018-06-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["B1"] * days,
            "item_id": [10] * days,
            "sales": [float(value) for value in range(days)],
        }
    )


def test_d2_h5_has_one_training_window_at_formal_boundary() -> None:
    frame = _one_d2_target_group(days=15)

    x_train, y_train = build_tabular_sequence(
        frame,
        horizon=5,
        window_size=10,
        feature_columns=["sales"],
    )

    assert x_train.shape == (1, 10, 1)
    assert y_train.shape == (1,)
    assert y_train.tolist() == [14.0]


def test_d2_runtime_fallback_matches_210_day_formal_split() -> None:
    assert STRICT_DATASET_PROTOCOL["Dataset2"]["target_split_days"] == {
        "train_days": 15,
        "val_days": 15,
        "test_days": 180,
    }


def test_formal_sequence_preflight_accepts_current_matrix() -> None:
    report = validate_formal_sequence_feasibility(ROOT)

    assert report["status"] == "passed"
    assert report["window_size"] == 10
    assert report["max_horizon"] == 5
    assert report["required_train_days"] == 15
    assert report["datasets"]["Dataset2"] == {"train_days": 15}


def test_formal_sequence_preflight_rejects_fourteen_day_d2(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs" / "default_config.json").read_text(encoding="utf-8")
    )
    config["paper_reproduction"]["strict_dataset_protocol"]["Dataset2"][
        "target_split_days"
    ] = {"train_days": 14, "val_days": 15, "test_days": 179}
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "default_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )

    with pytest.raises(DeploymentManifestError) as error:
        validate_formal_sequence_feasibility(tmp_path)

    assert error.value.code == "FORMAL_SEQUENCE_WINDOW_INFEASIBLE"
    assert "Dataset2 train_days=14 required=15" in str(error.value)
