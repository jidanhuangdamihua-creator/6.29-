from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.constants import SOLIDIFIED_TARGET_WINDOWS
from src.utils.parquet_data_loader import load_knn_results, read_dataset_windows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNN_ROOT = PROJECT_ROOT / "configs" / "solidified" / "knn"
SCENARIO_IDENTITY_FIELDS = [
    "target_val_window",
    "target_test_window",
    "source_history_window",
    "observed_days",
    "train_days",
    "val_days",
    "test_days",
]

# PR-B active-scenario specification to convert into tests once production
# supports the new parameters:
# - read_dataset_windows(dataset_id, dataset_dir, info_sharing="with") reads only with.
# - read_dataset_windows(dataset_id, dataset_dir, info_sharing="without") reads only without.
# - read_dataset_windows(..., knn_payload=payload) uses the payload without reading disk.
# - info_sharing=None preserves the current without + with dual-read compatibility path.


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("dataset_id", [4, 5, 6])
def test_read_dataset_windows_uses_scenario_identical_target_train_window(
    dataset_id: int,
) -> None:
    dataset_dir = KNN_ROOT / f"Dataset{dataset_id}"
    without_payload = _read_json(dataset_dir / "knn_without_info_sharing.json")
    with_payload = _read_json(dataset_dir / "knn_with_info_sharing.json")

    without_window = without_payload["target_train_window"]
    with_window = with_payload["target_train_window"]
    assert without_window == with_window

    for field in SCENARIO_IDENTITY_FIELDS:
        if field in without_payload or field in with_payload:
            assert field in without_payload, f"{field} missing from without payload"
            assert field in with_payload, f"{field} missing from with payload"
            assert without_payload[field] == with_payload[field]

    windows = read_dataset_windows(dataset_id, dataset_dir)
    expected_solidified_window = SOLIDIFIED_TARGET_WINDOWS[dataset_id]

    assert windows["target_train_window"] == without_window
    assert windows["target_train_window"] == with_window
    assert without_window["start"] == expected_solidified_window["train_start"]
    assert windows["train_start"] == expected_solidified_window["train_start"]
    assert windows["test_end"] == expected_solidified_window["test_end"]


@pytest.mark.parametrize("dataset_id", [4, 5, 6])
def test_read_dataset_windows_preserves_legacy_dual_scenario_read_order(
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: int,
) -> None:
    dataset_dir = KNN_ROOT / f"Dataset{dataset_id}"
    expected_solidified_window = SOLIDIFIED_TARGET_WINDOWS[dataset_id]
    shared_window = {
        "start": expected_solidified_window["train_start"],
        "end": "scenario-shared-end",
    }
    calls: list[str] = []

    def fake_load_knn_results(
        knn_json_dir: str | Path,
        info_sharing: str,
    ) -> dict[str, Any]:
        assert Path(knn_json_dir) == dataset_dir
        calls.append(info_sharing)
        return {
            "target_train_window": dict(shared_window),
            "source_pool_size": 10 if info_sharing == "without" else 20,
            "domain_filter": {"scenario": info_sharing},
        }

    monkeypatch.setattr(
        "src.utils.parquet_data_loader.load_knn_results",
        fake_load_knn_results,
    )

    windows = read_dataset_windows(dataset_id, dataset_dir)

    assert calls == ["without", "with"]
    assert windows["without_target_train_window"] == shared_window
    assert windows["with_target_train_window"] == shared_window
    assert windows["target_train_window"] == shared_window
    for key, value in expected_solidified_window.items():
        assert windows[key] == value


@pytest.mark.parametrize("info_sharing", ["with", "without"])
def test_read_dataset_windows_active_scenario_reads_only_requested_scenario(
    monkeypatch: pytest.MonkeyPatch,
    info_sharing: str,
) -> None:
    dataset_id = 5
    dataset_dir = KNN_ROOT / f"Dataset{dataset_id}"
    expected_solidified_window = SOLIDIFIED_TARGET_WINDOWS[dataset_id]
    calls: list[str] = []

    def fake_load_knn_results(
        knn_json_dir: str | Path,
        scenario: str,
    ) -> dict[str, Any]:
        assert Path(knn_json_dir) == dataset_dir
        calls.append(scenario)
        return {
            "target_train_window": {
                "start": expected_solidified_window["train_start"],
                "end": f"{scenario}-active-end",
            },
            "source_pool_size": 10 if scenario == "without" else 20,
            "domain_filter": {"scenario": scenario},
        }

    monkeypatch.setattr(
        "src.utils.parquet_data_loader.load_knn_results",
        fake_load_knn_results,
    )

    windows = read_dataset_windows(dataset_id, dataset_dir, info_sharing=info_sharing)

    assert calls == [info_sharing]
    assert windows["target_train_window"] == {
        "start": expected_solidified_window["train_start"],
        "end": f"{info_sharing}-active-end",
    }
    assert windows[f"{info_sharing}_target_train_window"] == windows["target_train_window"]
    inactive = "without" if info_sharing == "with" else "with"
    assert f"{inactive}_target_train_window" not in windows
    assert windows[f"{info_sharing}_source_pool_size"] == (10 if info_sharing == "without" else 20)
    assert windows[f"{info_sharing}_domain_filter"] == {"scenario": info_sharing}
    for key, value in expected_solidified_window.items():
        assert windows[key] == value


@pytest.mark.parametrize("info_sharing", ["with", "without"])
def test_read_dataset_windows_active_scenario_uses_payload_without_loading(
    monkeypatch: pytest.MonkeyPatch,
    info_sharing: str,
) -> None:
    dataset_id = 5
    dataset_dir = KNN_ROOT / f"Dataset{dataset_id}"
    expected_solidified_window = SOLIDIFIED_TARGET_WINDOWS[dataset_id]
    payload = {
        "target_train_window": {
            "start": expected_solidified_window["train_start"],
            "end": f"{info_sharing}-payload-end",
        },
        "source_pool_size": 123,
        "domain_filter": {"scenario": info_sharing, "source": "payload"},
    }

    def fail_load_knn_results(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("read_dataset_windows should reuse knn_payload")

    monkeypatch.setattr(
        "src.utils.parquet_data_loader.load_knn_results",
        fail_load_knn_results,
    )

    windows = read_dataset_windows(
        dataset_id,
        dataset_dir,
        info_sharing=info_sharing,
        knn_payload=payload,
    )

    assert windows["target_train_window"] == payload["target_train_window"]
    assert windows[f"{info_sharing}_target_train_window"] == payload["target_train_window"]
    assert windows[f"{info_sharing}_source_pool_size"] == 123
    assert windows[f"{info_sharing}_domain_filter"] == {"scenario": info_sharing, "source": "payload"}
    for key, value in expected_solidified_window.items():
        assert windows[key] == value


def test_load_knn_results_payload_branch_adds_path_without_reading_disk(tmp_path: Path) -> None:
    payload = {
        "results": {"target-a": [{"source_entity": "source-a"}]},
        "selected_features": ["sales"],
        "feature_cols": ["sales"],
        "target_train_window": {"start": "2020-01-01", "end": "2020-01-30"},
    }

    loaded = load_knn_results(tmp_path / "missing", "with", payload=payload)

    assert loaded is not payload
    assert loaded["results"] == payload["results"]
    assert loaded["selected_features"] == payload["selected_features"]
    assert loaded["feature_cols"] == payload["feature_cols"]
    assert loaded["target_train_window"] == payload["target_train_window"]
    assert loaded["_path"] == str(tmp_path / "missing" / "knn_with_info_sharing.json")
    assert "_path" not in payload
