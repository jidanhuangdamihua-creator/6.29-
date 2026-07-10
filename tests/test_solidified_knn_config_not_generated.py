from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLIDIFIED_KNN_ROOT = PROJECT_ROOT / "configs" / "solidified" / "knn"
GENERATED_KNN_ROOT = (
    PROJECT_ROOT / "outputs" / "feature_consistency_full" / "generated_json"
)

PROMOTION_ERROR = (
    "generated KNN appears to have been promoted into solidified config; "
    "this is an experiment-change and must not be included in the perf/refactor PR."
)

D4_D5_D6_DATASET_IDS = (4, 5, 6)
INFO_SHARING_MODES = ("with", "without")


def _iter_d4_d5_d6_knn_paths() -> list[tuple[int, str, Path, Path]]:
    pairs: list[tuple[int, str, Path, Path]] = []
    for dataset_id in D4_D5_D6_DATASET_IDS:
        for mode in INFO_SHARING_MODES:
            filename = f"knn_{mode}_info_sharing.json"
            solidified = SOLIDIFIED_KNN_ROOT / f"Dataset{dataset_id}" / filename
            generated = GENERATED_KNN_ROOT / f"Dataset{dataset_id}" / filename
            pairs.append((dataset_id, mode, solidified, generated))
    return pairs


def test_solidified_knn_config_not_promoted_from_generated() -> None:
    """Guard against copying exploratory KNN JSON into solidified configs."""
    existing_generated = [
        (dataset_id, mode, solidified, generated)
        for dataset_id, mode, solidified, generated in _iter_d4_d5_d6_knn_paths()
        if generated.exists()
    ]

    if not existing_generated:
        pytest.skip("no exploratory generated KNN JSON found")

    for dataset_id, mode, solidified, generated in existing_generated:
        assert solidified.exists(), f"missing solidified config: {solidified}"

        solidified_payload = json.loads(solidified.read_text(encoding="utf-8"))
        generated_payload = json.loads(generated.read_text(encoding="utf-8"))

        assert solidified_payload != generated_payload, (
            f"Dataset{dataset_id} knn_{mode}_info_sharing.json: {PROMOTION_ERROR} "
            f"(solidified={solidified}, generated={generated})"
        )
