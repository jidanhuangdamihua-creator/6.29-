"""Guard solidified D4-D6 KNN configs from exploratory generated JSON promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLIDIFIED_KNN_ROOT = PROJECT_ROOT / "configs" / "solidified" / "knn"

GENERATED_KNN_ROOTS = (
    PROJECT_ROOT / "outputs" / "feature_consistency" / "generated_json",
    PROJECT_ROOT / "outputs" / "feature_consistency_full" / "generated_json",
)

PROMOTION_ERROR = (
    "generated KNN appears to have been promoted into solidified config; "
    "this is an experiment-change and must not be included in the perf/refactor PR."
)

D4_D5_D6_DATASET_IDS = (4, 5, 6)
INFO_SHARING_MODES = ("with", "without")


def _iter_d4_d5_d6_knn_paths(
    solidified_root: Path = SOLIDIFIED_KNN_ROOT,
    generated_roots: Sequence[Path] = GENERATED_KNN_ROOTS,
) -> list[tuple[int, str, Path, Path]]:
    """Return every unique solidified/generated KNN path pair."""
    pairs: list[tuple[int, str, Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()

    for dataset_id in D4_D5_D6_DATASET_IDS:
        for mode in INFO_SHARING_MODES:
            filename = f"knn_{mode}_info_sharing.json"
            solidified = solidified_root / f"Dataset{dataset_id}" / filename

            for generated_root in generated_roots:
                generated = generated_root / f"Dataset{dataset_id}" / filename
                pair_key = (solidified.resolve(), generated.resolve())

                if pair_key in seen:
                    continue

                seen.add(pair_key)
                pairs.append((dataset_id, mode, solidified, generated))

    return pairs


def verify_generated_knn_not_promoted(
    solidified_root: Path,
    generated_roots: Sequence[Path],
) -> int:
    """Check all existing generated KNN files and return the checked count."""
    checked = 0

    for dataset_id, mode, solidified, generated in _iter_d4_d5_d6_knn_paths(
        solidified_root=solidified_root,
        generated_roots=generated_roots,
    ):
        if not generated.exists():
            continue

        checked += 1

        assert solidified.exists(), f"missing solidified config: {solidified}"

        solidified_payload = json.loads(solidified.read_text(encoding="utf-8"))
        generated_payload = json.loads(generated.read_text(encoding="utf-8"))

        assert solidified_payload != generated_payload, (
            f"Dataset{dataset_id} knn_{mode}_info_sharing.json: "
            f"{PROMOTION_ERROR} "
            f"(solidified={solidified}, generated={generated})"
        )

    return checked


def test_solidified_knn_config_not_promoted_from_generated() -> None:
    """Guard against copying exploratory KNN JSON into solidified configs."""
    checked = verify_generated_knn_not_promoted(
        solidified_root=SOLIDIFIED_KNN_ROOT,
        generated_roots=GENERATED_KNN_ROOTS,
    )

    if checked == 0:
        pytest.skip("no exploratory generated KNN JSON found in known roots")


def test_empty_first_root_populated_second_root_is_checked(
    tmp_path: Path,
) -> None:
    """The second generated root must be checked when the first is empty."""
    solidified_root = tmp_path / "solidified"
    generated_root_1 = tmp_path / "generated_empty"
    generated_root_2 = tmp_path / "generated_populated"

    filename = "knn_with_info_sharing.json"

    solidified_path = solidified_root / "Dataset4" / filename
    generated_path = generated_root_2 / "Dataset4" / filename

    solidified_path.parent.mkdir(parents=True, exist_ok=True)
    generated_root_1.mkdir(parents=True, exist_ok=True)
    generated_path.parent.mkdir(parents=True, exist_ok=True)

    promoted_payload = {
        "dataset_id": 4,
        "info_sharing": "with",
        "results": {"target": [{"source_entity": "source"}]},
    }

    solidified_path.write_text(
        json.dumps(promoted_payload),
        encoding="utf-8",
    )
    generated_path.write_text(
        json.dumps(promoted_payload),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="generated KNN appears"):
        verify_generated_knn_not_promoted(
            solidified_root=solidified_root,
            generated_roots=(generated_root_1, generated_root_2),
        )

    generated_path.write_text(
        json.dumps(
            {
                "dataset_id": 4,
                "info_sharing": "with",
                "results": {
                    "target": [{"source_entity": "different_source"}]
                },
            }
        ),
        encoding="utf-8",
    )

    checked = verify_generated_knn_not_promoted(
        solidified_root=solidified_root,
        generated_roots=(generated_root_1, generated_root_2),
    )

    assert checked == 1
