from __future__ import annotations

import pandas as pd
import pytest

from scripts import aggregate_d1_d6_results as aggregate


def _write_result(path, *, dataset_id: int, mode: str, method: str = "No-TL") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "information_sharing": mode,
                "scenario": f"{mode}_information_sharing",
                "target_entity_key": f"target-{dataset_id}-{mode}",
                "method": method,
                "rmse": 1.0,
                "smape": 2.0,
                "metric_space_used": "normalized_minmax_space",
                "paper_metric_aligned": False,
                "error": "",
            }
        ]
    ).to_csv(path, index=False)


def test_discover_source_csvs_recurses_nested_mode_results(tmp_path) -> None:
    without = tmp_path / "d5_without" / "results" / "dataset5_without_results.csv"
    with_ = tmp_path / "d5_with" / "results" / "dataset5_with_results.csv"
    _write_result(without, dataset_id=5, mode="without")
    _write_result(with_, dataset_id=5, mode="with")

    selected, audit = aggregate.discover_source_csvs(tmp_path, strict=False)

    assert selected[(5, "without")] == without
    assert selected[(5, "with")] == with_
    assert not [row for row in audit if row["status"] == "missing" and row["dataset_id"] == 5]


def test_normalize_information_sharing_canonicalizes_legacy_values() -> None:
    assert aggregate.normalize_information_sharing("without_information_sharing") == "without"
    assert aggregate.normalize_information_sharing("no_information") == "without"
    assert aggregate.normalize_information_sharing("with_information_sharing") == "with"
    assert aggregate.normalize_information_sharing("info_sharing") == "with"


def test_discover_source_csvs_strict_fails_on_missing_dataset_mode(tmp_path) -> None:
    _write_result(
        tmp_path / "d5_without" / "results" / "dataset5_without_results.csv",
        dataset_id=5,
        mode="without",
    )

    with pytest.raises(FileNotFoundError, match="Missing result CSV"):
        aggregate.discover_source_csvs(tmp_path, strict=True)


def test_aggregate_allow_missing_continues_and_records_missing_audit(tmp_path) -> None:
    output = tmp_path / "summary.csv"
    _write_result(
        tmp_path / "d5_without" / "results" / "dataset5_without_results.csv",
        dataset_id=5,
        mode="without",
    )

    result = aggregate.aggregate(run_dir=tmp_path, output=output, allow_missing=True)

    assert output.is_file()
    assert (output.parent / "summary_audit.csv").is_file()
    assert any(row["status"] == "missing" for row in result["audit_rows"])


def test_discover_source_csvs_does_not_default_to_legacy_source_csvs(tmp_path, monkeypatch) -> None:
    legacy = tmp_path / "legacy" / "results" / "dataset5_without_results.csv"
    _write_result(legacy, dataset_id=5, mode="without")
    monkeypatch.setattr(aggregate, "SOURCE_CSVS", {5: legacy})

    selected, audit = aggregate.discover_source_csvs(tmp_path / "empty", allow_missing=True)

    assert selected == {}
    assert all(row.get("source_csv_path") != str(legacy) for row in audit)


def test_discover_source_csvs_prefers_mode_path_then_newest_mtime(tmp_path) -> None:
    preferred = tmp_path / "d5_without" / "results" / "dataset5_results.csv"
    other = tmp_path / "misc" / "results" / "dataset5_without_results.csv"
    _write_result(preferred, dataset_id=5, mode="without", method="preferred")
    _write_result(other, dataset_id=5, mode="without", method="other")

    selected, _ = aggregate.discover_source_csvs(tmp_path, allow_missing=True)

    assert selected[(5, "without")] == preferred
