from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import time

import pandas as pd
import pytest

import scripts.dryrun_raw_wide_knn_compare as compare_script
from scripts.dryrun_raw_wide_knn_compare import (
    compare_feature_schema,
    compute_pool_domain_diagnostics,
    reconstruct_runtime_window_metadata,
    normalize_wide_clean_schema,
    stable_cap_entities,
    topk_overlap_ratio,
    validate_output_dir,
    wide_with_non_configured_domain_count,
    WideCleanUnavailable,
    build_or_load_wide_clean,
)


def test_single_domain_narrow_pool_is_vacuous() -> None:
    source = pd.DataFrame(
        {
            "store_nbr": [1, 1, 2],
            "item_nbr": [10, 10, 20],
            "family": ["GROCERY I", "GROCERY I", "GROCERY I"],
        }
    )

    diagnostics = compute_pool_domain_diagnostics(
        source,
        domain_column="family",
        domain_value="GROCERY I",
        group_cols=["store_nbr", "item_nbr"],
    )

    assert diagnostics["source_domain_nunique"] == 1
    assert diagnostics["after_filter_rows"] == 3
    assert diagnostics["after_filter_entities"] == 2
    assert diagnostics["domain_filter_vacuous"] is True
    assert diagnostics["domain_filter_effective"] is False


def test_multi_domain_wide_pool_is_effective() -> None:
    source = pd.DataFrame(
        {
            "store_id": ["A", "A", "B", "C"],
            "item_id": ["x", "x", "y", "z"],
            "dept_id": ["FOODS_3", "FOODS_3", "FOODS_2", "FOODS_3"],
        }
    )

    diagnostics = compute_pool_domain_diagnostics(
        source,
        domain_column="dept_id",
        domain_value="FOODS_3",
        group_cols=["store_id", "item_id"],
    )

    assert diagnostics["source_domain_nunique"] == 2
    assert diagnostics["after_filter_rows"] == 3
    assert diagnostics["after_filter_entities"] == 2
    assert diagnostics["domain_filter_vacuous"] is False
    assert diagnostics["domain_filter_effective"] is True


def test_cap_is_stable_after_window_eligibility() -> None:
    source = pd.DataFrame(
        {
            "store_id": ["C", "A", "B", "D", "A"],
            "item_id": ["3", "1", "2", "4", "1"],
            "date": pd.date_range("2020-01-01", periods=5),
        }
    )

    first, first_meta = stable_cap_entities(source, ["store_id", "item_id"], cap=2)
    second, second_meta = stable_cap_entities(
        source.sample(frac=1.0, random_state=7), ["store_id", "item_id"], cap=2
    )

    assert set(map(tuple, first[["store_id", "item_id"]].drop_duplicates().to_numpy())) == set(
        map(tuple, second[["store_id", "item_id"]].drop_duplicates().to_numpy())
    )
    assert first_meta == second_meta
    assert first_meta == {
        "max_source_entities": 2,
        "pre_cap_source_entities": 4,
        "post_cap_source_entities": 2,
        "cap_applied": True,
        "hash_key_columns": ["store_id", "item_id"],
    }


def test_entity_key_builder_does_not_use_rowwise_pandas_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_rowwise_operation(*args: object, **kwargs: object) -> object:
        if kwargs.get("axis") == 1 or (args and args[0] == 1):
            raise AssertionError("row-wise pandas operation used for entity keys")
        raise AssertionError("unexpected pandas operation")

    monkeypatch.setattr(pd.DataFrame, "agg", fail_rowwise_operation)
    monkeypatch.setattr(pd.DataFrame, "apply", fail_rowwise_operation)
    source = pd.DataFrame({"store": [1, None], "item": [2, 3]})

    result, _ = stable_cap_entities(source, ["store", "item"], cap=None)

    assert result.index.tolist() == [0, 1]


def test_stable_cap_entities_is_fast_and_deterministic_on_large_frame() -> None:
    rows = 100_000
    source = pd.DataFrame(
        {
            "store": [index % 250 for index in range(rows)],
            "item": [index % 400 for index in range(rows)],
            "value": range(rows),
        }
    )

    started = time.perf_counter()
    first, first_meta = stable_cap_entities(source, ["store", "item"], cap=75)
    elapsed = time.perf_counter() - started
    second, second_meta = stable_cap_entities(
        source.sample(frac=1.0, random_state=7), ["store", "item"], cap=75
    )

    assert elapsed < 3.0
    assert first_meta == second_meta
    assert set(map(tuple, first[["store", "item"]].drop_duplicates().to_numpy())) == set(
        map(tuple, second[["store", "item"]].drop_duplicates().to_numpy())
    )


def test_schema_and_overlap_diagnostics_are_explicit() -> None:
    schema = compare_feature_schema(
        ["sales", "week", "sell_price"],
        ["sales", "week", "snap"],
    )

    assert schema == {
        "narrow_feature_columns_count": 3,
        "wide_feature_columns_count": 3,
        "feature_columns_missing_in_wide": ["sell_price"],
        "feature_columns_extra_in_wide": ["snap"],
        "feature_schema_match": False,
    }
    assert topk_overlap_ratio(["a", "b", "c"], ["a", "b", "c"], top_k=3) == 1.0
    assert topk_overlap_ratio(["a", "b", "c"], ["b", "x", "y"], top_k=3) == pytest.approx(1 / 3)
    assert topk_overlap_ratio(["a"], ["x"], top_k=3) == 0.0
    assert wide_with_non_configured_domain_count(
        ["FOODS_3", "FOODS_2", "FOODS_3"], "FOODS_3"
    ) == 1


def test_missing_json_feature_fails_wide_clean_schema_explicitly() -> None:
    payload = {
        "group_cols": ["store_nbr", "item_nbr"],
        "domain_filter": {"column": "family", "value": "GROCERY I"},
        "feature_cols": ["sales", "oil_price"],
    }
    incomplete = pd.DataFrame(
        {
            "date": ["2017-01-01"],
            "store_nbr": [1],
            "item_nbr": [2],
            "family": ["GROCERY I"],
            "sales": [1.0],
        }
    )

    with pytest.raises(ValueError, match=r"missing columns: \['oil_price'\]"):
        normalize_wide_clean_schema(incomplete, payload=payload, dataset_id=5)


def test_runtime_windows_and_output_safety() -> None:
    metadata = reconstruct_runtime_window_metadata(
        {"dataset_id": 5, "train_start": "2017-01-17"}
    )

    assert metadata["target_feature_window_days"] == 30
    assert metadata["source_history_days"] == 300
    assert metadata["target_observed_start"] == "2017-01-17"
    assert metadata["target_observed_end"] == "2017-02-15"
    assert metadata["source_history_start"] == "2016-04-22"
    assert metadata["source_history_end"] == "2017-02-15"

    safe = Path("outputs/feature_consistency/raw_wide_knn_dryrun_20260710")
    assert validate_output_dir(safe) == safe
    with pytest.raises(ValueError, match="protected"):
        validate_output_dir(Path("configs/solidified/knn/raw_wide_knn_dryrun"))
    with pytest.raises(ValueError, match="protected"):
        validate_output_dir(Path("数据集/固化数据/raw_wide_knn_dryrun"))


def test_cli_writes_only_diagnostic_artifacts_from_a_reusable_wide_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_root = tmp_path / "parquet"
    knn_root = tmp_path / "knn"
    clean_root = tmp_path / "clean"
    output_root = tmp_path / "outputs" / "feature_consistency" / "raw_wide_knn_dryrun_test"
    parquet_root.mkdir()
    clean_root.mkdir()
    (knn_root / "Dataset5").mkdir(parents=True)
    monkeypatch.setattr(compare_script, "ALLOWED_OUTPUT_ROOT", tmp_path / "outputs" / "feature_consistency")

    source_dates = pd.date_range("2016-04-22", periods=300, freq="D")
    target_dates = pd.date_range("2017-01-17", periods=30, freq="D")

    def frame(store: int, item: int, family: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        data: dict[str, object] = {
            "date": dates,
            "store_nbr": store,
            "item_nbr": item,
            "entity_id": f"{store}_{item}",
            "family": family,
        }
        for index, feature in enumerate(
            [
                "sales", "year", "month", "week", "day", "class", "perishable",
                "cluster", "transactions", "oil_price", "is_holiday",
            ]
        ):
            data[feature] = float(index + 1)
        return pd.DataFrame(data)

    narrow_source = frame(1, 10, "GROCERY I", source_dates)
    target = frame(9, 99, "GROCERY I", target_dates)
    wide = pd.concat(
        [narrow_source, frame(2, 20, "BEVERAGES", source_dates)], ignore_index=True
    )
    narrow_source.to_parquet(parquet_root / "dataset5-source.parquet", index=False)
    target.to_parquet(parquet_root / "dataset5-target.parquet", index=False)
    wide.to_parquet(clean_root / "dataset5-wide-clean.parquet", index=False)

    payload = {
        "dataset_id": 5,
        "dataset": "D5",
        "info_sharing": "with",
        "k": 1,
        "target_train_window": {"start": "2017-01-17", "end": "2017-01-31"},
        "domain_filter": {"column": "family", "value": "GROCERY I"},
        "group_cols": ["store_nbr", "item_nbr"],
        "feature_cols": [
            "sales", "year", "month", "week", "day", "class", "perishable",
            "cluster", "transactions", "oil_price", "is_holiday",
        ],
        "results": {"9_99": [{"source_entity": "1_10", "distance": 0.0, "weight": 1.0}]},
    }
    for mode in ("with", "without"):
        scenario_payload = dict(payload, info_sharing=mode)
        (knn_root / "Dataset5" / f"knn_{mode}_info_sharing.json").write_text(
            json.dumps(scenario_payload), encoding="utf-8"
        )

    compare_script.main(
        [
            "--dataset", "5", "--mode", "both", "--top-k", "1",
            "--clean-input-root", str(clean_root), "--output-dir", str(output_root),
            "--parquet-root", str(parquet_root), "--knn-root", str(knn_root),
            "--max-source-entities", "1",
        ]
    )

    csv_path = output_root / "dataset5_raw_wide_knn_compare.csv"
    json_path = output_root / "dataset5_raw_wide_knn_compare_summary.json"
    markdown_path = output_root / "raw_wide_knn_compare_report.md"
    assert csv_path.exists() and json_path.exists() and markdown_path.exists()
    result = pd.read_csv(csv_path)
    assert result.loc[0, "domain_filter_vacuous_on_narrow"]
    assert "Runtime-window comparability check" in markdown_path.read_text(encoding="utf-8")
    assert "Schema-contract check" in markdown_path.read_text(encoding="utf-8")
    assert result["wide_with_post_cap_source_entities"].eq(1).all()
    assert result["wide_without_post_cap_source_entities"].eq(1).all()


def test_script_is_directly_invocable_from_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dryrun_raw_wide_knn_compare.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--max-source-entities" in result.stdout


def test_cli_fails_explicitly_when_wide_clean_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs" / "feature_consistency" / "raw_wide_knn_dryrun_failed"
    monkeypatch.setattr(compare_script, "ALLOWED_OUTPUT_ROOT", tmp_path / "outputs" / "feature_consistency")

    with pytest.raises(SystemExit) as failure:
        compare_script.main(
            [
                "--dataset", "5", "--output-dir", str(output_root),
                "--raw-input-root", str(tmp_path / "missing_raw"),
                "--reuse-existing-wide-clean",
                "--parquet-root", str(tmp_path / "missing_parquet"),
                "--knn-root", str(tmp_path / "missing_knn"),
            ]
        )

    assert failure.value.code == 1
    summary = json.loads((output_root / "dataset5_raw_wide_knn_compare_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "reuse-existing-wide-clean requires --clean-input-root" in summary["error_message"]


def test_reuse_existing_wide_clean_requires_the_expected_parquet(tmp_path: Path) -> None:
    payload = {
        "group_cols": ["store_nbr", "item_nbr"],
        "domain_filter": {"column": "family", "value": "GROCERY I"},
        "feature_cols": ["sales"],
    }

    with pytest.raises(WideCleanUnavailable, match="requested but missing"):
        build_or_load_wide_clean(
            dataset_id=5,
            payload=payload,
            raw_root=tmp_path / "raw",
            clean_input_root=tmp_path / "clean",
            run_dir=tmp_path / "run",
            runtime={},
            reuse_existing_wide_clean=True,
        )
