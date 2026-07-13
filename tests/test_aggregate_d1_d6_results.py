from __future__ import annotations

import pandas as pd
import pytest

from scripts import aggregate_d1_d6_results as aggregate
from src.constants import RESULT_CONTRACT_VERSION
from src.evaluation.metric_contract import METRIC_CONTRACT_VERSION, SMAPE_DEFINITION_ID
from src.evaluation.metric_contract import MetricProtocolError
from src.protocols.experiment_protocol import FORMAL_SEEDS


def _formal_row(**overrides):
    row = {
        "dataset": "D1",
        "dataset_id": 1,
        "target_entity_key": "target-a",
        "method": "No-TL",
        "horizon": 1,
        "seed": 42,
        "information_sharing": "without",
        "smape": 20.0,
        "strict_paper_metrics": True,
        "paper_metric_space_requested": "original_sales_space",
        "paper_metric_space_actual": "original_sales_space",
        "primary_metric_space_actual": "original_sales_space",
        "smape_metric_space": "original_sales_space",
        "inverse_transform_status": "applied",
        "paper_metric_computed_valid": True,
        "paper_metric_status": "valid",
        "paper_metric_error": "",
        "metric_sample_count": 2,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "smape_definition_id": SMAPE_DEFINITION_ID,
        "smape_unit": "percent",
        "smape_epsilon": 1e-8,
        "smape_range_min": 0.0,
        "smape_range_max": 200.0,
        "sales_value_policy": "clip_negative_to_zero_v1",
        "target_negative_count": 0,
        "metric_target_key": "target-a",
        "metric_horizon": 1,
        "metric_date_start": "2024-01-02",
        "metric_date_end": "2024-01-03",
        "metric_index_digest": "digest-target-a-42",
    }
    row.update(overrides)
    row["metric_target_key"] = overrides.get(
        "metric_target_key", row["target_entity_key"]
    )
    row["metric_horizon"] = overrides.get("metric_horizon", row["horizon"])
    return row


def _formal_seed_rows(
    *,
    dataset: str = "D1",
    target: str = "target-a",
    method: str = "Method-A",
    smapes=(10.0, 11.0, 12.0, 13.0, 14.0),
):
    return [
        _formal_row(
            dataset=dataset,
            dataset_id=int(dataset[1:]),
            target_entity_key=target,
            method=method,
            seed=seed,
            smape=smape,
            metric_index_digest=f"{dataset}-{target}-{method}-{seed}",
        )
        for seed, smape in zip(FORMAL_SEEDS, smapes)
    ]


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


def test_aggregate_full_canonical_uses_superset_order_and_preserves_extra_columns(tmp_path) -> None:
    d1_path = tmp_path / "d1_without" / "results" / "dataset1_without_results.csv"
    d4_path = tmp_path / "d4_without" / "results" / "dataset4_without_results.csv"
    d1_path.parent.mkdir(parents=True, exist_ok=True)
    d4_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset_id": 1,
                "information_sharing": "without",
                "scenario": "without_information_sharing",
                "target_entity_key": "GLOBAL",
                "method": "No-TL",
                "rmse": 1.0,
                "smape": 2.0,
                "metric_space_current": "normalized_minmax_space",
                "target_window_days": 210,
                "metric_protocol": "{}",
                "legacy_d1_only_metric": "keep-me",
                "error": "",
            }
        ]
    ).to_csv(d1_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": 4,
                "information_sharing": "without",
                "scenario": "without_information_sharing",
                "target_entity_key": "store-a",
                "method": "MSWA-TL",
                "rmse": 3.0,
                "smape": 4.0,
                "selected_sources": "[]",
                "source_identifier": "unknown",
                "source_selection_feature_cols": "[\"sales\"]",
                "y_pred_nan_count": 0,
                "legacy_d4_only_diagnostic": "keep-too",
                "error": "",
            }
        ]
    ).to_csv(d4_path, index=False)

    output = tmp_path / "canonical.csv"
    aggregate.aggregate(run_dir=tmp_path, output=output, allow_missing=True)

    df = pd.read_csv(output, dtype=str, keep_default_na=False)
    for column in (
        "result_contract_version",
        "schema_family",
        "result_status",
        "failure_type",
        "metric_protocol",
        "target_window_days",
        "selected_sources",
        "source_identifier",
        "source_selection_feature_cols",
        "y_pred_nan_count",
        "legacy_d1_only_metric",
        "legacy_d4_only_diagnostic",
    ):
        assert column in df.columns
    assert df.loc[0, "result_contract_version"] == RESULT_CONTRACT_VERSION
    assert df.columns.get_loc("result_contract_version") < df.columns.get_loc("legacy_d1_only_metric")
    assert df.columns.get_loc("selected_sources") < df.columns.get_loc("legacy_d4_only_diagnostic")


def test_aggregate_canonical_information_sharing_does_not_leak_legacy_aliases(tmp_path) -> None:
    d1_path = tmp_path / "d1_without" / "results" / "dataset1_without_results.csv"
    d4_path = tmp_path / "d4_with" / "results" / "dataset4_with_results.csv"
    d1_path.parent.mkdir(parents=True, exist_ok=True)
    d4_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset_id": 1,
                "information_sharing": "without_information_sharing",
                "scenario": "without_information_sharing",
                "target_entity_key": "GLOBAL",
                "method": "No-TL",
                "rmse": 1.0,
                "smape": 2.0,
                "error": "",
            }
        ]
    ).to_csv(d1_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": 4,
                "information_sharing": "with",
                "scenario": "with_information_sharing",
                "target_entity_key": "store-a",
                "method": "MSWA-TL",
                "rmse": 3.0,
                "smape": 4.0,
                "error": "",
            }
        ]
    ).to_csv(d4_path, index=False)

    output = tmp_path / "canonical.csv"
    result = aggregate.aggregate(run_dir=tmp_path, output=output, allow_missing=True)

    canonical_df = pd.read_csv(output, dtype=str, keep_default_na=False)
    best_df = pd.read_csv(result["extra_paths"][2], dtype=str, keep_default_na=False)
    for df in (canonical_df, best_df):
        assert set(df["information_sharing"]).issubset({"with", "without"})
        assert not df["information_sharing"].isin(
            {"with_information_sharing", "without_information_sharing"}
        ).any()


def test_aggregate_reuses_discovery_csv_read_for_source_rows(tmp_path, monkeypatch) -> None:
    source = tmp_path / "d5_mixed" / "results" / "dataset5_results.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset_id": 5,
                "information_sharing": "without",
                "scenario": "without_information_sharing",
                "target_entity_key": "store-a",
                "method": "No-TL",
                "rmse": 1.0,
                "smape": 2.0,
                "error": "",
            },
            {
                "dataset_id": 5,
                "information_sharing": "with",
                "scenario": "with_information_sharing",
                "target_entity_key": "store-b",
                "method": "MSWA-TL",
                "rmse": 3.0,
                "smape": 4.0,
                "error": "",
            },
        ]
    ).to_csv(source, index=False)

    original_read_csv = aggregate.pd.read_csv
    read_counts = {source: 0}

    def counting_read_csv(path, *args, **kwargs):
        path_obj = path if isinstance(path, type(source)) else None
        if path_obj == source:
            read_counts[source] += 1
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(aggregate.pd, "read_csv", counting_read_csv)

    output = tmp_path / "summary.csv"
    aggregate.aggregate(run_dir=tmp_path, output=output, allow_missing=True)

    assert read_counts[source] == 1
    df = original_read_csv(output, dtype=str, keep_default_na=False)
    assert len(df) == 2
    assert df["information_sharing"].tolist() == ["without", "with"]
    assert df["method"].tolist() == ["No-TL", "MSWA-TL"]


def test_csv_cache_reloads_when_source_size_changes(tmp_path) -> None:
    source = tmp_path / "results" / "dataset5_results.csv"
    _write_result(source, dataset_id=5, mode="without", method="A")
    csv_cache = {}

    first_rows = aggregate._read_source(source, dataset_hint=5, csv_cache=csv_cache)
    assert [row["method"] for row in first_rows] == ["A"]

    _write_result(source, dataset_id=5, mode="without", method="longer-method-name")
    second_rows = aggregate._read_source(source, dataset_hint=5, csv_cache=csv_cache)

    assert [row["method"] for row in second_rows] == ["longer-method-name"]


def test_formal_aggregation_excludes_numeric_non_strict_and_legacy_rows() -> None:
    frame = pd.DataFrame(
        [
            _formal_row(smape=20.0),
            _formal_row(method="SS-TL", smape=1.0, strict_paper_metrics=False),
            {"dataset": "D1", "method": "MSWA-TL", "smape": 0.5},
        ]
    )

    result = aggregate.aggregate_formal_smape(frame)

    assert result["eligible_rows"]["method"].tolist() == ["No-TL"]
    assert result["cross_dataset_macro"]["smape"].tolist() == [20.0]
    assert result["exclusion_reason_counts"]["invalid:strict_paper_metrics"] == 1
    assert result["exclusion_reason_counts"]["missing:metric_contract_version"] == 1


def test_formal_aggregation_keeps_horizon_and_sharing_scenario_separate() -> None:
    rows = []
    for horizon in (1, 5):
        for sharing in ("without", "with"):
            rows.append(
                _formal_row(
                    horizon=horizon,
                    information_sharing=sharing,
                    smape=10.0 + horizon,
                )
            )

    result = aggregate.aggregate_formal_smape(pd.DataFrame(rows))
    output = result["cross_dataset_macro"]

    assert output.groupby(["horizon", "sharing_scenario"]).ngroups == 4
    assert len(output) == 4


def test_written_formal_summaries_preserve_horizon_and_sharing_dimensions(tmp_path) -> None:
    rows = []
    for horizon in (1, 5):
        for sharing in ("without", "with"):
            rows.append(
                _formal_row(
                    horizon=horizon,
                    information_sharing=sharing,
                    smape=10.0 + horizon,
                )
            )
    output = tmp_path / "formal.csv"

    metric_paths = aggregate._write_metric_summaries(output, rows)
    best_paths = aggregate._write_best_method_outputs(output, rows)

    dataset_summary = pd.read_csv(metric_paths[0])
    best_by_target = pd.read_csv(best_paths[0])
    assert {"horizon", "sharing_scenario"}.issubset(dataset_summary.columns)
    assert dataset_summary.groupby(["horizon", "sharing_scenario"]).ngroups == 4
    assert {"horizon", "information_sharing"}.issubset(best_by_target.columns)
    assert best_by_target.groupby(["horizon", "information_sharing"]).ngroups == 4


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_formal_aggregation_rejects_invalid_seed_matrix(mutation) -> None:
    rows = _formal_seed_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(
            _formal_row(
                dataset="D1",
                dataset_id=1,
                target_entity_key="target-a",
                method="Method-A",
                seed=99,
                smape=15.0,
                metric_index_digest="D1-target-a-Method-A-99",
            )
        )
    else:
        rows.append(dict(rows[0]))

    with pytest.raises(MetricProtocolError, match="formal_seed|duplicate_formal"):
        aggregate.aggregate_formal_smape(
            pd.DataFrame(rows),
            expected_seeds=FORMAL_SEEDS,
        )


def test_formal_aggregation_uses_seed_then_target_then_dataset_macro() -> None:
    rows = []
    rows.extend(_formal_seed_rows(dataset="D1", target="t1", smapes=(1, 2, 3, 4, 5)))
    rows.extend(_formal_seed_rows(dataset="D1", target="t2", smapes=(3, 4, 5, 6, 7)))
    rows.extend(_formal_seed_rows(dataset="D2", target="t1", smapes=(5, 6, 7, 8, 9)))
    rows.extend(_formal_seed_rows(dataset="D2", target="t2", smapes=(7, 8, 9, 10, 11)))

    result = aggregate.aggregate_formal_smape(
        pd.DataFrame(rows),
        expected_seeds=FORMAL_SEEDS,
    )

    assert result["seed_mean"].sort_values(["dataset", "target"])["smape"].tolist() == [3.0, 5.0, 7.0, 9.0]
    assert result["dataset_macro"].sort_values("dataset")["smape"].tolist() == [4.0, 8.0]
    assert result["cross_dataset_macro"]["smape"].tolist() == [6.0]


def test_best_method_outputs_preserve_seed_rows_but_choose_by_seed_mean(tmp_path) -> None:
    rows = []
    rows.extend(_formal_seed_rows(method="Method-A", smapes=(0, 100, 100, 100, 100)))
    rows.extend(_formal_seed_rows(method="Method-B", smapes=(20, 20, 20, 20, 20)))
    output = tmp_path / "formal.csv"

    paths = aggregate._write_best_method_outputs(
        output,
        rows,
        expected_seeds=FORMAL_SEEDS,
    )

    best_by_target = pd.read_csv(paths[0])
    seed_detail = pd.read_csv(paths[2])
    assert len(seed_detail) == len(rows)
    assert {"seed", "seed_rank"}.issubset(seed_detail.columns)
    assert best_by_target.loc[0, "best_method"] == "Method-B"
    assert best_by_target.loc[0, "candidate_method_count"] == 2
