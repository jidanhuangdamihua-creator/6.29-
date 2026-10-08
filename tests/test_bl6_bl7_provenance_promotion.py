from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.build_paper_results_master_v1_2 import EXPECTED_METHODS, norm_method
from tools.promote_bl6_bl7_formal_results import (
    FORMAL_HORIZONS,
    FORMAL_SEEDS,
    FORMAL_TARGETS,
    PromotionError,
    aggregate_five_seed,
    build_code_identity_sidecar,
    filter_formal_scope,
    promote_candidate,
    validate_raw_evidence,
    validate_promotion_inputs,
    verify_code_identity_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = ROOT.parent / "论文文稿相关" / "原始实验结果" / "BL6_BL7_FULL_RESULTS_COMBINED.xlsx"
RESULTS = ROOT / "outputs" / "baselines" / "bl6_bl7_full_20260907"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_frame() -> pd.DataFrame:
    return pd.read_excel(INPUT_XLSX, sheet_name="BL6_BL7_Results")


def _formal_raw_by_dataset() -> dict[str, pd.DataFrame]:
    return {
        dataset: pd.read_csv(RESULTS / f"d{dataset[1:]}_baseline_predictions.csv")
        for dataset in FORMAL_TARGETS
    }


def test_derived_code_identity_sidecar_is_complete_and_original_manifest_unchanged(tmp_path: Path) -> None:
    source_manifest = RESULTS / "d1_baseline_run_manifest.json"
    source_summary = RESULTS / "d1_baselines_multiseed.csv"
    source_raw = RESULTS / "d1_baseline_predictions.csv"
    before = _sha256(source_manifest)
    sidecar = build_code_identity_sidecar(
        source_manifest,
        source_summary_path=source_summary,
        source_raw_path=source_raw,
        output_path=tmp_path / "derived.json",
        repo_root=ROOT,
    )
    required = {
        "provenance_type", "source_run_id", "source_manifest_file", "source_manifest_sha256",
        "source_summary_file", "source_summary_sha256", "source_raw_prediction_file",
        "source_raw_prediction_sha256", "git_branch", "git_head", "git_status_clean",
        "baseline_runner_path", "baseline_runner_sha256", "bl6_module_path", "bl6_module_sha256",
        "bl7_module_path", "bl7_module_sha256", "metric_authority_path", "metric_authority_sha256",
        "promotion_tool_path", "promotion_tool_sha256", "derived_at_utc", "derivation_basis",
        "original_manifest_modified",
    }
    assert required.issubset(sidecar)
    assert sidecar["provenance_type"] == "derived_code_identity"
    assert sidecar["original_manifest_modified"] is False
    assert sidecar["source_manifest_sha256"] == before
    assert _sha256(source_manifest) == before
    assert json.loads((tmp_path / "derived.json").read_text())["git_head"]


def test_derived_code_identity_and_source_summary_raw_hashes_verify(tmp_path: Path) -> None:
    source_manifest = RESULTS / "d1_baseline_run_manifest.json"
    source_summary = RESULTS / "d1_baselines_multiseed.csv"
    source_raw = RESULTS / "d1_baseline_predictions.csv"
    before = {_path: _sha256(_path) for _path in (source_manifest, source_summary, source_raw)}
    build_code_identity_sidecar(
        source_manifest,
        source_summary_path=source_summary,
        source_raw_path=source_raw,
        output_path=tmp_path / "derived.json",
        repo_root=ROOT,
    )
    assert verify_code_identity_sidecar(tmp_path / "derived.json", repo_root=ROOT)
    assert before == {_path: _sha256(_path) for _path in (source_manifest, source_summary, source_raw)}


def test_formal_scope_has_exact_120_rows_and_excludes_archive_rows() -> None:
    frame = _source_frame()
    formal = filter_formal_scope(frame)
    assert len(formal) == 120
    assert set(formal["method"]) == {"BL6_DLinear", "BL7_TCN"}
    assert set(formal["horizon"]) == set(FORMAL_HORIZONS)
    assert set(formal["seed"]) == set(FORMAL_SEEDS)
    assert set(zip(formal["dataset_id"], formal["target_entity_key"])) == set(FORMAL_TARGETS.items())
    assert len(frame) - len(formal) == 420


def test_promotion_status_is_derived_and_original_trial_is_preserved() -> None:
    formal = filter_formal_scope(_source_frame())
    candidate = promote_candidate(
        formal,
        audit_path="audit.md",
        code_identity_files={"D1": "d1.json"},
        promotion_timestamp_utc="2026-09-07T00:00:00+00:00",
    )
    assert set(candidate["result_status"]) == {"confirmed_baseline"}
    assert set(candidate["original_result_status"]) == {"trial"}
    assert set(candidate["promotion_mode"]) == {"derived_paper_scope"}
    keys = ["dataset_id", "target_entity_key", "method", "horizon", "seed"]
    left = candidate.set_index(keys).sort_index()
    right = formal.set_index(keys).sort_index()
    assert left["rmse"].tolist() == right["rmse"].tolist()
    assert left["smape"].tolist() == right["smape"].tolist()


def test_five_seed_aggregation_is_formal_only_and_complete() -> None:
    aggregate = aggregate_five_seed(filter_formal_scope(_source_frame()))
    assert len(aggregate) == 24
    assert set(aggregate["seed_count"]) == {5}
    assert set(aggregate["horizon"]) == set(FORMAL_HORIZONS)
    assert set(zip(aggregate["dataset_id"], aggregate["target_entity_key"])) == set(FORMAL_TARGETS.items())
    assert np.isfinite(aggregate[["rmse_mean", "rmse_std", "smape_mean", "smape_std"]].to_numpy()).all()


def test_promotion_inputs_fail_closed_for_missing_seed_and_horizon() -> None:
    frame = filter_formal_scope(_source_frame()).copy()
    frame = frame[~((frame["method"] == "BL6_DLinear") & (frame["horizon"] == 5) & (frame["seed"] == 46))]
    with pytest.raises(PromotionError, match="cell completeness"):
        validate_promotion_inputs(frame, raw_by_dataset={}, repo_root=ROOT, require_raw=False)


def test_promotion_inputs_fail_closed_for_missing_horizon() -> None:
    frame = filter_formal_scope(_source_frame())
    frame = frame[frame["horizon"] != 5]
    with pytest.raises(PromotionError, match="cell completeness"):
        validate_promotion_inputs(frame, raw_by_dataset={}, repo_root=ROOT, require_raw=False)


def test_promotion_inputs_fail_closed_for_duplicate_cell() -> None:
    frame = pd.concat([filter_formal_scope(_source_frame()), filter_formal_scope(_source_frame()).iloc[[0]]], ignore_index=True)
    with pytest.raises(PromotionError, match="cell completeness"):
        validate_promotion_inputs(frame, raw_by_dataset={}, repo_root=ROOT, require_raw=False)


def test_promotion_inputs_fail_closed_for_missing_raw_dataset() -> None:
    with pytest.raises(PromotionError, match="raw prediction completeness"):
        validate_promotion_inputs(filter_formal_scope(_source_frame()), raw_by_dataset={"D1": pd.DataFrame()}, repo_root=ROOT)


def test_promotion_inputs_fail_closed_for_metric_and_hash_defects() -> None:
    frame = filter_formal_scope(_source_frame()).copy()
    frame.loc[frame.index[0], "rmse"] = np.nan
    with pytest.raises(PromotionError, match="finite"):
        validate_promotion_inputs(frame, raw_by_dataset={}, repo_root=ROOT, require_raw=False)


def test_raw_metric_parity_fails_closed() -> None:
    raw_by_dataset = _formal_raw_by_dataset()
    raw_by_dataset["D1"] = raw_by_dataset["D1"].copy()
    raw_by_dataset["D1"].loc[raw_by_dataset["D1"].index[0], "y_pred"] += 1.0
    with pytest.raises(PromotionError, match="metric parity"):
        validate_raw_evidence(filter_formal_scope(_source_frame()), raw_by_dataset)


def test_formal_promotion_preserves_all_numeric_values_and_trial_status() -> None:
    source = filter_formal_scope(_source_frame())
    candidate = promote_candidate(source, audit_path="audit.md", code_identity_files={}, promotion_timestamp_utc="now")
    keys = ["dataset_id", "target_entity_key", "method", "horizon", "seed"]
    left = candidate.set_index(keys).sort_index()
    right = source.set_index(keys).sort_index()
    for column in ("rmse", "mae", "smape", "accuracy", "sample_count"):
        assert left[column].tolist() == right[column].tolist()
    assert set(candidate["original_result_status"]) == {"trial"}


def test_formal_scope_is_exactly_four_targets_three_horizons_five_seeds_two_methods() -> None:
    formal = filter_formal_scope(_source_frame())
    assert len(formal) == 4 * 3 * 5 * 2
    assert set(formal["dataset_id"]) == set(FORMAL_TARGETS)
    assert set(formal["horizon"]) == {1, 3, 5}
    assert set(formal["seed"]) == {42, 43, 44, 45, 46}
    assert set(formal["method"]) == {"BL6_DLinear", "BL7_TCN"}


def test_strict_loader_aliases_accept_new_paper_names_without_changing_old_names() -> None:
    assert norm_method("BL6_DLinear") == "DLinear"
    assert norm_method("BL7_TCN") == "TCN"
    assert {norm_method(name) for name in ("LSTM", "LightGBM", "Seasonal-Naive")} == {
        "LSTM", "LightGBM", "Seasonal-Naive"
    }
    assert {"DLinear", "TCN"}.issubset(EXPECTED_METHODS)


def test_strict_loader_candidate_validation_preserves_120_rows() -> None:
    from tools.build_paper_results_master_v1_2 import validate_external_baseline_rows

    candidate = promote_candidate(
        filter_formal_scope(_source_frame()),
        audit_path="audit.md",
        code_identity_files={"D1": "d1.json", "D4": "d4.json", "D5": "d5.json", "D6": "d6.json"},
        promotion_timestamp_utc="2026-09-07T00:00:00+00:00",
    )
    accepted = validate_external_baseline_rows(candidate)
    assert len(accepted) == 120
    assert set(accepted["method"]) == {"DLinear", "TCN"}
    assert accepted.duplicated(["dataset", "target", "method", "horizon", "seed"]).sum() == 0


def test_original_combined_workbook_sha256_is_unchanged() -> None:
    assert _sha256(INPUT_XLSX) == "64196d73e20c326ecd5ae2099431e2dd8bc3697ae67d7f7680c7d881c859ab7a"
