from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import SCHEMA_FAMILY_D1_D3, SCHEMA_FAMILY_D4_D6
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.utils.result_acceptance import (
    AcceptanceScope,
    AggregateProfile,
    ExpectedResultContract,
    ResultAcceptanceError,
    _read_csv,
    _same_candidate_content,
    accept_cell_csv,
    accept_global_aggregate,
    accept_mode_matrix,
    build_formal_cell_contract,
    require_accepted,
)
from test_strict_result_contract import _strict_row


def _cell_rows(
    *,
    dataset_id: int = 1,
    mode: str = "without",
    target: str = "1/10",
    horizon: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for method in FORMAL_METHODS:
        row = _strict_row(horizon=horizon, seed=seed)
        row.update(
            {
                "dataset_id": f"D{dataset_id}",
                "target_entity_key": target,
                "scenario": mode,
                "information_sharing": mode,
                "method": method,
                "schema_family": SCHEMA_FAMILY_D1_D3,
                "result_status": "trial",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _contract(
    *,
    scope: AcceptanceScope = AcceptanceScope.CELL,
    dataset_ids: tuple[int, ...] = (1,),
    modes: tuple[str, ...] = ("without",),
    targets: dict[tuple[int, str], tuple[str, ...]] | None = None,
    horizons: tuple[int, ...] = (1,),
    seeds: tuple[int, ...] = (42,),
    profile: AggregateProfile | None = None,
) -> ExpectedResultContract:
    return ExpectedResultContract(
        scope=scope,
        formal=True,
        dataset_ids=dataset_ids,
        modes=modes,
        protocol_tracks=("strict_paper",),
        targets_by_dataset_mode=targets
        or {(dataset_id, mode): ("1/10",) for dataset_id in dataset_ids for mode in modes},
        methods=FORMAL_METHODS,
        horizons=horizons,
        seeds=seeds,
        confirmation_eligible=True,
        aggregate_profile=profile,
    )


def _write(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


CSV_FLOAT64_SENTINEL = np.float64(99.55002834343927)
REAL_FLOAT64_REPRESENTATIVES = (
    1.8733843701179975,
    3988.5717485263062,
    228158973078.64197,
    2119980886335.1516,
    23067.708026457934,
    190.21684898242836,
    30.586227816995237,
)


def test_candidate_csv_reader_preserves_legal_float64_content(tmp_path: Path) -> None:
    frame = _cell_rows().copy()
    frame.loc[0, "rmse"] = CSV_FLOAT64_SENTINEL
    frame.loc[0, "smape"] = REAL_FLOAT64_REPRESENTATIVES[0]
    frame.loc[1, "rmse"] = REAL_FLOAT64_REPRESENTATIVES[1]
    frame.loc[1, "smape"] = REAL_FLOAT64_REPRESENTATIVES[2]
    frame.loc[2, "rmse"] = REAL_FLOAT64_REPRESENTATIVES[3]
    frame.loc[2, "smape"] = REAL_FLOAT64_REPRESENTATIVES[4]
    frame.loc[3, "rmse"] = REAL_FLOAT64_REPRESENTATIVES[5]
    frame.loc[3, "smape"] = REAL_FLOAT64_REPRESENTATIVES[6]
    path = _write(tmp_path / "candidate.csv", frame)

    reread, reasons = _read_csv(path)

    assert reasons == []
    assert np.array_equal(
        reread["rmse"].to_numpy(dtype=np.float64).view(np.int64),
        frame["rmse"].to_numpy(dtype=np.float64).view(np.int64),
    )
    assert _same_candidate_content(frame, reread)


def test_cell_acceptance_requires_exact_formal_coverage(tmp_path: Path) -> None:
    path = _write(tmp_path / "cell.csv", _cell_rows())

    outcome = accept_cell_csv(path, expected=_contract())

    assert outcome.report.passed
    assert outcome.report.reasons == ()
    assert outcome.report.counts["rows"] == 6
    assert set(outcome.accepted_rows["result_status"]) == {"trial"}

    built = build_formal_cell_contract(
        dataset_id=1,
        mode="without_information_sharing",
        targets=("1/10",),
        horizon=1,
        seed=42,
    )
    assert built == _contract()
    assert len(require_accepted(outcome)) == 6


def test_d4_runtime_audit_columns_are_accepted_as_registered_schema(
    tmp_path: Path,
) -> None:
    frame = _cell_rows(dataset_id=4, target="166/258").assign(
        schema_family=SCHEMA_FAMILY_D4_D6,
        source_pool_track="extended",
        domain_filter_applied_to_source=False,
        domain_filter_scope="target_only",
        domain_filter_column="first_category_id",
        domain_filter_value=15,
        target_domain_validation_passed=True,
        target_domain_validation_target_count=5,
        source_pool_policy="without_information_sharing_same_store",
    )
    path = _write(tmp_path / "d4_cell.csv", frame)

    outcome = accept_cell_csv(
        path,
        expected=_contract(
            dataset_ids=(4,),
            targets={(4, "without"): ("166/258",)},
        ),
    )

    assert outcome.report.passed
    assert outcome.report.reasons == ()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda frame: frame.iloc[0:0], "csv_empty"),
        (lambda frame: frame.assign(error="training failed", smape=float("nan")), "terminal_error"),
        (lambda frame: frame.iloc[:-1], "method_coverage_mismatch"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate_formal_key"),
        (lambda frame: frame.assign(smape=float("inf")), "nonfinite_primary_metric"),
        (lambda frame: frame.assign(horizon=2), "unexpected_horizon"),
        (lambda frame: frame.assign(result_status="confirmed_baseline"), "cell_status_must_be_trial"),
        (lambda frame: frame.assign(unregistered_extra="x"), "unregistered_columns"),
    ],
)
def test_cell_acceptance_fails_closed_with_stable_reason_codes(
    tmp_path: Path, mutation, reason: str
) -> None:
    path = _write(tmp_path / f"{reason}.csv", mutation(_cell_rows()))

    outcome = accept_cell_csv(path, expected=_contract())

    assert not outcome.report.passed
    assert reason in outcome.report.reasons
    if reason == "terminal_error":
        assert outcome.report.reasons[0] == "terminal_error"


def test_diagnostic_cell_is_never_confirmation_eligible(tmp_path: Path) -> None:
    path = _write(tmp_path / "diagnostic.csv", _cell_rows())
    expected = _contract()
    expected = ExpectedResultContract(
        **{**expected.__dict__, "formal": False, "confirmation_eligible": False}
    )

    outcome = accept_cell_csv(path, expected=expected)

    assert not outcome.report.passed
    assert "confirmation_ineligible" in outcome.report.reasons
    with pytest.raises(ResultAcceptanceError, match="confirmation_ineligible"):
        require_accepted(outcome)


def test_mode_matrix_requires_25_accepted_cells_and_is_only_promoter(tmp_path: Path) -> None:
    paths = []
    for horizon in range(1, 6):
        for seed in range(42, 47):
            paths.append(
                _write(
                    tmp_path / f"h{horizon}_s{seed}.csv",
                    _cell_rows(horizon=horizon, seed=seed),
                )
            )
    expected = _contract(
        scope=AcceptanceScope.MODE_MATRIX,
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
    )

    outcome = accept_mode_matrix(paths, expected=expected)

    assert outcome.report.passed
    assert outcome.report.counts["cells"] == 25
    assert len(outcome.accepted_rows) == 150
    assert set(outcome.accepted_rows["result_status"]) == {"confirmed_baseline"}

    incomplete = accept_mode_matrix(paths[:-1], expected=expected)
    assert not incomplete.report.passed
    assert "cell_matrix_mismatch" in incomplete.report.reasons


def _with_representative_floats(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched.loc[0, "rmse"] = CSV_FLOAT64_SENTINEL
    enriched.loc[0, "smape"] = REAL_FLOAT64_REPRESENTATIVES[0]
    enriched.loc[1, "rmse"] = REAL_FLOAT64_REPRESENTATIVES[1]
    enriched.loc[1, "smape"] = REAL_FLOAT64_REPRESENTATIVES[2]
    enriched.loc[2, "rmse"] = REAL_FLOAT64_REPRESENTATIVES[3]
    enriched.loc[2, "smape"] = REAL_FLOAT64_REPRESENTATIVES[4]
    enriched.loc[3, "rmse"] = REAL_FLOAT64_REPRESENTATIVES[5]
    enriched.loc[3, "smape"] = REAL_FLOAT64_REPRESENTATIVES[6]
    return enriched


def _mode_matrix_candidate_fixture(
    tmp_path: Path,
) -> tuple[list[Path], ExpectedResultContract, pd.DataFrame, Path]:
    paths = []
    for horizon in range(1, 6):
        for seed in range(42, 47):
            frame = _cell_rows(horizon=horizon, seed=seed)
            if (horizon, seed) == (1, 42):
                frame = _with_representative_floats(frame)
            paths.append(_write(tmp_path / f"h{horizon}_s{seed}.csv", frame))
    expected = _contract(
        scope=AcceptanceScope.MODE_MATRIX,
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
    )
    accepted = accept_mode_matrix(paths, expected=expected)
    assert accepted.report.passed
    candidate = _write(tmp_path / "candidate_mode.csv", accepted.accepted_rows)
    return paths, expected, accepted.accepted_rows, candidate


def test_mode_matrix_candidate_accepts_legal_float64_round_trip(tmp_path: Path) -> None:
    paths, expected, accepted_rows, candidate = _mode_matrix_candidate_fixture(tmp_path)

    outcome = accept_mode_matrix(
        paths,
        expected=expected,
        candidate_mode_csv=candidate,
    )

    assert outcome.report.passed
    assert outcome.report.reasons == ()
    assert outcome.report.counts == {"cells": 25, "rows": 150}
    assert len(accepted_rows) == 150


def test_mode_matrix_candidate_rejects_material_numeric_mutation(tmp_path: Path) -> None:
    paths, expected, accepted_rows, _ = _mode_matrix_candidate_fixture(tmp_path)
    candidate_frame = accepted_rows.copy()
    candidate_frame.loc[0, "smape"] = float(candidate_frame.loc[0, "smape"]) + 0.01
    candidate = _write(tmp_path / "candidate_mode_smape_mutation.csv", candidate_frame)

    outcome = accept_mode_matrix(
        paths,
        expected=expected,
        candidate_mode_csv=candidate,
    )

    assert not outcome.report.passed
    assert outcome.report.reasons == ("candidate_mode_mismatch",)


def test_mode_matrix_candidate_rejects_one_float64_unit_mutation(tmp_path: Path) -> None:
    paths, expected, accepted_rows, _ = _mode_matrix_candidate_fixture(tmp_path)
    candidate_frame = accepted_rows.copy()
    original = np.float64(candidate_frame.loc[0, "smape"])
    mutated = np.nextafter(original, np.inf)
    assert mutated != original
    candidate_frame.loc[0, "smape"] = mutated
    candidate = _write(tmp_path / "candidate_mode_nextafter_mutation.csv", candidate_frame)

    outcome = accept_mode_matrix(
        paths,
        expected=expected,
        candidate_mode_csv=candidate,
    )

    assert not outcome.report.passed
    assert outcome.report.reasons == ("candidate_mode_mismatch",)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("scenario", "with"),
        ("information_sharing", "with"),
        ("method", "changed-method"),
        ("target_entity_key", "changed-target"),
        ("horizon", 2),
        ("seed", 99),
        ("result_status", "trial"),
        ("schema_family", "changed-schema-family"),
    ],
)
def test_mode_matrix_candidate_rejects_identity_or_schema_mutation(
    tmp_path: Path, column: str, value: object
) -> None:
    paths, expected, accepted_rows, _ = _mode_matrix_candidate_fixture(tmp_path)
    candidate_frame = accepted_rows.copy()
    candidate_frame.loc[0, column] = value
    candidate = _write(tmp_path / f"candidate_{column}_mutation.csv", candidate_frame)

    outcome = accept_mode_matrix(
        paths,
        expected=expected,
        candidate_mode_csv=candidate,
    )

    assert not outcome.report.passed
    assert "candidate_mode_mismatch" in outcome.report.reasons


FULL_TARGETS = {
    (dataset_id, mode): tuple(
        f"D{dataset_id}-T{index}"
        for index in range(1, (1 if dataset_id <= 3 else 5) + 1)
    )
    for dataset_id in range(1, 7)
    for mode in ("without", "with")
}


def _confirmed_mode_rows(dataset_id: int, mode: str, targets: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for method in FORMAL_METHODS:
            for horizon in range(1, 6):
                for seed in range(42, 47):
                    rows.append(
                        {
                            "dataset_id": f"D{dataset_id}",
                            "protocol_track": "strict_paper",
                            "scenario": mode,
                            "information_sharing": mode,
                            "target_entity_key": target,
                            "method": method,
                            "horizon": horizon,
                            "seed": seed,
                            "result_status": "confirmed_baseline",
                            "schema_family": SCHEMA_FAMILY_D1_D3,
                            "rmse": 1.0,
                            "smape": 1.0,
                            "error": "",
                        }
                    )
    return pd.DataFrame(rows)


def _selection_aggregate_candidate_fixture(
    tmp_path: Path,
) -> tuple[list[Path], ExpectedResultContract, pd.DataFrame, Path]:
    d5_targets = FULL_TARGETS[(5, "without")]
    mode_paths = []
    for mode in ("without", "with"):
        frame = _confirmed_mode_rows(5, mode, d5_targets)
        if mode == "without":
            frame = _with_representative_floats(frame)
        mode_paths.append(_write(tmp_path / f"d5_{mode}.csv", frame))
    expected = _contract(
        scope=AcceptanceScope.GLOBAL_AGGREGATE,
        dataset_ids=(5,),
        modes=("without", "with"),
        targets={(5, mode): d5_targets for mode in ("without", "with")},
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
        profile=AggregateProfile.RUN_SELECTION_AGGREGATE,
    )
    accepted = accept_global_aggregate(mode_paths, expected=expected)
    assert accepted.report.passed
    candidate = _write(tmp_path / "candidate_aggregate.csv", accepted.accepted_rows)
    return mode_paths, expected, accepted.accepted_rows, candidate


def test_global_aggregate_candidate_accepts_and_rejects_float64_content(
    tmp_path: Path,
) -> None:
    mode_paths, expected, accepted_rows, candidate = _selection_aggregate_candidate_fixture(
        tmp_path
    )

    accepted = accept_global_aggregate(
        mode_paths,
        expected=expected,
        candidate_aggregate_csv=candidate,
    )

    assert accepted.report.passed
    assert accepted.report.reasons == ()
    assert accepted.report.counts == {
        "mode_files": 2,
        "rows": 1500,
        "dataset_mode_groups": 2,
    }

    mutated_frame = accepted_rows.copy()
    mutated_frame.loc[0, "smape"] = float(mutated_frame.loc[0, "smape"]) + 0.01
    mutated_candidate = _write(
        tmp_path / "candidate_aggregate_smape_mutation.csv",
        mutated_frame,
    )
    rejected = accept_global_aggregate(
        mode_paths,
        expected=expected,
        candidate_aggregate_csv=mutated_candidate,
    )

    assert not rejected.report.passed
    assert rejected.report.reasons == ("candidate_aggregate_mismatch",)


def test_global_profiles_distinguish_selection_from_full_baseline(tmp_path: Path) -> None:
    d5_targets = FULL_TARGETS[(5, "without")]
    subset_paths = [
        _write(
            tmp_path / f"d5_{mode}.csv",
            _confirmed_mode_rows(5, mode, d5_targets),
        )
        for mode in ("without", "with")
    ]
    selection = _contract(
        scope=AcceptanceScope.GLOBAL_AGGREGATE,
        dataset_ids=(5,),
        modes=("without", "with"),
        targets={(5, mode): d5_targets for mode in ("without", "with")},
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
        profile=AggregateProfile.RUN_SELECTION_AGGREGATE,
    )

    selected = accept_global_aggregate(subset_paths, expected=selection)

    assert selected.report.passed
    assert len(selected.accepted_rows) == 1500

    full_contract = _contract(
        scope=AcceptanceScope.GLOBAL_AGGREGATE,
        dataset_ids=tuple(range(1, 7)),
        modes=("without", "with"),
        targets=FULL_TARGETS,
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
        profile=AggregateProfile.FULL_D1_D6_BASELINE,
    )
    subset_as_full = accept_global_aggregate(subset_paths, expected=full_contract)
    assert not subset_as_full.report.passed
    assert "dataset_mode_coverage_mismatch" in subset_as_full.report.reasons


def test_full_global_profile_accepts_exactly_12_groups_and_5400_rows(tmp_path: Path) -> None:
    paths = []
    for (dataset_id, mode), targets in FULL_TARGETS.items():
        paths.append(
            _write(
                tmp_path / f"d{dataset_id}_{mode}.csv",
                _confirmed_mode_rows(dataset_id, mode, targets),
            )
        )
    expected = _contract(
        scope=AcceptanceScope.GLOBAL_AGGREGATE,
        dataset_ids=tuple(range(1, 7)),
        modes=("without", "with"),
        targets=FULL_TARGETS,
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
        profile=AggregateProfile.FULL_D1_D6_BASELINE,
    )

    outcome = accept_global_aggregate(paths, expected=expected)

    assert outcome.report.passed
    assert outcome.report.counts["dataset_mode_groups"] == 12
    assert outcome.report.counts["rows"] == 5400
