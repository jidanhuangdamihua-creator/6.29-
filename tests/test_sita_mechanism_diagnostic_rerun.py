"""Plan-only tests for the 42-cell mechanism diagnostic rerun matrix."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.sita_mechanism_diagnostic_rerun import (
    EXPECTED_DIAGNOSTIC_RERUN_CELLS,
    enumerate_mechanism_diagnostic_rerun_cells,
    expected_matrix_group_counts,
    format_plan,
    matrix_group_counts,
    plan_mechanism_diagnostic_rerun,
    resolve_diagnostic_only_cell,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import MechanismDiagnosticsContractError


def test_diagnostic_rerun_matrix_is_exactly_42():
    cells = enumerate_mechanism_diagnostic_rerun_cells()
    assert len(cells) == EXPECTED_DIAGNOSTIC_RERUN_CELLS == 42
    assert matrix_group_counts(cells) == expected_matrix_group_counts()
    assert len({cell.cell_key for cell in cells}) == 42


def test_resolve_diagnostic_only_cell_from_42_universe():
    cells = enumerate_mechanism_diagnostic_rerun_cells()
    smoke = resolve_diagnostic_only_cell("D1-H1-seed42-B", cells)
    assert smoke.cell_key == "D1|1/10|h1|s42|B"
    assert resolve_diagnostic_only_cell(smoke.cell_key, cells) == smoke
    assert len(enumerate_mechanism_diagnostic_rerun_cells()) == 42
    try:
        resolve_diagnostic_only_cell("D2-H1-seed42-B", cells)
        raise AssertionError("expected unknown-cell failure")
    except MechanismDiagnosticsContractError as exc:
        assert "not in the 42-cell diagnostic matrix" in str(exc)


def test_diagnostic_rerun_pilot_and_d6_slice():
    cells = enumerate_mechanism_diagnostic_rerun_cells()
    d1 = [c for c in cells if c.dataset == "D1"]
    assert {(c.horizon, c.seed, c.arm) for c in d1} == {(1, 42, "A"), (1, 42, "B"), (1, 42, "C")}
    assert {c.target for c in d1} == {"1/10"}
    d6_a = [c for c in cells if c.dataset == "D6" and c.arm == "A"]
    d6_bc = [c for c in cells if c.dataset == "D6" and c.arm in {"B", "C"}]
    assert {c.seed for c in d6_a} == {42}
    assert {c.horizon for c in d6_a} == {1, 3, 5}
    assert {c.seed for c in d6_bc} == {42, 43, 44, 45, 46}
    assert {c.target for c in d6_bc} == {"CA_1/FOODS_3_080"}


def test_plan_only_report_prints_readiness_fields():
    report = plan_mechanism_diagnostic_rerun()
    text = format_plan(report)
    assert report["matrix_match"] is True
    assert report["group_counts"]["TOTAL"] == 42
    for field in (
        "Resolved cells",
        "Formal available seeds",
        "Diagnostic selected seeds",
        "EMA semantics",
        "PRIMARY weight variant",
        "PRIMARY CKA layer",
        "Supplementary CKA layers",
        "C_INIT_B_FINAL comparability",
        "reference sample source/count/digest",
        "training-dynamics hooks",
        "prediction hooks",
        "resume/output root",
    ):
        assert field in text
    assert "EXECUTE_READY=" in text
    assert "FINAL_EXECUTE_READY=" in text
    assert "DIAGNOSTICS_DISABLED_FORMAL_INTERFACE_UNCHANGED=" in text
    assert "outputs/mechanism_diagnostics/mechanism_rerun_YYYYMMDD_HHMMSS" in text
    assert "rerun_42cell" not in text
    assert "frozen A-E root untouched = YES" in text
    assert "MATRIX_MATCH=" in text
    assert "EMA_SEMANTICS_READY=" in text
    assert "REFERENCE_INPUTS_READY=" in text
    assert "Formal seeds D1/D4/D5" not in text
    assert "D1/D4/D5 = [42]" not in text


def test_formal_available_seeds_are_parsed_not_hardcoded_subset():
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        assert_diagnostic_seeds_are_formal_subset,
        assert_formal_universe_schema,
        diagnostic_selected_seeds,
        parse_formal_ae_uniform_universe,
    )

    universe = parse_formal_ae_uniform_universe()
    assert_formal_universe_schema(universe)
    assert universe["n_rows"] == 300
    assert universe["n_unique_cells"] == 300
    assert universe["extra_or_duplicate_rows"] is False
    assert universe["n_conditions"] == 12
    assert universe["seed_count_per_condition_arm_min"] == 5
    assert universe["seed_count_per_condition_arm_max"] == 5
    cells = enumerate_mechanism_diagnostic_rerun_cells()
    selected = diagnostic_selected_seeds(cells)
    assert_diagnostic_seeds_are_formal_subset(universe, selected)
    report = plan_mechanism_diagnostic_rerun()
    assert report["formal_available_seeds"]["D1"] == universe["seeds_by_dataset"]["D1"]
    assert report["diagnostic_selected_seeds"]["D1 A/B/C"] != universe["seeds_by_dataset"]["D1"]
    assert report["consistency_audit"]["lookup_scope"] == "diagnostic_selected_cells_only"
    assert report["consistency_audit"]["formal_seed_universe_recorded"] is True
