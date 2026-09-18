"""Synthetic, no-training tests for corrected A-E runtime hardening."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.ms_katl_ae_formal import (
    build_formal_run_provenance,
    finalize_formal_cell_attempt,
    prepare_formal_cell_attempt,
    validate_formal_resume_provenance,
)
from src.transfer_methods.ms_katl_sita_ablation_3e4 import (
    select_uniform_3e4_cell,
)


def _cell():
    return SimpleNamespace(
        dataset="D1",
        target="1/10",
        horizon=1,
        seed=42,
        scenario="without",
        stage_a_key="D1|1/10|h1|s42|without",
        selected_lr=3e-4,
        selected_lr_label="3e-4",
    )


def _digests():
    return {
        "matrix_digest": "matrix",
        "protocol_digest": "protocol",
        "critical_runtime_digest": "runtime",
        "working_tree_protocol_digest": "working",
    }


def test_failed_attempt_is_preserved_and_next_attempt_is_allocated(tmp_path):
    first_action, first = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=False, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert first_action == "RUN"
    assert first is not None
    finalize_formal_cell_attempt(first, status="FAILED", error=RuntimeError("synthetic"))

    second_action, second = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=True, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert second_action == "RUN"
    assert second is not None and second.name == "attempt_002"
    assert json.loads((first / "status.json").read_text())["status"] == "FAILED"
    assert (first / "result.json").is_file()
    assert (first / "provenance.json").is_file()


def test_interrupted_attempt_is_preserved_and_complete_is_skipped(tmp_path):
    _, first = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=False, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert first is not None
    status_path = first / "status.json"
    status = json.loads(status_path.read_text())
    status["pid"] = 99999999
    status_path.write_text(json.dumps(status), encoding="utf-8")

    _, second = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=True, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert json.loads(status_path.read_text())["status"] == "INTERRUPTED"
    assert second is not None and second.name == "attempt_002"
    finalize_formal_cell_attempt(second, status="COMPLETE", result={"status": "COMPLETE"})
    action, skipped = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=True, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert action == "SKIPPED_COMPLETE"
    assert skipped == second


@pytest.mark.parametrize("field", ["matrix_digest", "protocol_digest", "critical_runtime_digest"])
def test_resume_digest_mismatch_is_refused(tmp_path, field):
    root = tmp_path / "root"
    root.mkdir()
    payload = {**_digests(), "matrix_kind": "UNIFORM_3E4_PAPER_5SEED", "expected_runs": 300}
    (root / "manifest_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")
    changed = dict(payload, **{field: "different"})
    with pytest.raises(Exception, match="digest mismatch"):
        validate_formal_resume_provenance(root, changed)


def test_root_provenance_contains_required_hardening_fields():
    payload = build_formal_run_provenance(
        PROJECT_ROOT,
        matrix_digest="matrix",
        matrix_kind="UNIFORM_3E4_PAPER_5SEED",
        expected_runs=300,
        cells=[_cell()],
    )
    for key in (
        "paper_method_name",
        "ablation_protocol_name",
        "branch",
        "HEAD",
        "runtime_commit",
        "matrix_digest",
        "protocol_digest",
        "critical_runtime_digest",
        "working_tree_protocol_digest",
        "dataset_target_map",
        "stage_a_rng_contract",
        "historical_source_weight_reuse",
        "ARM_D_IS_FORMAL_MAIN_SITA",
        "ARM_D_REASON",
    ):
        assert key in payload


def test_single_cell_selector_uses_formal_target_mapping():
    selected = select_uniform_3e4_cell(dataset="D4", horizon=1, seed=42, arm="D")
    assert selected["target"] == "166/258"
    assert selected["arm"] == "D"
    assert selected["stage_a_key"] == "D4|166/258|h1|s42|without"

    with pytest.raises(Exception, match="D4 target must be 166/258"):
        select_uniform_3e4_cell(dataset="D4", horizon=1, seed=42, arm="D", target="166/311")
