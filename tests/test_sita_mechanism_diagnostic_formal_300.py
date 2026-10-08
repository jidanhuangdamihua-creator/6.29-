from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
    CELL_JSON_ARTIFACT_NAMES,
    error_by_sales_bin,
)
from src.transfer_methods import sita_mechanism_diagnostic_rerun as rerun


def test_formal_300_matrix_is_exact_frozen_universe():
    cells = rerun.enumerate_formal_300_cells()
    assert len(cells) == 300
    assert len({cell.canonical_cell_key for cell in cells}) == 60
    assert all(
        {item.arm for item in cells if item.canonical_cell_key == key} == set("ABCDE")
        for key in {cell.canonical_cell_key for cell in cells}
    )
    assert len({cell.condition_key for cell in cells}) == 12
    assert {cell.arm for cell in cells} == {"A", "B", "C", "D", "E"}
    assert {int(cell.seed) for cell in cells} == {42, 43, 44, 45, 46}


def test_default_matrix_remains_diagnostic_42():
    assert len(rerun.enumerate_mechanism_diagnostic_rerun_cells()) == 42
    assert rerun.DEFAULT_MATRIX_MODE == "diagnostic-42"


def test_formal_only_cell_can_resolve_arm_e():
    cell = rerun.resolve_diagnostic_only_cell(
        "D1-H3-seed46-E",
        matrix_mode="formal-300",
    )
    assert (cell.dataset, cell.horizon, cell.seed, cell.arm) == ("D1", 3, 46, "E")


def test_live_formal_arm_contract_is_A_to_E_and_B_is_frozen():
    contract = rerun.formal_arm_contract()
    assert set(contract) == set("ABCDE")
    assert contract["B"]["init_mode"] == "fused"
    assert contract["B"]["target_backbone_trainable"] is False
    assert contract["C"]["target_backbone_trainable"] is True
    assert contract["D"]["revin_enabled"] is True
    assert contract["E"]["init_mode"] == "best_source"
    assert all(not item["transfer_branch_enabled"] for item in contract.values())


def test_sales_bin_rows_keep_all_identifiers():
    rows = [
        {
            "dataset": "D1",
            "target": "1/10",
            "horizon": 1,
            "seed": 42,
            "arm": "B",
            "y_true": float(i + 1),
            "absolute_error": 0.1,
            "squared_error": 0.01,
        }
        for i in range(6)
    ]
    output = error_by_sales_bin(rows)
    assert output
    assert all(
        row[field] not in (None, "")
        for row in output
        for field in ("dataset", "target", "horizon", "seed", "arm", "bin")
    )


def test_required_cell_outputs_include_result_status_and_provenance():
    assert set(CELL_JSON_ARTIFACT_NAMES) >= {"result.json", "status.json", "provenance.json"}


def test_formal_only_cell_subset_keeps_exact_reference_key_audit(tmp_path):
    from src.transfer_methods.sita_mechanism_diagnostic_execution import (
        FakeDiagnosticCellExecutor,
        run_mechanism_diagnostic_cell_loop,
    )
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import prepare_run_root

    cell = rerun.resolve_diagnostic_only_cell("D1-H3-seed46-E", matrix_mode="formal-300")
    prepare_run_root(
        tmp_path,
        config_digest="cfg",
        matrix_digest="mat",
        code_digest="code",
        resume=False,
        matrix_mode="formal-300",
    )
    result = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=(cell,),
        executor=FakeDiagnosticCellExecutor(),
        config_digest="cfg",
        matrix_digest="mat",
        code_digest="code",
        allow_cell_subset=True,
        matrix_mode="formal-300",
    )
    assert result["n_cells"] == 1
    assert result["n_failed"] == 0


def test_formal_300_fake_run_is_independent_and_historical_mismatch_is_nonblocking(tmp_path):
    from src.transfer_methods.sita_mechanism_diagnostic_execution import (
        FakeDiagnosticCellExecutor,
        run_mechanism_diagnostic_cell_loop,
        mechanism_rerun_code_digest,
    )
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        prepare_run_root,
        required_cell_outputs_complete,
    )

    cells = rerun.enumerate_formal_300_cells()
    cfg = rerun.mechanism_rerun_config_digest(cells, matrix_mode="formal-300")
    mat = rerun.matrix_digest_for_cells(cells)
    code = mechanism_rerun_code_digest()
    prepare_run_root(
        tmp_path,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
        resume=False,
        matrix_mode="formal-300",
    )
    result = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=cells,
        executor=FakeDiagnosticCellExecutor(),
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
        matrix_mode="formal-300",
    )
    assert result["n_cells"] == 300
    assert result["n_canonical_cells"] == 60
    assert result["n_ran"] == 300
    assert result["n_failed"] == 0
    assert result["finalized"] == "YES"
    assert len(list((tmp_path / "stage_a").glob("*.json"))) == 60
    assert all(required_cell_outputs_complete(tmp_path, cell.cell_key) for cell in cells)
    manifest = __import__("json").loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["historical_frozen_reproduction_required"] is False
    assert manifest["historical_trend_match_required"] is False
    assert manifest["all_finalized"] is True
    assert (tmp_path / "summaries" / "formal_consistency.csv").is_file()
    assert not (tmp_path / "summaries" / "historical_frozen_comparison.csv").exists()
    assert (tmp_path / "summaries" / "parameter_drift_raw.csv").is_file()
    assert (tmp_path / "summaries" / "parameter_drift_ema.csv").is_file()
    assert (tmp_path / "summaries" / "cka_flatten.csv").is_file()
    assert (tmp_path / "summaries" / "cka_conv1.csv").is_file()
    assert (tmp_path / "summaries" / "cka_conv2.csv").is_file()
    assert (tmp_path / "summaries" / "cka_conv3.csv").is_file()
    assert (tmp_path / "summaries" / "pointwise_summary.csv").is_file()
    assert manifest.get("formal_reference_result", {}).get("authority_kind") == "FORMAL_PER_DATASET_LR_AE"
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import filesystem_safe_id

    first = cells[0]
    ckpt = __import__("json").loads(
        (tmp_path / "checkpoints" / f"{filesystem_safe_id(first.cell_key)}.json").read_text()
    )
    init = ckpt["checkpoints"][f"{first.arm}_INIT"]
    raw = ckpt["checkpoints"][f"{first.arm}_RAW_FINAL"]
    final = ckpt["checkpoints"][f"{first.arm}_FINAL"]
    assert init["weight_variant"] == "INITIAL"
    assert init["optimizer_update_count"] == 0
    assert raw["optimizer_update_count"] == final["optimizer_update_count"]
    assert raw["optimizer_update_count"] > 0
    for layer in ("flatten", "conv1", "conv2", "conv3"):
        assert (tmp_path / "representations" / layer / f"{filesystem_safe_id(first.cell_key)}.npz").is_file()


def test_formal_300_plan_only_prints_authority_and_capture_flags():
    report = rerun.plan_mechanism_diagnostic_rerun(matrix_mode="formal-300")
    text = rerun.format_plan(report)
    assert report["execute_ready"] is True
    assert report["formal_reference"]["authority_kind"] == "FORMAL_PER_DATASET_LR_AE"
    assert report["target_learning_rates"]["D1"]["label"] == "3e-5"
    assert report["target_learning_rates"]["D5"]["label"] == "3e-4"
    assert "MATRIX=300" in text
    assert "CANONICAL_STAGE_A_UNITS=60" in text
    assert "D1 LR = 3e-5" in text
    assert "D5 LR = 3e-4" in text
    assert "CKA_CONV3_CAPTURE_READY=YES" in text
    assert "RAW_DRIFT_READY=YES" in text
    assert "EMA_DRIFT_READY=YES" in text
    assert "FORMAL_CONSISTENCY_READY=YES" in text
    assert "OUTPUT_ISOLATION_READY=YES" in text
