from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.protocols.gate1_transformation import (
    AuthorityProducer,
    AvailabilityResolver,
    FormalInputLoader,
    FormalPreflight,
    ForecastBlindProducer,
    Gate1Failure,
    HistoryReconstructionProducer,
    ProofWriter,
    SafeTargetViewOperator,
    SchemaRegistry,
    SourcePoolOperator,
    UnifiedRunner,
    build_d5_holiday,
    build_d6_calendar_view,
    join_d6_sell_price,
)


def _rows(dates: list[str], **columns: object) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    for name, value in columns.items():
        if isinstance(value, (list, tuple)):
            frame[name] = value
        else:
            frame[name] = [value] * len(frame)
    return frame


def _raw_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_TC_001_authority_precedence_and_temp_path_rejection(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    loader = FormalInputLoader(root)
    formal_paths = (
        "docs/protocol/gate1_frozen_transformation_contract.md",
        "docs/protocol/gate1_implementation_scope.md",
        "docs/protocol/gate1_contract_traceability_matrix.md",
    )
    assert tuple(path for path in formal_paths if (root / path).is_file()) == formal_paths
    contract = root / formal_paths[0]
    actual_digest = hashlib.sha256(contract.read_bytes()).hexdigest()
    sidecar = (root / "docs/protocol/gate1_frozen_transformation_contract.sha256").read_text(encoding="utf-8")
    assert f"contract_digest: sha256:{actual_digest}" in sidecar
    assert "contract_version: 1R.1.0" in sidecar
    with pytest.raises(Gate1Failure, match="AUTHORITY_PATH"):
        loader.load([tmp_path / "gate1_frozen_transformation_contract.md"])


def _formal_doc_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        "contract": (root / "docs/protocol/gate1_frozen_transformation_contract.md").read_text(encoding="utf-8"),
        "scope": (root / "docs/protocol/gate1_implementation_scope.md").read_text(encoding="utf-8"),
        "matrix": (root / "docs/protocol/gate1_contract_traceability_matrix.md").read_text(encoding="utf-8"),
    }


def test_gate1r_formal_docs_have_unique_global_and_dataset_sections() -> None:
    contract = _formal_doc_texts()["contract"]
    for decision_id in (f"G{index:02d}" for index in range(1, 17)):
        assert len(re.findall(rf"^### {decision_id} —", contract, flags=re.MULTILINE)) == 1
    for dataset in ("D1", "D2", "D3", "D4", "D5", "D6"):
        assert len(re.findall(rf"^### {dataset} —", contract, flags=re.MULTILINE)) == 1


def test_gate1r_each_dataset_chapter_has_the_required_contract_fields() -> None:
    contract = _formal_doc_texts()["contract"]
    labels = (
        "Raw authority",
        "Main backtest authority",
        "Extra benchmark/holdout role",
        "Forecast origin",
        "Origin authority",
        "Origin role",
        "Source history interval",
        "Target train interval",
        "Validation interval",
        "Blind interval",
        "Target canonical keys",
        "Source eligibility",
        "KNN window and fields",
        "Expected cardinality",
        "Calendar/missing rules",
        "Field-specific availability",
        "Worker-safe fields",
        "Evaluator truth",
        "Audit-only fields",
        "Forbidden fields",
        "Canonical digest objects",
        "Fail-closed conditions",
        "Unique executable conclusion",
    )
    headings = list(re.finditer(r"^### D[1-6] —", contract, flags=re.MULTILINE))
    assert len(headings) == 6
    for index, match in enumerate(headings):
        section = contract[match.start() : headings[index + 1].start() if index + 1 < len(headings) else len(contract)]
        for label in labels:
            assert f"**{label}.**" in section, label


def test_gate1r_final_human_overrides_are_affirmative_and_testable() -> None:
    contract = _formal_doc_texts()["contract"]
    assert "D2 and D5 missing-day rules below close their observed 175/180 and 885/900 gaps." in contract
    assert "sales=0" in contract and "rebuild the approved covariates" in contract
    assert "48/1159415" in contract
    assert "forecast PROMO is unavailable and must be excluded" in contract
    assert "transactions never enter worker/KNN/model" in contract
    assert "history-only forward fill followed by lag-one" in contract
    assert "week` is deleted" in contract
    assert "`weekday` is the original `calendar.csv` string/object and `wday` is the original `calendar.csv` integer" in contract
    assert "Order is `weekday`, `wday`, `wm_yr_wk`." in contract
    assert "sell_price` joins exactly on `(store_id,item_id,wm_yr_wk)`" in contract
    assert "`entity_id` is display/serialization metadata and cannot replace the composite source key." in contract


def test_gate1r_formal_docs_have_no_unresolved_markers() -> None:
    markers = (
        "待确认",
        "等待确认",
        "按绿色方案",
        "是否保留",
        "候选方案",
        "TBD",
        "TODO",
        "大致",
        "视情况",
        "按现有逻辑",
        "后续决定",
    )
    for name, text in _formal_doc_texts().items():
        for marker in markers:
            assert marker not in text, f"{name}: {marker}"


def test_gate1r_traceability_is_closed() -> None:
    docs = _formal_doc_texts()
    matrix = docs["matrix"]
    required_columns = (
        "Decision ID",
        "Dataset",
        "Contract clause",
        "Human decision source",
        "Implementation component",
        "Expected artifact/view",
        "Proof/digest",
        "Acceptance test",
        "Real-input readiness check",
        "Publication gate",
        "Failure code",
        "Status",
    )
    header = next(line for line in matrix.splitlines() if line.startswith("| Decision ID |"))
    assert all(column in header for column in required_columns)
    decision_ids = [f"G{index:02d}" for index in range(1, 17)]
    decision_ids += [
        *(f"D1-{index:02d}" for index in range(1, 8)),
        *(f"D2-{index:02d}" for index in range(1, 8)),
        *(f"D3-{index:02d}" for index in range(1, 8)),
        *(f"D4-{index:02d}" for index in range(1, 7)),
        *(f"D5-{index:02d}" for index in range(1, 11)),
        *(f"D6-{index:02d}" for index in range(1, 8)),
        "P-01",
        "P-02",
        "P-03",
    ]
    for decision_id in decision_ids:
        assert f"| {decision_id} |" in matrix, decision_id
    rows = [
        line for line in matrix.splitlines()
        if line.startswith("| ") and not line.startswith("|---") and not line.startswith("| Decision ID")
    ]
    assert rows
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 12, row
        assert all(cells), row
    assert "same as above" not in matrix
    assert "现有逻辑" not in matrix
    assert "后续补充" not in matrix


def test_gate1r_sidecar_record_and_combined_identity_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = _formal_doc_texts()
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in docs.items()}
    sidecar = (root / "docs/protocol/gate1_frozen_transformation_contract.sha256").read_text(encoding="utf-8")
    record = (root / "docs/protocol/gate1r_contract_refreeze_record.md").read_text(encoding="utf-8")
    decision_book_hash = "4aaebe5f07d3dc0e61ada72dbe0625c82615ba74577e91b72b46b07a709c689d"
    combined_payload = (
        f"decision_book_sha256={decision_book_hash}\n"
        f"contract_sha256={hashes['contract']}\n"
        f"scope_sha256={hashes['scope']}\n"
        f"matrix_sha256={hashes['matrix']}\n"
    ).encode("utf-8")
    combined = hashlib.sha256(combined_payload).hexdigest()
    assert f"contract_digest: sha256:{hashes['contract']}" in sidecar
    assert f"scope_sha256: sha256:{hashes['scope']}" in sidecar
    assert f"matrix_sha256: sha256:{hashes['matrix']}" in sidecar
    assert f"combined_formal_identity_digest: sha256:{combined}" in sidecar
    assert f"Contract SHA-256: `sha256:{hashes['contract']}`" in record
    assert f"Scope SHA-256: `sha256:{hashes['scope']}`" in record
    assert f"Matrix SHA-256: `sha256:{hashes['matrix']}`" in record
    assert f"Combined formal identity digest: `sha256:{combined}`" in record
    assert "old_contract_status: SUPERSEDED" in sidecar
    assert "Old contract status: `SUPERSEDED`" in record
    assert "all green-highlighted entries are approved final decisions" in record
    assert "Freeze commit: `resolved by the commit containing this record`" in record


def test_gate1r_scope_contains_the_complete_one_time_implementation() -> None:
    scope = _formal_doc_texts()["scope"]
    for index in range(1, 23):
        assert f"| I{index:02d} |" in scope
    assert "one implementation and one commit" in scope
    assert "D2 175/180" in scope
    assert "D5 885/900" in scope
    assert "D6's dual weekday/wday schema" in scope
    assert "tools/operations/materialize_d1_d6_sealed_authority.py" in scope
    assert "scripts/adopt_and_seal_d3_d6.py" in scope
    assert "src/protocols/gate1_transformation.py" in scope


def test_TC_002_history_forecast_producer_isolation() -> None:
    history = _rows(["2020-01-01"], sales=[3.0], Open=[None])
    forecast = _rows(["2020-01-02"], sales=[999.0], Open=[1])
    history_view = HistoryReconstructionProducer().build("D3", history, origin="2020-01-01")
    blind = ForecastBlindProducer().build("D3", history_view, forecast, origin="2020-01-01")
    mutated = ForecastBlindProducer().build(
        "D3", history_view, forecast.assign(sales=[-12345.0]), origin="2020-01-01"
    )
    pd.testing.assert_frame_equal(blind.worker, mutated.worker)
    assert "sales" not in blind.worker.columns
    assert "Open" not in blind.worker.columns


def test_TC_003_explicit_schema_rejects_auto_expansion() -> None:
    registry = SchemaRegistry()
    valid = _rows(["2020-01-01"], year=[2020], month=[1], day=[1])
    registry.validate("D1", "worker", valid)
    for injected in ("numeric_noise", "date_extra", "json_fallback"):
        with pytest.raises(Gate1Failure, match="SCHEMA_EXTRA"):
            registry.validate("D1", "worker", valid.assign(**{injected: 1}))
    with pytest.raises(Gate1Failure, match="SCHEMA_DTYPE"):
        registry.validate("D1", "worker", valid.assign(year=["2020"]))


def test_TC_004_field_specific_missing_rules_reject_generic_fill() -> None:
    resolver = AvailabilityResolver()
    with pytest.raises(Gate1Failure, match="MISSING_RULE"):
        resolver.resolve("D3", "Promo", available_at="future", fill_method="ffill")
    with pytest.raises(Gate1Failure, match="MISSING_RULE"):
        resolver.resolve("D5", "oil_price", available_at="future", fill_method="bfill")


def test_TC_005_raw_authority_hash_drift_blocks_preflight() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=project_root, prefix=".gate1-authority-") as fixture_root_text:
        fixture_root = Path(fixture_root_text)
        path = fixture_root / "authority.csv"
        path.write_bytes(b"date,value\n2020-01-01,1\n")
        authority = AuthorityProducer(
            root=fixture_root,
            files={"D1.train": Path("authority.csv")},
            expected_hashes={"D1.train": _raw_hash(path)},
        ).load()
        path.write_bytes(b"date,value\n2020-01-01,2\n")
        with pytest.raises(Gate1Failure, match="RAW_HASH_DRIFT"):
            AuthorityProducer(
                root=fixture_root,
                files={"D1.train": Path("authority.csv")},
                expected_hashes=authority.expected_hashes,
            ).load()


def test_TC_D1_01_target_truth_isolation() -> None:
    views = SafeTargetViewOperator().build(
        "D1",
        _rows(["2020-01-01"], sales=[3.0], year=[2020], month=[1], day=[1]),
        _rows(["2020-01-02"], sales=[4.0], year=[2020], month=[1], day=[2]),
    )
    assert "sales" not in views.worker.columns
    assert "sales" not in views.forecast.columns
    assert "sales" in views.label_truth.columns
    assert "sales" in views.evaluator_truth.columns


def test_TC_D2_01_history_promo_kept_but_forecast_promo_excluded() -> None:
    history = HistoryReconstructionProducer().build(
        "D2", _rows(["2020-01-01"], sales=[1.0], PROMO=[1]), origin="2020-01-01"
    )
    blind = ForecastBlindProducer().build(
        "D2", history, _rows(["2020-01-02"], PROMO=[1]), origin="2020-01-01"
    )
    assert "PROMO" in history.frame.columns
    assert "PROMO" not in blind.worker.columns


def test_TC_D2_02_future_actual_promo_excluded() -> None:
    history = HistoryReconstructionProducer().build("D2", _rows(["2020-01-01"], sales=[1.0]))
    clean = ForecastBlindProducer().build("D2", history, _rows(["2020-01-02"]), origin="2020-01-01")
    actual = ForecastBlindProducer().build(
        "D2", history, _rows(["2020-01-02"], PROMO=[99]), origin="2020-01-01"
    )
    pd.testing.assert_frame_equal(clean.worker, actual.worker)


def test_TC_D2_03_no_future_promo_fill() -> None:
    history = HistoryReconstructionProducer().build("D2", _rows(["2020-01-01"], PROMO=[1]))
    blind = ForecastBlindProducer().build("D2", history, _rows(["2020-01-02"]), origin="2020-01-01")
    assert "PROMO" not in blind.worker.columns


def test_TC_D3_01_open_historical_reconstruction_only() -> None:
    history = HistoryReconstructionProducer().build(
        "D3",
        _rows(["2020-01-01", "2020-01-02"], sales=[4.0, 0.0], Open=[None, None]),
        origin="2020-01-02",
    )
    assert history.frame["Open"].tolist() == [1, 0]
    assert history.repair_counts["Open"] == 2


def test_TC_D3_02_open_excluded_from_source_model_and_sample_scope() -> None:
    base = _rows(["2020-01-01"], sales=[1.0], Open=[1])
    changed = base.assign(Open=[0])
    first = SafeTargetViewOperator().build("D3", base, base)
    second = SafeTargetViewOperator().build("D3", changed, changed)
    assert "Open" not in first.worker.columns and "Open" not in first.knn.columns
    assert first.sample_range == second.sample_range


def test_TC_D3_03_open_does_not_read_target_day_actual() -> None:
    history = HistoryReconstructionProducer().build("D3", _rows(["2020-01-01"], sales=[1.0], Open=[1]))
    clean = ForecastBlindProducer().build("D3", history, _rows(["2020-01-02"]), origin="2020-01-01")
    poisoned = ForecastBlindProducer().build(
        "D3", history, _rows(["2020-01-02"], sales=[999], Open=[0]), origin="2020-01-01"
    )
    pd.testing.assert_frame_equal(clean.worker, poisoned.worker)


def test_TC_D3_04_promo_long_term_fields_do_not_replace_daily_promo() -> None:
    history = HistoryReconstructionProducer().build(
        "D3", _rows(["2020-01-01"], sales=[1.0], Promo=[1], Promo2=[0], PromoInterval=["Jan"])
    )
    blind = ForecastBlindProducer().build(
        "D3", history, _rows(["2020-01-02"], Promo2=[1], PromoInterval=["Feb"]), origin="2020-01-01"
    )
    assert "Promo" in history.frame.columns
    assert all(name not in blind.worker.columns for name in ("Promo", "Promo2", "PromoInterval"))


def test_TC_D3_05_customers_excluded_from_all_model_paths() -> None:
    views = SafeTargetViewOperator().build(
        "D3", _rows(["2020-01-01"], sales=[1.0], Customers=[10]), _rows(["2020-01-02"], Customers=[11])
    )
    assert all("Customers" not in frame.columns for frame in (views.worker, views.knn, views.forecast))


def test_TC_D3_06_schoolholiday_authority_zero_fill_and_dtype() -> None:
    frame = _rows(["2020-01-01", "2020-01-02"], sales=[1.0, 0.0], SchoolHoliday=[None, 1])
    result = HistoryReconstructionProducer().build("D3", frame, origin="2020-01-02")
    assert result.frame["SchoolHoliday"].tolist() == [0, 1]
    assert str(result.frame["SchoolHoliday"].dtype) == "int64"


def test_TC_D3_07_synthetic_region_mapping() -> None:
    operator = SourcePoolOperator()
    assert operator.region_for_store(1) == 1
    assert operator.region_for_store(10) == 1
    assert operator.region_for_store(11) == 2
    assert operator.region_for_store(20) == 2
    assert operator.region_for_store(21) == 3
    assert operator.region_for_store(30) == 3
    assert operator.target_store == 10


def test_TC_D3_08_with_without_candidate_pools() -> None:
    operator = SourcePoolOperator()
    without = operator.candidates("without")
    with_sharing = operator.candidates("with")
    assert without == tuple(range(1, 10))
    assert with_sharing == tuple(store for store in range(1, 31) if store != 10)
    assert 10 not in without and 10 not in with_sharing


def test_TC_D3_09_forbidden_legacy_domain_tokens() -> None:
    with pytest.raises(Gate1Failure, match="DOMAIN"):
        SourcePoolOperator().validate_domain("region = 1")
    with pytest.raises(Gate1Failure, match="DOMAIN"):
        SourcePoolOperator().validate_domain("TODO_REGION_UNAVAILABLE")


def test_TC_D4_01_only_approved_benchmark_future_covariates() -> None:
    approved = (
        "activity_flag", "discount", "holiday_flag", "precpt",
        "avg_temperature", "avg_humidity", "avg_wind_level",
    )
    blind = ForecastBlindProducer().build(
        "D4", HistoryReconstructionProducer().build("D4", _rows(["2020-01-01"])),
        _rows(["2020-01-02"], **{name: [1] for name in approved}), origin="2020-01-01",
    )
    assert tuple(name for name in blind.worker.columns if name in approved) == approved
    assert blind.manifest_classes == {name: "benchmark-provided future covariate" for name in approved}


def test_TC_D4_02_hourly_and_stock_fields_are_audit_only() -> None:
    frame = _rows(
        ["2020-01-01"], hours_sale=[1], hours_stock_status=[1], stock_hour6_22_cnt=[1],
        hours_sale_sum=[1], hours_stock_max=[1],
    )
    views = SafeTargetViewOperator().build("D4", frame, frame)
    assert all(name not in views.worker.columns for name in frame.columns if "hour" in name or "stock" in name)
    assert set(views.audit.columns) >= {"hours_sale", "hours_stock_status", "stock_hour6_22_cnt"}


def test_TC_D4_03_generic_fill_rejected() -> None:
    with pytest.raises(Gate1Failure, match="GENERIC_FILL"):
        ForecastBlindProducer().build(
            "D4", HistoryReconstructionProducer().build("D4", _rows(["2020-01-01"])),
            _rows(["2020-01-02"], provenance="bfill"), origin="2020-01-01",
        )


def test_TC_D5_01_onpromotion_future_known_zero_fill() -> None:
    history = HistoryReconstructionProducer().build("D5", _rows(["2020-01-01"], sales=[1]))
    blind = ForecastBlindProducer().build("D5", history, _rows(["2020-01-02"], onpromotion=[None]), origin="2020-01-01")
    assert blind.worker["onpromotion"].tolist() == [0]


def test_TC_D5_02_transactions_history_zero_fill_forecast_excluded() -> None:
    history = HistoryReconstructionProducer().build("D5", _rows(["2020-01-01"], transactions=[None]))
    blind = ForecastBlindProducer().build("D5", history, _rows(["2020-01-02"], transactions=[99]), origin="2020-01-01")
    assert history.frame["transactions"].tolist() == [0]
    assert "transactions" not in blind.worker.columns


def test_TC_D5_03_oil_prior_only_lag_one_forward_fill() -> None:
    history = HistoryReconstructionProducer().build(
        "D5", _rows(["2020-01-01", "2020-01-02", "2020-01-03"], oil_price=[90.0, 91.0, None]),
    )
    assert pd.isna(history.frame["oil_price"].iloc[0])
    assert history.frame["oil_price"].iloc[1:].tolist() == [90.0, 91.0]


def test_TC_D5_04_oil_no_prior_fail_closed() -> None:
    with pytest.raises(Gate1Failure, match="OIL_NO_PRIOR"):
        HistoryReconstructionProducer().build("D5", _rows(["2020-01-01"], oil_price=[None]))


def test_TC_D5_05_week_removed_and_date_fields_retained() -> None:
    views = SafeTargetViewOperator().build(
        "D5", _rows(["2020-01-01"], week=[1], year=[2020], month=[1], day=[1]),
        _rows(["2020-01-02"], week=[2], year=[2020], month=[1], day=[2]),
    )
    assert all("week" not in frame.columns for frame in (views.worker, views.knn, views.forecast))
    assert set(("year", "month", "day")) <= set(views.forecast.columns)


def test_TC_D5_06_holiday_mapping_and_scope() -> None:
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "type": ["Holiday", "Additional", "Bridge", "Work Day"],
            "locale": ["National"] * 4,
            "locale_name": [""] * 4,
            "transferred": [False] * 4,
        }
    )
    result = build_d5_holiday(holidays, pd.date_range("2020-01-01", "2020-01-04"), store_state="X", store_city="X")
    assert result.tolist() == [1, 1, 1, 0]


def test_TC_D5_07_transferred_holiday_deduplicates_without_row_expansion() -> None:
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "type": ["Holiday", "Transfer", "Additional"],
            "locale": ["National"] * 3,
            "locale_name": [""] * 3,
            "transferred": [True, False, False],
        }
    )
    result = build_d5_holiday(holidays, pd.date_range("2020-01-01", "2020-01-02"), store_state="X", store_city="X")
    assert len(result) == 2 and result.tolist() == [0, 1]


def test_TC_D6_01_sell_price_exact_key_join() -> None:
    target = pd.DataFrame({"store_id": ["CA_1", "CA_1"], "item_id": ["FOODS_1", "FOODS_1"], "wm_yr_wk": [1, 2]})
    prices = pd.DataFrame({"store_id": ["CA_1", "CA_1"], "item_id": ["FOODS_1", "FOODS_1"], "wm_yr_wk": [1, 2], "sell_price": [2.0, 3.0]})
    joined = join_d6_sell_price(target, prices)
    assert joined["sell_price"].tolist() == [2.0, 3.0]


def test_TC_D6_02_sell_price_fail_closed_conditions() -> None:
    target = pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1"], "wm_yr_wk": [1]})
    for prices in (
        pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1"], "wm_yr_wk": [2], "sell_price": [2.0]}),
        pd.DataFrame({"store_id": ["CA_1", "CA_1"], "item_id": ["FOODS_1", "FOODS_1"], "wm_yr_wk": [1, 1], "sell_price": [2.0, 3.0]}),
        pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1"], "wm_yr_wk": [1], "sell_price": [None]}),
    ):
        with pytest.raises(Gate1Failure, match="PRICE_"):
            join_d6_sell_price(target, prices)


def test_TC_D6_03_calendar_authority_and_allowed_fields() -> None:
    calendar = pd.DataFrame(
        {name: [1] for name in ("weekday", "wday", "wm_yr_wk", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI")}
    )
    view = build_d6_calendar_view(calendar, store_state="CA")
    assert set(view.columns) == {"weekday", "wday", "wm_yr_wk", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap"}
    with pytest.raises(Gate1Failure, match="CALENDAR_"):
        build_d6_calendar_view(calendar.assign(unapproved=1), store_state="CA")


def test_TC_D6_04_state_specific_snap_mapping_and_unknown_state_failure() -> None:
    calendar = pd.DataFrame({"weekday": [1], "wday": [1], "wm_yr_wk": [1], "event_name_1": [None], "event_type_1": [None], "event_name_2": [None], "event_type_2": [None], "snap_CA": [1], "snap_TX": [2], "snap_WI": [3]})
    assert build_d6_calendar_view(calendar, store_state="CA")["snap"].tolist() == [1]
    assert build_d6_calendar_view(calendar, store_state="TX")["snap"].tolist() == [2]
    assert build_d6_calendar_view(calendar, store_state="WI")["snap"].tolist() == [3]
    with pytest.raises(Gate1Failure, match="SNAP_STATE"):
        build_d6_calendar_view(calendar, store_state="XX")


def test_TC_PROOF_01_proof_completeness_and_identity_binding() -> None:
    proof = ProofWriter().build(
        contract_digest="sha256:b145028c2b3f8314e66fc73be9795269644d016a7a1cf258a9f62f1b7443d09e",
        authority={"snapshot_id": "mini", "files": {"a": "a"}},
        schemas={"worker": "w", "knn": "k"},
        resolver={"date": {"status": "accepted"}},
        views={"worker": ["date"], "knn": ["sales"], "forecast": ["date"], "label": ["sales"], "audit": []},
        artifacts={"model": {"sha256": "a" * 64}},
    )
    assert proof["contract_digest"].startswith("sha256:")
    with pytest.raises(Gate1Failure, match="PROOF_"):
        FormalPreflight().check({"proof": {**proof, "contract_digest": "sha256:bad"}})


def test_TC_PREFLIGHT_01_formal_preflight_blocks_forbidden_states() -> None:
    preflight = FormalPreflight()
    for state in (
        {"target_day_actual": True},
        {"forbidden_fields": ["transactions"]},
        {"generic_fill": "bfill"},
        {"candidate_sources": [10]},
        {"row_count_before": 1, "row_count_after": 2},
        {"proof_complete": False},
    ):
        with pytest.raises(Gate1Failure):
            preflight.check(state)


def test_TC_RUNNER_01_unified_runner_never_calls_legacy_d4_d6_runners() -> None:
    calls: list[str] = []
    runner = UnifiedRunner(legacy_runners={dataset: lambda: calls.append(dataset) for dataset in ("D4", "D5", "D6")})
    proof = ProofWriter().build(
        contract_digest="sha256:b145028c2b3f8314e66fc73be9795269644d016a7a1cf258a9f62f1b7443d09e",
        authority={"snapshot_id": "mini", "files": {"a": "a"}},
        schemas={"worker": "w", "knn": "k"},
        resolver={"date": {"status": "accepted"}},
        views={"worker": ["date"], "knn": ["sales"], "forecast": ["date"], "label": ["sales"], "audit": []},
        artifacts={"model": {"sha256": "a" * 64}},
    )
    report = runner.dry_run(
        dataset="D4",
        state={"contract_digest": "sha256:b145028c2b3f8314e66fc73be9795269644d016a7a1cf258a9f62f1b7443d09e", "proof_complete": True, "proof": proof},
    )
    assert report["path"] == "frozen schema -> availability gate -> safe target view -> model"
    assert calls == []
