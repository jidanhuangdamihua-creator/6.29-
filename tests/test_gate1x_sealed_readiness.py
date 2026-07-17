from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
import pytest

from src.protocols.gate1_transformation import (
    COMBINED_FORMAL_IDENTITY_DIGEST,
    CONTRACT_DIGEST,
    FREEZE_COMMIT_SHA,
    FormalPreflight,
    Gate1Failure,
    ProofWriter,
    SchemaRegistry,
    build_contract_views,
    build_d6_calendar_view,
    build_d5_holiday,
    build_d5_oil_price,
    dataset_contract,
    join_d6_sell_price,
    load_formal_identity,
    normalize_onpromotion,
    rebuild_d2_wide_frame,
    slice_dataset_roles,
    source_pool_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame(key_fields: tuple[str, ...], key: tuple[object, ...], dates: pd.DatetimeIndex, **extra: object) -> pd.DataFrame:
    result = {field: [value] * len(dates) for field, value in zip(key_fields, key)}
    result["date"] = dates
    for name, value in extra.items():
        result[name] = [value] * len(dates)
    return pd.DataFrame(result)


def test_formal_identity_uses_frozen_authority_and_new_starting_head_is_separate() -> None:
    identity = load_formal_identity(ROOT)
    assert identity["contract_digest"] == CONTRACT_DIGEST
    assert identity["combined_formal_identity_digest"] == COMBINED_FORMAL_IDENTITY_DIGEST
    assert identity["freeze_commit_sha"] == FREEZE_COMMIT_SHA
    assert identity["contract_version"] == "1R.1.0"


def test_formal_identity_sidecar_tamper_fails_closed(tmp_path: Path) -> None:
    for relative in (
        "docs/protocol/gate1_frozen_transformation_contract.md",
        "docs/protocol/gate1_implementation_scope.md",
        "docs/protocol/gate1_contract_traceability_matrix.md",
        "docs/protocol/gate1_frozen_transformation_contract.sha256",
        "docs/protocol/gate1r_contract_refreeze_record.md",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{FREEZE_COMMIT_SHA}:{relative}"]))
    sidecar = tmp_path / "docs/protocol/gate1_frozen_transformation_contract.sha256"
    sidecar.write_text(sidecar.read_text(encoding="utf-8").replace("1R.1.0", "1R.tampered"), encoding="utf-8")
    with pytest.raises(Gate1Failure, match="FORMAL_IDENTITY"):
        load_formal_identity(tmp_path)


@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4", "D5", "D6"])
def test_frozen_windows_are_180_15_15_180_and_knn_30(dataset: str) -> None:
    spec = dataset_contract(dataset)
    assert (spec.origin - spec.source_history_start).days == 179
    assert (spec.target_train_end - spec.target_train_start).days == 14
    assert (spec.validation_end - spec.validation_start).days == 14
    assert (spec.blind_end - spec.blind_start).days == 179
    assert (spec.knn_end - spec.knn_start).days == 29
    assert spec.expected_blind_rows == (180 if dataset in {"D1", "D2", "D3"} else 900)


def test_d3_role_slicing_isolates_history_and_truth() -> None:
    spec = dataset_contract("D3")
    source_dates = pd.date_range(spec.source_history_start, spec.origin)
    target_dates = pd.date_range(spec.target_train_start, spec.blind_end)
    source = _frame(spec.key_fields, (1,), source_dates, sales=1.0)
    target = _frame(spec.key_fields, (10,), target_dates, sales=2.0, SchoolHoliday=0)
    roles = slice_dataset_roles("D3", source, target)
    assert roles.source_history["date"].max() == pd.Timestamp(spec.origin)
    assert (roles.worker_safe_blind["date"] > pd.Timestamp(spec.origin)).all()
    assert "sales" not in roles.worker_safe_blind.columns
    assert "Customers" not in roles.worker_safe_blind.columns
    assert len(roles.target_train) == 15
    assert len(roles.target_validation) == 15
    assert len(roles.worker_safe_blind) == 180
    assert len(roles.evaluator_truth) == 180


def test_d2_calendarization_closes_missing_day_and_removes_forecast_promo() -> None:
    spec = dataset_contract("D2")
    source = _frame(spec.key_fields, (1, 9), pd.date_range(spec.source_history_start, spec.origin), sales=1.0, PROMO=1)
    dates = pd.date_range(spec.target_train_start, spec.blind_end).difference(pd.DatetimeIndex([pd.Timestamp("2018-07-03")]))
    target = _frame(spec.key_fields, (1, 10), dates, sales=2.0, PROMO=1)
    roles = slice_dataset_roles("D2", source, target)
    assert len(roles.worker_safe_blind) == 180
    assert "PROMO" not in roles.worker_safe_blind.columns
    repaired = roles.evaluator_truth.loc[roles.evaluator_truth["date"] == pd.Timestamp("2018-07-03")]
    assert repaired["sales"].eq(0).all()
    assert len(roles.repairs) == 1


def test_d2_wide_rebuild_keeps_empty_entity_date_rows() -> None:
    wide = pd.DataFrame({"DATE": ["2018-01-02"], "QTY_B1_1": [None], "QTY_B1_2": [3]})
    rebuilt = rebuild_d2_wide_frame(wide)
    assert len(rebuilt) == 2
    assert rebuilt.loc[rebuilt["item_id"].eq("1"), "sales"].eq(0).all()


def test_d5_onpromotion_is_strict_and_target_set_preserves_1159415() -> None:
    assert normalize_onpromotion(pd.Series([True, "false", 1, None, float("nan")])).tolist() == [1, 0, 1, 0, 0]
    with pytest.raises(Gate1Failure, match="ONPROMOTION_ENCODING"):
        normalize_onpromotion(pd.Series(["unknown"]))
    assert (48, 1159415) in tuple(tuple(int(part) for part in key) for key in dataset_contract("D5").target_keys)


def test_d5_oil_is_global_history_only_and_holiday_is_date_folded() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2017-02-14", "2017-02-15"]), "store_nbr": [48, 48], "item_nbr": [1, 1]})
    oil = pd.DataFrame({"date": pd.to_datetime(["2017-02-13", "2017-02-14"]), "dcoilwtico": [50.0, 51.0]})
    repaired = build_d5_oil_price(frame, oil, origin="2017-02-15")
    assert repaired["oil_price"].tolist() == [50.0, 51.0]
    holidays = pd.DataFrame({"date": pd.to_datetime(["2017-02-14", "2017-02-14", "2017-02-15"]), "type": ["Holiday", "Bridge", "Work Day"], "transferred": [1, 1, 0]})
    assert build_d5_holiday(frame, holidays)["is_holiday"].tolist() == [1, 0]


def test_d4_schema_excludes_hourly_and_stock_fields() -> None:
    allowed = SchemaRegistry().allowed("D4", "worker")
    assert set(allowed).issuperset({"activity_flag", "discount", "holiday_flag", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"})
    assert not set(allowed).intersection({"hours_sale_sum_leakage_risk", "stock_hour6_22_cnt"})
    assert SchemaRegistry().allowed("D4", "knn") == ("date", "sales")


def test_d6_calendar_preserves_weekday_wday_order_and_exact_price_join() -> None:
    calendar = pd.DataFrame({"date": pd.to_datetime(["2015-01-01"]), "weekday": ["Thursday"], "wday": [5], "wm_yr_wk": [1], "snap_CA": [1]})
    view = build_d6_calendar_view(calendar, store_state="CA")
    assert list(view.columns[:4]) == ["date", "weekday", "wday", "wm_yr_wk"]
    target = pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_3_586"], "wm_yr_wk": [1]})
    prices = target.assign(sell_price=[2.5])
    assert join_d6_sell_price(target, prices)["sell_price"].tolist() == [2.5]
    with pytest.raises(Gate1Failure, match="D6_PRICE_DUPLICATE"):
        join_d6_sell_price(target, pd.concat([prices, prices], ignore_index=True))


def test_source_pool_rules_are_frozen() -> None:
    assert source_pool_candidates("D1", "without-sharing") == (1,)
    assert source_pool_candidates("D1", "with-sharing") == (1, 2, 3)
    assert source_pool_candidates("D2", "with-sharing") == (1, 2, 3)
    assert source_pool_candidates("D3", "without-sharing") == tuple(range(1, 10))
    assert 10 not in source_pool_candidates("D3", "with-sharing")


def test_proof_writer_and_formal_preflight_require_all_layers() -> None:
    names = ("source_history", "target_observed", "worker_safe_blind", "evaluator_truth", "audit_view")
    proof = ProofWriter().build(contract_digest=CONTRACT_DIGEST, authority={"dataset": "D1", "raw": {"files": ["raw"]}}, schemas={"worker": {}, "knn": {}}, resolver={"status": "passed"}, views={name: {} for name in names}, artifacts={"physical_hash": "a" * 64})
    for name in ProofWriter.REQUIRED:
        assert name in proof
    assert FormalPreflight().check({"proof": proof})["status"] == "passed"


def test_actual_contract_views_have_five_frames() -> None:
    spec = dataset_contract("D1")
    dates = pd.date_range(spec.target_train_start, spec.blind_end)
    roles = slice_dataset_roles("D1", _frame(spec.key_fields, (2, 9), pd.date_range(spec.source_history_start, spec.origin), sales=1.0), _frame(spec.key_fields, (1, 10), dates, sales=2.0))
    views = build_contract_views("D1", roles)
    assert tuple(views) == ("source_history", "target_observed", "worker_safe_blind", "evaluator_truth", "audit_view")
