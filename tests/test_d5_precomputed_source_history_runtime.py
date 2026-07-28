from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.utils.parquet_data_loader as loader
from scripts.precompute_d5_source_history import build_output_manifest
from src.protocols.source_history import (
    build_exact_source_history_candidate_frame,
    source_history_frame_digest,
)
from src.utils.d5_calendar_reconstruction import (
    load_d5_authorities,
    reconstruct_d5_source_history_calendar,
)
from src.utils.d5_precomputed_source_history import (
    D5_PRECOMPUTED_MANIFEST_FILENAME,
    D5_PRECOMPUTED_SOURCE_FILENAME,
    D5PrecomputedSourceHistoryHashMismatch,
    load_precomputed_d5_source_history,
)

from tests.test_d5_runtime_reconstruction_contract import _write_runtime_fixture


def _write_precomputed_fixture(tmp_path: Path):
    raw_dir, parquet_dir, windows, expected_dates = _write_runtime_fixture(tmp_path)
    authorities = load_d5_authorities(raw_dir, use_holidays=True)
    source_path = parquet_dir / "dataset5-source.parquet"
    source = pd.read_parquet(source_path)
    source_dates = pd.date_range("2016-08-20", "2017-02-15", freq="D")
    reconstructed, report = reconstruct_d5_source_history_calendar(
        source,
        expected_dates=source_dates,
        authorities=authorities,
    )
    eligibility = build_exact_source_history_candidate_frame(
        reconstructed,
        key_fields=("store_nbr", "item_nbr"),
        origin=source_dates[-1],
        source_history_days=180,
    )
    output_path = parquet_dir / D5_PRECOMPUTED_SOURCE_FILENAME
    persisted = eligibility.candidate_frame.copy(deep=False)
    persisted.attrs = {}
    persisted.to_parquet(output_path, index=False)
    manifest = build_output_manifest(
        source_path=source_path,
        auxiliary_files={
            name: Path(evidence.path)
            for name, evidence in authorities.files.items()
            if evidence.used
        },
        output_path=output_path,
        frame=eligibility.candidate_frame,
        source_history_start=source_dates[0],
        source_history_end=source_dates[-1],
        source_history_days=180,
        key_fields=("store_nbr", "item_nbr"),
        generated_at="2026-07-26T00:00:00Z",
        source_history_frame_digest=source_history_frame_digest(
            eligibility.candidate_frame,
            key_fields=("store_nbr", "item_nbr"),
        ),
        synthetic_row_count=report.synthetic_row_count,
        repository_root=tmp_path,
    )
    manifest_path = parquet_dir / D5_PRECOMPUTED_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return (
        raw_dir,
        parquet_dir,
        windows,
        expected_dates,
        authorities,
        source_path,
        output_path,
        manifest_path,
        eligibility.candidate_frame,
    )


def test_precomputed_resolver_validates_all_inputs_and_loads_frame(tmp_path: Path) -> None:
    (
        _,
        _,
        _,
        _,
        authorities,
        source_path,
        _,
        _,
        expected_frame,
    ) = _write_precomputed_fixture(tmp_path)

    loaded = load_precomputed_d5_source_history(
        source_path=source_path,
        authorities=authorities,
        source_history_start=pd.Timestamp("2016-08-20"),
        source_history_end=pd.Timestamp("2017-02-15"),
        source_history_days=180,
        key_fields=("store_nbr", "item_nbr"),
    )

    assert loaded is not None
    frame, _ = loaded
    assert frame.equals(expected_frame)


def test_precomputed_resolver_fails_closed_on_specific_csv_hash_mismatch(tmp_path: Path) -> None:
    raw_dir, _, _, _, authorities, source_path, _, _, _ = _write_precomputed_fixture(tmp_path)
    transactions_path = raw_dir / "transactions.csv"
    transactions_path.write_text(
        transactions_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        D5PrecomputedSourceHistoryHashMismatch,
        match=r"file=transactions\.csv",
    ):
        load_precomputed_d5_source_history(
            source_path=source_path,
            authorities=authorities,
            source_history_start=pd.Timestamp("2016-08-20"),
            source_history_end=pd.Timestamp("2017-02-15"),
            source_history_days=180,
            key_fields=("store_nbr", "item_nbr"),
        )


def test_force_recompute_ignores_existing_precomputed_files(tmp_path: Path, monkeypatch) -> None:
    _, _, _, _, authorities, source_path, _, _, _ = _write_precomputed_fixture(tmp_path)
    monkeypatch.setenv("D5_FORCE_RECOMPUTE", "1")

    loaded = load_precomputed_d5_source_history(
        source_path=source_path,
        authorities=authorities,
        source_history_start=pd.Timestamp("2016-08-20"),
        source_history_end=pd.Timestamp("2017-02-15"),
        source_history_days=180,
        key_fields=("store_nbr", "item_nbr"),
    )

    assert loaded is None


def test_missing_precomputed_pair_keeps_legacy_fallback(tmp_path: Path) -> None:
    raw_dir, parquet_dir, _, _ = _write_runtime_fixture(tmp_path)
    authorities = load_d5_authorities(raw_dir, use_holidays=True)
    source_path = parquet_dir / "dataset5-source.parquet"
    assert not (parquet_dir / D5_PRECOMPUTED_SOURCE_FILENAME).exists()
    assert not (parquet_dir / D5_PRECOMPUTED_MANIFEST_FILENAME).exists()

    loaded = load_precomputed_d5_source_history(
        source_path=source_path,
        authorities=authorities,
        source_history_start=pd.Timestamp("2016-08-20"),
        source_history_end=pd.Timestamp("2017-02-15"),
        source_history_days=180,
        key_fields=("store_nbr", "item_nbr"),
    )

    assert loaded is None


def test_loader_skips_source_reconstruction_when_precomputed_frame_is_valid(
    tmp_path: Path, monkeypatch
) -> None:
    (
        raw_dir,
        parquet_dir,
        windows,
        expected_dates,
        authorities,
        source_path,
        _,
        _,
        expected_frame,
    ) = _write_precomputed_fixture(tmp_path)
    monkeypatch.setattr(
        loader,
        "reconstruct_d5_source_history_calendar",
        lambda *args, **kwargs: pytest.fail("source calendarization must be skipped"),
    )
    monkeypatch.setattr(
        loader,
        "build_exact_source_history_candidate_frame",
        lambda *args, **kwargs: pytest.fail("source eligibility must be skipped"),
    )

    loaded = loader.load_parquet_source_target_with_diagnostics(
        dataset_id=5,
        source_path=source_path,
        target_path=parquet_dir / "dataset5-target.parquet",
        windows=windows,
        source_history_days=180,
        expected_dates=expected_dates,
        d5_authorities=authorities,
    )

    assert loaded.source_df.equals(expected_frame)
    assert loaded.source_df.attrs["source_history_calendarization_rule"] == (
        "D5_APPROVED_SOURCE_HISTORY_CALENDARIZATION"
    )


def test_loader_force_recompute_keeps_original_source_path(
    tmp_path: Path, monkeypatch
) -> None:
    (
        _,
        parquet_dir,
        windows,
        expected_dates,
        authorities,
        source_path,
        _,
        _,
        _,
    ) = _write_precomputed_fixture(tmp_path)
    real_reconstruct = loader.reconstruct_d5_source_history_calendar
    real_build = loader.build_exact_source_history_candidate_frame
    reconstruct_calls = []
    build_calls = []

    def tracked_reconstruct(*args, **kwargs):
        reconstruct_calls.append(True)
        return real_reconstruct(*args, **kwargs)

    def tracked_build(*args, **kwargs):
        build_calls.append(True)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(loader, "reconstruct_d5_source_history_calendar", tracked_reconstruct)
    monkeypatch.setattr(loader, "build_exact_source_history_candidate_frame", tracked_build)
    monkeypatch.setenv("D5_FORCE_RECOMPUTE", "1")

    loader.load_parquet_source_target_with_diagnostics(
        dataset_id=5,
        source_path=source_path,
        target_path=parquet_dir / "dataset5-target.parquet",
        windows=windows,
        source_history_days=180,
        expected_dates=expected_dates,
        d5_authorities=authorities,
    )

    assert reconstruct_calls
    assert build_calls
