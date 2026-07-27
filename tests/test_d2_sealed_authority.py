from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.protocols.formal_input_paths import formal_input_paths


ROOT = Path(__file__).resolve().parents[1]


def test_d2_sealed_manifest_binds_the_declared_current_bytes() -> None:
    paths = formal_input_paths(ROOT, 2)
    manifest_path = paths["source"].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "source": (48654, "04c316a7519e37c6f6712b5c34d25edb38e82833568102a22bd3961081d07409"),
        "target": (1807, "e4893b3e2cc7c2d173cba4b76fc58ceac99515af2b89e999ce62531a01f8e009"),
    }
    for role, (rows, digest) in expected.items():
        path = paths[role]
        artifact = manifest["artifacts"][role]
        assert artifact["path"] == path.name
        assert artifact["row_count"] == rows
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["sha256"] == digest
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_d2_sealed_manifest_records_precalendarized_zero_dates() -> None:
    paths = formal_input_paths(ROOT, 2)
    manifest = json.loads(
        (paths["source"].parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["content_validation_level"] == "sealed_bytes_verified"
    assert manifest["dataset_canonicalization"]["runtime_calendarization"] is False
    assert manifest["dataset_canonicalization"]["verified_zero_sales_dates"] == [
        "2018-04-01",
        "2018-04-25",
        "2018-05-01",
        "2018-06-02",
    ]


def test_d2_sealed_source_allowed_dates_have_finite_date_fields() -> None:
    paths = formal_input_paths(ROOT, 2)
    source = pd.read_parquet(paths["source"])
    dates = pd.to_datetime(source["date"]).dt.normalize()
    approved = pd.to_datetime(
        ["2018-04-01", "2018-04-25", "2018-05-01", "2018-06-02"]
    )
    rows = source.loc[dates.isin(approved)]
    assert len(rows) == 27 * len(approved)
    assert rows["sales"].eq(0).all()
    assert rows["promo"].eq(0).all()
    assert rows[["year", "month", "week", "day"]].notna().all().all()
    assert np.isfinite(rows[["year", "month", "week", "day"]].to_numpy(dtype=float)).all()
