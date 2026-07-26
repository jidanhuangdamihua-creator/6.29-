from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.protocols.formal_input_paths import formal_input_paths


ROOT = Path(__file__).resolve().parents[1]


def test_d2_sealed_manifest_binds_the_declared_current_bytes() -> None:
    paths = formal_input_paths(ROOT, 2)
    manifest_path = paths["source"].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "source": (48654, "466391bb7e89067663d2d8f882834819896620c56bbbdc1959b81df938080ab2"),
        "target": (1802, "fbfe0df5a5624504b00a8ea701ca7dd250ab46232d29f82473dcf4d0df712588"),
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
