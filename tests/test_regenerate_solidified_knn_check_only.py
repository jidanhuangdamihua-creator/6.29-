from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.regenerate_solidified_knn import (
    _write_json,
    snapshot_knn_config_files,
    verify_knn_config_unchanged,
)


def test_write_json_converts_numpy_pandas_and_nonfinite_values(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"
    payload = {
        "np_int": np.int64(7),
        "np_float": np.float64(1.5),
        "np_bool": np.bool_(True),
        "array": np.array([np.int64(1), np.float64(2.5)]),
        "nan": np.nan,
        "pos_inf": np.inf,
        "neg_inf": -np.inf,
        "pd_na": pd.NA,
        "timestamp": pd.Timestamp("2024-01-02 03:04:05"),
        "date": date(2024, 1, 3),
        "path": tmp_path / "x",
        "tuple_key_dict": {
            ("b", 2): np.int64(9),
            np.int64(3): "three",
        },
        "set_values": {"b", "a"},
        "frozenset_values": frozenset([np.int64(2), np.int64(1)]),
    }

    _write_json(output, payload)

    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "-Infinity" not in text

    loaded = json.loads(text)
    assert loaded["np_int"] == 7
    assert loaded["np_float"] == 1.5
    assert loaded["np_bool"] is True
    assert loaded["array"] == [1, 2.5]
    assert loaded["nan"] is None
    assert loaded["pos_inf"] is None
    assert loaded["neg_inf"] is None
    assert loaded["pd_na"] is None
    assert loaded["timestamp"] == "2024-01-02T03:04:05"
    assert loaded["date"] == "2024-01-03"
    assert loaded["path"] == str(tmp_path / "x")
    assert loaded["tuple_key_dict"]["('b', 2)"] == 9
    assert loaded["tuple_key_dict"]["3"] == "three"
    assert loaded["set_values"] == ["a", "b"]
    assert loaded["frozenset_values"] == [1, 2]


def test_check_only_snapshot_detects_unchanged_config_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Dataset5"
    dataset_dir.mkdir()
    path = dataset_dir / "knn_without_info_sharing.json"
    path.write_text(json.dumps({"results": {}}), encoding="utf-8")

    before = snapshot_knn_config_files(tmp_path)

    generated_dir = tmp_path.parent / "outputs" / "generated_json" / "Dataset5"
    generated_dir.mkdir(parents=True)
    (generated_dir / path.name).write_text(json.dumps({"results": {"new": []}}), encoding="utf-8")

    verify_knn_config_unchanged(tmp_path, before)
