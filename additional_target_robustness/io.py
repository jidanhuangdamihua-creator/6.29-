"""Canary artifacts. Source calendars are never written here."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from additional_target_robustness.protocol import ProtocolError


def write_canary(directory: Path, artifacts: dict[str, object]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _json(directory / "protocol_check.json", artifacts["protocol_check"])
    _json(directory / "source_selection.json", artifacts["source_selection"])
    _json(directory / "cache_usage.json", artifacts["cache_usage"])
    artifacts["evaluation_index"].to_parquet(directory / "evaluation_index.parquet", index=False)
    artifacts["predictions"].to_parquet(directory / "predictions.parquet", index=False)
    artifacts["metrics"].to_csv(directory / "metrics.csv", index=False)
    (directory / "run_log.txt").write_text(str(artifacts["run_log"]), encoding="utf-8")
    status = str(artifacts["protocol_check"]["status"])
    if status == "READY_FOR_FULL_RUN":
        _require_ready(artifacts)
    _json(directory / "DONE.json", {"status": status, "canary": "D1/2/6"})


def _require_ready(artifacts: dict[str, object]) -> None:
    metrics = artifacts["metrics"]
    if len(metrics) != 12:
        raise ProtocolError("canary must contain 12 method records")
    if set(metrics["n_test"].tolist()) != {180}:
        raise ProtocolError("canary n_test is not 180")
    check = artifacts["protocol_check"]
    for horizon in ("1", "3", "5"):
        block = check["horizons"][horizon]
        if not block["target_dates_equal"] or set(block["n_test"].values()) != {180}:
            raise ProtocolError(f"horizon {horizon} dates are not shared")


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
