#!/usr/bin/env python3
"""Compact D5/D6 old-10 missing-BL progress snapshot."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = {
    "D5": ROOT / "outputs" / "additional_target_robustness_20261004" / "old10_missing_baselines" / "D5" / "baselines",
    "D6": ROOT / "outputs" / "additional_target_robustness_20261004" / "old10_missing_baselines" / "D6" / "baselines",
}
METHODS = (
    "LightGBM",
    "LSTM",
    "DLinear",
    "TCN",
    "MSML-TL",
)
BL_PER = 150
EXPECTED = 750


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size < 8:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tail(path: Path, n: int = 1) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n] if lines else ""


def _baselines(dataset: str) -> dict:
    rows = _read_csv(OUT[dataset] / "baseline_progress.csv")
    by_method = {}
    for method in METHODS:
        done = sum(
            1
            for row in rows
            if str(row.get("method")) == method
            and str(row.get("status")) in {"COMPLETED", "SKIP"}
            and str(row.get("n_test", "180")) in {"180", "180.0"}
        )
        failed = sum(
            1
            for row in rows
            if str(row.get("method")) == method and str(row.get("status")) not in {"COMPLETED", "SKIP", ""}
        )
        by_method[method] = (done, failed)
    last = rows[-1] if rows else {}
    completed = sum(item[0] for item in by_method.values())
    failed = sum(item[1] for item in by_method.values())
    return {
        "completed": completed,
        "failed": failed,
        "expected": EXPECTED,
        "by_method": by_method,
        "last": (
            f"{last.get('method', '')} {last.get('target_identity', '')} "
            f"h={last.get('horizon', '')} s={last.get('seed', '')} {last.get('status', '')}"
        ).strip(),
    }


def render() -> str:
    lines = [f"D5+D6 old10 missing BL monitor {_utc()}"]
    b5, b6 = _baselines("D5"), _baselines("D6")
    lines.append("BASELINES")
    lines.append("  method              D5      D6    TOT")
    for method in METHODS:
        d5, _f5 = b5["by_method"][method]
        d6, _f6 = b6["by_method"][method]
        lines.append(f"  {method:<18} {d5:3d}/150 {d6:3d}/150 {d5 + d6:3d}/300")
    lines.append(f"  D5 {b5['completed']}/750 fail={b5['failed']} last={b5['last']}")
    lines.append(f"  D6 {b6['completed']}/750 fail={b6['failed']} last={b6['last']}")
    lines.append(f"  TOT {b5['completed'] + b6['completed']}/1500 fail={b5['failed'] + b6['failed']}")
    lines.append("LOG")
    lines.append("  d5_bl   " + _tail(OUT["D5"] / "baseline_run_log.txt"))
    lines.append("  d6_bl   " + _tail(OUT["D6"] / "baseline_run_log.txt"))
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
