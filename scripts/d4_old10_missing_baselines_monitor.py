#!/usr/bin/env python3
"""Compact D4 old-10 missing-BL progress snapshot."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "additional_target_robustness_20261004" / "old10_missing_baselines" / "D4" / "baselines"
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


def _baselines() -> dict:
    rows = _read_csv(OUT / "baseline_progress.csv")
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
    lines = [f"D4 old10 missing BL monitor {_utc()}"]
    bl = _baselines()
    lines.append("BASELINES")
    lines.append("  method              D4")
    for method in METHODS:
        done, failed = bl["by_method"][method]
        lines.append(f"  {method:<18} {done:3d}/150 fail={failed}")
    lines.append(f"  D4 {bl['completed']}/750 fail={bl['failed']} last={bl['last']}")
    lines.append("LOG")
    lines.append("  d4_bl   " + _tail(OUT / "baseline_run_log.txt"))
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
