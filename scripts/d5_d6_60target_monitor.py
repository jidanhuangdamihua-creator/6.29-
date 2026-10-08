#!/usr/bin/env python3
"""Compact D5/D6 Stage A + baseline + SITA progress snapshot."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = {
    "D5": ROOT / "outputs" / "d5_60target_robustness_20261004",
    "D6": ROOT / "outputs" / "d6_60target_robustness_20261004",
}
STAGE_A = {"D5": (1725, 1500, 225), "D6": (1845, 1740, 105)}
METHODS = (
    "Seasonal Naive",
    "LightGBM",
    "LSTM",
    "DLinear",
    "TCN",
    "PatchTST",
    "TimesNet",
    "MSML-TL",
)
BL_PER = 900
SITA_PER = 900


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


def _count(rows: list[dict], field: str, values: set[str]) -> int:
    return sum(1 for row in rows if str(row.get(field, "")) in values)


def _stage_a(dataset: str) -> dict:
    required, n_miss, n_hit = STAGE_A[dataset]
    rows = _read_csv(OUT[dataset] / "stage_a_training_progress.csv")
    trained = _count(rows, "final_status", {"TRAINED", "TRAINED_IN_CANARY", "CACHE_HIT_AFTER_RESUME"})
    failed = _count(rows, "final_status", {"FAILED", "CACHE_HASH_MISMATCH", "LOAD_FAILURE", "BLOCKED_SOURCE_DATA_PROTOCOL"})
    last = rows[-1] if rows else {}
    available = n_hit + trained
    return {
        "required": required,
        "original_hit": n_hit,
        "progress_rows": len(rows),
        "trained_logged": trained,
        "failed": failed,
        "available_est": min(required, available),
        "missing_est": max(0, required - min(required, available)),
        "last": f"{last.get('source','')} h={last.get('horizon','')} s={last.get('seed','')} {last.get('final_status','')}",
    }


def _baselines(dataset: str) -> dict:
    rows = _read_csv(OUT[dataset] / "baselines" / "baseline_progress.csv")
    by_method = {}
    for method in METHODS:
        done = sum(
            1
            for row in rows
            if str(row.get("method")) == method and str(row.get("status")) in {"COMPLETED", "SKIP"}
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
        "expected": 7200,
        "by_method": by_method,
        "last": f"{last.get('method','')} {last.get('target_identity','')} h={last.get('horizon','')} s={last.get('seed','')} {last.get('status','')}",
    }


def _sita(dataset: str) -> dict:
    root = OUT[dataset] / "sita"
    rows = _read_csv(root / "sita_progress.csv") if (root / "sita_progress.csv").is_file() else _read_csv(root / "progress.csv")
    if not rows:
        cells = list((root / "cells").glob("*/DONE.json")) if (root / "cells").exists() else []
        n = len(cells)
        return {"completed": n, "failed": 0, "expected": SITA_PER, "status": "NOT_STARTED" if n == 0 else "RUNNING", "last": ""}
    done = _count(rows, "status", {"COMPLETED", "SKIP", "CACHE_HIT"})
    failed = sum(1 for row in rows if str(row.get("status", "")) not in {"COMPLETED", "SKIP", "CACHE_HIT", ""})
    last = rows[-1]
    return {
        "completed": done,
        "failed": failed,
        "expected": SITA_PER,
        "status": "RUNNING" if done < SITA_PER else "COMPLETE",
        "last": f"{last.get('target_identity', last.get('target',''))} h={last.get('horizon','')} s={last.get('seed','')} {last.get('status','')}",
    }


def render() -> str:
    lines = [f"D5+D6 monitor {_utc()}"]
    a5, a6 = _stage_a("D5"), _stage_a("D6")
    lines.append("STAGE A")
    lines.append(f"  D5 {a5['available_est']}/{a5['required']} trained_logged={a5['trained_logged']} fail={a5['failed']} last={a5['last']}")
    lines.append(f"  D6 {a6['available_est']}/{a6['required']} trained_logged={a6['trained_logged']} fail={a6['failed']} last={a6['last']}")
    lines.append(
        f"  TOT {a5['available_est']+a6['available_est']}/3570 "
        f"logged={a5['trained_logged']+a6['trained_logged']} fail={a5['failed']+a6['failed']}"
    )
    b5, b6 = _baselines("D5"), _baselines("D6")
    lines.append("BASELINES")
    lines.append("  method              D5     D6   TOT")
    for method in METHODS:
        d5, _f5 = b5["by_method"][method]
        d6, _f6 = b6["by_method"][method]
        lines.append(f"  {method:<18} {d5:4d}/900 {d6:4d}/900 {d5+d6:4d}/1800")
    lines.append(f"  D5 {b5['completed']}/7200 fail={b5['failed']} last={b5['last']}")
    lines.append(f"  D6 {b6['completed']}/7200 fail={b6['failed']} last={b6['last']}")
    lines.append(f"  TOT {b5['completed']+b6['completed']}/14400 fail={b5['failed']+b6['failed']}")
    s5, s6 = _sita("D5"), _sita("D6")
    lines.append("SITA")
    lines.append(f"  D5 {s5['completed']}/{s5['expected']} {s5['status']} fail={s5['failed']} last={s5['last']}")
    lines.append(f"  D6 {s6['completed']}/{s6['expected']} {s6['status']} fail={s6['failed']} last={s6['last']}")
    lines.append(f"  TOT {s5['completed']+s6['completed']}/1800")
    lines.append("LOG")
    lines.append("  stage_a " + _tail(OUT["D5"] / "stage_a_run_log.txt"))
    lines.append("  d5_bl   " + _tail(OUT["D5"] / "baselines" / "baseline_run_log.txt"))
    lines.append("  d6_bl   " + _tail(OUT["D6"] / "baselines" / "baseline_run_log.txt"))
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
