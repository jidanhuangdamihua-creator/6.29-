#!/usr/bin/env python3
"""Every 30s: 10-target ablation + 40-target +RevIN progress."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = (
    {
        "name": "消融 10-target",
        "expected": 2700,
        "cells": ROOT / "outputs" / "ablation_10targets",
        "log": ROOT / "logs" / "ablation_10targets.log",
        "failures": ROOT / "results" / "ablation_10targets_failures.csv",
        "pattern": "scripts/run_ablation_10targets.py",
    },
    {
        "name": "+RevIN 40-target",
        "expected": 7200,
        "cells": ROOT / "outputs" / "baselines_revin_40target",
        "log": ROOT / "logs" / "revin_baselines_40target.log",
        "failures": ROOT / "results" / "revin_baseline_failures.csv",
        "pattern": "scripts/run_baselines_revin_40target.py",
    },
    {
        "name": "SITA sales-only",
        "expected": 1800,
        "cells": ROOT / "outputs" / "sita_sales_only_ablation_20261007" / "predictions",
        "log": ROOT / "logs" / "sita_sales_only.log",
        "failures": ROOT / "outputs" / "sita_sales_only_ablation_20261007" / "failures.csv",
        "pattern": "scripts/run_sita_sales_only_ablation.py",
    },
)


def count_done(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.rglob("DONE.json"))


def running(pattern: str) -> str:
    try:
        out = subprocess.check_output(["pgrep", "-fl", pattern], text=True)
    except subprocess.CalledProcessError:
        return "未在跑"
    lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "watch_ablation_revin" not in ln]
    if not lines:
        return "未在跑"
    pid = lines[0].split(None, 1)[0]
    return f"在跑 pid={pid}"


def fail_n(path: Path) -> int:
    if not path.is_file():
        return 0
    n = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
    return max(0, n - 1)


def tail_log(path: Path) -> str:
    if not path.is_file():
        return "(无日志)"
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(读日志失败)"
    for line in reversed(text):
        line = line.strip()
        if line:
            return line[:160]
    return "(空日志)"


def eta(done: int, expected: int, rate: float | None) -> str:
    left = max(0, expected - done)
    if left == 0:
        return "已完成"
    if rate is None or rate <= 0:
        return "—"
    sec = left / rate
    hours, rem = divmod(int(sec), 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def snapshot(prev: dict[str, tuple[float, int]]) -> dict[str, tuple[float, int]]:
    now = time.time()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    print("\033[2J\033[H", end="")
    print(f"{stamp}  每 30 秒刷新  Ctrl+C 退出")
    print("-" * 72)
    out = {}
    for job in JOBS:
        done = count_done(job["cells"])
        expected = int(job["expected"])
        pct = 100.0 * done / expected if expected else 0.0
        rate = None
        if job["name"] in prev:
            t0, n0 = prev[job["name"]]
            dt = now - t0
            if dt > 0:
                rate = (done - n0) / dt
        rate_txt = f"{rate:.2f} 格/s" if rate is not None else "速率待第二拍"
        print(f"{job['name']}  {running(job['pattern'])}")
        print(
            f"  {done}/{expected}  {pct:.1f}%  失败 {fail_n(job['failures'])}  "
            f"{rate_txt}  剩余约 {eta(done, expected, rate)}"
        )
        print(f"  日志: {tail_log(job['log'])}")
        print()
        out[job["name"]] = (now, done)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    prev: dict[str, tuple[float, int]] = {}
    try:
        while True:
            prev = snapshot(prev)
            if args.once:
                return 0
            time.sleep(max(1, int(args.interval)))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
