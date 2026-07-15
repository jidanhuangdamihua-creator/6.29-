#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time


def _append(event: str, task: str, **extra: object) -> None:
    path = Path(os.environ["FAKE_EVENTS"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "event": event,
            "task": task,
            "pid": os.getpid(),
            "time": time.time(),
            **extra,
        },
        sort_keys=True,
    ) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", required=True)
    parser.add_argument("--only")
    parser.add_argument("--info-sharing")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scheduler-outcome")
    parser.add_argument("--scheduler-reason")
    parser.add_argument("--scheduler-task-status", action="append", default=[])
    args = parser.parse_args()

    if args.operation == "prepare":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run_plan.json").write_text("{}\n", encoding="utf-8")
        _append("prepare", "parent")
        return 0
    if args.operation == "aggregate":
        _append("aggregate", "parent")
        return 0
    if args.operation == "scheduler-finalize":
        statuses = dict(value.split("=", 1) for value in args.scheduler_task_status)
        _append(
            str(args.scheduler_outcome),
            "parent",
            statuses=statuses,
            reason=args.scheduler_reason,
        )
        return 0
    if args.operation != "mode-worker":
        return 2

    task = f"{args.only}_{args.info_sharing}"

    def interrupted(signum, frame):
        _append("terminated", task)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    _append("start", task, threads=threads)
    sleep_seconds = os.environ.get(
        "FAKE_D5_SLEEP" if task.startswith("d5_") else "FAKE_ORDINARY_SLEEP",
        os.environ.get("FAKE_SLEEP", "0.08"),
    )
    time.sleep(float(sleep_seconds))
    if task == os.environ.get("FAKE_FAIL_MODE"):
        _append("fail", task)
        return int(os.environ.get("FAKE_FAIL_CODE", "7"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "accepted.marker").write_text("accepted\n", encoding="utf-8")
    _append("finish", task, threads=threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
