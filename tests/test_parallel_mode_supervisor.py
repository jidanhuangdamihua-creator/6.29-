from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "parallel_mode_runner.sh"
FAKE_WORKER = ROOT / "tests" / "fixtures" / "fake_formal_worker.py"
FAKE_SETSID = ROOT / "tests" / "fixtures" / "fake_setsid.py"


def _supervisor_environment(tmp_path: Path, **overrides: str) -> dict[str, str]:
    events = tmp_path / "events.jsonl"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "UNIFIED_RUNNER_BIN": str(FAKE_WORKER),
            "SETSID_BIN": str(FAKE_SETSID),
            "RUN_ROOT": str(tmp_path / "formal-run"),
            "FAKE_EVENTS": str(events),
            "TERMINATION_GRACE_SECONDS": "1",
        }
    )
    environment.update(overrides)
    return environment


def _run_supervisor(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=_supervisor_environment(tmp_path, **overrides),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _events(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _peaks(events: list[dict[str, object]]) -> tuple[int, int, int]:
    running: set[str] = set()
    peak = 0
    d5_peak = 0
    thread_peak = 0
    for event in events:
        task = str(event["task"])
        if event["event"] == "start":
            running.add(task)
            peak = max(peak, len(running))
            d5_peak = max(d5_peak, len([name for name in running if name.startswith("d5_")]))
            thread_peak = max(
                thread_peak,
                sum(
                    int(start["threads"])
                    for start in events
                    if start["event"] == "start" and str(start["task"]) in running
                ),
            )
        elif event["event"] in {"finish", "fail", "terminated"}:
            running.discard(task)
    return peak, d5_peak, thread_peak


def test_dry_run_prints_twelve_unique_mode_workers_without_launch(tmp_path: Path) -> None:
    completed = _run_supervisor(tmp_path, DRY_RUN="1", MAX_JOBS="6")

    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("[MODE]")]
    assert len(lines) == 12
    assert len(set(lines)) == 12
    assert "cells=60 unique=60" in completed.stdout
    assert "MAX_JOBS=6" in completed.stdout
    assert "THREAD_BUDGET=16" in completed.stdout
    assert "D5_THREADS=6" in completed.stdout
    assert "ORDINARY_THREADS=2" in completed.stdout
    assert not (tmp_path / "formal-run").exists()
    assert _events(tmp_path) == []


@pytest.mark.parametrize("value", ["0", "13", "x", "1.5", ""])
def test_invalid_max_jobs_fails_before_run_root_creation(
    tmp_path: Path,
    value: str,
) -> None:
    completed = _run_supervisor(tmp_path, MAX_JOBS=value)

    assert completed.returncode == 2
    assert "MAX_JOBS" in completed.stderr
    assert not (tmp_path / "formal-run").exists()
    assert _events(tmp_path) == []


def test_probe_is_fixed_to_four_modes_and_never_aggregates(tmp_path: Path) -> None:
    completed = _run_supervisor(
        tmp_path,
        DRY_RUN="1",
        PROBE="1",
        PUBLISH_GLOBAL="0",
        MAX_JOBS="4",
    )

    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("[MODE]")]
    assert len(lines) == 4
    assert {line.split()[1] for line in lines} == {
        "d1_without",
        "d1_with",
        "d2_without",
        "d2_with",
    }
    assert "aggregate" not in completed.stdout.lower()


def test_scheduler_runs_d5_dependency_lane_first_and_within_thread_budget(
    tmp_path: Path,
) -> None:
    completed = _run_supervisor(
        tmp_path,
        MAX_JOBS="12",
        FAKE_D5_SLEEP="0.8",
        FAKE_ORDINARY_SLEEP="1.2",
    )

    assert completed.returncode == 0, completed.stderr
    events = _events(tmp_path)
    peak, d5_peak, thread_peak = _peaks(events)
    assert 2 <= peak <= 12
    assert d5_peak == 1
    assert thread_peak <= 16
    starts = [event for event in events if event["event"] == "start"]
    assert starts[0]["task"] == "d5_without"
    without_finish = next(
        float(event["time"])
        for event in events
        if event["event"] == "finish" and event["task"] == "d5_without"
    )
    with_start = next(
        float(event["time"])
        for event in events
        if event["event"] == "start" and event["task"] == "d5_with"
    )
    assert with_start >= without_finish
    assert not any(
        event["event"] == "start"
        and not str(event["task"]).startswith("d5_")
        and without_finish < float(event["time"]) < with_start
        for event in events
    )
    assert {
        int(event["threads"])
        for event in starts
        if str(event["task"]).startswith("d5_")
    } == {6}
    assert {
        int(event["threads"])
        for event in starts
        if not str(event["task"]).startswith("d5_")
    } == {2}
    assert [event["event"] for event in events].count("finish") == 12
    assert events[-1]["event"] == "aggregate"
    pid_files = list((tmp_path / "formal-run" / "supervisor").glob("*/pids.tsv"))
    assert len(pid_files) == 1
    pid_file = pid_files[0]
    assert len(pid_file.read_text(encoding="utf-8").splitlines()) >= 25


def test_worker_failure_stops_new_launches_but_preserves_inflight_completions(
    tmp_path: Path,
) -> None:
    completed = _run_supervisor(
        tmp_path,
        MAX_JOBS="4",
        FAKE_D5_SLEEP="0.6",
        FAKE_ORDINARY_SLEEP="0.35",
        FAKE_FAIL_MODE="d1_without",
    )

    assert completed.returncode == 7, completed.stderr
    events = _events(tmp_path)
    failure = next(event for event in events if event["event"] == "fail")
    assert not any(event["event"] == "terminated" for event in events)
    assert any(
        event["event"] == "finish" and float(event["time"]) >= float(failure["time"])
        for event in events
    )
    assert not any(
        event["event"] == "start" and float(event["time"]) > float(failure["time"])
        for event in events
    )
    assert not any(event["event"] == "aggregate" for event in events)
    final = next(event for event in events if event["event"] == "partial_failed")
    assert final["statuses"]["d1_without"] == "failed"
    finished_tasks = {
        str(event["task"]) for event in events if event["event"] == "finish"
    }
    assert finished_tasks
    for task in finished_tasks:
        assert (tmp_path / "formal-run" / task / "accepted.marker").is_file()


def test_d5_without_failure_blocks_d5_with(tmp_path: Path) -> None:
    completed = _run_supervisor(
        tmp_path,
        MAX_JOBS="4",
        FAKE_SLEEP="0.25",
        FAKE_FAIL_MODE="d5_without",
    )

    assert completed.returncode == 7, completed.stderr
    events = _events(tmp_path)
    assert not any(
        event["event"] == "start" and event["task"] == "d5_with"
        for event in events
    )
    final = next(event for event in events if event["event"] == "partial_failed")
    assert final["statuses"]["d5_without"] == "failed"
    assert final["statuses"]["d5_with"] == "blocked"


def test_sigterm_reaches_every_active_worker_group(tmp_path: Path) -> None:
    process = subprocess.Popen(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=_supervisor_environment(
            tmp_path,
            MAX_JOBS="3",
            FAKE_SLEEP="5",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        started = [event for event in _events(tmp_path) if event["event"] == "start"]
        if len(started) == 3:
            break
        time.sleep(0.05)
    else:
        process.kill()
        process.communicate(timeout=5)
        pytest.fail("supervisor did not launch three workers before signal deadline")

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 143, (stdout, stderr)
    events = _events(tmp_path)
    started_tasks = {str(event["task"]) for event in events if event["event"] == "start"}
    terminated_tasks = {
        str(event["task"]) for event in events if event["event"] == "terminated"
    }
    assert terminated_tasks == started_tasks
    assert not any(event["event"] == "aggregate" for event in events)
