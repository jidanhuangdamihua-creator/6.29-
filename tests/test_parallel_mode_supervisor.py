from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "parallel_mode_runner.sh"
FAKE_WORKER = ROOT / "tests" / "fixtures" / "fake_formal_worker.py"
FAKE_SETSID = ROOT / "tests" / "fixtures" / "fake_setsid.py"


def _process_group_support_available() -> bool:
    with tempfile.TemporaryDirectory(prefix="parallel-mode-probe-") as probe_dir:
        probe_root = Path(probe_dir)
        probe_environment = os.environ.copy()
        probe_environment.update(
            {
                "FAKE_EVENTS": str(probe_root / "events.jsonl"),
                "FAKE_SLEEP": "0.5",
            }
        )
        probe_script = r'''
set -Eeuo pipefail

setsid_bin="$1"
python_bin="$2"
worker_bin="$3"
worker_output="$4"
supervisor_pgid="$(ps -o pgid= -p $$ | awk '{gsub(/[[:space:]]/, "", $0); print}')"
worker_pgid=""

nohup "${setsid_bin}" --wait \
    "${python_bin}" "${worker_bin}" \
    --operation mode-worker \
    --only 1 \
    --info-sharing without \
    --output-dir "${worker_output}" \
    >"${worker_output}.log" 2>&1 </dev/null &
launcher_pid=$!

cleanup() {
    if [[ "${worker_pgid}" =~ ^[0-9]+$ && "${worker_pgid}" -gt 1 ]]; then
        kill -TERM -- "-${worker_pgid}" 2>/dev/null || true
    fi
    kill -TERM "${launcher_pid}" 2>/dev/null || true
    wait "${launcher_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for ((attempt = 0; attempt < 50; attempt++)); do
    worker_pid="$(pgrep -P "${launcher_pid}" 2>/dev/null | head -n 1 || true)"
    if [[ "${worker_pid}" =~ ^[0-9]+$ ]]; then
        worker_pgid="$(ps -o pgid= -p "${worker_pid}" \
            | awk '{gsub(/[[:space:]]/, "", $0); print}')"
        if [[ "${worker_pgid}" =~ ^[0-9]+$ \
            && "${worker_pgid}" != "${supervisor_pgid}" ]]; then
            exit 0
        fi
    fi
    kill -0 "${launcher_pid}" 2>/dev/null || exit 1
    sleep 0.1
done
exit 1
'''
        try:
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    probe_script,
                    "parallel-mode-probe",
                    str(FAKE_SETSID),
                    sys.executable,
                    str(FAKE_WORKER),
                    str(probe_root / "worker"),
                ],
                cwd=ROOT,
                env=probe_environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
            return completed.returncode == 0
        except (OSError, ValueError):
            return False
        except subprocess.TimeoutExpired:
            return False


PROCESS_GROUP_TEST = pytest.mark.skipif(
    not _process_group_support_available(),
    reason="sandbox does not permit process-group creation",
)


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


def _peaks(events: list[dict[str, object]]) -> tuple[int, int]:
    running: set[str] = set()
    peak = 0
    d5_peak = 0
    for event in events:
        task = str(event["task"])
        if event["event"] == "start":
            running.add(task)
            peak = max(peak, len(running))
            d5_peak = max(d5_peak, len([name for name in running if name.startswith("d5_")]))
        elif event["event"] in {"finish", "fail", "terminated"}:
            running.discard(task)
    return peak, d5_peak


def test_dry_run_prints_twelve_unique_mode_workers_without_launch(tmp_path: Path) -> None:
    completed = _run_supervisor(tmp_path, DRY_RUN="1", MAX_JOBS="6")

    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("[MODE]")]
    assert len(lines) == 12
    assert len(set(lines)) == 12
    assert "cells=300 unique=300" in completed.stdout
    assert "MAX_JOBS=6" in completed.stdout
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


@PROCESS_GROUP_TEST
def test_scheduler_ignores_transient_setsid_shim(tmp_path: Path) -> None:
    completed = _run_supervisor(
        tmp_path,
        MAX_JOBS="1",
        FAKE_SLEEP="0.5",
        FAKE_SETSID_PREEXEC_SHIM="1",
        FAKE_SETSID_SHIM_DELAY="0.2",
    )

    assert completed.returncode == 0, completed.stderr
    events = _events(tmp_path)
    assert [event["event"] for event in events].count("finish") == 12
    assert events[-1]["event"] == "aggregate"


@PROCESS_GROUP_TEST
def test_scheduler_enforces_global_and_d5_caps_and_overlaps(tmp_path: Path) -> None:
    completed = _run_supervisor(tmp_path, MAX_JOBS="4", FAKE_SLEEP="0.5")

    assert completed.returncode == 0, completed.stderr
    events = _events(tmp_path)
    peak, d5_peak = _peaks(events)
    assert 2 <= peak <= 4
    assert d5_peak == 1
    assert [event["event"] for event in events].count("finish") == 12
    assert events[-1]["event"] == "aggregate"
    pid_files = list((tmp_path / "formal-run" / "supervisor").glob("*/pids.tsv"))
    assert len(pid_files) == 1
    pid_file = pid_files[0]
    assert len(pid_file.read_text(encoding="utf-8").splitlines()) >= 25


@PROCESS_GROUP_TEST
def test_first_worker_failure_terminates_peers_and_skips_global(tmp_path: Path) -> None:
    completed = _run_supervisor(
        tmp_path,
        MAX_JOBS="4",
        FAKE_SLEEP="0.35",
        FAKE_FAIL_MODE="d1_without",
    )

    assert completed.returncode == 7, completed.stderr
    events = _events(tmp_path)
    assert any(event["event"] == "fail" for event in events)
    assert any(event["event"] == "terminated" for event in events)
    assert not any(event["event"] == "aggregate" for event in events)
    assert len([event for event in events if event["event"] == "start"]) <= 4


@PROCESS_GROUP_TEST
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
