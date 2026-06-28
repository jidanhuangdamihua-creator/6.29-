from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import time


@dataclass
class ProgressSnapshot:
    total_runs: int
    completed_runs: int
    current_dataset: str
    current_method: str
    elapsed_seconds: float
    average_seconds_per_run: float
    eta_seconds: float


class ExperimentProgressTracker:
    """Track experiment progress and provide ETA snapshots."""

    def __init__(self, total_runs: int) -> None:
        if total_runs <= 0:
            raise ValueError("total_runs must be positive")
        self.total_runs = int(total_runs)
        self.completed_runs = 0
        self.start_time = time.time()

    def update(self, current_dataset: str, current_method: str) -> ProgressSnapshot:
        elapsed = max(0.0, time.time() - self.start_time)
        completed = max(0, self.completed_runs)
        avg = (elapsed / completed) if completed > 0 else 0.0
        remaining = max(0, self.total_runs - completed)
        eta = avg * remaining if completed > 0 else 0.0

        return ProgressSnapshot(
            total_runs=self.total_runs,
            completed_runs=completed,
            current_dataset=str(current_dataset),
            current_method=str(current_method),
            elapsed_seconds=elapsed,
            average_seconds_per_run=avg,
            eta_seconds=eta,
        )

    def mark_completed(self) -> None:
        if self.completed_runs < self.total_runs:
            self.completed_runs += 1



def format_seconds(seconds: float) -> str:
    seconds_i = max(0, int(round(float(seconds))))
    return str(timedelta(seconds=seconds_i))
