"""Canonical filesystem layout for one formal D1-D6 run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.protocols.experiment_protocol import normalize_scenario


@dataclass(frozen=True)
class RunLayout:
    run_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", Path(self.run_root))

    @staticmethod
    def _dataset_id(dataset_id: int) -> int:
        value = int(dataset_id)
        if value not in range(1, 7):
            raise ValueError("dataset_id must be between 1 and 6")
        return value

    @staticmethod
    def _mode(mode: str) -> str:
        try:
            normalized = normalize_scenario(mode)
        except ValueError as exc:
            raise ValueError("mode must be without or with") from exc
        if normalized not in {"without", "with"}:
            raise ValueError("mode must be without or with")
        return normalized

    @staticmethod
    def _cell_value(name: str, value: int, allowed: range) -> int:
        normalized = int(value)
        if normalized not in allowed:
            raise ValueError(f"{name} is outside the formal protocol")
        return normalized

    def mode_dir(self, dataset_id: int, mode: str) -> Path:
        dataset = self._dataset_id(dataset_id)
        normalized_mode = self._mode(mode)
        return self.run_root / f"d{dataset}_{normalized_mode}"

    @property
    def attempts_dir(self) -> Path:
        return self.run_root / "attempts"

    def attempt_dir(self, attempt_id: str) -> Path:
        value = str(attempt_id)
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("attempt_id must be one safe path component")
        return self.attempts_dir / value

    def attempt_manifest(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "attempt_manifest.json"

    def attempt_events_dir(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "scheduler_events"

    def attempt_worker_logs(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "worker_logs"

    def attempt_cells_dir(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "cells"

    def attempt_result(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "attempt_result.json"

    @property
    def state(self) -> Path:
        return self.run_root / "state.json"

    @property
    def lease(self) -> Path:
        return self.run_root / "lease.json"

    @property
    def recovery_lock(self) -> Path:
        return self.run_root / ".recovery.lock"

    @property
    def sealed_success(self) -> Path:
        return self.run_root / "SEALED_SUCCESS"

    def cell_dir(self, dataset_id: int, mode: str, horizon: int, seed: int) -> Path:
        normalized_horizon = self._cell_value("horizon", horizon, range(1, 6))
        normalized_seed = self._cell_value("seed", seed, range(42, 47))
        return (
            self.mode_dir(dataset_id, mode)
            / "cells"
            / f"h{normalized_horizon}_s{normalized_seed}"
        )

    def cell_result(self, dataset_id: int, mode: str, horizon: int, seed: int) -> Path:
        dataset = self._dataset_id(dataset_id)
        normalized_mode = self._mode(mode)
        return (
            self.cell_dir(dataset, normalized_mode, horizon, seed)
            / "results"
            / f"dataset{dataset}_{normalized_mode}_results.csv"
        )

    def cell_manifest(self, dataset_id: int, mode: str, horizon: int, seed: int) -> Path:
        return self.cell_result(dataset_id, mode, horizon, seed).with_suffix(".manifest.json")

    def cell_acceptance_report(
        self, dataset_id: int, mode: str, horizon: int, seed: int
    ) -> Path:
        return self.cell_result(dataset_id, mode, horizon, seed).with_suffix(
            ".acceptance.json"
        )

    def mode_result(self, dataset_id: int, mode: str) -> Path:
        dataset = self._dataset_id(dataset_id)
        normalized_mode = self._mode(mode)
        return (
            self.mode_dir(dataset, normalized_mode)
            / "results"
            / f"dataset{dataset}_{normalized_mode}_results.csv"
        )

    def mode_manifest(self, dataset_id: int, mode: str) -> Path:
        return self.mode_result(dataset_id, mode).with_suffix(".manifest.json")

    def mode_acceptance_report(self, dataset_id: int, mode: str) -> Path:
        return self.mode_result(dataset_id, mode).with_suffix(".acceptance.json")

    @property
    def aggregate_result(self) -> Path:
        return self.run_root / "results" / "d1_d6_results.csv"

    @property
    def aggregate_manifest(self) -> Path:
        return self.aggregate_result.with_suffix(".manifest.json")

    @property
    def aggregate_acceptance_report(self) -> Path:
        return self.aggregate_result.with_suffix(".acceptance.json")
