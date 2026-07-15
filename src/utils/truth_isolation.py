"""Runtime tripwires for proving worker code never loads evaluator truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


class SealedTruthAccessError(RuntimeError):
    """A worker attempted to access evaluator-only truth."""


@dataclass
class TruthAccessTripwire:
    """Record every forbidden access before raising.

    The counter is deliberately external to the fitting callable.  Catching the
    exception inside a model adapter therefore cannot make a truth access pass.
    """

    access_log: List[Dict[str, Any]] = field(default_factory=list)
    evaluator_loader_call_count: int = 0

    @property
    def attempted_access_count(self) -> int:
        return len(self.access_log)

    def access(self, resource: str, detail: Optional[str] = None) -> None:
        entry: Dict[str, Any] = {
            "resource": str(resource),
            "accessed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if detail is not None:
            entry["detail"] = str(detail)
        self.access_log.append(entry)
        raise SealedTruthAccessError(f"worker truth access is forbidden: {resource}")

    def evaluator_loader_called(self, detail: Optional[str] = None) -> None:
        self.evaluator_loader_call_count += 1
        self.access("evaluator_loader", detail)

    def assert_clean(self) -> None:
        assert self.attempted_access_count == 0, (
            "attempted_access_count must be zero; "
            f"observed {self.attempted_access_count} access(es)"
        )
        assert self.evaluator_loader_call_count == 0, (
            "evaluator_loader_call_count must be zero; "
            f"observed {self.evaluator_loader_call_count} call(s)"
        )


def run_truth_free_fit(
    fit: Callable[[], Any],
    *,
    tripwire: TruthAccessTripwire | None = None,
) -> Any:
    """Run a fitting callable and enforce the external truth-access contract."""

    monitor = tripwire or TruthAccessTripwire()
    result = fit()
    monitor.assert_clean()
    return result


__all__ = [
    "SealedTruthAccessError",
    "TruthAccessTripwire",
    "run_truth_free_fit",
]
