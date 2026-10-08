"""Paired relative improvement, Wilcoxon, percentile bootstrap, and Holm.

The confirmatory analysis passes 12 dataset-by-horizon conditions. The
functions themselves accept any equal-length sample so the formulas can be
checked on a hand-sized example. Seeds and test days are not the resample
unit. Bootstrap draws conditions with replacement from ``numpy.random.default_rng``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


BOOTSTRAP_SEED = 20261001
BOOTSTRAP_RESAMPLES = 100_000
PERCENTILE_METHOD = "linear"


class StatisticsError(ValueError):
    """A statistical input violates the frozen formula."""


@dataclass(frozen=True)
class WinLossTie:
    wins: int
    losses: int
    ties: int


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    n_resamples: int
    seed: int
    percentile_method: str = PERCENTILE_METHOD


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    pvalue: float
    method: str
    n: int
    n_zero_discarded: int
    alternative: str = "two-sided"
    zero_method: str = "wilcox"


def condition_mean(errors: np.ndarray) -> np.ndarray:
    """Mean across the five seeds of each condition.

    ``errors`` has shape ``(n_conditions, 5)``. The returned vector keeps the
    condition order and does not treat seeds as separate samples.
    """
    values = _float_matrix(errors, "errors")
    if values.ndim != 2 or values.shape[1] != 5:
        raise StatisticsError(
            f"condition means require shape (n_conditions, 5), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise StatisticsError("condition means require finite errors")
    return np.mean(values, axis=1, dtype=np.float64)


def relative_improvement(error_x: np.ndarray, error_y: np.ndarray) -> np.ndarray:
    """Return ``(E_Y - E_X) / E_Y * 100``.

    ``error_x`` is the candidate and ``error_y`` is the control. Positive RI
    means the candidate has the lower error. A control error of 0 raises;
    it is not replaced by infinity or zero.
    """
    candidate = _float_vector(error_x, "candidate error")
    control = _float_vector(error_y, "control error")
    if candidate.shape != control.shape:
        raise StatisticsError(
            f"RI requires paired errors of equal length, got {candidate.shape} and {control.shape}"
        )
    if candidate.size == 0:
        raise StatisticsError("RI requires at least one paired condition")
    if not np.isfinite(candidate).all() or not np.isfinite(control).all():
        raise StatisticsError("RI requires finite errors")
    if np.any(control == 0.0):
        raise StatisticsError("relative improvement is undefined when the control error is 0")
    return (control - candidate) / control * np.float64(100.0)


def win_loss_tie(ri: np.ndarray) -> WinLossTie:
    """Count RI > 0 as wins, RI < 0 as losses, and exact zeros as ties."""
    values = _float_vector(ri, "RI")
    if not np.isfinite(values).all():
        raise StatisticsError("win/loss counts require finite RI")
    return WinLossTie(
        wins=int(np.sum(values > 0.0)),
        losses=int(np.sum(values < 0.0)),
        ties=int(np.sum(values == 0.0)),
    )


def paired_wilcoxon(ri: np.ndarray) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test on paired condition RIs.

    SciPy is called with ``zero_method='wilcox'``, ``method='auto'``, and
    ``alternative='two-sided'``. SciPy 1.13 selects the exact null when the
    sample length is at most 50 and no RI is zero; otherwise it selects the
    normal approximation. Zeros are discarded by the wilcox rule. The method
    actually selected is stored on the result.
    """
    values = _float_vector(ri, "RI")
    if not np.isfinite(values).all():
        raise StatisticsError("Wilcoxon requires finite RI")
    n_zero = int(np.sum(values == 0.0))
    if values.size <= 50 and n_zero == 0:
        method = "exact"
    else:
        method = "approx"
    result = stats.wilcoxon(
        values,
        zero_method="wilcox",
        method="auto",
        alternative="two-sided",
    )
    return WilcoxonResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        method=method,
        n=int(values.size - n_zero),
        n_zero_discarded=n_zero,
    )


def percentile_bootstrap(
    ri: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Percentile interval for the mean RI.

    Each resample draws the observed conditions with replacement and takes
    their arithmetic mean. The interval is the 2.5 and 97.5 percentiles of
    those means, using NumPy's linear percentile. The generator is restarted
    from ``seed`` on every call.
    """
    values = _float_vector(ri, "RI")
    if not np.isfinite(values).all():
        raise StatisticsError("bootstrap requires finite RI")
    if values.size == 0:
        raise StatisticsError("bootstrap requires at least one condition")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or int(n_resamples) < 1:
        raise StatisticsError("bootstrap resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise StatisticsError("bootstrap seed must be an integer")
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, values.size, size=(int(n_resamples), values.size))
    means = np.mean(values[draws], axis=1, dtype=np.float64)
    low, high = np.percentile(means, [2.5, 97.5], method=PERCENTILE_METHOD)
    return Interval(
        low=float(low),
        high=float(high),
        n_resamples=int(n_resamples),
        seed=int(seed),
    )


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values in the original order.

    Sorted p-values are multiplied by ``m, m-1, ..., 1`` and capped at 1.
    The adjusted values are then made nondecreasing in sorted order. Ties
    keep their original relative order. This adjustment is only valid inside
    one predeclared family and one metric.
    """
    values = _float_vector(p_values, "p-values")
    if np.any((values < 0.0) | (values > 1.0) | ~np.isfinite(values)):
        raise StatisticsError("Holm requires finite p-values in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    m = values.size
    for rank, index in enumerate(order):
        factor = m - rank
        candidate = min(1.0, float(factor) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def five_number(values: np.ndarray) -> dict[str, float]:
    """Minimum, quartiles, and maximum of one paired-difference vector."""
    sample = _float_vector(values, "paired differences")
    if not np.isfinite(sample).all() or sample.size == 0:
        raise StatisticsError("a paired-difference summary requires finite values")
    quartiles = np.percentile(sample, [0.0, 25.0, 50.0, 75.0, 100.0], method=PERCENTILE_METHOD)
    return {
        "min": float(quartiles[0]),
        "q1": float(quartiles[1]),
        "median": float(quartiles[2]),
        "q3": float(quartiles[3]),
        "max": float(quartiles[4]),
    }


def _float_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise StatisticsError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _float_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise StatisticsError(f"{name} must be two-dimensional, got shape {array.shape}")
    return array


def main() -> int:
    """Run the full frozen analysis. Holm families are not narrowed at runtime."""
    from analysis.run_statistics import main as run_main

    return run_main()


if __name__ == "__main__":
    raise SystemExit(main())
