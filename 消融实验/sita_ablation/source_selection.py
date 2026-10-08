"""Rank legal sources on the frozen comparison window.

Distances and both weight vectors are computed here. Alpha entropy and signed
error stay unresolved (OPEN-1, OPEN-2) and are not calculated. This module does
not fuse convolution parameters.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from sita_ablation.d5_source_history import (
    D5_HISTORY_DAYS,
    D5_HISTORY_END,
    D5_HISTORY_START,
    ReconstructionSummary,
    expected_source_history_dates,
    inactive_reconstruction_summary,
    reconstruct_d5_source_history,
)
from sita_ablation.data import (
    DATASETS,
    ComparisonDays,
    ConditionFrames,
    ContractError,
    DatasetId,
    DateSpan,
    load_dataset_contract,
    target_series,
)
from sita_ablation.data import (
    _between,
    _component,
    _encode_similarity,
    _iso,
    _normalize_key,
    _similarity_window,
    _target_group_value,
)
from sita_ablation.scaling import SelectionScaler, fit_selection_scaler, transform_selection

CellSeed = Literal[42, 43, 44, 45, 46]
SelectionRule = Literal["top", "bottom", "random3"]
CELL_SEEDS: tuple[CellSeed, ...] = (42, 43, 44, 45, 46)
SELECTION_RULES: tuple[SelectionRule, ...] = ("top", "bottom", "random3")
# Formal anchored ranking. src/protocols/experiment_protocol.py:205 and INTERFACES.md §5.
TIE_TOLERANCE = 1e-12
# Formal inverse-distance epsilon. src/protocols/experiment_protocol.py:207 and INTERFACES.md §5.
WEIGHT_EPSILON = 1e-8
STATUS_COMPLETED = "COMPLETED"
STATUS_SKIPPED = "SKIPPED_INSUFFICIENT_CANDIDATES"
RANDOM3_DRAW = "sorted_permutation_v1"


@dataclass(frozen=True)
class RankedSource:
    source_key: tuple[str, ...]
    distance: float
    tie_group: int


@dataclass(frozen=True)
class SelectedSource:
    source_key: tuple[str, ...]
    rank: int
    distance: float
    tie_group: int


@dataclass(frozen=True)
class ExclusionRecord:
    source_key: tuple[str, ...]
    reason: str
    missing_dates: tuple[str, ...] = ()
    variable: str | None = None
    gate: str = ""


@dataclass(frozen=True)
class GateCounts:
    identity: int
    source_history: int
    comparison_window: int
    comparison_variables: int


@dataclass(frozen=True)
class SelectionResult:
    dataset: DatasetId
    status: str
    rule: SelectionRule
    requested_k: int
    seed: CellSeed
    comparison_days: ComparisonDays
    comparison_start: str
    comparison_end: str
    comparison_dates: tuple[str, ...]
    similarity_variables: tuple[str, ...]
    target_key: tuple[str, ...]
    selected: tuple[SelectedSource, ...]
    distances: np.ndarray
    inverse_distance_weights: np.ndarray
    uniform_weights: np.ndarray
    selection_scaler: SelectionScaler | None
    exclusions: tuple[ExclusionRecord, ...]
    n_legal: int
    legal_keys: tuple[tuple[str, ...], ...]
    source_history_start: str
    source_history_end: str
    gate_counts: GateCounts
    reconstruction: ReconstructionSummary
    scaler_population_keys: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.status not in {STATUS_COMPLETED, STATUS_SKIPPED}:
            raise ContractError(f"unknown selection status {self.status!r}")
        _require_float_vector("distances", self.distances)
        _require_float_vector("inverse_distance_weights", self.inverse_distance_weights)
        _require_float_vector("uniform_weights", self.uniform_weights)
        if len(self.comparison_dates) != self.comparison_days:
            raise ContractError("comparison date count does not match the requested window")
        if self.comparison_dates[0] != self.comparison_start or self.comparison_dates[-1] != self.comparison_end:
            raise ContractError("comparison endpoints do not match the date list")
        if self.n_legal != len(self.legal_keys):
            raise ContractError("n_legal does not match legal_keys")
        if self.gate_counts.comparison_variables != self.n_legal:
            raise ContractError("the variable gate count does not match the legal candidate count")
        if self.gate_counts.identity < self.gate_counts.source_history:
            raise ContractError("source-history count exceeds the identity count")
        if self.gate_counts.source_history < self.gate_counts.comparison_window:
            raise ContractError("comparison-window count exceeds the source-history count")
        if self.gate_counts.comparison_window < self.gate_counts.comparison_variables:
            raise ContractError("variable count exceeds the comparison-window count")
        if self.source_history_start > self.source_history_end:
            raise ContractError("source history ends before it starts")
        if self.dataset == "D5" and (
            self.source_history_start != D5_HISTORY_START or self.source_history_end != D5_HISTORY_END
        ):
            raise ContractError("D5 source history must stay on 2016-08-20 through 2017-02-15")
        if self.dataset == "D5" and not self.reconstruction.applied:
            raise ContractError("D5 selection did not apply source-history reconstruction")
        if self.dataset != "D5" and self.reconstruction.applied:
            raise ContractError(f"{self.dataset} selection applied D5 reconstruction")
        if set(self.reconstruction.identities_only_after_cutoff) & set(self.legal_keys):
            raise ContractError("an identity from after the cutoff entered the legal pool")
        if self.status == STATUS_SKIPPED:
            if self.selected or self.n_legal >= self.requested_k:
                raise ContractError("a skipped selection must have fewer legal candidates than K and no selected sources")
            if self.distances.shape != (0,) or self.inverse_distance_weights.shape != (0,) or self.uniform_weights.shape != (0,):
                raise ContractError("a skipped selection has no weight vector to pad")
            if self.selection_scaler is not None:
                raise ContractError("a skipped selection does not fit a selection scaler")
            if self.scaler_population_keys:
                raise ContractError("a skipped selection must not fit a scaler population")
            return
        if self.scaler_population_keys != self.legal_keys:
            raise ContractError("the scaler population is not the final legal candidate set")
        if len(self.selected) != self.requested_k or self.distances.shape != (self.requested_k,):
            raise ContractError("completed selection must return exactly K sources and distances")
        if self.inverse_distance_weights.shape != (self.requested_k,) or self.uniform_weights.shape != (self.requested_k,):
            raise ContractError("completed selection weights must have length K")
        if self.selection_scaler is None:
            raise ContractError("completed selection is missing the selection scaler")
        if tuple(item.rank for item in self.selected) != tuple(range(self.requested_k)):
            raise ContractError("selected ranks must be 0..K-1 in training order")
        if [item.distance for item in self.selected] != self.distances.tolist():
            raise ContractError("selected distances and the distance vector disagree")
        expected_inverse = inverse_distance_weights(self.distances)
        expected_uniform = uniform_weights(self.requested_k)
        if not np.array_equal(self.inverse_distance_weights, expected_inverse):
            raise ContractError("inverse-distance weights do not match 1/(distance+1e-8) normalized")
        if not np.array_equal(self.uniform_weights, expected_uniform):
            raise ContractError("uniform weights are not 1/K")


def rank_candidates(
    dataset: DatasetId,
    frames: ConditionFrames,
    *,
    comparison_days: Literal[15, 30],
    rule: Literal["top", "bottom", "random3"],
    k: int,
    seed: CellSeed,
) -> SelectionResult:
    """Select K legal sources, or skip when the legal pool is smaller than K."""
    if dataset not in DATASETS:
        raise ContractError(f"unsupported dataset {dataset!r}")
    if type(comparison_days) is not int or comparison_days not in (15, 30):
        raise ContractError(f"comparison window must be 15 or 30, got {comparison_days!r}")
    if rule not in SELECTION_RULES:
        raise ContractError(f"selection rule must be top, bottom, or random3, got {rule!r}")
    if type(k) is not int or k < 1:
        raise ContractError(f"K must be a positive integer, got {k!r}")
    if rule == "random3" and k != 3:
        raise ContractError("random3 draws exactly 3 sources")
    if seed not in CELL_SEEDS:
        raise ContractError(f"cell seed must be one of {CELL_SEEDS}, got {seed!r}")
    if frames.dataset != dataset:
        raise ContractError(f"frames dataset {frames.dataset!r} does not match {dataset!r}")

    contract = load_dataset_contract(dataset)
    _require_frames_match_contract(frames, contract)
    span, comparison_dates = _comparison_calendar(frames, comparison_days)
    history_dates = _source_history_dates(frames)
    variables = contract.similarity_variables
    screened, gate_counts, reconstruction = _screen_candidates(frames, span, history_dates)
    legal_pairs: list[tuple[tuple[str, ...], np.ndarray]] = []
    exclusion_records: list[ExclusionRecord] = []
    for item in screened:
        if item.exclusion is None:
            if item.window is None:
                raise ContractError(f"{dataset} legal candidate {item.source_key!r} has no comparison window")
            legal_pairs.append((item.source_key, item.window))
        else:
            exclusion_records.append(item.exclusion)
    legal_pairs.sort(key=lambda item: item[0])
    exclusion_records.sort(key=lambda record: (record.source_key, record.reason, record.variable or "", record.gate))
    legal_keys = tuple(key for key, _window in legal_pairs)
    if gate_counts.comparison_variables != len(legal_keys):
        raise ContractError(f"{dataset} variable gate count does not match the screened legal keys")
    target_window = _require_target_window(frames, span, variables)
    common = {
        "dataset": dataset,
        "rule": rule,
        "requested_k": k,
        "seed": seed,
        "comparison_days": comparison_days,
        "comparison_start": comparison_dates[0],
        "comparison_end": comparison_dates[-1],
        "comparison_dates": comparison_dates,
        "similarity_variables": variables,
        "target_key": contract.target_key,
        "exclusions": tuple(exclusion_records),
        "n_legal": len(legal_keys),
        "legal_keys": legal_keys,
        "source_history_start": _iso(history_dates[0]),
        "source_history_end": _iso(history_dates[-1]),
        "gate_counts": gate_counts,
        "reconstruction": reconstruction,
    }
    if len(legal_pairs) < k:
        return SelectionResult(
            status=STATUS_SKIPPED,
            selected=(),
            distances=np.empty(0, dtype=np.float64),
            inverse_distance_weights=np.empty(0, dtype=np.float64),
            uniform_weights=np.empty(0, dtype=np.float64),
            selection_scaler=None,
            scaler_population_keys=(),
            **common,
        )

    candidate_windows = np.stack([window for _key, window in legal_pairs]).astype(np.float64, copy=False)
    scaler = fit_selection_scaler(target_window, candidate_windows, variables)
    scaled_target = transform_selection(scaler, target_window)
    scaled_candidates = transform_selection(scaler, candidate_windows)
    if scaled_target.shape != (len(variables) * len(comparison_dates),):
        raise ContractError("scaled target vector does not follow the frozen variable order")
    if scaled_candidates.shape != (len(legal_pairs), len(variables) * len(comparison_dates)):
        raise ContractError("scaled candidate matrix does not follow the frozen variable order")
    distances = np.linalg.norm(scaled_candidates - scaled_target, axis=1).astype(np.float64, copy=False)
    ranked = anchored_rank(legal_keys, distances)
    chosen = _choose(ranked, legal_keys, rule=rule, k=k, seed=seed)
    selected_distances = np.array([item.distance for item in chosen], dtype=np.float64)
    selected = tuple(
        SelectedSource(item.source_key, index, item.distance, item.tie_group)
        for index, item in enumerate(chosen)
    )
    return SelectionResult(
        status=STATUS_COMPLETED,
        selected=selected,
        distances=selected_distances,
        inverse_distance_weights=inverse_distance_weights(selected_distances),
        uniform_weights=uniform_weights(k),
        selection_scaler=scaler,
        scaler_population_keys=legal_keys,
        **common,
    )


def anchored_rank(
    source_keys: Sequence[Sequence[str]],
    distances: np.ndarray,
    *,
    tie_tolerance: float = TIE_TOLERANCE,
) -> tuple[RankedSource, ...]:
    """Rank by ascending distance with the formal anchored, non-chaining tie groups.

    Keys are the only order inside a tie group. No further tie-break is applied.
    """
    keys = tuple(tuple(str(part) for part in key) for key in source_keys)
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if len(keys) != values.size:
        raise ContractError("source key and distance counts differ")
    if len(set(keys)) != len(keys):
        raise ContractError("distance ranking contains duplicate source keys")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ContractError("source distances must be finite and non-negative")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ContractError("tie_tolerance must be finite and non-negative")

    ordered = sorted(zip(keys, values.tolist()), key=lambda item: (item[1], item[0]))
    ranked: list[RankedSource] = []
    tie_group = 1
    start = 0
    while start < len(ordered):
        anchor = float(ordered[start][1])
        stop = start + 1
        while stop < len(ordered) and float(ordered[stop][1]) - anchor <= tie_tolerance:
            stop += 1
        current = [(key, float(raw_distance)) for key, raw_distance in ordered[start:stop]]
        current.sort(key=lambda item: item[0])
        ranked.extend(RankedSource(key, raw_distance, tie_group) for key, raw_distance in current)
        start = stop
        tie_group += 1
    return tuple(ranked)


def derived_random3_seed(cell_seed: int) -> int:
    """32-bit seed from SHA256(f"{cell_seed}|random3|v1"), little-endian first four bytes."""
    if cell_seed not in CELL_SEEDS:
        raise ContractError(f"cell seed must be one of {CELL_SEEDS}, got {cell_seed!r}")
    digest = hashlib.sha256(f"{cell_seed}|random3|v1".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ContractError("inverse-distance weights require finite non-negative distances")
    scores = np.float64(1.0) / (values + np.float64(WEIGHT_EPSILON))
    total = np.sum(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(total) or total == 0.0:
        raise ContractError("inverse-distance weights are non-finite")
    return np.array(scores / total, dtype=np.float64, copy=True)


def uniform_weights(k: int) -> np.ndarray:
    if type(k) is not int or k < 1:
        raise ContractError(f"uniform weights require a positive K, got {k!r}")
    if k == 1:
        return np.array([1.0], dtype=np.float64)
    return np.full(k, np.float64(1.0) / np.float64(k), dtype=np.float64)


def source_key_overlap(left: SelectionResult, right: SelectionResult) -> int:
    """Size of the intersection of two selected key sets. Random-3 uses the same draw stream."""
    left_keys = {item.source_key for item in left.selected}
    right_keys = {item.source_key for item in right.selected}
    return len(left_keys & right_keys)


def top3_overlap_count(
    cell: SelectionResult,
    reference: SelectionResult | None,
    *,
    horizon: int,
    reference_horizon: int,
) -> int | None:
    """Intersection of a random3 or bottom selection with the completed top/main selection.

    Returns an integer only when both selections are ``COMPLETED`` and share
    dataset, horizon, seed, a 30-day window, and ``K=3``. Otherwise returns
    ``None``. A missing comparison is not reported as 0. ``io`` does not call
    this function.
    """
    if reference is None or cell.status != STATUS_COMPLETED or reference.status != STATUS_COMPLETED:
        return None
    if cell.rule not in {"random3", "bottom"} or reference.rule != "top":
        return None
    if cell.requested_k != 3 or reference.requested_k != 3:
        return None
    if cell.comparison_days != 30 or reference.comparison_days != 30:
        return None
    if cell.dataset != reference.dataset or cell.seed != reference.seed:
        return None
    if type(horizon) is not int or type(reference_horizon) is not int or isinstance(horizon, bool):
        return None
    if horizon != reference_horizon or horizon not in (1, 3, 5):
        return None
    count = source_key_overlap(cell, reference)
    if count > 3:
        raise ContractError("top3 overlap count must stay in 0..3")
    return count


@dataclass(frozen=True)
class _ScreenedCandidate:
    source_key: tuple[str, ...]
    window: np.ndarray | None
    exclusion: ExclusionRecord | None


def _choose(
    ranked: tuple[RankedSource, ...],
    legal_keys: tuple[tuple[str, ...], ...],
    *,
    rule: SelectionRule,
    k: int,
    seed: CellSeed,
) -> tuple[RankedSource, ...]:
    by_key = {item.source_key: item for item in ranked}
    if rule == "top":
        return ranked[:k]
    if rule == "bottom":
        return ranked[-k:]
    drawn = _draw_random3(legal_keys, seed)
    return tuple(
        RankedSource(key, by_key[key].distance, by_key[key].tie_group)
        for key in drawn
    )


def _draw_random3(legal_keys: Sequence[tuple[str, ...]], cell_seed: CellSeed) -> tuple[tuple[str, ...], ...]:
    ordered = tuple(sorted(tuple(key) for key in legal_keys))
    if len(ordered) < 3:
        raise ContractError("random3 cannot draw from fewer than 3 legal candidates")
    generator = np.random.default_rng(derived_random3_seed(cell_seed))
    permutation = generator.permutation(len(ordered))
    return tuple(ordered[int(index)] for index in permutation[:3])


def _comparison_calendar(frames: ConditionFrames, comparison_days: ComparisonDays) -> tuple[DateSpan, tuple[str, ...]]:
    span = frames.comparison_30 if comparison_days == 30 else frames.comparison_15
    dates = pd.date_range(span.start, span.end, freq="D")
    if len(dates) != comparison_days:
        raise ContractError(f"{frames.dataset} comparison window has {len(dates)} days, expected {comparison_days}")
    if span.end >= frames.target_test.start:
        raise ContractError(
            f"{frames.dataset} comparison window {_iso(span.start)}..{_iso(span.end)} "
            f"reaches test start {_iso(frames.target_test.start)}"
        )
    rendered = tuple(_iso(day) for day in dates)
    test_start = _iso(frames.target_test.start)
    if any(day >= test_start for day in rendered):
        raise ContractError(f"{frames.dataset} comparison dates include test date {test_start}")
    return span, rendered


def _require_target_window(frames: ConditionFrames, span: DateSpan, variables: tuple[str, ...]) -> np.ndarray:
    window = _similarity_window(frames, target_series(frames), span)
    if window is None:
        raise ContractError(
            f"{frames.dataset} target {frames.target_key!r} comparison window {span.iso()} "
            "is incomplete or has a non-finite similarity value"
        )
    if window.shape != (len(variables), len(pd.date_range(span.start, span.end, freq="D"))):
        raise ContractError(f"{frames.dataset} target comparison window has shape {window.shape}")
    if window.shape[0] != len(variables):
        raise ContractError("target comparison window variable axis does not match the frozen order")
    return window


def _screen_candidates(
    frames: ConditionFrames,
    span: DateSpan,
    history_dates: pd.DatetimeIndex,
) -> tuple[tuple[_ScreenedCandidate, ...], GateCounts, ReconstructionSummary]:
    """Apply identity, 180-day history, comparison dates, then comparison variables.

    The selection scaler is not fit here. D5 identities are taken from rows on
    or before the cutoff, and missing natural days are reconstructed before the
    180-day gate. ``comparison_days`` does not shorten that history.
    """
    groups = _scoped_groups(frames)
    groups, reconstruction = _reconstruct_d5_groups(frames, groups, history_dates)
    comparison_index = pd.DatetimeIndex(pd.date_range(span.start, span.end, freq="D")).normalize()
    screened: list[_ScreenedCandidate] = []
    n_history = 0
    n_comparison = 0
    n_variables = 0
    for key, series in groups:
        exclusion, window, passed_history, passed_comparison = _classify_candidate(
            frames,
            key,
            series,
            span,
            history_dates,
            comparison_index,
        )
        if passed_history:
            n_history += 1
        if passed_comparison:
            n_comparison += 1
        if exclusion is None:
            n_variables += 1
        screened.append(_ScreenedCandidate(key, window, exclusion))
    counts = GateCounts(
        identity=len(groups),
        source_history=n_history,
        comparison_window=n_comparison,
        comparison_variables=n_variables,
    )
    return tuple(screened), counts, reconstruction


def _source_history_dates(frames: ConditionFrames) -> pd.DatetimeIndex:
    dates = expected_source_history_dates(frames.source_history.end, days=D5_HISTORY_DAYS)
    if (
        len(dates) != D5_HISTORY_DAYS
        or dates[0] != frames.source_history.start
        or dates[-1] != frames.source_history.end
    ):
        raise ContractError(
            f"{frames.dataset} frozen source history {frames.source_history.iso()} "
            "is not the formal 180-day grid"
        )
    if frames.dataset == "D5" and (_iso(dates[0]) != D5_HISTORY_START or _iso(dates[-1]) != D5_HISTORY_END):
        raise ContractError("D5 source history grid drifted from 2016-08-20 through 2017-02-15")
    return dates


def _reconstruct_d5_groups(
    frames: ConditionFrames,
    groups: list[tuple[tuple[str, ...], pd.DataFrame]],
    history_dates: pd.DatetimeIndex,
) -> tuple[list[tuple[tuple[str, ...], pd.DataFrame]], ReconstructionSummary]:
    start = _iso(history_dates[0])
    end = _iso(history_dates[-1])
    if frames.dataset != "D5":
        return groups, inactive_reconstruction_summary(start, end)
    only_after = _cutoff_only_identities(frames)
    outside_rows = _rows_outside_history(frames)
    duplicate_keys = {key for key, series in groups if series["date"].duplicated().any()}
    clean = [(key, series) for key, series in groups if key not in duplicate_keys]
    if not clean:
        return groups, ReconstructionSummary(
            applied=True,
            source_history_start=start,
            source_history_end=end,
            expected_day_count=len(history_dates),
            input_entity_count=0,
            synthetic_row_count=0,
            original_row_count=0,
            original_rows_unchanged=True,
            rows_outside_window=outside_rows,
            identities_only_after_cutoff=only_after,
            authority_sha256=(),
        )
    observed = pd.concat([series for _key, series in clean], ignore_index=True)
    reconstructed, summary = reconstruct_d5_source_history(observed)
    summary = replace(
        summary,
        identities_only_after_cutoff=only_after,
        rows_outside_window=outside_rows,
    )
    by_key = {key: _slice_reconstructed(reconstructed, frames, key) for key, _series in clean}
    rebuilt = [(key, by_key[key] if key in by_key else series) for key, series in groups]
    return rebuilt, summary


def _cutoff_only_identities(frames: ConditionFrames) -> tuple[tuple[str, ...], ...]:
    source = frames.source_rows
    if source.empty or "date" not in source.columns:
        return ()
    start = frames.source_history.start
    end = frames.source_history.end
    after = _key_set(source.loc[source["date"] > end], frames.key_fields)
    inside = _key_set(source.loc[(source["date"] >= start) & (source["date"] <= end)], frames.key_fields)
    return tuple(sorted(after - inside))


def _rows_outside_history(frames: ConditionFrames) -> int:
    source = frames.source_rows
    if source.empty:
        return 0
    start = frames.source_history.start
    end = frames.source_history.end
    outside = (source["date"] < start) | (source["date"] > end)
    return int(outside.sum())


def _key_set(frame: pd.DataFrame, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    if frame.empty:
        return set()
    keys: set[tuple[str, ...]] = set()
    for key_values, _group in frame.groupby(list(key_fields), sort=False, dropna=False):
        keys.add(_normalize_key(key_values, key_fields))
    return keys


def _slice_reconstructed(
    frame: pd.DataFrame,
    frames: ConditionFrames,
    key: tuple[str, ...],
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for field, value in zip(frames.key_fields, key):
        mask &= frame[field].map(_component) == value
    sliced = frame.loc[mask].sort_values("date", kind="mergesort").reset_index(drop=True)
    if sliced.empty:
        raise ContractError(f"{frames.dataset} lost reconstructed rows for {key!r}")
    return sliced


def _scoped_groups(frames: ConditionFrames) -> list[tuple[tuple[str, ...], pd.DataFrame]]:
    """Domain pool from rows inside the frozen source history. The target is not a candidate."""
    source = _between(frames.source_rows, frames.source_history)
    if frames.dataset == "D1":
        mask = source["store_id"].map(_component) == "1"
        mask &= source["item_id"].map(_component).isin({str(item) for item in range(1, 10)})
    elif frames.dataset == "D4":
        mask = source["store_id"].map(_component) == frames.target_key[0]
    elif frames.dataset == "D5":
        family = _target_group_value(frames)
        mask = source["store_nbr"].map(_component) == frames.target_key[0]
        mask &= source["family"].map(_component) == family
    elif frames.dataset == "D6":
        department = _target_group_value(frames)
        mask = source["store_id"].map(_component) == frames.target_key[0]
        mask &= source["dept_id"].map(_component) == department
    else:
        raise ContractError(f"unsupported dataset {frames.dataset}")
    narrowed = source.loc[mask]
    groups: list[tuple[tuple[str, ...], pd.DataFrame]] = []
    if narrowed.empty:
        return groups
    for key_values, group in narrowed.groupby(list(frames.key_fields), sort=False, dropna=False):
        key = _normalize_key(key_values, frames.key_fields)
        if key == frames.target_key:
            continue
        ordered = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        groups.append((key, ordered))
    groups.sort(key=lambda item: item[0])
    return groups


def _classify_candidate(
    frames: ConditionFrames,
    key: tuple[str, ...],
    series: pd.DataFrame,
    span: DateSpan,
    history_dates: pd.DatetimeIndex,
    comparison_index: pd.DatetimeIndex,
) -> tuple[ExclusionRecord | None, np.ndarray | None, bool, bool]:
    history_exclusion = _history_exclusion(key, series, history_dates, comparison_index)
    if history_exclusion is not None:
        return history_exclusion, None, False, False
    comparison_exclusion = _comparison_exclusion(key, series, comparison_index)
    if comparison_exclusion is not None:
        return comparison_exclusion, None, True, False
    variable_exclusion = _variable_exclusion(frames, key, series, span)
    if variable_exclusion is not None:
        return variable_exclusion, None, True, True
    matrix = _similarity_window(frames, series, span)
    if matrix is None:
        raise ContractError(f"{frames.dataset} candidate {key!r} failed the comparison window after it was screened")
    return None, matrix, True, True


def _history_exclusion(
    key: tuple[str, ...],
    series: pd.DataFrame,
    history_dates: pd.DatetimeIndex,
    comparison_index: pd.DatetimeIndex,
) -> ExclusionRecord | None:
    """Require the exact 180-day set. A hole only inside the comparison window keeps that reason."""
    if series["date"].duplicated().any():
        return ExclusionRecord(key, "duplicate_source_history", gate="source_history")
    actual = set(pd.to_datetime(series["date"]).dt.normalize())
    expected = set(pd.DatetimeIndex(history_dates).normalize())
    missing = expected - actual
    if actual - expected:
        return ExclusionRecord(key, "incomplete_source_history", gate="source_history")
    if not missing:
        if len(series) != len(expected):
            return ExclusionRecord(key, "duplicate_source_history", gate="source_history")
        return None
    comparison = set(pd.DatetimeIndex(comparison_index).normalize())
    outside = expected - comparison
    missing_dates = tuple(_iso(day) for day in sorted(missing))
    if missing <= comparison and outside <= actual:
        return ExclusionRecord(
            key,
            "missing_observed_dates",
            missing_dates=missing_dates,
            gate="comparison_window",
        )
    return ExclusionRecord(
        key,
        "incomplete_source_history",
        missing_dates=missing_dates,
        gate="source_history",
    )


def _comparison_exclusion(
    key: tuple[str, ...],
    series: pd.DataFrame,
    comparison_index: pd.DatetimeIndex,
) -> ExclusionRecord | None:
    window = series.loc[series["date"].isin(comparison_index)]
    if window["date"].duplicated().any():
        return ExclusionRecord(key, "duplicate_observed_dates", gate="comparison_window")
    actual = set(pd.to_datetime(window["date"]).dt.normalize())
    expected = set(pd.DatetimeIndex(comparison_index).normalize())
    missing = expected - actual
    if missing or actual - expected:
        missing_dates = tuple(_iso(day) for day in sorted(missing))
        return ExclusionRecord(
            key,
            "missing_observed_dates",
            missing_dates=missing_dates,
            gate="comparison_window",
        )
    return None


def _variable_exclusion(
    frames: ConditionFrames,
    key: tuple[str, ...],
    series: pd.DataFrame,
    span: DateSpan,
) -> ExclusionRecord | None:
    window = _between(series, span)
    for name in frames.similarity_variables:
        if name not in window.columns:
            raise ContractError(f"{frames.dataset} candidate {key!r} is missing similarity column {name!r}")
        encoded = _encode_similarity(frames.dataset, name, window[name])
        if encoded is None:
            return ExclusionRecord(key, "invalid_similarity_value", variable=name, gate="comparison_variables")
        if not np.isfinite(encoded).all():
            return ExclusionRecord(key, "non_finite_similarity", variable=name, gate="comparison_variables")
    return None


def _require_frames_match_contract(frames: ConditionFrames, contract) -> None:
    if frames.key_fields != contract.key_fields or frames.target_key != contract.target_key:
        raise ContractError(
            f"{contract.dataset} frames identity {frames.key_fields}/{frames.target_key} "
            f"does not match frozen {contract.key_fields}/{contract.target_key}"
        )
    if frames.similarity_variables != contract.similarity_variables:
        raise ContractError(
            f"{contract.dataset} similarity variables {frames.similarity_variables!r} "
            f"do not match frozen order {contract.similarity_variables!r}"
        )
    if frames.grouping_field != contract.grouping_field:
        raise ContractError(f"{contract.dataset} grouping field does not match the frozen contract")
    expected_30 = DateSpan(contract.target_train.start, contract.target_val.end)
    expected_15 = DateSpan(contract.target_train.start, contract.target_train.end)
    if frames.comparison_30 != expected_30 or frames.comparison_15 != expected_15:
        raise ContractError(f"{contract.dataset} comparison spans do not match the frozen train and validation dates")
    if (
        frames.target_train != contract.target_train
        or frames.target_val != contract.target_val
        or frames.target_test != contract.target_test
        or frames.source_history != contract.source_history
    ):
        raise ContractError(f"{contract.dataset} frame dates do not match the frozen calendar")
    if contract.target_val.start != contract.target_train.end + pd.Timedelta(days=1):
        raise ContractError(f"{contract.dataset} frozen validation does not start the day after training")
    if contract.target_test.start != contract.target_val.end + pd.Timedelta(days=1):
        raise ContractError(f"{contract.dataset} frozen test does not start the day after validation")
    if _span_days(contract.target_train) != 15 or _span_days(contract.target_val) != 15 or _span_days(expected_30) != 30:
        raise ContractError(f"{contract.dataset} frozen train, validation, and 30-day windows are not 15/15/30 days")


def _span_days(span: DateSpan) -> int:
    return int((span.end - span.start).days) + 1


def _require_float_vector(name: str, values: np.ndarray) -> None:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype != np.float64:
        raise ContractError(f"{name} must be a one-dimensional float64 vector")
