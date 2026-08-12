"""Construct and retrieve the date-bounded KNN observation frames."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from .experiment_protocol import (
    ObservationWindow,
    ProtocolViolation,
    SourceKey,
    normalize_source_key,
)
from .gate1_transformation import normalized_frame_digest
from src.utils.dataframe_attrs import (
    get_protocol_frame_context,
    lightweight_frame_attrs,
    temporarily_detached_attrs,
)


_CONFIGURED_FRAME_ATTR = "protocol_knn_observed_frame"
_DIGEST_EVIDENCE_ATTR = "_protocol_knn_digest_evidence"
_DIGEST_IDENTITY_ATTR = "_protocol_knn_digest_identity"
_DIGEST_EVIDENCE_ISSUER = object()


@dataclass(frozen=True)
class CanonicalKnnDigestEvidence:
    frame_id: int
    issuer: object
    trusted: bool
    group_cols: tuple[str, ...]
    feature_cols: tuple[str, ...] | None
    ignore_columns: tuple[str, ...]
    shape: tuple[int, int]
    columns: tuple[str, ...]
    dtypes: tuple[str, ...]
    digest: str
    role: str
    observed_start: str
    observed_end: str
    candidate_scope: tuple[SourceKey, ...]
    complete_candidate_scope: tuple[SourceKey, ...]
    excluded_candidate_scope: tuple[SourceKey, ...]
    context_identity: tuple[tuple[str, object], ...]

    def __deepcopy__(self, memo):
        copied = replace(self, trusted=False, frame_id=-1)
        memo[id(self)] = copied
        return copied


def _normalized_dates(frame: pd.DataFrame, *, role: str) -> pd.Series:
    if "date" not in frame.columns:
        raise ProtocolViolation(f"{role} frame requires date column")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ProtocolViolation(f"{role} frame contains invalid dates")
    return dates


def _compute_canonical_knn_frame_digest(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
    ignore_columns: Sequence[str] = (),
) -> str:
    """Digest the actual KNN frame after deterministic key/date ordering."""
    with temporarily_detached_attrs(frame):
        dates = _normalized_dates(frame, role="KNN")
        ordered = frame.copy()
    ordered["date"] = dates
    if ignore_columns:
        ordered = ordered.drop(columns=list(ignore_columns), errors="ignore")
    if feature_cols is not None:
        normalized_features = tuple(str(column) for column in feature_cols)
        missing = [column for column in normalized_features if column not in ordered.columns]
        if missing:
            raise ProtocolViolation(
                f"KNN frame is missing declared feature columns: {missing!r}"
            )
        ordered = ordered.loc[:, [*group_cols, "date", *normalized_features]].copy()
    sort_cols = [column for column in (*group_cols, "date") if column in ordered.columns]
    if not sort_cols:
        raise ProtocolViolation("KNN frame digest requires group or date columns")
    ordered = ordered.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return normalized_frame_digest(ordered)


def canonical_knn_frame_digest(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
    ignore_columns: Sequence[str] = (),
) -> str:
    """Return an exact digest, reusing only builder-owned trusted evidence."""

    normalized_groups = tuple(str(column) for column in group_cols)
    normalized_features = (
        None if feature_cols is None else tuple(str(column) for column in feature_cols)
    )
    normalized_ignored = tuple(str(column) for column in ignore_columns)
    evidence = frame.attrs.get(_DIGEST_EVIDENCE_ATTR)
    public_digest = frame.attrs.get("knn_frame_digest")
    if isinstance(evidence, CanonicalKnnDigestEvidence):
        if public_digest is not None and str(public_digest) != evidence.digest:
            raise ProtocolViolation("KNN frame digest metadata disagrees with trusted evidence")
        evidence_matches = (
            evidence.trusted
            and evidence.issuer is _DIGEST_EVIDENCE_ISSUER
            and evidence.frame_id == id(frame)
            and evidence.group_cols == normalized_groups
            and evidence.feature_cols == normalized_features
            and evidence.ignore_columns == normalized_ignored
            and evidence.shape == tuple(frame.shape)
            and evidence.columns == tuple(str(column) for column in frame.columns)
            and evidence.dtypes == tuple(str(dtype) for dtype in frame.dtypes)
            and frame.attrs.get(_DIGEST_IDENTITY_ATTR) == evidence.context_identity
        )
        if evidence_matches:
            return evidence.digest

    computed = _compute_canonical_knn_frame_digest(
        frame,
        group_cols=normalized_groups,
        feature_cols=normalized_features,
        ignore_columns=normalized_ignored,
    )
    if public_digest is not None and str(public_digest) != computed:
        raise ProtocolViolation("KNN frame digest metadata differs from actual frame content")
    return computed


def _prepared_complete_row_mask(
    observed: pd.DataFrame,
    *,
    window: ObservationWindow,
    group_cols: tuple[str, ...],
    feature_cols: tuple[str, ...],
    eligibility_proof: object,
    pool_identity: int,
) -> tuple[np.ndarray, tuple[SourceKey, ...], tuple[SourceKey, ...], tuple[SourceKey, ...]]:
    """Validate a prepared-pool proof and its full aligned candidate frame."""

    required_attributes = (
        "pool_id",
        "group_cols",
        "required_dates",
        "feature_cols",
        "candidate_scope",
        "complete_candidate_scope",
        "excluded_candidate_scope",
        "missing_dates_by_candidate",
    )
    missing_attributes = [
        name for name in required_attributes if not hasattr(eligibility_proof, name)
    ]
    if missing_attributes:
        raise ProtocolViolation(
            "prepared-pool date eligibility proof is incomplete: "
            f"{missing_attributes!r}"
        )

    candidate_scope = tuple(
        normalize_source_key(key) for key in eligibility_proof.candidate_scope
    )
    complete_scope = tuple(
        normalize_source_key(key)
        for key in eligibility_proof.complete_candidate_scope
    )
    excluded_scope = tuple(
        normalize_source_key(key)
        for key in eligibility_proof.excluded_candidate_scope
    )
    required_dates = tuple(
        pd.date_range(
            window.knn_observed_start,
            window.knn_observed_end,
            freq="D",
        ).strftime("%Y-%m-%d")
    )
    if (
        eligibility_proof.pool_id != pool_identity
        or tuple(eligibility_proof.group_cols) != group_cols
        or tuple(eligibility_proof.required_dates) != required_dates
        or tuple(eligibility_proof.feature_cols) != feature_cols
    ):
        raise ProtocolViolation(
            "prepared-pool date eligibility proof identity does not match builder input"
        )
    if len(set(candidate_scope)) != len(candidate_scope):
        raise ProtocolViolation("prepared-pool eligibility contains duplicate candidate keys")
    complete_set = set(complete_scope)
    excluded_set = set(excluded_scope)
    if complete_set & excluded_set or complete_set | excluded_set != set(candidate_scope):
        raise ProtocolViolation(
            "prepared-pool eligibility does not partition the candidate scope"
        )
    if tuple(key for key in candidate_scope if key in complete_set) != complete_scope:
        raise ProtocolViolation("prepared-pool complete candidate order changed")
    if tuple(key for key in candidate_scope if key in excluded_set) != excluded_scope:
        raise ProtocolViolation("prepared-pool excluded candidate order changed")
    missing_scope = tuple(
        normalize_source_key(key)
        for key, missing_dates in eligibility_proof.missing_dates_by_candidate
        if tuple(missing_dates)
    )
    if missing_scope != excluded_scope:
        raise ProtocolViolation(
            "prepared-pool missing-date facts do not match excluded candidates"
        )

    rows_per_candidate = len(required_dates)
    if len(observed) != len(candidate_scope) * rows_per_candidate:
        raise ProtocolViolation(
            "prepared-pool observed frame row count does not match candidate scope"
        )
    complete_rows = np.zeros(len(observed), dtype=bool)
    expected_dates = pd.DatetimeIndex(pd.to_datetime(required_dates))
    for position, candidate_key in enumerate(candidate_scope):
        start = position * rows_per_candidate
        stop = start + rows_per_candidate
        block = observed.iloc[start:stop]
        actual_keys = tuple(
            normalize_source_key(tuple(row))
            for row in block.loc[:, list(group_cols)].itertuples(index=False, name=None)
        )
        if any(actual_key != candidate_key for actual_key in actual_keys):
            raise ProtocolViolation(
                "prepared-pool observed frame candidate scope/order changed"
            )
        actual_dates = pd.DatetimeIndex(block["date"])
        if not actual_dates.equals(expected_dates):
            raise ProtocolViolation(
                f"source {candidate_key!r} prepared observed dates changed"
            )
        if candidate_key in complete_set:
            complete_rows[start:stop] = True
    return complete_rows, candidate_scope, complete_scope, excluded_scope


def _build_observed_knn_frame(
    frame: pd.DataFrame,
    *,
    window: ObservationWindow,
    role: str,
    group_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
    eligibility_proof: object | None = None,
    pool_identity: int | None = None,
) -> pd.DataFrame:
    """Build one observed frame, optionally using verified prepared-pool eligibility."""

    normalized_groups = tuple(str(column) for column in group_cols)
    normalized_features = tuple(str(column) for column in (feature_cols or ()))
    with temporarily_detached_attrs(frame):
        parsed_dates = _normalized_dates(frame, role=role)
    observed_mask = parsed_dates.between(
        pd.Timestamp(window.knn_observed_start),
        pd.Timestamp(window.knn_observed_end),
        inclusive="both",
    )
    with temporarily_detached_attrs(frame):
        observed = frame.loc[observed_mask].copy()
    if observed.empty:
        raise ProtocolViolation(f"{role} KNN observed frame is empty")
    complete_row_mask = None
    candidate_scope: tuple[SourceKey, ...] = ()
    complete_candidate_scope: tuple[SourceKey, ...] = ()
    excluded_candidate_scope: tuple[SourceKey, ...] = ()
    if eligibility_proof is not None:
        if pool_identity is None:
            raise ProtocolViolation("prepared-pool builder requires pool identity")
        (
            complete_row_mask,
            candidate_scope,
            complete_candidate_scope,
            excluded_candidate_scope,
        ) = _prepared_complete_row_mask(
            observed,
            window=window,
            group_cols=normalized_groups,
            feature_cols=normalized_features,
            eligibility_proof=eligibility_proof,
            pool_identity=pool_identity,
        )
    if feature_cols is not None:
        missing = [column for column in normalized_features if column not in observed.columns]
        if missing:
            raise ProtocolViolation(
                f"{role} KNN observed frame is missing declared feature columns: {missing!r}"
            )
        numeric_observed = (
            observed
            if complete_row_mask is None
            else observed.loc[complete_row_mask]
        )
        for column in normalized_features:
            numeric = pd.to_numeric(numeric_observed[column], errors="coerce")
            if numeric.isna().any():
                raise ProtocolViolation(
                    f"{role} KNN observed feature {column!r} contains non-numeric values"
                )
    observed["date"] = parsed_dates.loc[observed_mask].to_numpy()
    observed.attrs = {
        key: value
        for key, value in lightweight_frame_attrs(frame.attrs).items()
        if key != _CONFIGURED_FRAME_ATTR
    }
    digest_feature_cols = (
        None if tuple(feature_cols or ()) == ("sales",) else tuple(feature_cols or ())
    )
    digest_ignored_columns = (
        ("promo",) if tuple(feature_cols or ()) == ("sales",) else ()
    )
    digest = canonical_knn_frame_digest(
        observed,
        group_cols=normalized_groups,
        feature_cols=digest_feature_cols,
        ignore_columns=digest_ignored_columns,
    )
    observed.attrs.update(
        {
            "knn_frame_role": str(role),
            "knn_observed_start": window.knn_observed_start.isoformat(),
            "knn_observed_end": window.knn_observed_end.isoformat(),
            "knn_observed_days": window.observed_days,
            "knn_boundary": "inclusive",
            "knn_feature_columns": list(feature_cols or ()),
            "feature_scope": "historical_observed",
            "max_allowed_date_relation": "date<=origin",
            "knn_frame_min_date": observed["date"].min().strftime("%Y-%m-%d"),
            "knn_frame_max_date": observed["date"].max().strftime("%Y-%m-%d"),
            "knn_frame_digest": digest,
        }
    )
    observed.attrs[_DIGEST_EVIDENCE_ATTR] = CanonicalKnnDigestEvidence(
        frame_id=id(observed),
        issuer=_DIGEST_EVIDENCE_ISSUER,
        trusted=True,
        group_cols=normalized_groups,
        feature_cols=digest_feature_cols,
        ignore_columns=digest_ignored_columns,
        shape=tuple(observed.shape),
        columns=tuple(str(column) for column in observed.columns),
        dtypes=tuple(str(dtype) for dtype in observed.dtypes),
        digest=digest,
        role=str(role),
        observed_start=window.knn_observed_start.isoformat(),
        observed_end=window.knn_observed_end.isoformat(),
        candidate_scope=candidate_scope,
        complete_candidate_scope=complete_candidate_scope,
        excluded_candidate_scope=excluded_candidate_scope,
        context_identity=(
            ("role", str(role)),
            ("observed_start", window.knn_observed_start.isoformat()),
            ("observed_end", window.knn_observed_end.isoformat()),
            ("group_cols", tuple(str(column) for column in group_cols)),
            ("feature_cols", tuple(str(column) for column in (feature_cols or ()))),
        ),
    )
    observed.attrs[_DIGEST_IDENTITY_ATTR] = observed.attrs[
        _DIGEST_EVIDENCE_ATTR
    ].context_identity
    return observed


def build_observed_knn_frame(
    frame: pd.DataFrame,
    *,
    window: ObservationWindow,
    role: str,
    group_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the inclusive observed copy used by KNN, never the full model frame."""

    return _build_observed_knn_frame(
        frame,
        window=window,
        role=role,
        group_cols=group_cols,
        feature_cols=feature_cols,
    )


def build_prepared_pool_observed_knn_frame(
    frame: pd.DataFrame,
    *,
    window: ObservationWindow,
    role: str,
    group_cols: Sequence[str],
    feature_cols: Sequence[str],
    eligibility_proof: object,
    pool_identity: int,
) -> pd.DataFrame:
    """Build a full prepared-pool frame while validating only complete candidates."""

    return _build_observed_knn_frame(
        frame,
        window=window,
        role=role,
        group_cols=group_cols,
        feature_cols=feature_cols,
        eligibility_proof=eligibility_proof,
        pool_identity=pool_identity,
    )


def seal_canonical_knn_digest_evidence(
    frame: pd.DataFrame,
    *,
    context_identity: Sequence[tuple[str, object]],
) -> None:
    """Bind trusted builder evidence to the completed configure identity."""

    evidence = frame.attrs.get(_DIGEST_EVIDENCE_ATTR)
    if (
        not isinstance(evidence, CanonicalKnnDigestEvidence)
        or not evidence.trusted
        or evidence.issuer is not _DIGEST_EVIDENCE_ISSUER
        or evidence.frame_id != id(frame)
    ):
        raise ProtocolViolation("only a builder-owned KNN frame may be sealed")
    normalized_identity = tuple((str(key), value) for key, value in context_identity)
    frame.attrs[_DIGEST_EVIDENCE_ATTR] = replace(
        evidence,
        context_identity=normalized_identity,
    )
    frame.attrs[_DIGEST_IDENTITY_ATTR] = normalized_identity


def get_configured_knn_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """Retrieve a configured observed frame and fail closed when absent."""
    normalized_role = str(role).lower()
    context = get_protocol_frame_context(frame)
    configured = None
    if context is not None:
        observed_frames = context.observed_frames or {}
        observed_carrier_ids = context.observed_carrier_ids or {}
        configured_lifecycle = (
            normalized_role in observed_frames
            or normalized_role in observed_carrier_ids
        )
        if configured_lifecycle:
            configured = observed_frames.get(normalized_role)
    if configured is None:
        configured = frame.attrs.get(_CONFIGURED_FRAME_ATTR)
    if not isinstance(configured, pd.DataFrame):
        raise ProtocolViolation(
            f"{role} frame is missing configured KNN observed frame"
        )
    evidence = configured.attrs.get(_DIGEST_EVIDENCE_ATTR)
    if (
        not isinstance(evidence, CanonicalKnnDigestEvidence)
        or not evidence.trusted
        or evidence.issuer is not _DIGEST_EVIDENCE_ISSUER
    ):
        raise ProtocolViolation(f"{role} configured KNN frame is not builder-owned")
    if evidence.frame_id != id(configured):
        raise ProtocolViolation(f"{role} configured KNN frame identity changed")
    observed = configured.copy()
    observed.attrs = configured.attrs.copy()
    return observed


__all__ = [
    "build_observed_knn_frame",
    "build_prepared_pool_observed_knn_frame",
    "CanonicalKnnDigestEvidence",
    "canonical_knn_frame_digest",
    "get_configured_knn_frame",
    "seal_canonical_knn_digest_evidence",
]
