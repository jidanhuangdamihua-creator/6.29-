"""Target-local reuse of selected K3 model-raw source frames.

This module deliberately stops before fill, raw splitting, normalization, RFE,
sequence construction, and tensor construction.  It owns immutable templates
for one target lifecycle and returns an isolated working copy to every method.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd

from src.protocols.experiment_protocol import (
    ProtocolViolation,
    normalize_scenario,
    normalize_source_key,
)
from src.protocols.runner_adapter import source_key_mask
from src.utils.dataframe_attrs import (
    copy_frame_with_lightweight_attrs,
    select_rows_with_lightweight_attrs,
)


SourceKey = Tuple[object, ...]


def _hash_frame_values(frame: pd.DataFrame) -> bytes:
    try:
        return pd.util.hash_pandas_object(
            frame,
            index=True,
            categorize=False,
        ).to_numpy(dtype="uint64", copy=False).tobytes()
    except (TypeError, ValueError):
        # Model-raw frames are expected to contain scalar values.  This fallback
        # remains deterministic for unusual scalar extension dtypes.
        return frame.to_json(
            orient="split",
            date_format="iso",
            date_unit="ns",
            default_handler=str,
        ).encode("utf-8")


def canonical_raw_frame_digest(frame: pd.DataFrame) -> str:
    """Digest exact raw values plus index, column order, and dtype order."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [repr(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "index_names": [repr(name) for name in frame.index.names],
                "index_dtype": (
                    [str(level.dtype) for level in frame.index.levels]
                    if isinstance(frame.index, pd.MultiIndex)
                    else str(frame.index.dtype)
                ),
                "rows": int(len(frame)),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(_hash_frame_values(frame))
    return digest.hexdigest()


def _row_identity_digest(frame: pd.DataFrame, group_cols: Sequence[str]) -> str:
    identity_columns = [*group_cols]
    if "date" in frame.columns:
        identity_columns.append("date")
    identity = frame.loc[:, identity_columns]
    return canonical_raw_frame_digest(identity)


def _normalize_dataset_id(value: object) -> str:
    normalized = str(value).strip().upper()
    if normalized.startswith("DATASET") and normalized.removeprefix("DATASET").isdigit():
        return f"D{normalized.removeprefix('DATASET')}"
    return normalized


def _selection_contract(selection_result: Mapping[str, object]) -> tuple[
    Tuple[SourceKey, ...], str, str
]:
    sources = selection_result.get("sources", ())
    meta = selection_result.get("meta", {})
    if not isinstance(sources, (list, tuple)) or not isinstance(meta, Mapping):
        raise ProtocolViolation("K3 raw reuse requires a structured selection result")
    if int(meta.get("requested_k", -1)) != 3 or int(meta.get("effective_k", -1)) != 3:
        raise ProtocolViolation("K3 raw reuse requires exact K=3 selection")
    keys = tuple(
        normalize_source_key(source.get("source_key", ()))
        for source in sources
        if isinstance(source, Mapping)
    )
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ProtocolViolation("K3 raw reuse requires three unique selected source keys")
    candidate_digest = str(meta.get("candidate_pool_digest", ""))
    selection_digest = str(meta.get("selection_result_digest", ""))
    if not candidate_digest or not selection_digest:
        raise ProtocolViolation("K3 raw reuse requires selection protocol digests")
    return keys, candidate_digest, selection_digest


@dataclass(frozen=True)
class RawPreprocessingIdentity:
    """Small immutable ownership/provenance proof for one raw source."""

    lifecycle_identity: Tuple[object, ...]
    dataset_id: str
    scenario: str
    horizon: int
    seed: int
    target_key: SourceKey
    source_key: SourceKey
    source_cutoff: str
    source_window: Tuple[str, str]
    group_cols: Tuple[str, ...]
    model_feature_cols: Tuple[str, ...]
    row_count: int
    row_identity_digest: str
    date_identity_digest: str
    columns: Tuple[object, ...]
    dtypes: Tuple[str, ...]
    candidate_pool_digest: str
    selection_result_digest: str
    source_protocol_digest: str
    raw_frame_digest: str


@dataclass(frozen=True)
class CanonicalRawSource:
    """Read-only-by-ownership raw template; callers only receive deep copies."""

    identity: RawPreprocessingIdentity
    _template: pd.DataFrame = field(repr=False, compare=False)

    def working_copy(
        self,
        *,
        lifecycle_identity: Sequence[object],
        source_key: Sequence[object],
        model_feature_cols: Sequence[str],
    ) -> pd.DataFrame:
        if tuple(lifecycle_identity) != self.identity.lifecycle_identity:
            raise ProtocolViolation("canonical raw lifecycle identity mismatch")
        if normalize_source_key(source_key) != self.identity.source_key:
            raise ProtocolViolation("canonical raw source key identity mismatch")
        if tuple(str(column) for column in model_feature_cols) != self.identity.model_feature_cols:
            raise ProtocolViolation("canonical raw model feature schema identity mismatch")
        copied = copy_frame_with_lightweight_attrs(self._template, deep=True)
        copied.attrs = deepcopy(copied.attrs)
        copied.attrs.pop("method", None)
        copied.attrs["protocol_raw_preprocessing_identity"] = self.identity
        copied.attrs["protocol_cell_identity"] = self.identity.lifecycle_identity
        return copied


@dataclass
class TargetK3RawSourceContext:
    """One target lifecycle; intentionally neither global nor persistent."""

    lifecycle_identity: Tuple[object, ...]
    _sources: Dict[SourceKey, CanonicalRawSource] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _selected_keys: Tuple[SourceKey, ...] = field(default=(), init=False, repr=False)
    _candidate_pool_digest: str = field(default="", init=False, repr=False)
    _selection_result_digest: str = field(default="", init=False, repr=False)
    _source_columns: Tuple[object, ...] = field(default=(), init=False, repr=False)
    _source_dtypes: Tuple[str, ...] = field(default=(), init=False, repr=False)
    _source_row_identity_digest: str = field(default="", init=False, repr=False)
    _source_raw_digest: str = field(default="", init=False, repr=False)
    _source_owner_id: int = field(default=0, init=False, repr=False)
    _group_cols: Tuple[str, ...] = field(default=(), init=False, repr=False)
    _model_feature_cols: Tuple[str, ...] = field(default=(), init=False, repr=False)
    _materialization_count: int = field(default=0, init=False, repr=False)

    @property
    def materialization_count(self) -> int:
        return self._materialization_count

    def canonical_identity_for(self, source_key: Sequence[object]) -> int:
        key = normalize_source_key(source_key)
        canonical = self._sources.get(key)
        if canonical is None:
            raise ProtocolViolation(f"canonical raw source key is not bound: {key!r}")
        return id(canonical)

    def provenance_for(self, source_key: Sequence[object]) -> RawPreprocessingIdentity:
        key = normalize_source_key(source_key)
        canonical = self._sources.get(key)
        if canonical is None:
            raise ProtocolViolation(f"canonical raw source key is not bound: {key!r}")
        return canonical.identity

    def _validate_lifecycle_frame(self, source_df: pd.DataFrame) -> None:
        if len(self.lifecycle_identity) != 5:
            raise ProtocolViolation("canonical raw lifecycle identity must have five fields")
        lifecycle_dataset, lifecycle_scenario, _, _, lifecycle_target = self.lifecycle_identity
        frame_dataset = source_df.attrs.get("protocol_dataset_id", lifecycle_dataset)
        if _normalize_dataset_id(frame_dataset) != _normalize_dataset_id(lifecycle_dataset):
            raise ProtocolViolation("canonical raw dataset identity mismatch")
        frame_scenario = source_df.attrs.get(
            "protocol_scenario",
            source_df.attrs.get("information_sharing_scenario", lifecycle_scenario),
        )
        if normalize_scenario(frame_scenario) != normalize_scenario(lifecycle_scenario):
            raise ProtocolViolation("canonical raw scenario identity mismatch")
        frame_target = source_df.attrs.get("protocol_target_key")
        if frame_target is not None and normalize_source_key(frame_target) != normalize_source_key(lifecycle_target):
            raise ProtocolViolation("canonical raw target identity mismatch")

    def _validate_bound_source(self, source_df: pd.DataFrame) -> None:
        columns = tuple(source_df.columns)
        if columns != self._source_columns:
            raise ProtocolViolation("canonical raw source schema identity mismatch")
        dtypes = tuple(map(str, source_df.dtypes))
        if dtypes != self._source_dtypes:
            raise ProtocolViolation("canonical raw source dtype identity mismatch")
        if id(source_df) != self._source_owner_id:
            row_digest = _row_identity_digest(source_df, self._group_cols)
            if row_digest != self._source_row_identity_digest:
                raise ProtocolViolation("canonical raw source row identity/order mismatch")
            raw_digest = canonical_raw_frame_digest(source_df)
            if raw_digest != self._source_raw_digest:
                raise ProtocolViolation("canonical raw source content identity mismatch")
            raise ProtocolViolation("canonical raw source owner identity mismatch")

    def _bind(
        self,
        *,
        selection_result: Mapping[str, object],
        source_df: pd.DataFrame,
        group_cols: Sequence[str],
        model_feature_cols: Sequence[str],
    ) -> None:
        self._validate_lifecycle_frame(source_df)
        if not source_df.columns.is_unique:
            raise ProtocolViolation("canonical raw source schema contains duplicate columns")
        normalized_group_cols = tuple(str(column) for column in group_cols)
        normalized_features = tuple(str(column) for column in model_feature_cols)
        missing = [
            column
            for column in (*normalized_group_cols, "date", *normalized_features)
            if column not in source_df.columns
        ]
        if missing:
            raise ProtocolViolation(f"canonical raw source schema missing columns: {missing!r}")
        keys, candidate_digest, selection_digest = _selection_contract(selection_result)

        self._selected_keys = keys
        self._candidate_pool_digest = candidate_digest
        self._selection_result_digest = selection_digest
        self._source_columns = tuple(source_df.columns)
        self._source_dtypes = tuple(map(str, source_df.dtypes))
        self._source_row_identity_digest = _row_identity_digest(source_df, normalized_group_cols)
        self._source_raw_digest = canonical_raw_frame_digest(source_df)
        self._source_owner_id = id(source_df)
        self._group_cols = normalized_group_cols
        self._model_feature_cols = normalized_features

        dataset_id = _normalize_dataset_id(
            source_df.attrs.get("protocol_dataset_id", self.lifecycle_identity[0])
        )
        scenario = str(
            source_df.attrs.get(
                "protocol_scenario",
                source_df.attrs.get("information_sharing_scenario", self.lifecycle_identity[1]),
            )
        )
        target_key = normalize_source_key(self.lifecycle_identity[4])
        source_cutoff = str(source_df.attrs.get("source_observation_cutoff", ""))
        source_window = (
            str(source_df.attrs.get("source_history_start", "")),
            str(source_df.attrs.get("source_history_end", "")),
        )
        source_protocol_digest = str(
            source_df.attrs.get(
                "source_history_frame_digest",
                source_df.attrs.get("source_frame_digest", ""),
            )
        )

        for key in keys:
            mask = source_key_mask(source_df, normalized_group_cols, key)
            template = select_rows_with_lightweight_attrs(source_df, mask, deep=True)
            template.attrs = deepcopy(template.attrs)
            template.attrs.pop("method", None)
            if template.empty:
                raise ProtocolViolation(f"selected canonical raw source key is absent: {key!r}")
            duplicate_dates = template.duplicated(
                subset=[*normalized_group_cols, "date"],
                keep=False,
            )
            if bool(duplicate_dates.any()):
                raise ProtocolViolation(f"selected canonical raw source has duplicate date: {key!r}")
            raw_digest = canonical_raw_frame_digest(template)
            row_digest = _row_identity_digest(template, normalized_group_cols)
            date_digest = canonical_raw_frame_digest(template.loc[:, ["date"]])
            identity = RawPreprocessingIdentity(
                lifecycle_identity=tuple(self.lifecycle_identity),
                dataset_id=dataset_id,
                scenario=scenario,
                horizon=int(self.lifecycle_identity[2]),
                seed=int(self.lifecycle_identity[3]),
                target_key=target_key,
                source_key=key,
                source_cutoff=source_cutoff,
                source_window=source_window,
                group_cols=normalized_group_cols,
                model_feature_cols=normalized_features,
                row_count=int(len(template)),
                row_identity_digest=row_digest,
                date_identity_digest=date_digest,
                columns=tuple(template.columns),
                dtypes=tuple(map(str, template.dtypes)),
                candidate_pool_digest=candidate_digest,
                selection_result_digest=selection_digest,
                source_protocol_digest=source_protocol_digest,
                raw_frame_digest=raw_digest,
            )
            self._sources[key] = CanonicalRawSource(identity=identity, _template=template)
            self._materialization_count += 1

    def working_source(
        self,
        *,
        selection_result: Mapping[str, object],
        source_df: pd.DataFrame,
        source_key: Sequence[object],
        group_cols: Sequence[str],
        model_feature_cols: Sequence[str],
    ) -> pd.DataFrame:
        keys, candidate_digest, selection_digest = _selection_contract(selection_result)
        normalized_group_cols = tuple(str(column) for column in group_cols)
        normalized_features = tuple(str(column) for column in model_feature_cols)
        if not self._sources:
            self._bind(
                selection_result=selection_result,
                source_df=source_df,
                group_cols=normalized_group_cols,
                model_feature_cols=normalized_features,
            )
        else:
            self._validate_lifecycle_frame(source_df)
            if keys != self._selected_keys:
                raise ProtocolViolation("canonical raw K3 selection key/order mismatch")
            if (
                candidate_digest != self._candidate_pool_digest
                or selection_digest != self._selection_result_digest
            ):
                raise ProtocolViolation("canonical raw K3 selection digest mismatch")
            if normalized_group_cols != self._group_cols:
                raise ProtocolViolation("canonical raw group schema identity mismatch")
            if normalized_features != self._model_feature_cols:
                raise ProtocolViolation("canonical raw model feature schema identity mismatch")
            self._validate_bound_source(source_df)

        key = normalize_source_key(source_key)
        if key not in self._selected_keys:
            raise ProtocolViolation(f"canonical raw source key is outside K3 selection: {key!r}")
        return self._sources[key].working_copy(
            lifecycle_identity=self.lifecycle_identity,
            source_key=key,
            model_feature_cols=normalized_features,
        )


__all__ = [
    "CanonicalRawSource",
    "RawPreprocessingIdentity",
    "TargetK3RawSourceContext",
    "canonical_raw_frame_digest",
]
