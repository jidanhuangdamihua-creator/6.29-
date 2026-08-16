"""Target-local normalization and sequence result reuse.

The context owns immutable/private canonical templates for exactly one
dataset/scenario/horizon/seed/target lifecycle.  Every caller receives fresh
mutable frames, scalers, and arrays.  No cache in this module has global,
cross-target, cross-cell, disk, or process lifetime.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.protocols.experiment_protocol import ProtocolViolation, normalize_source_key
from src.protocols.transformation_identity import (
    NORMALIZATION_EVIDENCE_ATTR,
    SEQUENCE_EVIDENCE_ATTR,
    NormalizationEvidence,
    NormalizationIdentity,
    SequenceEvidence,
    SequenceIdentity,
    exact_array_digest,
    require_same_identity,
    scaler_parameter_evidence,
    validate_normalized_partition_evidence,
)
from src.utils.dataframe_attrs import (
    context_with,
    copy_frame_with_lightweight_attrs,
    get_protocol_frame_context,
    lightweight_frame_attrs,
    set_protocol_frame_context,
)


NormalizationResult = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    MinMaxScaler,
    list[str],
]
SequenceResult = Tuple[np.ndarray, np.ndarray]


def _canonical_lifecycle(value: Sequence[Any]) -> Tuple[Any, ...]:
    result = tuple(value)
    if len(result) != 5:
        raise ProtocolViolation("transformation reuse lifecycle identity must have five fields")
    return (
        str(result[0]),
        str(result[1]),
        int(result[2]),
        int(result[3]),
        normalize_source_key(result[4]),
    )


def _require_private_scaler_clone(template: MinMaxScaler) -> MinMaxScaler:
    clone = deepcopy(template)
    if clone is template or scaler_parameter_evidence(clone) != scaler_parameter_evidence(template):
        raise ProtocolViolation("NORMALIZATION_REUSE_SCALER_CLONE_MISMATCH")
    for name in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
        template_value = np.asarray(getattr(template, name))
        clone_value = np.asarray(getattr(clone, name))
        if np.shares_memory(template_value, clone_value):
            raise ProtocolViolation("NORMALIZATION_REUSE_SCALER_CLONE_ALIAS")
    if hasattr(template, "feature_names_in_") and np.shares_memory(
        np.asarray(template.feature_names_in_), np.asarray(clone.feature_names_in_)
    ):
        raise ProtocolViolation("NORMALIZATION_REUSE_SCALER_CLONE_ALIAS")
    return clone


@dataclass
class _CanonicalNormalizationResult:
    identity: NormalizationIdentity
    evidence: NormalizationEvidence
    feature_columns: Tuple[str, ...]
    _train_template: pd.DataFrame = field(repr=False)
    _validation_template: pd.DataFrame = field(repr=False)
    _test_template: pd.DataFrame = field(repr=False)
    _scaler_template: MinMaxScaler = field(repr=False)

    @classmethod
    def from_heavy_result(
        cls,
        identity: NormalizationIdentity,
        result: NormalizationResult,
    ) -> "_CanonicalNormalizationResult":
        train, validation, test, scaler, feature_columns = result
        evidences = tuple(frame.attrs.get(NORMALIZATION_EVIDENCE_ATTR) for frame in (train, validation, test))
        if not all(isinstance(item, NormalizationEvidence) for item in evidences):
            raise ProtocolViolation("NORMALIZATION_REUSE_EVIDENCE_MISSING")
        evidence = evidences[0]
        if any(item != evidence for item in evidences[1:]):
            raise ProtocolViolation("NORMALIZATION_REUSE_EVIDENCE_MISMATCH")
        require_same_identity(evidence.identity, identity, contract="normalization reuse")
        if tuple(feature_columns) != identity.actual_feature_cols:
            raise ProtocolViolation("NORMALIZATION_REUSE_FEATURE_MISMATCH")

        templates = []
        for frame in (train, validation, test):
            template = copy_frame_with_lightweight_attrs(frame)
            template.attrs = {NORMALIZATION_EVIDENCE_ATTR: evidence}
            templates.append(template)
        for template, partition_evidence in zip(
            templates,
            (evidence.train, evidence.validation, evidence.test),
        ):
            validate_normalized_partition_evidence(template, partition_evidence)
        scaler_template = deepcopy(scaler)
        if scaler_parameter_evidence(scaler_template) != evidence.scaler_parameters:
            raise ProtocolViolation("NORMALIZATION_REUSE_SCALER_EVIDENCE_MISMATCH")
        return cls(
            identity=identity,
            evidence=evidence,
            feature_columns=tuple(feature_columns),
            _train_template=templates[0],
            _validation_template=templates[1],
            _test_template=templates[2],
            _scaler_template=scaler_template,
        )

    def copy_for_consumer(
        self,
        identity: NormalizationIdentity,
        raw_frames: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> NormalizationResult:
        # The MISS path built exact evidence and then copied every canonical
        # value into this owner-private entry.  Consumers only receive deep
        # frame/scaler copies, so a HIT compares immutable identity/evidence
        # tokens instead of rescanning the same canonical values.
        require_same_identity(self.identity, identity, contract="normalization reuse")
        scaler = _require_private_scaler_clone(self._scaler_template)
        working_frames = []
        for raw_frame, template in zip(
            raw_frames,
            (self._train_template, self._validation_template, self._test_template),
        ):
            working = copy_frame_with_lightweight_attrs(template)
            working.attrs = lightweight_frame_attrs(raw_frame.attrs)
            working.attrs[NORMALIZATION_EVIDENCE_ATTR] = self.evidence
            raw_context = get_protocol_frame_context(raw_frame)
            actual_source_key = (
                raw_context.actual_source_key
                if raw_context is not None
                else raw_frame.attrs.get("protocol_actual_source_key")
            )
            if actual_source_key is not None:
                set_protocol_frame_context(
                    working,
                    context_with(
                        raw_context,
                        actual_source_key=tuple(actual_source_key),
                        raw_partition=raw_frame,
                        fitted_scaler=scaler,
                        scaler_feature_cols=self.feature_columns,
                    ),
                )
            working_frames.append(working)
        return (
            working_frames[0],
            working_frames[1],
            working_frames[2],
            scaler,
            list(self.feature_columns),
        )


@dataclass
class _CanonicalSequenceResult:
    identity: SequenceIdentity
    evidence: SequenceEvidence
    _x_template: np.ndarray = field(repr=False)
    _y_template: np.ndarray = field(repr=False)

    @classmethod
    def from_heavy_result(
        cls,
        identity: SequenceIdentity,
        frame: pd.DataFrame,
        result: SequenceResult,
    ) -> "_CanonicalSequenceResult":
        evidence = frame.attrs.get(SEQUENCE_EVIDENCE_ATTR)
        if not isinstance(evidence, SequenceEvidence):
            raise ProtocolViolation("SEQUENCE_REUSE_EVIDENCE_MISSING")
        require_same_identity(evidence.identity, identity, contract="sequence reuse")
        x_template = np.array(result[0], copy=True)
        y_template = np.array(result[1], copy=True)
        if (
            exact_array_digest(x_template) != evidence.x_exact_digest
            or exact_array_digest(y_template) != evidence.y_exact_digest
        ):
            raise ProtocolViolation("SEQUENCE_REUSE_EVIDENCE_MISMATCH")
        x_template.flags.writeable = False
        y_template.flags.writeable = False
        return cls(identity, evidence, x_template, y_template)

    @property
    def template_writeable_flags(self) -> Tuple[bool, bool]:
        return (bool(self._x_template.flags.writeable), bool(self._y_template.flags.writeable))

    def copy_for_consumer(self, identity: SequenceIdentity, frame: pd.DataFrame) -> SequenceResult:
        require_same_identity(self.identity, identity, contract="sequence reuse")
        if (
            exact_array_digest(self._x_template) != self.evidence.x_exact_digest
            or exact_array_digest(self._y_template) != self.evidence.y_exact_digest
        ):
            raise ProtocolViolation("SEQUENCE_REUSE_EVIDENCE_MISMATCH")
        current_normalization = frame.attrs.get(NORMALIZATION_EVIDENCE_ATTR)
        if not isinstance(current_normalization, NormalizationEvidence):
            raise ProtocolViolation("SEQUENCE_REUSE_NORMALIZATION_EVIDENCE_MISSING")
        if identity.normalized_partition_evidence not in (
            current_normalization.train,
            current_normalization.validation,
            current_normalization.test,
        ):
            raise ProtocolViolation("SEQUENCE_REUSE_NORMALIZATION_EVIDENCE_MISMATCH")
        frame.attrs[SEQUENCE_EVIDENCE_ATTR] = self.evidence
        x_work = np.array(self._x_template, copy=True)
        y_work = np.array(self._y_template, copy=True)
        from src.protocols.provenance import bind_reused_actual_cnn_arrays

        bind_reused_actual_cnn_arrays(
            frame,
            input_tensor=x_work,
            labels=y_work,
            evidence=self.evidence,
        )
        return x_work, y_work


@dataclass
class TargetTransformationReuseContext:
    """One explicitly passed normalization/sequence owner for one target."""

    lifecycle_identity: Tuple[Any, ...]
    _normalizations: Dict[NormalizationIdentity, _CanonicalNormalizationResult] = field(
        default_factory=dict, init=False, repr=False
    )
    _sequences: Dict[SequenceIdentity, _CanonicalSequenceResult] = field(
        default_factory=dict, init=False, repr=False
    )
    normalization_requests: int = field(default=0, init=False)
    normalization_heavy_builds: int = field(default=0, init=False)
    normalization_hits: int = field(default=0, init=False)
    normalization_misses: int = field(default=0, init=False)
    normalization_consumer_frame_copies: int = field(default=0, init=False)
    normalization_consumer_scaler_copies: int = field(default=0, init=False)
    sequence_requests: int = field(default=0, init=False)
    sequence_heavy_builds: int = field(default=0, init=False)
    sequence_hits: int = field(default=0, init=False)
    sequence_misses: int = field(default=0, init=False)
    sequence_consumer_array_copies: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.lifecycle_identity = _canonical_lifecycle(self.lifecycle_identity)

    def _require_owner(self, lifecycle_identity: Sequence[Any]) -> None:
        if _canonical_lifecycle(lifecycle_identity) != self.lifecycle_identity:
            raise ProtocolViolation("TRANSFORMATION_REUSE_CONTEXT_OWNER_MISMATCH")

    def normalize(
        self,
        *,
        identity: NormalizationIdentity,
        raw_frames: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        heavy_builder: Callable[[], NormalizationResult],
    ) -> NormalizationResult:
        self._require_owner(identity.train_partition_identity.lifecycle_identity)
        for partition in (
            identity.validation_partition_identity,
            identity.test_partition_identity,
        ):
            self._require_owner(partition.lifecycle_identity)
        self.normalization_requests += 1
        canonical = self._normalizations.get(identity)
        if canonical is None:
            self.normalization_misses += 1
            result = heavy_builder()
            self.normalization_heavy_builds += 1
            canonical = _CanonicalNormalizationResult.from_heavy_result(identity, result)
            if identity in self._normalizations:
                raise ProtocolViolation("NORMALIZATION_REUSE_DUPLICATE_INSERT")
            self._normalizations[identity] = canonical
        else:
            self.normalization_hits += 1
        consumer = canonical.copy_for_consumer(identity, raw_frames)
        self.normalization_consumer_frame_copies += 3
        self.normalization_consumer_scaler_copies += 1
        return consumer

    def sequence(
        self,
        *,
        identity: SequenceIdentity,
        frame: pd.DataFrame,
        heavy_builder: Callable[[], SequenceResult],
    ) -> SequenceResult:
        self._require_owner(identity.lifecycle_identity)
        self.sequence_requests += 1
        canonical = self._sequences.get(identity)
        if canonical is None:
            self.sequence_misses += 1
            result = heavy_builder()
            self.sequence_heavy_builds += 1
            canonical = _CanonicalSequenceResult.from_heavy_result(identity, frame, result)
            if identity in self._sequences:
                raise ProtocolViolation("SEQUENCE_REUSE_DUPLICATE_INSERT")
            self._sequences[identity] = canonical
        else:
            self.sequence_hits += 1
        consumer = canonical.copy_for_consumer(identity, frame)
        self.sequence_consumer_array_copies += 2
        return consumer

    def counts(self) -> Dict[str, int]:
        return {
            "normalization_requests": self.normalization_requests,
            "normalization_heavy_builds": self.normalization_heavy_builds,
            "normalization_hits": self.normalization_hits,
            "normalization_misses": self.normalization_misses,
            "normalization_consumer_frame_copies": self.normalization_consumer_frame_copies,
            "normalization_consumer_scaler_copies": self.normalization_consumer_scaler_copies,
            "sequence_requests": self.sequence_requests,
            "sequence_heavy_builds": self.sequence_heavy_builds,
            "sequence_hits": self.sequence_hits,
            "sequence_misses": self.sequence_misses,
            "sequence_consumer_array_copies": self.sequence_consumer_array_copies,
        }


__all__ = ["TargetTransformationReuseContext"]
