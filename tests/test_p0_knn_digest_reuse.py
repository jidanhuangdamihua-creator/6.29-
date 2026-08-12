from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.protocols.knn_frames as knn_frames
from src.protocols.experiment_protocol import ObservationWindow, ProtocolViolation
from src.protocols.knn_frames import (
    build_observed_knn_frame,
    canonical_knn_frame_digest,
    get_configured_knn_frame,
)
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector
from src.utils.dataframe_attrs import get_protocol_frame_context


DIGEST_CASES = {
    "D1": (("store_id", "item_id"), ("sales",), "d3d3706aefcdb6aa25f98d07f6e5b31807fa1de5fda92a39c6320404016ff1b6"),
    "D2": (("brand_id", "item_id"), ("sales", "promo"), "c13e12c7d99d2d810c2c53136d5d988d6f67c0d563a9a7a919ad60f9ba606def"),
    "D3": (("store_id",), ("sales",), "c560e2aa42909c7b7b95868b48a9b47e2ec99d91415c81484ece3375071c7b0f"),
    "D4": (("store_id", "product_id"), ("sales",), "e70a88bed7a23099e8fc315d200cce367c8d12ec9d98f48f09e4f130c2537509"),
    "D5": (("store_nbr", "item_nbr"), ("sales", "onpromotion", "oil_price"), "cab39ea1876c5ce331641d47007bb062a29102006e060b9238c333ae36bc425f"),
    "D6": (("store_id", "item_id"), ("sales",), "d23344b5a59bf6f7d0882f0aea08558ad6b67c0a1e47259e7f1713aeb258f872"),
}


@pytest.mark.parametrize("dataset_id", DIGEST_CASES)
def test_d1_d6_canonical_digest_bytes_match_pre_p0_golden(dataset_id: str) -> None:
    group_cols, feature_cols, expected = DIGEST_CASES[dataset_id]
    number = int(dataset_id[1:])
    payload: dict[str, object] = {
        "date": pd.date_range("2020-01-01", periods=3, freq="D")
    }
    for offset, column in enumerate(group_cols):
        payload[column] = [number * 10 + offset] * 3
    for offset, column in enumerate(feature_cols):
        payload[column] = np.asarray(
            [number + offset + 0.125, number + offset + 1.25, number + offset + 2.5],
            dtype=float,
        )
    frame = pd.DataFrame(payload)

    actual = canonical_knn_frame_digest(
        frame,
        group_cols=group_cols,
        feature_cols=None if feature_cols == ("sales",) else feature_cols,
        ignore_columns=(),
    )
    assert actual == expected


def _observed_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": 1,
            "product_id": 2,
            "date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "sales": np.arange(30, dtype=float),
        }
    )


def test_builder_owned_observed_digest_is_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_compute = knn_frames._compute_canonical_knn_frame_digest

    def counting_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(knn_frames, "_compute_canonical_knn_frame_digest", counting_compute)
    observed = build_observed_knn_frame(
        _observed_source(),
        window=ObservationWindow.from_start("2020-01-01"),
        role="source",
        group_cols=("store_id", "product_id"),
        feature_cols=("sales",),
    )
    first = observed.attrs["knn_frame_digest"]
    assert canonical_knn_frame_digest(
        observed,
        group_cols=("store_id", "product_id"),
        feature_cols=None,
        ignore_columns=("promo",),
    ) == first
    assert calls == 1


def test_copied_or_tampered_frame_cannot_reuse_trusted_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = build_observed_knn_frame(
        _observed_source(),
        window=ObservationWindow.from_start("2020-01-01"),
        role="source",
        group_cols=("store_id", "product_id"),
        feature_cols=("sales",),
    )
    copied = observed.copy()
    copied.loc[copied.index[0], "sales"] += 1.0
    with pytest.raises(ProtocolViolation, match="digest metadata"):
        canonical_knn_frame_digest(
            copied,
            group_cols=("store_id", "product_id"),
            feature_cols=None,
            ignore_columns=("promo",),
        )


def test_public_digest_without_builder_evidence_is_never_trusted() -> None:
    frame = _observed_source()
    frame.attrs["knn_frame_digest"] = "0" * 64
    with pytest.raises(ProtocolViolation, match="digest metadata"):
        canonical_knn_frame_digest(
            frame,
            group_cols=("store_id", "product_id"),
            feature_cols=None,
            ignore_columns=("promo",),
        )


def test_public_context_identity_mismatch_cannot_reuse_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_compute = knn_frames._compute_canonical_knn_frame_digest

    def counting_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(knn_frames, "_compute_canonical_knn_frame_digest", counting_compute)
    observed = build_observed_knn_frame(
        _observed_source(),
        window=ObservationWindow.from_start("2020-01-01"),
        role="source",
        group_cols=("store_id", "product_id"),
        feature_cols=("sales",),
    )
    observed.attrs[knn_frames._DIGEST_IDENTITY_ATTR] = (("dataset_id", "wrong"),)
    canonical_knn_frame_digest(
        observed,
        group_cols=("store_id", "product_id"),
        feature_cols=None,
        ignore_columns=("promo",),
    )
    assert calls == 2


def test_copy_slice_concat_materialize_and_spec_mismatch_recompute_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_compute = knn_frames._compute_canonical_knn_frame_digest

    def counting_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(knn_frames, "_compute_canonical_knn_frame_digest", counting_compute)
    observed = build_observed_knn_frame(
        _observed_source(),
        window=ObservationWindow.from_start("2020-01-01"),
        role="source",
        group_cols=("store_id", "product_id"),
        feature_cols=("sales",),
    )
    assert calls == 1

    copied = observed.copy()
    assert canonical_knn_frame_digest(
        copied,
        group_cols=("store_id", "product_id"),
        feature_cols=None,
        ignore_columns=("promo",),
    ) == observed.attrs["knn_frame_digest"]

    sliced = observed.iloc[:10]
    with pytest.raises(
        ProtocolViolation,
        match="KNN frame digest metadata differs from actual frame content",
    ):
        canonical_knn_frame_digest(
            sliced,
            group_cols=("store_id", "product_id"),
            feature_cols=None,
            ignore_columns=("promo",),
        )

    concatenated = pd.concat(
        [observed.iloc[:15], observed.iloc[15:]],
        ignore_index=True,
    )
    assert canonical_knn_frame_digest(
        concatenated,
        group_cols=("store_id", "product_id"),
        feature_cols=None,
        ignore_columns=("promo",),
    ) == observed.attrs["knn_frame_digest"]

    materialized = pd.DataFrame(observed.to_dict(orient="list"))
    assert canonical_knn_frame_digest(
        materialized,
        group_cols=("store_id", "product_id"),
        feature_cols=None,
        ignore_columns=("promo",),
    ) == observed.attrs["knn_frame_digest"]
    assert calls == 5

    canonical_knn_frame_digest(
        observed,
        group_cols=("store_id", "product_id"),
        feature_cols=("sales",),
        ignore_columns=(),
    )
    assert calls == 6


def _configured_d1_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_dates = pd.date_range("2017-06-01", periods=30, freq="D")
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "store_id": store,
                    "item_id": item,
                    "date": observed_dates,
                    "sales": float(item),
                }
            )
            for store in range(1, 4)
            for item in range(1, 10)
        ],
        ignore_index=True,
    )
    target_dates = pd.date_range("2017-06-01", periods=31, freq="D")
    target = pd.DataFrame(
        {
            "store_id": 1,
            "item_id": 10,
            "date": target_dates,
            "sales": 0.0,
        }
    )
    return configure_protocol_frames(
        source,
        target,
        dataset_id="D1",
        scenario="with",
        group_cols=("store_id", "item_id"),
        observed_start="2017-06-01",
    )


def test_builder_owned_carrier_returns_independent_working_copy() -> None:
    configured_source, _ = _configured_d1_frames()
    context = get_protocol_frame_context(configured_source)
    assert context is not None
    trusted = context.observed_frames["source"]

    working = get_configured_knn_frame(configured_source, "source")
    assert working is not trusted
    pd.testing.assert_frame_equal(working, trusted)
    working.attrs["working_copy_only"] = True
    assert "working_copy_only" not in trusted.attrs


def test_legacy_configured_attr_without_context_remains_supported() -> None:
    carrier = _observed_source()
    trusted = build_observed_knn_frame(
        carrier,
        window=ObservationWindow.from_start("2020-01-01"),
        role="source",
        group_cols=("store_id", "product_id"),
        feature_cols=("sales",),
    )
    carrier.attrs[knn_frames._CONFIGURED_FRAME_ATTR] = trusted

    working = get_configured_knn_frame(carrier, "source")
    assert working is not trusted
    pd.testing.assert_frame_equal(working, trusted)


@pytest.mark.parametrize(
    "derivative_kind",
    ("copy", "slice", "concat", "materialized"),
)
@pytest.mark.parametrize("tampered", (False, True))
def test_configured_carrier_derivatives_use_working_copy_semantics(
    derivative_kind: str,
    tampered: bool,
) -> None:
    configured_source, _ = _configured_d1_frames()
    if derivative_kind == "copy":
        derivative = configured_source.copy()
    elif derivative_kind == "slice":
        derivative = configured_source.iloc[:].copy()
    elif derivative_kind == "concat":
        midpoint = len(configured_source) // 2
        derivative = pd.concat(
            [configured_source.iloc[:midpoint], configured_source.iloc[midpoint:]],
            ignore_index=True,
        )
    else:
        derivative = pd.DataFrame(configured_source.to_dict(orient="list"))
    derivative.attrs = configured_source.attrs.copy()

    working = get_configured_knn_frame(derivative, "source")
    context = get_protocol_frame_context(configured_source)
    assert context is not None
    assert working is not context.observed_frames["source"]
    if tampered:
        working.loc[working.index[0], "sales"] += 1.0
        with pytest.raises(
            ProtocolViolation,
            match="KNN frame digest metadata differs from actual frame content",
        ):
            canonical_knn_frame_digest(
                working,
                group_cols=("store_id", "item_id"),
                feature_cols=None,
                ignore_columns=("promo",),
            )
    else:
        pd.testing.assert_frame_equal(working, context.observed_frames["source"])


def test_selector_accepts_copied_configured_carrier_without_rebuilding() -> None:
    configured_source, configured_target = _configured_d1_frames()
    copied_source = configured_source.copy()
    assert id(copied_source) != id(configured_source)
    assert get_protocol_frame_context(copied_source) is get_protocol_frame_context(
        configured_source
    )

    selected = SourceSelector().select_top_k_sources(
        configured_target,
        copied_source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_id", "item_id"),
    )
    assert len(selected["sources"]) == 2


def test_configure_and_selector_hash_each_builder_owned_role_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_compute = knn_frames._compute_canonical_knn_frame_digest

    def counting_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(knn_frames, "_compute_canonical_knn_frame_digest", counting_compute)
    configured_source, configured_target = _configured_d1_frames()
    SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_id", "item_id"),
    )
    assert calls == 4
