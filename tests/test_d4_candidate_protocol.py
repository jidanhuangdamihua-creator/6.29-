from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import (
    SourceIdentity,
    build_candidate_keys,
    get_experiment_protocol,
)
from src.protocols.runner_adapter import configure_protocol_frames


DATES = pd.date_range("2020-01-01", periods=35, freq="D")
OBSERVED_DATES = DATES[:30]
TARGET_KEY = ("166", "258")
SAME_STORE_DIFFERENT_CATEGORY_KEY = ("166", "259")
SAME_STORE_SAME_CATEGORY_KEY = ("166", "261")
CROSS_STORE_DIFFERENT_CATEGORY_KEY = ("167", "260")
CROSS_STORE_SAME_PRODUCT_KEY = ("168", "258")


def _rows(
    store_id: int,
    product_id: int,
    second_category_id: int,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": store_id,
            "product_id": product_id,
            "second_category_id": second_category_id,
            "date": dates,
            "sales": np.full(len(dates), float(product_id)),
        }
    )


def _d4_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    target = _rows(166, 258, 20, DATES)
    source = pd.concat(
        [
            _rows(166, 258, 20, OBSERVED_DATES),
            _rows(166, 259, 30, OBSERVED_DATES),
            _rows(166, 261, 20, OBSERVED_DATES),
            _rows(167, 260, 40, OBSERVED_DATES),
            _rows(168, 258, 50, OBSERVED_DATES),
        ],
        ignore_index=True,
    )
    return source, target


def _candidate_keys(
    *, scenario: str, prepared: bool = False
) -> tuple[tuple[str, str], ...]:
    source, target = _d4_frames()
    pool = (
        prepare_daily_sequence_pool(
            source,
            group_cols=("store_id", "product_id"),
            observed_start="2020-01-01",
            metadata_cols=("second_category_id",),
        )
        if prepared
        else None
    )
    configured_source, _ = configure_protocol_frames(
        source,
        target,
        dataset_id="D4",
        scenario=scenario,
        group_cols=("store_id", "product_id"),
        grouping_col="second_category_id",
        observed_start="2020-01-01",
        prepared_pool=pool,
    )
    return configured_source.attrs["protocol_candidate_keys"]


class Dataset4CandidateProtocolTest(unittest.TestCase):
    def test_shared_d4_protocol_uses_product_not_second_category_for_eligibility(
        self,
    ) -> None:
        identities = (
            SourceIdentity(TARGET_KEY, "20"),
            SourceIdentity(SAME_STORE_DIFFERENT_CATEGORY_KEY, "30"),
            SourceIdentity(SAME_STORE_SAME_CATEGORY_KEY, "20"),
            SourceIdentity(CROSS_STORE_DIFFERENT_CATEGORY_KEY, "40"),
            SourceIdentity(CROSS_STORE_SAME_PRODUCT_KEY, "50"),
        )
        protocol = get_experiment_protocol("D4")

        without_candidate_keys = build_candidate_keys(
            protocol, "without", TARGET_KEY, identities
        )
        with_candidate_keys = build_candidate_keys(protocol, "with", TARGET_KEY, identities)

        self.assertEqual(
            without_candidate_keys,
            (SAME_STORE_DIFFERENT_CATEGORY_KEY, SAME_STORE_SAME_CATEGORY_KEY),
        )
        self.assertEqual(
            with_candidate_keys,
            (
                SAME_STORE_DIFFERENT_CATEGORY_KEY,
                SAME_STORE_SAME_CATEGORY_KEY,
                CROSS_STORE_DIFFERENT_CATEGORY_KEY,
            ),
        )

    def test_d5_still_excludes_different_group_candidates(self) -> None:
        target = _rows(166, 258, 20, DATES).assign(family="F1")
        source = pd.concat(
            [
                _rows(166, 259, 30, OBSERVED_DATES).assign(family="F1"),
                _rows(167, 260, 40, OBSERVED_DATES).assign(family="F1"),
                _rows(166, 261, 20, OBSERVED_DATES).assign(family="F2"),
            ],
            ignore_index=True,
        )

        configured_source, _ = configure_protocol_frames(
            source,
            target,
            dataset_id="D5",
            scenario="with",
            group_cols=("store_id", "product_id"),
            grouping_col="family",
            observed_start="2020-01-01",
        )

        self.assertEqual(
            configured_source.attrs["protocol_candidate_keys"],
            (("166", "259"), ("167", "260")),
        )

    def test_without_allows_different_second_category_but_not_cross_store(
        self,
    ) -> None:
        candidate_keys = _candidate_keys(scenario="without")

        self.assertIn(SAME_STORE_DIFFERENT_CATEGORY_KEY, candidate_keys)
        self.assertNotIn(TARGET_KEY, candidate_keys)
        self.assertNotIn(CROSS_STORE_DIFFERENT_CATEGORY_KEY, candidate_keys)
        self.assertNotIn(CROSS_STORE_SAME_PRODUCT_KEY, candidate_keys)

    def test_with_allows_cross_store_different_second_category_but_not_same_product(
        self,
    ) -> None:
        candidate_keys = _candidate_keys(scenario="with")

        self.assertIn(CROSS_STORE_DIFFERENT_CATEGORY_KEY, candidate_keys)
        self.assertNotIn(TARGET_KEY, candidate_keys)
        self.assertNotIn(CROSS_STORE_SAME_PRODUCT_KEY, candidate_keys)

    def test_prepared_pool_does_not_restore_second_category_restriction(
        self,
    ) -> None:
        without_candidate_keys = _candidate_keys(scenario="without", prepared=True)
        with_candidate_keys = _candidate_keys(scenario="with", prepared=True)

        self.assertIn(SAME_STORE_DIFFERENT_CATEGORY_KEY, without_candidate_keys)
        self.assertIn(CROSS_STORE_DIFFERENT_CATEGORY_KEY, with_candidate_keys)
