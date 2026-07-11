from __future__ import annotations

import unittest

import pandas as pd

from src.constants import STRICT_PROTOCOL_FIELDS
from src.utils.result_schema import align_d1_d3_result_records
from src.utils.result_validation import (
    classify_protocol_result,
    confirmed_baseline_rows,
    promote_complete_baseline_groups,
    validate_confirmed_baseline_group,
)


def _strict_row(*, horizon: int = 1, seed: int = 42) -> dict:
    return {
        "dataset_id": "D1",
        "target_entity_key": "Store1/Item10",
        "scenario": "without",
        "method": "MSWA-TL",
        "protocol_track": "strict_paper",
        "protocol_version": "d1_d6_protocol_v1",
        "knn_observed_start": "2017-06-05",
        "knn_observed_end": "2017-07-04",
        "knn_representation": "daily_sales_flattened_30d",
        "target_test_excluded": True,
        "source_future_excluded": True,
        "candidate_pool_digest": "a" * 64,
        "selection_result_digest": "b" * 64,
        "horizon": horizon,
        "seed": seed,
        "primary_metric_space": "original_sales",
        "sample_manifest_digest": "c" * 64,
        "rmse": 1.0,
        "mae": 0.5,
        "smape": 2.0,
        "accuracy": 1.0,
        "error": "",
    }


class StrictResultContractTest(unittest.TestCase):
    def test_mandatory_fields_match_approved_contract(self) -> None:
        self.assertEqual(
            STRICT_PROTOCOL_FIELDS,
            (
                "protocol_track",
                "protocol_version",
                "knn_observed_start",
                "knn_observed_end",
                "knn_representation",
                "target_test_excluded",
                "source_future_excluded",
                "candidate_pool_digest",
                "selection_result_digest",
                "horizon",
                "seed",
                "primary_metric_space",
                "sample_manifest_digest",
            ),
        )

    def test_old_rows_are_legacy_unverified_and_not_silently_filled(self) -> None:
        aligned = align_d1_d3_result_records(
            [
                {
                    "dataset_id": 1,
                    "method": "No-TL",
                    "information_sharing": "without",
                    "rmse": 1.0,
                    "smape": 2.0,
                }
            ]
        )
        self.assertEqual(aligned.loc[0, "result_status"], "legacy_unverified")
        self.assertEqual(aligned.loc[0, "protocol_version"], "")
        self.assertEqual(classify_protocol_result(aligned.iloc[0].to_dict()), "legacy_unverified")

    def test_complete_row_is_trial_until_full_group_is_validated(self) -> None:
        self.assertEqual(classify_protocol_result(_strict_row()), "trial")

    def test_only_complete_five_seed_five_horizon_group_is_confirmed(self) -> None:
        rows = pd.DataFrame(
            [
                _strict_row(horizon=horizon, seed=seed)
                for horizon in range(1, 6)
                for seed in range(42, 47)
            ]
        )
        confirmed = validate_confirmed_baseline_group(rows)
        self.assertTrue((confirmed["result_status"] == "confirmed_baseline").all())
        self.assertEqual(len(confirmed_baseline_rows(confirmed)), 25)

        incomplete = rows.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "five horizons.*five seeds"):
            validate_confirmed_baseline_group(incomplete)
        mixed = pd.concat(
            [confirmed, pd.DataFrame([{"result_status": "legacy_unverified"}])],
            ignore_index=True,
        )
        self.assertEqual(len(confirmed_baseline_rows(mixed)), 25)

    def test_group_promotion_keeps_legacy_and_incomplete_trials_out(self) -> None:
        complete = pd.DataFrame(
            [
                _strict_row(horizon=horizon, seed=seed)
                for horizon in range(1, 6)
                for seed in range(42, 47)
            ]
        )
        incomplete = complete.iloc[:-1].copy()
        incomplete["method"] = "MSSB-TL"
        legacy = pd.DataFrame([{"dataset_id": "D1", "method": "old"}])
        promoted = promote_complete_baseline_groups(
            pd.concat([complete, incomplete, legacy], ignore_index=True)
        )

        self.assertEqual(
            set(promoted.loc[promoted["method"] == "MSWA-TL", "result_status"]),
            {"confirmed_baseline"},
        )
        self.assertEqual(
            set(promoted.loc[promoted["method"] == "MSSB-TL", "result_status"]),
            {"trial"},
        )
        self.assertEqual(
            promoted.loc[promoted["method"] == "old", "result_status"].iloc[0],
            "legacy_unverified",
        )


if __name__ == "__main__":
    unittest.main()
