from __future__ import annotations

import random
import unittest

import numpy as np
import pandas as pd

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    temporal_split_by_ratio_or_dates,
)
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.rolling_origin import (
    assert_same_sample_manifest,
    build_sample_manifest,
    validate_feature_availability,
)


class RollingOriginProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=40, freq="D"),
                "sales": np.arange(40, dtype=float),
            }
        )

    def test_horizons_dates_labels_and_digest_are_exact(self) -> None:
        manifest = build_sample_manifest(
            self.frame,
            dataset_id="D1",
            track="strict_paper",
            scenario="without",
            target_key=("Store1", "Item10"),
            observed_end="2020-01-30",
            input_window=5,
        )

        self.assertEqual(manifest.horizons, (1, 2, 3, 4, 5))
        self.assertEqual(len(manifest.for_horizon(1)), 10)
        self.assertEqual(len(manifest.for_horizon(5)), 6)
        first = manifest.for_horizon(1)[0]
        self.assertEqual(first.forecast_origin, "2020-01-30")
        self.assertEqual(first.input_start, "2020-01-26")
        self.assertEqual(first.input_end, "2020-01-30")
        self.assertEqual(first.label_date, "2020-01-31")
        self.assertEqual(first.label, 30.0)
        self.assertEqual(len(manifest.digest), 64)
        self.assertEqual(
            manifest.digest,
            build_sample_manifest(
                self.frame.sample(frac=1.0, random_state=7),
                dataset_id="D1",
                track="strict_paper",
                scenario="without",
                target_key=("Store1", "Item10"),
                observed_end="2020-01-30",
                input_window=5,
            ).digest,
        )

    def test_methods_must_use_same_ordered_sample_keys(self) -> None:
        manifest = build_sample_manifest(
            self.frame,
            dataset_id="D4",
            track="extended",
            scenario="with",
            target_key=("S1", "I1"),
            observed_end="2020-01-30",
            input_window=10,
        )
        keys = manifest.sample_keys
        assert_same_sample_manifest(manifest, keys, method="CNN")
        with self.assertRaisesRegex(ProtocolViolation, "sample manifest mismatch"):
            assert_same_sample_manifest(manifest, keys[:-1], method="BL1")

    def test_manifest_can_start_at_common_model_valid_origin(self) -> None:
        manifest = build_sample_manifest(
            self.frame,
            dataset_id="D1",
            track="strict_paper",
            scenario="without",
            target_key=("Store1", "Item10"),
            observed_end="2020-01-30",
            first_forecast_origin="2020-02-04",
            input_window=5,
        )
        first = manifest.for_horizon(1)[0]
        self.assertEqual(first.forecast_origin, "2020-02-04")
        self.assertEqual(first.label_date, "2020-02-05")

    def test_feature_availability_is_allowlist_only(self) -> None:
        allowlist = {
            "day_of_week": "known_in_advance",
            "planned_price": "known_in_advance",
        }
        validate_feature_availability(
            ("day_of_week", "planned_price"),
            allowlist=allowlist,
        )
        for forbidden in ("sales", "transactions", "stock", "customers"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ProtocolViolation, "not available at forecast origin"):
                    validate_feature_availability((forbidden,), allowlist=allowlist)
        with self.assertRaisesRegex(ProtocolViolation, "not available at forecast origin"):
            validate_feature_availability(("unreviewed_numeric_field",), allowlist=allowlist)

    def test_protocol_seed_resets_python_and_numpy(self) -> None:
        set_protocol_seed(43, include_frameworks=False)
        first = (random.random(), float(np.random.random()))
        set_protocol_seed(43, include_frameworks=False)
        second = (random.random(), float(np.random.random()))
        self.assertEqual(first, second)

    def test_manifest_label_identity_matches_cnn_sequence_builder(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=210, freq="D"),
                "entity_id": "1",
                "item_id": "10",
                "sales": np.arange(210, dtype=float),
            }
        )
        frame.attrs.update(
            {
                "split_role": "target",
                "split_mode": "days",
                "split_config": {"train_days": 15, "val_days": 15, "test_days": 180},
            }
        )
        manifest = build_sample_manifest(
            frame,
            dataset_id="D1",
            track="strict_paper",
            scenario="without",
            target_key=("1", "10"),
            observed_end="2020-01-30",
            first_forecast_origin="2020-02-09",
            input_window=10,
        )
        frame.attrs["protocol_sample_manifest"] = manifest
        _, _, test = temporal_split_by_ratio_or_dates(frame)
        for horizon in range(1, 6):
            _, cnn_labels = build_tabular_sequence(
                test,
                horizon=horizon,
                window_size=10,
                feature_columns=("sales",),
            )
            manifest_labels = np.asarray(
                [record.label for record in manifest.for_horizon(horizon)]
            )
            np.testing.assert_array_equal(cnn_labels, manifest_labels)

    def test_cnn_sequence_rejects_manifest_input_date_mismatch(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=210, freq="D"),
                "entity_id": "1",
                "item_id": "10",
                "sales": np.arange(210, dtype=float),
            }
        )
        frame.attrs.update(
            {
                "split_role": "target",
                "split_mode": "days",
                "split_config": {"train_days": 15, "val_days": 15, "test_days": 180},
                "protocol_sample_manifest": build_sample_manifest(
                    frame,
                    dataset_id="D1",
                    track="strict_paper",
                    scenario="without",
                    target_key=("1", "10"),
                    observed_end="2020-01-30",
                    first_forecast_origin="2020-02-09",
                    input_window=30,
                ),
            }
        )
        _, _, test = temporal_split_by_ratio_or_dates(frame)
        with self.assertRaisesRegex(ValueError, "does not consume"):
            build_tabular_sequence(
                test,
                horizon=1,
                window_size=10,
                feature_columns=("sales",),
            )


if __name__ == "__main__":
    unittest.main()
