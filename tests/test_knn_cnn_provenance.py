from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from src.protocols.candidate_pool import select_daily_sequence_sources
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol
from src.protocols.provenance import (
    assert_actual_cnn_training_validated,
    bind_actual_cnn_source_frame,
    build_cnn_tensor_provenance,
    extract_selected_source_slices,
    validate_cnn_tensor_provenance,
)
from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
)


class KnnCnnProvenanceTest(unittest.TestCase):
    def test_actual_normalized_cnn_arrays_are_validated_against_raw_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "entity_id": "S1",
                "item_id": "I1",
                "date": pd.date_range("2020-01-01", periods=40, freq="D"),
                "sales": np.arange(40, dtype=float),
                "planned_price": np.arange(40, dtype=float) + 100.0,
            }
        )
        bind_actual_cnn_source_frame(
            frame,
            source_key=("S1", "I1"),
            group_cols=("entity_id", "item_id"),
            feature_cols=("sales", "planned_price"),
        )
        frame.attrs.update(
            {
                "split_role": "source",
                "split_mode": "ratio",
                "split_config": {"train_ratio": 0.8, "val_ratio": 0.1},
            }
        )
        train, validation, test = temporal_split_by_ratio_or_dates(frame)
        train, validation, test, _, features = normalize_features(
            train,
            validation,
            test,
            feature_columns=("sales", "planned_price"),
        )
        x_source, y_source = build_tabular_sequence(
            train,
            horizon=1,
            window_size=10,
            feature_columns=features,
        )
        self.assertGreater(len(y_source), 0)
        assert_actual_cnn_training_validated(train, source_key=("S1", "I1"))

        tampered = train.copy()
        tampered.attrs = dict(train.attrs)
        tampered.loc[tampered.index[0], "sales"] += 0.25
        with self.assertRaisesRegex(ProtocolViolation, "actual CNN input tensor"):
            build_tabular_sequence(
                tampered,
                horizon=1,
                window_size=10,
                feature_columns=features,
            )

    def setUp(self) -> None:
        dates = pd.date_range("2020-01-01", periods=42, freq="D")
        self.source = pd.concat(
            [
                pd.DataFrame(
                    {
                        "store": "S1",
                        "item": "I1",
                        "date": dates,
                        "sales": np.arange(42, dtype=float),
                        "planned_price": np.arange(42, dtype=float) + 100.0,
                    }
                ),
                pd.DataFrame(
                    {
                        "store": "S2",
                        "item": "I2",
                        "date": dates,
                        "sales": np.arange(42, dtype=float) + 50.0,
                        "planned_price": np.arange(42, dtype=float) + 200.0,
                    }
                ),
            ],
            ignore_index=True,
        )
        observed_dates = pd.date_range("2020-01-11", periods=30, freq="D")
        target = pd.DataFrame(
            {"date": observed_dates, "sales": np.arange(10, 40, dtype=float)}
        )
        self.selection = select_daily_sequence_sources(
            target_df=target,
            source_df=self.source,
            protocol=get_experiment_protocol("D4"),
            scenario="with",
            target_key=("T", "I0"),
            candidate_keys=(("S1", "I1"), ("S2", "I2")),
            group_cols=("store", "item"),
            observed_start="2020-01-11",
            feature_cols=("sales",),
            k=1,
        )

    def test_selected_slice_and_knn_vector_match_raw_rows_exactly(self) -> None:
        slices = extract_selected_source_slices(
            self.selection,
            self.source,
            training_start="2020-01-01",
            model_feature_cols=("sales", "planned_price"),
        )

        self.assertEqual(len(slices), 1)
        selected = slices[0]
        self.assertEqual(selected.source_key, self.selection.ordered_source_keys[0])
        self.assertEqual(
            (selected.source_key[0], selected.source_key[1], selected.date_start, selected.date_end),
            ("S1", "I1", "2020-01-01", "2020-02-09"),
        )
        raw = self.source[
            (self.source["store"] == "S1")
            & (self.source["item"] == "I1")
            & self.source["date"].between("2020-01-11", "2020-02-09")
        ].sort_values("date")
        np.testing.assert_array_equal(
            np.asarray(self.selection.entries[0].raw_vector),
            raw["sales"].to_numpy(dtype=float),
        )
        np.testing.assert_array_equal(
            np.asarray(selected.values),
            self.source[
                (self.source["store"] == "S1")
                & (self.source["item"] == "I1")
                & self.source["date"].between("2020-01-01", "2020-02-09")
            ].sort_values("date")[["sales", "planned_price"]].to_numpy(dtype=float),
        )

    def test_cnn_tensor_dates_features_and_labels_validate_elementwise(self) -> None:
        selected = extract_selected_source_slices(
            self.selection,
            self.source,
            training_start="2020-01-01",
            model_feature_cols=("sales", "planned_price"),
        )[0]
        provenance = build_cnn_tensor_provenance(
            selected,
            window_size=3,
            horizon=2,
            label_col="sales",
        )

        validate_cnn_tensor_provenance(
            provenance,
            self.source,
            group_cols=("store", "item"),
        )
        self.assertEqual(provenance.input_dates[0], ("2020-01-01", "2020-01-02", "2020-01-03"))
        self.assertEqual(provenance.label_dates[0], "2020-01-05")
        np.testing.assert_array_equal(
            provenance.input_tensor[0],
            np.asarray([[0.0, 100.0], [1.0, 101.0], [2.0, 102.0]]),
        )
        self.assertEqual(float(provenance.labels[0]), 4.0)

    def test_key_substitution_date_reordering_and_value_mutation_fail(self) -> None:
        selected = extract_selected_source_slices(
            self.selection,
            self.source,
            training_start="2020-01-01",
            model_feature_cols=("sales", "planned_price"),
        )[0]
        provenance = build_cnn_tensor_provenance(
            selected,
            window_size=3,
            horizon=1,
            label_col="sales",
        )

        with self.assertRaisesRegex(ProtocolViolation, "source key"):
            validate_cnn_tensor_provenance(
                replace(provenance, source_key=("S9", "I9")),
                self.source,
                group_cols=("store", "item"),
            )

        reordered_dates = list(provenance.input_dates)
        reordered_dates[0] = tuple(reversed(reordered_dates[0]))
        with self.assertRaisesRegex(ProtocolViolation, "date order"):
            validate_cnn_tensor_provenance(
                replace(provenance, input_dates=tuple(reordered_dates)),
                self.source,
                group_cols=("store", "item"),
            )

        changed_tensor = provenance.input_tensor.copy()
        changed_tensor[0, 0, 0] += 1.0
        with self.assertRaisesRegex(ProtocolViolation, "input tensor"):
            validate_cnn_tensor_provenance(
                replace(provenance, input_tensor=changed_tensor),
                self.source,
                group_cols=("store", "item"),
            )


if __name__ == "__main__":
    unittest.main()
