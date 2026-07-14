from __future__ import annotations

import unittest
import json
from pathlib import Path

import pandas as pd

from src.protocols.candidate_pool import (
    SelectionEntry,
    build_candidate_pool_digest,
    build_selection_result_digest,
)

from scripts.run_strict_protocol_baseline import (
    build_mode_expected_contract,
    build_matrix_tasks,
    combine_result_frames,
)
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.utils.result_acceptance import AcceptanceScope


def _row(horizon: int, seed: int, method: str = "MSWA-TL") -> dict:
    digest_input = {
        "protocol_version": "d1_d6_protocol_v1",
        "dataset_id": "D1",
        "scenario": "without",
        "target_key": ["1", "10"],
        "group_cols": ["store_id", "item_id"],
        "candidate_keys": [["1", "1"]],
        "observed_start": "2017-06-05",
        "observed_end": "2017-07-04",
        "feature_cols": ["sales"],
    }
    candidate_digest = build_candidate_pool_digest(**digest_input)
    selected = {
        "source_rank": 1,
        "source_key": ["1", "1"],
        "distance": 0.5,
        "weight": 1.0,
        "tie_group": 1,
    }
    selection_digest = build_selection_result_digest(
        protocol_version="d1_d6_protocol_v1",
        candidate_pool_digest=candidate_digest,
        k=1,
        weight_mode="inverse_distance",
        weight_epsilon=1e-8,
        entries=(SelectionEntry(1, ("1", "1"), 0.5, 1.0, 1, "", "", (), ()),),
    )
    return {
        "dataset_id": "D1",
        "target_entity_key": "1/10",
        "scenario": "without",
        "method": method,
        "protocol_track": "strict_paper",
        "protocol_version": "d1_d6_protocol_v1",
        "knn_observed_start": "2017-06-05",
        "knn_observed_end": "2017-07-04",
        "knn_representation": "daily_sales_flattened_30d",
        "target_test_excluded": True,
        "source_future_excluded": True,
        "candidate_pool_digest": candidate_digest,
        "candidate_pool_digest_input": json.dumps(digest_input),
        "selection_result_digest": selection_digest,
        "selected_sources_runtime": json.dumps([selected]),
        "selection_authority": "shared_protocol",
        "requested_k": 1,
        "effective_k": 1,
        "failed_source_count": 0,
        "skipped_source_count": 0,
        "cnn_provenance_validated": True,
        "cnn_provenance_source_keys": json.dumps([["1", "1"]]),
        "cnn_provenance_sample_counts": json.dumps([20]),
        "horizon": horizon,
        "seed": seed,
        "primary_metric_space": "original_sales",
        "rmse_metric_space": "original_sales_space",
        "smape_metric_space": "original_sales_space",
        "sample_manifest_digest": "c" * 64,
        "sample_count": 170 - horizon + 1,
        "rmse": 1.0,
        "mae": 1.0,
        "smape": 1.0,
        "accuracy": 1.0,
        "error": "",
    }


class FormalProtocolMatrixTest(unittest.TestCase):
    def test_matrix_has_25_unique_horizon_seed_cells_and_cli_overrides(self) -> None:
        tasks = build_matrix_tasks(
            dataset="d5",
            scenario="with",
            output_dir=Path("outputs/formal"),
        )
        self.assertEqual(len(tasks), 25)
        self.assertEqual({task.horizon for task in tasks}, set(range(1, 6)))
        self.assertEqual({task.seed for task in tasks}, set(range(42, 47)))
        self.assertEqual(len({task.output_dir for task in tasks}), 25)
        for task in tasks:
            self.assertIn("--horizon", task.command)
            self.assertIn(str(task.horizon), task.command)
            self.assertIn("--seed", task.command)
            self.assertIn(str(task.seed), task.command)

    def test_combiner_promotes_only_complete_matrix(self) -> None:
        frames = [
            pd.DataFrame([_row(horizon, seed, method) for method in FORMAL_METHODS])
            for horizon in range(1, 6)
            for seed in range(42, 47)
        ]
        combined = combine_result_frames(frames)
        self.assertEqual(len(combined), 150)
        self.assertEqual(set(combined["result_status"]), {"confirmed_baseline"})

        with self.assertRaisesRegex(ValueError, "incomplete groups"):
            combine_result_frames(frames[:-1])

        missing_method = [frame[frame["method"] != "MSML-TL-RFE"] for frame in frames]
        with self.assertRaisesRegex(ValueError, "method coverage"):
            combine_result_frames(missing_method)

    def test_mode_contract_uses_authoritative_targets_and_correct_methods(self) -> None:
        expected = build_mode_expected_contract(
            dataset="d5",
            scenario="with",
        )

        self.assertEqual(expected.scope, AcceptanceScope.MODE_MATRIX)
        self.assertEqual(expected.methods, FORMAL_METHODS)
        self.assertEqual(len(expected.targets_by_dataset_mode[(5, "with")]), 5)
        self.assertEqual(expected.protocol_tracks, ("strict_paper",))

    def test_d1_matrix_uses_child_cli_dataset_choice(self) -> None:
        task = build_matrix_tasks(
            dataset="d1",
            scenario="without",
            output_dir=Path("outputs/formal"),
        )[0]
        index = task.command.index("--only-dataset")
        self.assertEqual(task.command[index + 1], "dataset1")


if __name__ == "__main__":
    unittest.main()
