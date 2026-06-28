import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import msml_tl
import msml_tl_rfe


class _FakeLayer:
    def __init__(self, name):
        self.name = name
        self.trainable = True


class _FakeModel:
    def __init__(self):
        self.layers = [_FakeLayer(name) for name in ["conv1", "pool1", "conv2", "pool2", "conv3"]]


class TestMsmlTlTransferLayers(unittest.TestCase):
    def _source_df(self):
        return pd.DataFrame(
            {
                "entity_id": [1, 1],
                "item_id": [2, 2],
                "sales": [10.0, 11.0],
                "feat": [1.0, 2.0],
            }
        )

    def _target_df(self):
        return pd.DataFrame(
            {
                "entity_id": [9, 9],
                "item_id": [8, 8],
                "sales": [20.0, 21.0],
                "feat": [3.0, 4.0],
            }
        )

    def test_run_msml_tl_fuses_weighted_layers_but_freezes_first_four_layers(self):
        source_model = _FakeModel()
        target_model = _FakeModel()
        selector = MagicMock()
        selector.select_top_k_sources.return_value = [
            {"source_key": (1, 2), "distance": 0.5, "weight": 1.0}
        ]

        with (
            patch.object(msml_tl, "SourceSelector", return_value=selector),
            patch.object(
                msml_tl,
                "train_source_cnn_for_msml",
                return_value={"model": source_model, "input_shape": (10, 1), "num_samples": 2},
            ),
            patch.object(msml_tl, "fuse_source_models_layerwise", return_value={}) as fuse_mock,
            patch.object(msml_tl, "build_base_cnn", return_value=target_model),
            patch.object(msml_tl, "load_fused_params_into_target_model", return_value=target_model),
            patch.object(msml_tl, "freeze_fused_layers", return_value=["conv1", "pool1", "conv2", "pool2"]) as freeze_mock,
            patch.object(
                msml_tl,
                "temporal_split_by_ratio_or_dates",
                return_value=(self._target_df(), self._target_df(), self._target_df()),
            ),
            patch.object(
                msml_tl,
                "normalize_features",
                return_value=(self._target_df(), self._target_df(), self._target_df(), None, ["feat"]),
            ),
            patch.object(msml_tl, "fine_tune_fused_target_model", return_value={"model": target_model}),
            patch.object(
                msml_tl,
                "evaluate_msml_model",
                return_value={"rmse": 0.0, "accuracy": 1.0, "prediction_shape": (1,)},
            ),
        ):
            result = msml_tl.run_msml_tl(
                source_df=self._source_df(),
                target_df=self._target_df(),
                feature_cols=["feat"],
                k=1,
            )

        self.assertEqual(["conv1", "conv2"], list(fuse_mock.call_args.args[2]))
        self.assertEqual(["conv1", "pool1", "conv2", "pool2"], list(freeze_mock.call_args.args[1]))
        self.assertEqual(["conv1", "conv2"], result["meta"]["fused_layers"])
        self.assertEqual(["conv1", "pool1", "conv2", "pool2"], result["frozen_layers"])

    def test_run_msml_tl_rfe_fuses_weighted_layers_but_freezes_first_four_layers(self):
        source_model = _FakeModel()
        target_model = _FakeModel()
        selector = MagicMock()
        selector.select_top_k_sources.return_value = [
            {"source_key": (1, 2), "distance": 0.5, "weight": 1.0}
        ]
        rfe_result = {
            "rfe_selected_features": ["feat"],
            "rfe_candidate_features": ["feat"],
            "final_selected_features": ["feat"],
            "num_original_features": 1,
            "num_rfe_selected_features": 1,
            "num_selected_features": 1,
            "target_col": "sales",
            "target_removed_from_rfe": True,
            "sales_added_back_as_history_input": False,
        }

        with (
            patch.object(msml_tl_rfe, "SourceSelector", return_value=selector),
            patch.object(
                msml_tl_rfe,
                "temporal_split_by_ratio_or_dates",
                return_value=(self._target_df(), self._target_df(), self._target_df()),
            ),
            patch.object(
                msml_tl_rfe,
                "_prepare_source_split",
                return_value=(self._source_df(), self._source_df(), self._source_df()),
            ),
            patch.object(msml_tl_rfe, "build_joint_rfe_training_dataframe", return_value=self._target_df()),
            patch.object(msml_tl_rfe, "run_rfe_feature_selection", return_value=rfe_result),
            patch.object(
                msml_tl_rfe,
                "train_source_cnn_for_msml_rfe",
                return_value={"model": source_model, "input_shape": (10, 1), "num_samples": 2},
            ),
            patch.object(msml_tl_rfe, "fuse_source_models_layerwise", return_value={}) as fuse_mock,
            patch.object(msml_tl_rfe, "build_base_cnn", return_value=target_model),
            patch.object(msml_tl_rfe, "load_fused_params_into_target_model", return_value=target_model),
            patch.object(msml_tl_rfe, "freeze_fused_layers", return_value=["conv1", "pool1", "conv2", "pool2"]) as freeze_mock,
            patch.object(
                msml_tl_rfe,
                "normalize_features",
                return_value=(self._target_df(), self._target_df(), self._target_df(), None, ["feat"]),
            ),
            patch.object(msml_tl_rfe, "fine_tune_fused_target_model_rfe", return_value={"model": target_model}),
            patch.object(
                msml_tl_rfe,
                "evaluate_msml_rfe_split",
                return_value={"rmse": 0.0, "accuracy": 1.0, "mae": 0.0},
            ),
            patch.object(
                msml_tl_rfe,
                "evaluate_msml_rfe_model",
                return_value={"rmse": 0.0, "accuracy": 1.0, "prediction_shape": (1,)},
            ),
        ):
            result = msml_tl_rfe.run_msml_tl_rfe(
                source_df=self._source_df(),
                target_df=self._target_df(),
                feature_cols=["feat"],
                k=1,
                source_selection_window="full_target_window",
            )

        self.assertEqual(["conv1", "conv2"], list(fuse_mock.call_args.args[2]))
        self.assertEqual(["conv1", "pool1", "conv2", "pool2"], list(freeze_mock.call_args.args[1]))
        self.assertEqual(["conv1", "conv2"], result["meta"]["fused_layers"])
        self.assertEqual(["conv1", "pool1", "conv2", "pool2"], result["frozen_layers"])

    def test_run_msml_tl_rfe_uses_full_target_df_for_source_selection_when_provided(self):
        source_model = _FakeModel()
        target_model = _FakeModel()
        selector = MagicMock()
        selector.select_top_k_sources.return_value = [
            {"source_key": (1, 2), "distance": 0.5, "weight": 1.0}
        ]
        split_target_df = self._target_df()
        full_target_df = self._target_df()
        full_target_df.attrs["paper_split_protocol"] = "solidified_d4_d6_target_train_window"
        rfe_result = {
            "rfe_selected_features": ["feat"],
            "rfe_candidate_features": ["feat"],
            "final_selected_features": ["feat"],
            "num_original_features": 1,
            "num_rfe_selected_features": 1,
            "num_selected_features": 1,
            "target_col": "sales",
            "target_removed_from_rfe": True,
            "sales_added_back_as_history_input": False,
        }

        with (
            patch.object(msml_tl_rfe, "SourceSelector", return_value=selector),
            patch.object(
                msml_tl_rfe,
                "temporal_split_by_ratio_or_dates",
                return_value=(split_target_df, split_target_df, split_target_df),
            ),
            patch.object(
                msml_tl_rfe,
                "_prepare_source_split",
                return_value=(self._source_df(), self._source_df(), self._source_df()),
            ),
            patch.object(msml_tl_rfe, "build_joint_rfe_training_dataframe", return_value=split_target_df),
            patch.object(msml_tl_rfe, "run_rfe_feature_selection", return_value=rfe_result),
            patch.object(
                msml_tl_rfe,
                "train_source_cnn_for_msml_rfe",
                return_value={"model": source_model, "input_shape": (10, 1), "num_samples": 2},
            ),
            patch.object(msml_tl_rfe, "fuse_source_models_layerwise", return_value={}),
            patch.object(msml_tl_rfe, "build_base_cnn", return_value=target_model),
            patch.object(msml_tl_rfe, "load_fused_params_into_target_model", return_value=target_model),
            patch.object(msml_tl_rfe, "freeze_fused_layers", return_value=["conv1", "pool1", "conv2", "pool2"]),
            patch.object(
                msml_tl_rfe,
                "normalize_features",
                return_value=(split_target_df, split_target_df, split_target_df, None, ["feat"]),
            ),
            patch.object(msml_tl_rfe, "fine_tune_fused_target_model_rfe", return_value={"model": target_model}),
            patch.object(
                msml_tl_rfe,
                "evaluate_msml_rfe_split",
                return_value={"rmse": 0.0, "accuracy": 1.0, "mae": 0.0},
            ),
            patch.object(
                msml_tl_rfe,
                "evaluate_msml_rfe_model",
                return_value={"rmse": 0.0, "accuracy": 1.0, "prediction_shape": (1,)},
            ),
        ):
            msml_tl_rfe.run_msml_tl_rfe(
                source_df=self._source_df(),
                target_df=split_target_df,
                full_target_df=full_target_df,
                feature_cols=["feat"],
                k=1,
            )

        self.assertIs(selector.select_top_k_sources.call_args.kwargs["target_df"], full_target_df)


if __name__ == "__main__":
    unittest.main()
