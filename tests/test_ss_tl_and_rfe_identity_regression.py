from __future__ import annotations

from typing import Sequence
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from src.experiment.experiment_runner import run_ss_tl_experiment
from src.transfer_methods import msml_tl_rfe


class _PredictingModel:
    def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.zeros((len(values), 1), dtype=float)


def _frame(
    group_cols: Sequence[str],
    store: str,
    product: str,
    *,
    periods: int = 48,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            group_cols[0]: store,
            group_cols[1]: product,
            "entity_id": store,
            "item_id": product,
            "date": pd.date_range("2021-01-01", periods=periods, freq="D"),
            "sales": np.arange(periods, dtype=float),
            "planned_price": np.arange(periods, dtype=float) + 100.0,
        }
    )


class SourceIdentityRegressionTest(TestCase):
    def test_ss_tl_result_meta_uses_normalized_selector_source_key(self) -> None:
        import src.transfer_methods.single_source_tl as single_source_tl

        group_cols = ("store_id", "product_id")
        source = pd.concat(
            [
                _frame(group_cols, "S1", "P1", periods=80),
                _frame(group_cols, "S2", "P2", periods=80),
            ],
            ignore_index=True,
        )
        target = _frame(group_cols, "T1", "P0", periods=80)
        selected_key = ("S1", "P1")

        with patch.object(
            msml_tl_rfe.SourceSelector,
            "select_top_k_sources",
            return_value={
                "sources": [{"source_key": list(selected_key), "distance": 0.25, "weight": 1.0}],
                "meta": {},
            },
        ), patch.object(single_source_tl, "train_source_model", return_value=object()), patch.object(
            single_source_tl,
            "build_target_model_from_source",
            return_value=(_PredictingModel(), ["conv1"]),
        ), patch.object(
            single_source_tl,
            "fine_tune_target_model",
            side_effect=lambda target_model, **_kwargs: target_model,
        ), patch.object(
            single_source_tl,
            "evaluate_regression_model",
            return_value={"rmse": 1.0, "accuracy": 0.5},
        ):
            result = run_ss_tl_experiment(
                source_df=source,
                target_df=target,
                feature_cols=("sales", "planned_price"),
                window_size=5,
                source_epochs=1,
                target_epochs=1,
                batch_size=4,
                group_cols=group_cols,
            )

        self.assertEqual(result["meta"]["source_key"], selected_key)

    def test_rfe_preserves_selected_source_identity_columns_for_cnn_provenance(self) -> None:
        for group_cols in (
            ("store_id", "product_id"),
            ("store_nbr", "item_nbr"),
            ("store_id", "item_id"),
        ):
            with self.subTest(group_cols=group_cols):
                self._assert_rfe_source_identity(group_cols)

    def _assert_rfe_source_identity(self, group_cols: tuple[str, str]) -> None:
        source_key = ("S1", "P1")
        source = _frame(group_cols, *source_key)
        target = _frame(group_cols, "T1", "P0")
        captured: dict[str, object] = {}

        def fake_train_source_cnn_for_msml_rfe(
            source_sequence_df: pd.DataFrame,
            feature_cols: Sequence[str],
            **_kwargs: object,
        ) -> dict[str, object]:
            captured["source_frame"] = source_sequence_df
            captured["feature_cols"] = tuple(feature_cols)
            return {"model": object(), "input_shape": (5, 2, 1), "num_samples": 12}

        with patch.object(
            msml_tl_rfe.SourceSelector,
            "select_top_k_sources",
            return_value={
                "sources": [{"source_key": source_key, "distance": 0.25, "weight": 1.0}],
                "meta": {},
            },
        ), patch.object(
            msml_tl_rfe,
            "run_rfe_feature_selection",
            return_value={
                "selected_feature_cols": ["sales", "planned_price"],
                "num_selected_features": 2,
                "num_original_features": 2,
                "keep_ratio": 1.0,
            },
        ), patch.object(
            msml_tl_rfe, "train_source_cnn_for_msml_rfe", side_effect=fake_train_source_cnn_for_msml_rfe
        ), patch.object(
            msml_tl_rfe,
            "summarize_model_weights",
            return_value={"model_weight_nan_count": 0, "model_weight_inf_count": 0},
        ), patch.object(msml_tl_rfe, "get_transferable_layer_names", return_value=["conv1"]), patch.object(
            msml_tl_rfe, "fuse_source_models_layerwise", return_value={"conv1": []}
        ), patch.object(msml_tl_rfe, "build_base_cnn", return_value=object()), patch.object(
            msml_tl_rfe, "load_fused_params_into_target_model", side_effect=lambda model, _params: model
        ), patch.object(msml_tl_rfe, "freeze_fused_layers", side_effect=lambda _model, layers: list(layers)), patch.object(
            msml_tl_rfe, "fine_tune_fused_target_model_rfe", side_effect=lambda target_model, **_kwargs: {"model": target_model}
        ), patch.object(
            msml_tl_rfe,
            "evaluate_msml_rfe_model",
            return_value={
                "rmse": 1.0,
                "accuracy": 0.5,
                "smape": 0.5,
                "y_true": np.array([0.0]),
                "y_pred": np.array([0.0]),
                "prediction_shape": (1,),
            },
        ):
            result = msml_tl_rfe.run_msml_tl_rfe(
                source_df=source,
                target_df=target,
                feature_cols=("sales", "planned_price"),
                k=1,
                window_size=5,
                source_epochs=1,
                target_epochs=1,
                batch_size=4,
                group_cols=group_cols,
            )

        source_frame = captured["source_frame"]
        self.assertIsInstance(source_frame, pd.DataFrame)
        self.assertTrue(set(group_cols).issubset(source_frame.columns))
        self.assertIn("date", source_frame.columns)
        self.assertFalse(set(group_cols).intersection(result["rfe_info"]["selected_feature_cols"]))
        self.assertFalse(set(group_cols).intersection(captured["feature_cols"]))
        self.assertTrue(source_frame.attrs["protocol_actual_cnn_audit"][source_key]["bound"])
