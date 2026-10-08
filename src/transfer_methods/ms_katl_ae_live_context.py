"""Live A–E cell contexts bound to frozen V3 D1–D6 frames. No training."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.transfer_methods.ms_katl_ae_ablation import (
    EXPECTED_CANONICAL_CELLS,
    STAGE_A_LEARNING_RATE,
    STAGE_B_BATCH_SIZE,
    STAGE_B_EPOCHS,
    AEContractError,
    project_root_from,
)
from src.transfer_methods.ms_katl_ae_execution import RealCanonicalCellPlan
from src.transfer_methods.ms_katl_lite import DEFAULT_TAU_K, prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_v3_frame_assembly import (
    REQUESTED_K,
    WINDOW_SIZE,
    assemble_v3_cell_frames,
)

SOURCE_POOL_AUTHORITY = "shared_protocol_selector"
FUSED_TARGET_SLOT = "RESERVED_FOR_STAGE_A"
CONTEXT_READY = "CONTEXT_READY"


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _frame_digest(frame: pd.DataFrame) -> str:
    dates = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame.columns else pd.Series(dtype="datetime64[ns]")
    sample = frame if len(frame) <= 128 else pd.concat([frame.head(64), frame.tail(64)], ignore_index=True)
    payload = {
        "shape": list(frame.shape),
        "columns": list(map(str, frame.columns)),
        "date_min": str(dates.min()) if not dates.empty else "",
        "date_max": str(dates.max()) if not dates.empty else "",
        "sample_hash": hashlib.sha256(
            pd.util.hash_pandas_object(sample, index=True).to_numpy().tobytes()
        ).hexdigest(),
    }
    return _json_digest(payload)


@dataclass
class SharedTargetPrepBundle:
    target_bundle_key: str
    stage_a_key: str
    target_identity: str
    feature_cols: Tuple[str, ...]
    target_scaler_digest: str
    target_split_digest: Dict[str, str]
    n_train: int
    n_val: int
    n_test: int
    train_dates: Tuple[str, ...]
    val_dates: Tuple[str, ...]
    test_dates: Tuple[str, ...]
    native_init_basis: Dict[str, Any]
    fused_target_slot: str
    metric_truth_context: Dict[str, Any]
    target_test_excluded: str
    source_future_excluded: str
    transformation_reuse_context: Any = None

    def digest(self) -> str:
        return _json_digest(
            {
                "target_bundle_key": self.target_bundle_key,
                "stage_a_key": self.stage_a_key,
                "feature_cols": list(self.feature_cols),
                "target_scaler_digest": self.target_scaler_digest,
                "target_split_digest": self.target_split_digest,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
                "fused_target_slot": self.fused_target_slot,
            }
        )


@dataclass
class LiveAECellContext:
    dataset: str
    target_identity: str
    scenario: str
    horizon: int
    seed: int
    selected_lr: float
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    group_cols: Tuple[str, ...]
    feature_cols: Tuple[str, ...]
    stage_a_kwargs: Dict[str, Any]
    shared_target_bundle: SharedTargetPrepBundle
    shared_target_bundle_inputs: Dict[str, Any]
    sealed_source_identity: str
    sealed_target_identity: str
    source_data_digest: str
    target_data_digest: str
    target_split_digest: Dict[str, str]
    stage_a_key: str
    target_bundle_key: str
    frame_authority: str
    source_pool_authority: str = SOURCE_POOL_AUTHORITY
    status: str = CONTEXT_READY
    rmse_original: Optional[float] = None
    mae_original: Optional[float] = None
    smape_original: Optional[float] = None

    def as_formal_context(self) -> Dict[str, Any]:
        return {
            "live": self,
            "stage_a_kwargs": dict(self.stage_a_kwargs),
            "bundle": self.shared_target_bundle,
            "cell": self.stage_a_key,
        }


def assert_stage_a_key_matches_bundle(stage_a_key: str, target_bundle_key: str) -> None:
    if stage_a_key != target_bundle_key:
        raise AEContractError(
            f"stage_a_key/target_bundle_key mismatch: {stage_a_key} != {target_bundle_key}"
        )


def _date_strings(frame: pd.DataFrame) -> Tuple[str, ...]:
    if "date" not in frame.columns or frame.empty:
        return tuple()
    return tuple(sorted({str(pd.Timestamp(item).normalize().date()) for item in frame["date"].tolist()}))


def prepare_shared_target_prep_bundle(
    *,
    cell: RealCanonicalCellPlan,
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_cols: Sequence[str],
    transformation_reuse_context: Any,
) -> SharedTargetPrepBundle:
    """Non-training target split / scaler / truth context. Does not call Stage A."""
    assert_stage_a_key_matches_bundle(cell.stage_a_key, cell.stage_a_key)
    raw_train_df, raw_val_df, raw_test_df = temporal_split_by_ratio_or_dates(target_df)
    target_train_df, target_val_df, target_test_df, target_scaler, target_feature_columns = normalize_features(
        raw_train_df,
        raw_val_df,
        raw_test_df,
        feature_columns=list(feature_cols),
        reuse_context=transformation_reuse_context,
    )
    x_train, y_train = build_tabular_sequence(
        target_train_df,
        horizon=int(cell.horizon),
        window_size=WINDOW_SIZE,
        feature_columns=list(feature_cols),
        reuse_context=transformation_reuse_context,
    )
    x_val, y_val = build_tabular_sequence(
        target_val_df,
        horizon=int(cell.horizon),
        window_size=WINDOW_SIZE,
        feature_columns=list(feature_cols),
        reuse_context=transformation_reuse_context,
    )
    x_test, y_test = build_tabular_sequence(
        target_test_df,
        horizon=int(cell.horizon),
        window_size=WINDOW_SIZE,
        feature_columns=list(feature_cols),
        reuse_context=transformation_reuse_context,
    )
    if len(y_train) == 0 or len(y_test) == 0:
        raise AEContractError(f"target windows empty for {cell.stage_a_key}")
    x_train_t = to_cnn_tensor(x_train)
    train_dates = set(_date_strings(raw_train_df))
    val_dates = set(_date_strings(raw_val_df))
    test_dates = set(_date_strings(raw_test_df))
    if train_dates & test_dates:
        raise AEContractError(f"target test leakage into train for {cell.stage_a_key}")
    target_test_excluded = "YES"
    source_dates = _date_strings(source_df)
    source_future_excluded = "YES"
    if source_dates and test_dates and max(source_dates) >= min(test_dates):
        # V3 still fail-closed later inside Stage A; record the observed guard path.
        source_future_excluded = "GUARD_DEFERRED_TO_STAGE_A"
    scaler_digest = hashlib.sha256(
        np_bytes(getattr(target_scaler, "data_min_", None))
        + np_bytes(getattr(target_scaler, "data_max_", None))
    ).hexdigest()
    split_digest = {
        "function": "temporal_split_by_ratio_or_dates",
        "authority": "FROZEN_V3_DATA_PROTOCOL",
        "n_train": str(len(y_train)),
        "n_val": str(len(y_val)),
        "n_test": str(len(y_test)),
        "target_scaler": scaler_digest,
        "train_dates": ",".join(sorted(train_dates)[:3]),
        "test_dates": ",".join(sorted(test_dates)[:3]),
    }
    return SharedTargetPrepBundle(
        target_bundle_key=cell.stage_a_key,
        stage_a_key=cell.stage_a_key,
        target_identity=cell.target_identity,
        feature_cols=tuple(str(col) for col in target_feature_columns),
        target_scaler_digest=scaler_digest,
        target_split_digest=split_digest,
        n_train=int(len(y_train)),
        n_val=int(len(y_val)),
        n_test=int(len(y_test)),
        train_dates=tuple(sorted(train_dates)),
        val_dates=tuple(sorted(val_dates)),
        test_dates=tuple(sorted(test_dates)),
        native_init_basis={
            "input_shape": tuple(int(dim) for dim in x_train_t.shape[1:]),
            "seed": int(cell.seed),
            "status": "SHAPE_ONLY_NO_WEIGHTS",
        },
        fused_target_slot=FUSED_TARGET_SLOT,
        metric_truth_context={
            "y_test_count": int(len(y_test)),
            "rmse_original": None,
            "mae_original": None,
            "smape_original": None,
        },
        target_test_excluded=target_test_excluded,
        source_future_excluded=source_future_excluded,
        transformation_reuse_context=transformation_reuse_context,
    )


def np_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    import numpy as np

    return np.ascontiguousarray(value).tobytes()


def build_stage_a_kwargs(cell: RealCanonicalCellPlan, frames: Mapping[str, Any]) -> Dict[str, Any]:
    kwargs = {
        "source_df": frames["source_model_df"],
        "target_df": frames["target_model_df"],
        "feature_cols": list(frames["model_feature_cols"]),
        "k": REQUESTED_K,
        "horizon": int(cell.horizon),
        "window_size": WINDOW_SIZE,
        "weight_mode": "inverse_distance",
        "learning_rate": float(STAGE_A_LEARNING_RATE),
        "source_epochs": int(STAGE_B_EPOCHS),
        "batch_size": int(STAGE_B_BATCH_SIZE),
        "tau_k": float(DEFAULT_TAU_K),
        "group_cols": tuple(frames["group_cols"]),
        "k3_selection_context": frames["k3_selection_context"],
        "k3_raw_source_context": frames["k3_raw_source_context"],
        "transformation_reuse_context": frames["transformation_reuse_context"],
        "seed": int(cell.seed),
    }
    validate_stage_a_kwargs_signature(kwargs)
    validate_stage_a_kwargs_objects(kwargs, cell)
    return kwargs


def validate_stage_a_kwargs_signature(kwargs: Mapping[str, Any]) -> None:
    signature = inspect.signature(prepare_ms_katl_stage_a)
    accepted = set(signature.parameters)
    unexpected = [key for key in kwargs if key not in accepted]
    if unexpected:
        raise AEContractError(f"unexpected prepare_ms_katl_stage_a args: {unexpected}")
    missing = []
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty and name not in kwargs:
            missing.append(name)
    if missing:
        raise AEContractError(f"missing required prepare_ms_katl_stage_a args: {missing}")


def validate_stage_a_kwargs_objects(kwargs: Mapping[str, Any], cell: RealCanonicalCellPlan) -> None:
    source_df = kwargs["source_df"]
    target_df = kwargs["target_df"]
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        raise AEContractError(f"source_df empty for {cell.stage_a_key}")
    if not isinstance(target_df, pd.DataFrame) or target_df.empty:
        raise AEContractError(f"target_df empty for {cell.stage_a_key}")
    feature_cols = list(kwargs["feature_cols"])
    missing_cols = [col for col in feature_cols if col not in source_df.columns or col not in target_df.columns]
    if missing_cols:
        raise AEContractError(f"feature cols missing for {cell.stage_a_key}: {missing_cols}")
    group_cols = tuple(kwargs["group_cols"])
    if not group_cols:
        raise AEContractError(f"group_cols missing for {cell.stage_a_key}")
    if int(kwargs["k"]) != REQUESTED_K:
        raise AEContractError(f"K contract drifted for {cell.stage_a_key}")
    if kwargs["k3_selection_context"] is None:
        raise AEContractError(f"source selection authority missing for {cell.stage_a_key}")
    if kwargs.get("seed") is None:
        raise AEContractError(f"stage_a seed missing for {cell.stage_a_key}")
    if int(kwargs["seed"]) != int(cell.seed):
        raise AEContractError(
            f"stage_a seed {kwargs['seed']} != cell.seed {cell.seed} for {cell.stage_a_key}"
        )
    scenario = str(target_df.attrs.get("information_sharing_scenario") or "")
    if "without" not in scenario.lower() and str(cell.scenario) != "without":
        raise AEContractError(f"scenario drifted for {cell.stage_a_key}: {scenario}")


def build_live_ae_cell_context(
    cell: RealCanonicalCellPlan,
    *,
    project_root: Optional[Path] = None,
) -> LiveAECellContext:
    root = project_root_from(project_root)
    frames = assemble_v3_cell_frames(
        cell.dataset,
        cell.target_identity,
        horizon=int(cell.horizon),
        seed=int(cell.seed),
        project_root=root,
    )
    kwargs = build_stage_a_kwargs(cell, frames)
    bundle = prepare_shared_target_prep_bundle(
        cell=cell,
        target_df=frames["target_model_df"],
        source_df=frames["source_model_df"],
        feature_cols=frames["model_feature_cols"],
        transformation_reuse_context=frames["transformation_reuse_context"],
    )
    assert_stage_a_key_matches_bundle(cell.stage_a_key, bundle.target_bundle_key)
    return LiveAECellContext(
        dataset=cell.dataset,
        target_identity=cell.target_identity,
        scenario=cell.scenario,
        horizon=int(cell.horizon),
        seed=int(cell.seed),
        selected_lr=float(cell.selected_lr),
        source_df=frames["source_model_df"],
        target_df=frames["target_model_df"],
        group_cols=tuple(frames["group_cols"]),
        feature_cols=tuple(str(col) for col in frames["model_feature_cols"]),
        stage_a_kwargs=kwargs,
        shared_target_bundle=bundle,
        shared_target_bundle_inputs={
            "target_identity": cell.target_identity,
            "entity_key": cell.entity_key,
            "target_split_identity": cell.target_split_identity,
            "knn_diagnostic_sources": tuple(cell.knn_diagnostic_sources),
            "source_pool_authority": SOURCE_POOL_AUTHORITY,
            "source_selection_contract": cell.source_selection_contract,
        },
        sealed_source_identity=str(frames["sealed_source_path"]),
        sealed_target_identity=str(frames["sealed_target_path"]),
        source_data_digest=_frame_digest(frames["source_model_df"]),
        target_data_digest=_frame_digest(frames["target_model_df"]),
        target_split_digest=dict(bundle.target_split_digest),
        stage_a_key=cell.stage_a_key,
        target_bundle_key=bundle.target_bundle_key,
        frame_authority=str(frames["frame_authority"]),
        source_pool_authority=SOURCE_POOL_AUTHORITY,
        status=CONTEXT_READY,
        rmse_original=None,
        mae_original=None,
        smape_original=None,
    )


def construct_all_live_ae_contexts(
    cells: Sequence[RealCanonicalCellPlan],
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    contexts: List[LiveAECellContext] = []
    failures: List[Dict[str, str]] = []
    dataset_status: Dict[str, str] = {f"D{idx}": "FAIL" for idx in range(1, 7)}
    for cell in cells:
        try:
            contexts.append(build_live_ae_cell_context(cell, project_root=project_root))
        except Exception as exc:
            failures.append({"stage_a_key": cell.stage_a_key, "reason": f"{type(exc).__name__}: {exc}"})
    by_dataset = {f"D{idx}": 0 for idx in range(1, 7)}
    for ctx in contexts:
        by_dataset[ctx.dataset] = by_dataset.get(ctx.dataset, 0) + 1
    for dataset, count in by_dataset.items():
        expected = 9
        dataset_status[dataset] = "PASS" if count == expected else "FAIL"
    compatible = 0
    for ctx in contexts:
        try:
            validate_stage_a_kwargs_signature(ctx.stage_a_kwargs)
            compatible += 1
        except AEContractError:
            pass
    ok = (
        not failures
        and len(contexts) == EXPECTED_CANONICAL_CELLS
        and compatible == EXPECTED_CANONICAL_CELLS
        and all(status == "PASS" for status in dataset_status.values())
    )
    matrix = [live_context_matrix_row(ctx) for ctx in contexts]
    return {
        "LIVE_CONTEXT_CONSTRUCTION": f"{len(contexts)}/{len(cells)} PASS" if ok else "FAIL",
        "REAL_STAGE_A_KWARGS_OBJECTS": f"{len(contexts)}/{len(cells)} PASS" if ok else "FAIL",
        "STAGE_A_SIGNATURE_COMPATIBILITY": f"{compatible}/{len(cells)} PASS" if compatible == EXPECTED_CANONICAL_CELLS else "FAIL",
        "REAL_SHARED_TARGET_BUNDLES": f"{len(contexts)}/{len(cells)} PASS" if ok else "FAIL",
        "dataset_status": dataset_status,
        "failures": failures,
        "contexts": contexts,
        "matrix": matrix,
        "ok": ok,
    }


def live_context_matrix_row(ctx: LiveAECellContext) -> Dict[str, Any]:
    return {
        "dataset": ctx.dataset,
        "target": ctx.target_identity,
        "horizon": ctx.horizon,
        "seed": ctx.seed,
        "scenario": ctx.scenario,
        "source_data_digest": ctx.source_data_digest,
        "target_data_digest": ctx.target_data_digest,
        "stage_a_kwargs_digest": _json_digest(
            {
                "k": ctx.stage_a_kwargs["k"],
                "horizon": ctx.stage_a_kwargs["horizon"],
                "window_size": ctx.stage_a_kwargs["window_size"],
                "learning_rate": ctx.stage_a_kwargs["learning_rate"],
                "source_epochs": ctx.stage_a_kwargs["source_epochs"],
                "batch_size": ctx.stage_a_kwargs["batch_size"],
                "group_cols": list(ctx.stage_a_kwargs["group_cols"]),
                "feature_cols": list(ctx.feature_cols),
                "seed": int(ctx.stage_a_kwargs["seed"]),
                "source_digest": ctx.source_data_digest,
                "target_digest": ctx.target_data_digest,
            }
        ),
        "target_bundle_digest": ctx.shared_target_bundle.digest(),
        "selected_lr": ctx.selected_lr,
        "status": CONTEXT_READY,
        "rmse_original": None,
        "mae_original": None,
        "smape_original": None,
    }


def representative_trace(ctx: LiveAECellContext) -> Dict[str, Any]:
    source_dates = _date_strings(ctx.source_df)
    target_dates = _date_strings(ctx.target_df)
    bundle = ctx.shared_target_bundle
    return {
        "dataset": ctx.dataset,
        "target_identity": ctx.target_identity,
        "horizon": ctx.horizon,
        "seed": ctx.seed,
        "scenario": ctx.scenario,
        "sealed_source_path": ctx.sealed_source_identity,
        "sealed_target_path": ctx.sealed_target_identity,
        "source_df_shape": list(ctx.source_df.shape),
        "target_df_shape": list(ctx.target_df.shape),
        "group_cols": list(ctx.group_cols),
        "feature_cols": list(ctx.feature_cols),
        "selected_lr": ctx.selected_lr,
        "source_date_min": source_dates[0] if source_dates else "",
        "source_date_max": source_dates[-1] if source_dates else "",
        "target_date_min": target_dates[0] if target_dates else "",
        "target_date_max": target_dates[-1] if target_dates else "",
        "train_dates": bundle.train_dates[:3],
        "val_dates": bundle.val_dates[:3],
        "test_dates": bundle.test_dates[:3],
        "stage_a_kwargs_keys": sorted(ctx.stage_a_kwargs),
        "shared_target_bundle_identity": bundle.target_bundle_key,
        "stage_a_key": ctx.stage_a_key,
        "target_bundle_key": ctx.target_bundle_key,
        "target_test_excluded": bundle.target_test_excluded,
        "source_future_excluded": bundle.source_future_excluded,
        "source_pool_authority": ctx.source_pool_authority,
        "frame_authority": ctx.frame_authority,
        "status": ctx.status,
        "STOP": "BEFORE prepare_ms_katl_stage_a",
    }
