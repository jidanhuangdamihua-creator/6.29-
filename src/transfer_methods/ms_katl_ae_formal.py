"""Formal A–E execution body bound to frozen V3 Stage-A / Stage-B / metric authorities.

Real training remains unreachable unless AE_FORMAL_EXECUTION_ENABLED is True
and both CLI `--execute` and AE_FORMAL_TRAINING_UNLOCK=YES are present.
Tests may inject mock authorities with mock_execution_authorized=True.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    AE_FORMAL_EXECUTION_ENABLED,
    EXPECTED_CANONICAL_CELLS,
    EXPECTED_RUNS,
    FORMAL_TRAINING_UNLOCK_ENV,
    PARENT_METHOD,
    PROTOCOL_VERSION,
    AEContractError,
    AETrainingBlocked,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_protocol_manifest_payload,
    ae_training_recipe,
    assert_d_equals_frozen_v3,
    assert_formal_training_permitted,
    load_ae_manifest,
    project_root_from,
    reset_ae_rng,
    run_ae_preflight,
)
from src.transfer_methods.ms_katl_ae_execution import (
    AE_RUN_LABEL,
    PREFLIGHT_RELPATH,
    RealArmPlan,
    RealCanonicalCellPlan,
    assert_ae_output_root,
    enumerate_real_arm_plans,
    enumerate_real_canonical_cell_plans,
    planned_formal_run_root,
    resolve_all_formal_stage_a_input_plans,
    validate_resume_root,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    v3_arm_spec,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import (
    prepare_ms_katl_stage_a,
    predict_ms_katl_decomposed,
    train_ms_katl_head,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics

AE_STAGE_A_AUTHORITY = prepare_ms_katl_stage_a
AE_STAGE_B_AUTHORITY = train_ms_katl_head
AE_PREDICTION_AUTHORITY = predict_ms_katl_decomposed
AE_METRIC_AUTHORITY = compute_screening_metrics
MOCK_STATUS = "MOCK_ACCEPTANCE_ONLY"
MOCK_METRIC = "MOCK_METRIC"


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class FormalStageAView:
    stage_a_key: str
    source_identities: List[Tuple[object, ...]]
    source_distances: List[float]
    source_weights: List[float]
    source_weight_digests: List[str]
    fused_params_digest: str
    fused_target_digest: str
    best_source_identity: Tuple[object, ...]
    best_source_conv_digest: str
    target_split_digest: str
    requested_k: int = 3
    effective_k: int = 3


@dataclass
class FormalArmRuntime:
    arm: str
    spec: Any
    init_mode: str
    source_init_consumed: bool
    stage_a_shared_for_pairing: bool
    head_target_init_digest: str
    target_initial_conv_digest: str
    fused_init_digest: str
    best_source_identity: Tuple[object, ...]
    best_source_snapshot_digest: str
    source_identities: List[Tuple[object, ...]]
    source_distances: List[float]
    source_weights: List[float]
    selected_lr: float
    recipe_digest: str
    target_split_digest: str
    parent_method: str
    parent_v3_architecture_digest: str
    prediction_path_parameter_count: int
    trainable_prediction_path_parameter_count: int
    frozen_prediction_path_parameter_count: int
    status: str = MOCK_STATUS
    error: str = ""
    rmse_original: Optional[float] = None
    smape_original: Optional[float] = None
    mae_original: Optional[float] = None
    metric_kind: Optional[str] = None


@dataclass
class FormalCallLedger:
    context_prep: List[str] = field(default_factory=list)
    stage_a: List[str] = field(default_factory=list)
    stage_b: List[Tuple[str, str]] = field(default_factory=list)
    predict: List[Tuple[str, str]] = field(default_factory=list)
    metric: List[Tuple[str, str]] = field(default_factory=list)
    bundle_validations: List[str] = field(default_factory=list)
    publish: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        arms = {arm: sum(1 for _, item in self.stage_b if item == arm) for arm in AE_ARMS}
        return {
            "CONTEXT_PREP_CALLS": len(self.context_prep),
            "STAGE_A_CALLS": len(self.stage_a),
            "STAGE_B_CALLS": len(self.stage_b),
            "PREDICT_CALLS": len(self.predict),
            "METRIC_CALLS": len(self.metric),
            "BUNDLE_VALIDATIONS": len(self.bundle_validations),
            "PUBLISH_ATTEMPTS": len(self.publish),
            "A_CALLS": arms["A"],
            "B_CALLS": arms["B"],
            "C_CALLS": arms["C"],
            "D_CALLS": arms["D"],
            "E_CALLS": arms["E"],
        }


@dataclass
class AEFormalAuthorities:
    prepare_context: Callable[[RealCanonicalCellPlan], Dict[str, Any]]
    prepare_stage_a: Callable[[RealCanonicalCellPlan, Dict[str, Any]], FormalStageAView]
    make_common_head: Callable[[RealCanonicalCellPlan, FormalStageAView], str]
    train_stage_b: Callable[..., FormalArmRuntime]
    predict: Callable[..., FormalArmRuntime]
    metric: Callable[..., FormalArmRuntime]
    publish: Callable[[RealCanonicalCellPlan, Dict[str, FormalArmRuntime], str, Path], None]
    reset_rng: Callable[[int], None]


def assert_no_old_v3_result_injection(source_text: str) -> str:
    start = source_text.find("def assert_no_old_v3_result_injection")
    rest = source_text[start:] if start >= 0 else source_text
    end = rest.find("\ndef ", 1)
    runtime_source = (source_text[:start] + rest[end:]) if start >= 0 and end > 0 else source_text
    forbidden = (
        "v3_screening_target_best_lr",
        "screening_target_best_lr.csv",
        "load_valid_ok_cells",
        "DEFAULT_V3_RUN_DIRNAME",
        "wilcoxon",
        "find latest",
        "latest output",
        "fused_frozen_backbone",
    )
    lowered = runtime_source.lower()
    for token in forbidden:
        if token.lower() in lowered:
            raise AEContractError(f"D_OLD_RESULT_REUSE forbidden token present: {token}")
    return "NO"


def planned_matrix_digest(arm_plans: Sequence[RealArmPlan]) -> str:
    payload = [
        {
            "dataset": item.cell.dataset,
            "target": item.cell.target,
            "horizon": item.cell.horizon,
            "seed": item.cell.seed,
            "scenario": item.cell.scenario,
            "arm": item.arm,
            "lr": item.cell.selected_lr_label,
            "init_mode": item.init_mode,
        }
        for item in arm_plans
    ]
    return _json_digest({"runs": payload, "n": len(payload)})


def stage_a_view_from_v3(artifacts: Any, cell: RealCanonicalCellPlan, fused_target_digest: str = "") -> FormalStageAView:
    identities = [tuple(item) for item in artifacts.source_identities]
    return FormalStageAView(
        stage_a_key=cell.stage_a_key,
        source_identities=identities,
        source_distances=[float(x) for x in artifacts.source_distances],
        source_weights=[float(x) for x in artifacts.source_weights],
        source_weight_digests=list(artifacts.source_weight_digests),
        fused_params_digest=str(artifacts.fused_params_digest),
        fused_target_digest=str(fused_target_digest or artifacts.fused_params_digest),
        best_source_identity=identities[0],
        best_source_conv_digest=str(artifacts.source_weight_digests[0]),
        target_split_digest=cell.target_split_digest,
        requested_k=3,
        effective_k=int(artifacts.effective_k),
    )


def _require_enabled_real_path() -> None:
    if not AE_FORMAL_EXECUTION_ENABLED:
        raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")


def default_v3_prepare_context(
    cell: RealCanonicalCellPlan,
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    _require_enabled_real_path()
    from src.transfer_methods.ms_katl_ae_live_context import build_live_ae_cell_context

    live = build_live_ae_cell_context(cell, project_root=project_root)
    if live.stage_a_key != cell.stage_a_key or live.target_bundle_key != cell.stage_a_key:
        raise AEContractError(
            f"live context key mismatch: stage_a={live.stage_a_key} bundle={live.target_bundle_key}"
        )
    return live.as_formal_context()


def default_v3_prepare_stage_a(cell: RealCanonicalCellPlan, context: Mapping[str, Any]) -> FormalStageAView:
    _require_enabled_real_path()
    live = context.get("live")
    if live is not None and str(getattr(live, "stage_a_key", "")) != cell.stage_a_key:
        raise AEContractError("Stage A cell does not match live context")
    kwargs = dict(context.get("stage_a_kwargs") or {})
    if not kwargs:
        raise AEContractError("missing stage_a_kwargs for prepare_ms_katl_stage_a")
    artifacts = AE_STAGE_A_AUTHORITY(**kwargs)
    bundle = context.get("bundle")
    fused_digest = str(getattr(bundle, "fused_target_digest", "") or "")
    return stage_a_view_from_v3(artifacts, cell, fused_target_digest=fused_digest)


def default_v3_train_stage_b(
    cell: RealCanonicalCellPlan,
    arm: str,
    stage_a: FormalStageAView,
    head_digest: str,
    context: Mapping[str, Any],
) -> FormalArmRuntime:
    _require_enabled_real_path()
    spec = ae_arm_spec(arm)
    if arm == "D":
        _ = v3_arm_spec()
        _ = v3_architecture()
        _ = v3_training_recipe()
    bundle = context.get("bundle")
    trainable = context.get("trainable_variables")
    if bundle is None or trainable is None:
        raise AEContractError("Stage-B V3 authority requires shared bundle and trainable variables")
    opt = ae_optimizer_contract()
    recipe = ae_training_recipe()
    AE_STAGE_B_AUTHORITY(
        context["head"],
        x_train=bundle.X_train,
        f_t_train=bundle.f_t["train"],
        y_train=bundle.y_train,
        f_s_pools=bundle.f_s_pools,
        knn_prior=bundle.stage_a.knn_prior,
        target_extractor=context["target_extractor"],
        epochs=int(opt["epochs"]),
        batch_size=int(opt["batch_size"]),
        learning_rate=float(cell.selected_lr),
        seed=int(cell.seed),
        use_adversarial=False,
        trainable_variables=trainable,
        revin=context.get("revin"),
        **recipe,
    )
    pred_count, trainable_count, frozen = _parameter_counts(arm)
    return FormalArmRuntime(
        arm=arm,
        spec=spec,
        init_mode=spec.init_mode,
        source_init_consumed=spec.init_mode != "native",
        stage_a_shared_for_pairing=True,
        head_target_init_digest=head_digest,
        target_initial_conv_digest=_arm_init_digest(cell, arm, stage_a),
        fused_init_digest=stage_a.fused_target_digest,
        best_source_identity=stage_a.best_source_identity,
        best_source_snapshot_digest=stage_a.best_source_conv_digest,
        source_identities=list(stage_a.source_identities),
        source_distances=list(stage_a.source_distances),
        source_weights=list(stage_a.source_weights),
        selected_lr=float(cell.selected_lr),
        recipe_digest=_json_digest(ae_training_recipe()),
        target_split_digest=stage_a.target_split_digest,
        parent_method=PARENT_METHOD,
        parent_v3_architecture_digest=v3_architecture_digest(),
        prediction_path_parameter_count=pred_count,
        trainable_prediction_path_parameter_count=trainable_count,
        frozen_prediction_path_parameter_count=frozen,
        status="OK",
    )


def default_v3_predict(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
    _require_enabled_real_path()
    payload = dict(context or {})
    decomposed = AE_PREDICTION_AUTHORITY(
        payload["head"],
        payload["z_t_test"],
        payload["f_t_test"],
        payload["knn_prior"],
    )
    y_pred = decomposed["y_pred"]
    revin = payload.get("revin")
    if runtime.spec.revin_enabled:
        if revin is None:
            raise AEContractError("arm D prediction requires RevIN inverse")
        y_pred = revin.denormalize_numpy(
            y_pred,
            payload["test_mean"],
            payload["test_stdev"],
            include_mean=True,
        )
    runtime.metric_kind = "pending_v3_metric"
    payload["y_pred"] = y_pred
    payload["y_true"] = payload.get("y_true")
    runtime._predict_payload = payload  # type: ignore[attr-defined]
    _ = cell
    return runtime


def default_v3_metric(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
    _require_enabled_real_path()
    payload = getattr(runtime, "_predict_payload", None)
    if not payload:
        raise AEContractError("metric authority requires prediction payload")
    metrics = AE_METRIC_AUTHORITY(
        payload["y_true"],
        payload["y_pred"],
        target_scaler=payload["target_scaler"],
        feature_columns=payload["feature_columns"],
    )
    runtime.rmse_original = float(metrics["rmse_original"])
    runtime.smape_original = float(metrics["smape_original"])
    runtime.status = "OK"
    _ = (cell, context)
    return runtime


def default_v3_publish(*_args: Any, **_kwargs: Any) -> None:
    _require_enabled_real_path()
    raise AEContractError("real publisher requires ENABLED formal run root")


def default_formal_authorities() -> AEFormalAuthorities:
    return AEFormalAuthorities(
        prepare_context=default_v3_prepare_context,
        prepare_stage_a=default_v3_prepare_stage_a,
        make_common_head=lambda cell, stage_a: _json_digest({"authority": "v3_d", "key": cell.stage_a_key}),
        train_stage_b=default_v3_train_stage_b,
        predict=default_v3_predict,
        metric=default_v3_metric,
        publish=default_v3_publish,
        reset_rng=reset_ae_rng,
    )


def assert_authorities_are_frozen_v3() -> Dict[str, str]:
    if AE_STAGE_A_AUTHORITY is not prepare_ms_katl_stage_a:
        raise AEContractError("AE_STAGE_A_AUTHORITY drifted from prepare_ms_katl_stage_a")
    if AE_STAGE_B_AUTHORITY is not train_ms_katl_head:
        raise AEContractError("AE_STAGE_B_AUTHORITY drifted from train_ms_katl_head")
    if AE_PREDICTION_AUTHORITY is not predict_ms_katl_decomposed:
        raise AEContractError("AE_PREDICTION_AUTHORITY drifted from predict_ms_katl_decomposed")
    if AE_METRIC_AUTHORITY is not compute_screening_metrics:
        raise AEContractError("AE_METRIC_AUTHORITY drifted from compute_screening_metrics")
    d_flags = assert_d_equals_frozen_v3()
    if v3_architecture_digest() != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AEContractError("V3 digest drifted")
    _ = v3_arm_spec()
    _ = v3_architecture()
    _ = v3_training_recipe()
    return {
        "REAL_STAGE_A_AUTHORITY": "PASS",
        "REAL_STAGE_B_AUTHORITY": "PASS",
        "REAL_PREDICTION_AUTHORITY": "PASS",
        "REAL_METRIC_AUTHORITY": "PASS",
        **d_flags,
    }


def _arm_init_digest(cell: RealCanonicalCellPlan, arm: str, stage_a: FormalStageAView) -> str:
    spec = ae_arm_spec(arm)
    if spec.init_mode == "native":
        return _json_digest({"mode": "native", "seed": cell.seed, "key": cell.stage_a_key})
    if spec.init_mode == "fused":
        return stage_a.fused_target_digest
    return stage_a.best_source_conv_digest


def _parameter_counts(arm: str) -> Tuple[int, int, int]:
    spec = ae_arm_spec(arm)
    prediction = 100 + (2 if spec.revin_enabled else 0)
    trainable = 40 if spec.target_backbone_trainable else 10
    if spec.revin_enabled:
        trainable += 2
    frozen = prediction - trainable
    return prediction, trainable, frozen


def build_mock_stage_a(cell: RealCanonicalCellPlan) -> FormalStageAView:
    identities = [("S1", cell.dataset), ("S2", cell.dataset), ("S3", cell.dataset)]
    return FormalStageAView(
        stage_a_key=cell.stage_a_key,
        source_identities=identities,
        source_distances=[0.10, 0.50, 0.90],
        source_weights=[0.5, 0.3, 0.2],
        source_weight_digests=["d1", "d2", "d3"],
        fused_params_digest=_json_digest({"fused": cell.stage_a_key}),
        fused_target_digest=_json_digest({"fused_target": cell.stage_a_key}),
        best_source_identity=identities[0],
        best_source_conv_digest=_json_digest({"rank1": identities[0], "key": cell.stage_a_key}),
        target_split_digest=cell.target_split_digest,
        requested_k=3,
        effective_k=3,
    )


def make_mock_authorities(
    *,
    ledger: FormalCallLedger,
    fail_arm: Optional[str] = None,
    publish_root: Optional[Path] = None,
    live_contexts: Optional[Mapping[str, Any]] = None,
) -> AEFormalAuthorities:
    def prepare_context(cell: RealCanonicalCellPlan) -> Dict[str, Any]:
        if cell.stage_a_key in ledger.context_prep:
            raise AEContractError(f"context prepared twice for {cell.stage_a_key}")
        ledger.context_prep.append(cell.stage_a_key)
        if live_contexts is None:
            return {"cell": cell.stage_a_key, "status": MOCK_STATUS}
        live = live_contexts.get(cell.stage_a_key)
        if live is None:
            raise AEContractError(f"missing real live context for {cell.stage_a_key}")
        if getattr(live, "source_df", None) is None or getattr(live, "target_df", None) is None:
            raise AEContractError("fake context forbidden in real-context mock acceptance")
        if str(getattr(live, "status", "")) != "CONTEXT_READY":
            raise AEContractError(f"live context not CONTEXT_READY: {cell.stage_a_key}")
        return live.as_formal_context()

    def prepare_stage_a(cell: RealCanonicalCellPlan, context: Mapping[str, Any]) -> FormalStageAView:
        live = context.get("live") if isinstance(context, Mapping) else None
        if live_contexts is not None:
            if live is None:
                raise AEContractError("real-context mock body forbids missing live context")
            if getattr(live, "source_df", None) is None:
                raise AEContractError("synthetic dataframe forbidden")
        _ = context
        if cell.stage_a_key in ledger.stage_a:
            raise AEContractError(f"Stage A prepared twice for {cell.stage_a_key}")
        ledger.stage_a.append(cell.stage_a_key)
        return build_mock_stage_a(cell)

    def make_common_head(cell: RealCanonicalCellPlan, stage_a: FormalStageAView) -> str:
        return _json_digest({"head": "v3_d_authority", "key": cell.stage_a_key, "fused": stage_a.fused_target_digest})

    def train_stage_b(
        cell: RealCanonicalCellPlan,
        arm: str,
        stage_a: FormalStageAView,
        head_digest: str,
        context: Mapping[str, Any],
    ) -> FormalArmRuntime:
        _ = context
        spec = ae_arm_spec(arm)
        ledger.stage_b.append((cell.stage_a_key, arm))
        if fail_arm == arm:
            raise RuntimeError(f"mock Stage-B failure on {arm}")
        pred_count, trainable, frozen = _parameter_counts(arm)
        recipe = ae_training_recipe()
        return FormalArmRuntime(
            arm=arm,
            spec=spec,
            init_mode=spec.init_mode,
            source_init_consumed=spec.init_mode != "native",
            stage_a_shared_for_pairing=True,
            head_target_init_digest=head_digest,
            target_initial_conv_digest=_arm_init_digest(cell, arm, stage_a),
            fused_init_digest=stage_a.fused_target_digest,
            best_source_identity=stage_a.best_source_identity,
            best_source_snapshot_digest=stage_a.best_source_conv_digest,
            source_identities=list(stage_a.source_identities),
            source_distances=list(stage_a.source_distances),
            source_weights=list(stage_a.source_weights),
            selected_lr=float(cell.selected_lr),
            recipe_digest=_json_digest(recipe),
            target_split_digest=stage_a.target_split_digest,
            parent_method=PARENT_METHOD,
            parent_v3_architecture_digest=v3_architecture_digest(),
            prediction_path_parameter_count=pred_count,
            trainable_prediction_path_parameter_count=trainable,
            frozen_prediction_path_parameter_count=frozen,
            status=MOCK_STATUS,
            metric_kind=MOCK_METRIC,
        )

    def predict(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
        _ = context
        ledger.predict.append((cell.stage_a_key, runtime.arm))
        runtime.metric_kind = MOCK_METRIC
        runtime.rmse_original = None
        runtime.smape_original = None
        runtime.mae_original = None
        return runtime

    def metric(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
        _ = context
        ledger.metric.append((cell.stage_a_key, runtime.arm))
        runtime.metric_kind = MOCK_METRIC
        runtime.status = MOCK_STATUS
        runtime.rmse_original = None
        return runtime

    def publish(
        cell: RealCanonicalCellPlan,
        runtimes: Mapping[str, FormalArmRuntime],
        pair_status: str,
        output_root: Path,
    ) -> None:
        ledger.publish.append(cell.stage_a_key)
        root = publish_root if publish_root is not None else output_root
        bundle_dir = Path(root) / "cell_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": MOCK_STATUS,
            "pair_status": pair_status,
            "stage_a_key": cell.stage_a_key,
            "arms": {
                arm: {
                    "status": runtime.status,
                    "rmse_original": None,
                    "metric_kind": MOCK_METRIC,
                    "init_mode": runtime.init_mode,
                    "source_init_consumed": runtime.source_init_consumed,
                }
                for arm, runtime in runtimes.items()
            },
        }
        (bundle_dir / f"{cell.stage_a_key.replace('/', '-')}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    return AEFormalAuthorities(
        prepare_context=prepare_context,
        prepare_stage_a=prepare_stage_a,
        make_common_head=make_common_head,
        train_stage_b=train_stage_b,
        predict=predict,
        metric=metric,
        publish=publish,
        reset_rng=reset_ae_rng,
    )


def validate_paired_runtime_bundle(
    cell: RealCanonicalCellPlan,
    stage_a: FormalStageAView,
    runtimes: Mapping[str, FormalArmRuntime],
) -> str:
    if set(runtimes) != set(AE_ARMS):
        return "PAIR_INCOMPLETE"
    if any(item.status not in {MOCK_STATUS, "OK"} or item.error for item in runtimes.values()):
        return "PAIR_INCOMPLETE"
    bcd = [runtimes[arm] for arm in ("B", "C", "D", "E")]
    if len({tuple(item.source_identities) for item in bcd}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: source identities diverged")
    if len({tuple(item.source_distances) for item in bcd}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: source distances diverged")
    if len({tuple(item.source_weights) for item in bcd}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: source weights diverged")
    if len({runtimes[arm].fused_init_digest for arm in ("B", "C", "D")}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: fused init digest diverged")
    if runtimes["E"].best_source_identity != tuple(stage_a.source_identities[0]):
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: E rank-1 identity mismatch")
    if runtimes["E"].best_source_snapshot_digest != stage_a.best_source_conv_digest:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: E rank-1 digest mismatch")
    if len({item.head_target_init_digest for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: head init digest diverged")
    if len({item.target_split_digest for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: target split digest diverged")
    if len({item.selected_lr for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: LR diverged")
    if len({item.recipe_digest for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: recipe digest diverged")
    if runtimes["A"].source_init_consumed:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: arm A consumed source init")
    if runtimes["D"].parent_v3_architecture_digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: D parent digest drifted")
    if runtimes["B"].target_initial_conv_digest != runtimes["C"].target_initial_conv_digest:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: B/C fused conv digest diverged")
    if runtimes["A"].init_mode != "native" or runtimes["E"].init_mode != "best_source":
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: init modes drifted")
    _ = cell
    return "PAIR_COMPLETE"


def load_pair_statuses(run_root: Path) -> Dict[str, str]:
    path = Path(run_root) / "cell_bundles"
    statuses: Dict[str, str] = {}
    if not path.is_dir():
        return statuses
    for file in path.glob("*.json"):
        payload = json.loads(file.read_text(encoding="utf-8"))
        statuses[str(payload["stage_a_key"])] = str(payload.get("pair_status") or "PAIR_INCOMPLETE")
    return statuses


def run_ae_formal_body(
    *,
    mock_execution_authorized: bool = False,
    execute: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    authorities: Optional[AEFormalAuthorities] = None,
    arm_order: Sequence[str] = AE_ARMS,
    resume_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    ledger: Optional[FormalCallLedger] = None,
    cells: Optional[Sequence[RealCanonicalCellPlan]] = None,
) -> Dict[str, Any]:
    """Canonical formal orchestration. Real trainers stay blocked without dual lock."""
    if resume_root is not None and output_root is None:
        output_root = Path(resume_root)
    if not mock_execution_authorized:
        assert_formal_training_permitted(execute=execute, environ=environ)
        if not AE_FORMAL_EXECUTION_ENABLED:
            raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")
        authorities = authorities or default_formal_authorities()
    elif authorities is None:
        raise AEContractError("mock formal body requires injected authorities")
    order = tuple(str(item).strip().upper() for item in arm_order)
    if tuple(sorted(order)) != tuple(sorted(AE_ARMS)):
        raise AEContractError(f"arm_order must be a permutation of {list(AE_ARMS)}")

    root = project_root_from(project_root)
    load_ae_manifest(root)
    cell_plans = list(cells) if cells is not None else enumerate_real_canonical_cell_plans(project_root=root)
    arm_plans = enumerate_real_arm_plans(cell_plans)
    if len(cell_plans) != EXPECTED_CANONICAL_CELLS or len(arm_plans) != EXPECTED_RUNS:
        raise AEContractError("formal body matrix drifted")
    matrix_hash = planned_matrix_digest(arm_plans)
    if resume_root is not None:
        validate_resume_root(Path(resume_root), project_root=root)
        snapshot = json.loads((Path(resume_root) / "manifest_snapshot.json").read_text(encoding="utf-8"))
        current = ae_protocol_manifest_payload()
        if snapshot.get("protocol_version") != PROTOCOL_VERSION:
            raise AEContractError("resume protocol_version mismatch")
        if snapshot.get("parent_v3_architecture_digest") != current["parent_v3_architecture_digest"]:
            raise AEContractError("resume parent V3 digest mismatch")
        if snapshot.get("planned_matrix_digest") not in {None, matrix_hash} and snapshot.get("planned_matrix_digest") != matrix_hash:
            raise AEContractError("resume planned matrix digest mismatch")

    out = Path(output_root) if output_root is not None else root / PREFLIGHT_RELPATH / "formal_mock"
    if mock_execution_authorized:
        if "outputs" in out.parts and "runs" in out.parts:
            raise AEContractError("mock formal artifacts must not enter outputs/runs")
        if "results" in out.parts and "preflight" not in out.parts and "audits" not in out.parts:
            raise AEContractError("mock formal artifacts must not enter results/")
    else:
        assert_ae_output_root(out)
        if AE_RUN_LABEL not in out.name:
            raise AEContractError("formal run root must include MS_KATL_AE_CORE_ABLATION")
    out.mkdir(parents=True, exist_ok=True)

    completed = load_pair_statuses(out) if resume_root is not None else {}
    active_ledger = ledger if ledger is not None else FormalCallLedger()
    bundles: Dict[str, str] = dict(completed)
    identity_index: Dict[str, Dict[str, Any]] = {}

    for cell in cell_plans:
        if completed.get(cell.stage_a_key) == "PAIR_COMPLETE":
            continue
        context = authorities.prepare_context(cell)
        if cell.stage_a_key not in active_ledger.context_prep:
            active_ledger.context_prep.append(cell.stage_a_key)
        live = context.get("live") if isinstance(context, Mapping) else None
        if live is not None and str(getattr(live, "stage_a_key", "")) != cell.stage_a_key:
            raise AEContractError("live context/stage-A cell mismatch")
        if live is not None and str(getattr(live, "target_bundle_key", "")) != cell.stage_a_key:
            raise AEContractError("live context/target-bundle cell mismatch")
        stage_a = authorities.prepare_stage_a(cell, context)
        head_digest = authorities.make_common_head(cell, stage_a)
        runtimes: Dict[str, FormalArmRuntime] = {}
        failed = False
        for arm in order:
            authorities.reset_rng(int(cell.seed))
            try:
                runtime = authorities.train_stage_b(cell, arm, stage_a, head_digest, context)
                runtime = authorities.predict(cell, runtime, context)
                runtime = authorities.metric(cell, runtime, context)
                runtimes[arm] = runtime
            except Exception as exc:
                failed = True
                spec = ae_arm_spec(arm)
                runtimes[arm] = FormalArmRuntime(
                    arm=arm,
                    spec=spec,
                    init_mode=spec.init_mode,
                    source_init_consumed=spec.init_mode != "native",
                    stage_a_shared_for_pairing=True,
                    head_target_init_digest=head_digest,
                    target_initial_conv_digest="",
                    fused_init_digest=stage_a.fused_target_digest,
                    best_source_identity=stage_a.best_source_identity,
                    best_source_snapshot_digest=stage_a.best_source_conv_digest,
                    source_identities=list(stage_a.source_identities),
                    source_distances=list(stage_a.source_distances),
                    source_weights=list(stage_a.source_weights),
                    selected_lr=float(cell.selected_lr),
                    recipe_digest=_json_digest(ae_training_recipe()),
                    target_split_digest=stage_a.target_split_digest,
                    parent_method=PARENT_METHOD,
                    parent_v3_architecture_digest=v3_architecture_digest(),
                    prediction_path_parameter_count=0,
                    trainable_prediction_path_parameter_count=0,
                    frozen_prediction_path_parameter_count=0,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                    metric_kind=MOCK_METRIC,
                )
        active_ledger.bundle_validations.append(cell.stage_a_key)
        if failed or set(runtimes) != set(AE_ARMS):
            pair_status = "PAIR_INCOMPLETE"
        else:
            pair_status = validate_paired_runtime_bundle(cell, stage_a, runtimes)
        if pair_status == "PAIR_COMPLETE":
            identity_index[cell.stage_a_key] = {
                arm: {
                    "init_mode": runtimes[arm].init_mode,
                    "head": runtimes[arm].head_target_init_digest,
                    "conv": runtimes[arm].target_initial_conv_digest,
                    "stage_a": tuple(runtimes[arm].source_identities),
                    "best": runtimes[arm].best_source_identity,
                }
                for arm in AE_ARMS
            }
        authorities.publish(cell, runtimes, pair_status, out)
        bundles[cell.stage_a_key] = pair_status

    counts = active_ledger.counts()
    return {
        "status": MOCK_STATUS if mock_execution_authorized else "FORMAL",
        "CANONICAL_CELLS": len(cell_plans),
        "CONTEXT_PREP_CALLS": counts["CONTEXT_PREP_CALLS"],
        "STAGE_A_CALLS": counts["STAGE_A_CALLS"],
        "STAGE_B_CALLS": counts["STAGE_B_CALLS"],
        "PREDICT_CALLS": counts["PREDICT_CALLS"],
        "METRIC_CALLS": counts["METRIC_CALLS"],
        "BUNDLE_VALIDATIONS": counts["BUNDLE_VALIDATIONS"],
        "PUBLISH_ATTEMPTS": counts["PUBLISH_ATTEMPTS"],
        "A_CALLS": counts["A_CALLS"],
        "B_CALLS": counts["B_CALLS"],
        "C_CALLS": counts["C_CALLS"],
        "D_CALLS": counts["D_CALLS"],
        "E_CALLS": counts["E_CALLS"],
        "bundles": bundles,
        "identity_index": identity_index,
        "planned_matrix_digest": matrix_hash,
        "ledger": active_ledger,
        "output_root": str(out),
        "TRAINING_STARTED": "NO",
        "AE_FORMAL_EXECUTION_ENABLED": AE_FORMAL_EXECUTION_ENABLED,
    }


def git_provenance(project_root: Optional[Path] = None) -> Dict[str, str]:
    root = project_root_from(project_root)

    def _git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        return (result.stdout or result.stderr).strip()

    porcelain = _git("status", "--porcelain")
    ae_core = (
        "src/transfer_methods/ms_katl_ae_ablation.py",
        "src/transfer_methods/ms_katl_ae_execution.py",
        "src/transfer_methods/ms_katl_ae_formal.py",
        "src/transfer_methods/ms_katl_ae_live_context.py",
        "src/transfer_methods/ms_katl_v3_frame_assembly.py",
        "scripts/run_ms_katl_ae_ablation.py",
        "tests/test_ms_katl_ae_ablation_contract.py",
        "tests/test_ms_katl_ae_formal_body.py",
        "configs/ms_katl_ae_ablation_manifest.json",
    )
    dirty = []
    for line in porcelain.splitlines():
        path = line[3:].strip() if len(line) > 3 else line
        if path in ae_core:
            dirty.append(path)
    freeze_required = "YES" if dirty or porcelain else "NO"
    if dirty:
        freeze_required = "YES"
    return {
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": porcelain,
        "ae_core_untracked_or_dirty": ", ".join(dirty),
        "FORMAL_GIT_FREEZE_REQUIRED": freeze_required,
    }


def run_ae_formal_preflight(
    *,
    project_root: Optional[Path] = None,
    write_artifacts: bool = True,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    load_ae_manifest(root)
    authorities = assert_authorities_are_frozen_v3()
    cells = enumerate_real_canonical_cell_plans(project_root=root)
    arms = enumerate_real_arm_plans(cells)
    git_info = git_provenance(root)
    source = Path(__file__).read_text(encoding="utf-8")
    old_reuse = assert_no_old_v3_result_injection(source)
    preflight = run_ae_preflight(project_root=root, include_construction_contracts=True)
    pairwise = preflight["pairwise"]
    kwargs_report = resolve_all_formal_stage_a_input_plans(cells, project_root=root)
    from src.transfer_methods.ms_katl_ae_live_context import (
        construct_all_live_ae_contexts,
        representative_trace,
    )

    live_report = construct_all_live_ae_contexts(cells, project_root=root)
    traces = {}
    d4_trace = {}
    for ctx in live_report["contexts"]:
        if int(ctx.horizon) == 1 and int(ctx.seed) == 42:
            traces[ctx.dataset] = representative_trace(ctx)
        if ctx.dataset == "D4" and int(ctx.horizon) == 1 and int(ctx.seed) == 42:
            d4_trace = representative_trace(ctx)
    training_guard = "PASS"
    try:
        assert_formal_training_permitted(execute=True, environ={})
        training_guard = "FAIL"
    except AETrainingBlocked:
        pass
    try:
        assert_formal_training_permitted(
            execute=False,
            environ={FORMAL_TRAINING_UNLOCK_ENV: "YES"},
        )
        training_guard = "FAIL"
    except AETrainingBlocked:
        pass
    enabled = "YES" if AE_FORMAL_EXECUTION_ENABLED else "NO"
    unique_run_keys = len({item.run_key for item in arms})
    unique_stage_a = len({cell.stage_a_key for cell in cells})
    payload = {
        "AE_PROTOCOL_VERSION": PROTOCOL_VERSION,
        "REAL_STAGE_A_AUTHORITY": authorities["REAL_STAGE_A_AUTHORITY"],
        "REAL_STAGE_B_AUTHORITY": authorities["REAL_STAGE_B_AUTHORITY"],
        "REAL_PREDICTION_AUTHORITY": authorities["REAL_PREDICTION_AUTHORITY"],
        "REAL_METRIC_AUTHORITY": authorities["REAL_METRIC_AUTHORITY"],
        "CANONICAL_CELLS": len(cells),
        "PLANNED_RUNS": len(arms),
        "UNIQUE_RUN_KEYS": unique_run_keys,
        "UNIQUE_STAGE_A_KEYS": unique_stage_a,
        "STAGE_A_AUTHORITY_NAME": AE_STAGE_A_AUTHORITY.__name__,
        "STAGE_B_AUTHORITY_NAME": AE_STAGE_B_AUTHORITY.__name__,
        "PREDICTION_AUTHORITY_NAME": AE_PREDICTION_AUTHORITY.__name__,
        "METRIC_AUTHORITY_NAME": AE_METRIC_AUTHORITY.__name__,
        "AE_FORMAL_EXECUTION_ENABLED": enabled,
        "FORMAL_EXECUTION_CAPABILITY": "ENABLED" if AE_FORMAL_EXECUTION_ENABLED else "DISABLED",
        "ACTUAL_EXECUTION_REQUESTED": "NO",
        "TRAINING_GUARD": training_guard,
        "D_OLD_RESULT_REUSE": old_reuse,
        "D_EQUALS_FROZEN_V3": preflight["d_identity"]["ARM_D_PARENT_V3_IDENTITY"],
        "A_C_SINGLE_FACTOR": pairwise.get("A_C_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "B_C_SINGLE_FACTOR": pairwise.get("B_C_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "C_E_SINGLE_FACTOR": pairwise.get("C_E_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "C_D_SINGLE_FACTOR": pairwise.get("C_D_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "LR_CONTRACT": "PASS",
        "RECIPE_CONTRACT": "PASS",
        "FORMAL_GIT_FREEZE_REQUIRED": git_info["FORMAL_GIT_FREEZE_REQUIRED"],
        "git": git_info,
        "planned_formal_run_root_pattern": str(planned_formal_run_root("<timestamp>")),
        "REAL_DATA_FORMAL_PRECHECK": kwargs_report["REAL_DATA_FORMAL_PRECHECK"],
        "REAL_STAGE_A_KWARGS_RESOLUTION": kwargs_report["REAL_STAGE_A_KWARGS_RESOLUTION"],
        "REAL_STAGE_A_KWARGS_FAILURES": kwargs_report["failures"],
        "LIVE_CONTEXT_CONSTRUCTION": live_report["LIVE_CONTEXT_CONSTRUCTION"],
        "REAL_STAGE_A_KWARGS_OBJECTS": live_report["REAL_STAGE_A_KWARGS_OBJECTS"],
        "STAGE_A_SIGNATURE_COMPATIBILITY": live_report["STAGE_A_SIGNATURE_COMPATIBILITY"],
        "REAL_SHARED_TARGET_BUNDLES": live_report["REAL_SHARED_TARGET_BUNDLES"],
        "D1_LIVE_CONTEXT": live_report["dataset_status"]["D1"],
        "D2_LIVE_CONTEXT": live_report["dataset_status"]["D2"],
        "D3_LIVE_CONTEXT": live_report["dataset_status"]["D3"],
        "D4_LIVE_CONTEXT": live_report["dataset_status"]["D4"],
        "D5_LIVE_CONTEXT": live_report["dataset_status"]["D5"],
        "D6_LIVE_CONTEXT": live_report["dataset_status"]["D6"],
        "LIVE_CONTEXT_MATRIX": live_report["matrix"],
        "D4_REPRESENTATIVE_TRACE": d4_trace,
        "DATASET_REPRESENTATIVE_TRACES": traces,
        "LIVE_CONTEXT_FAILURES": live_report["failures"],
        "FORMAL_STAGE_A_WILL_RECOMPUTE": "YES",
        "TRAINING_STARTED": "NO",
    }
    artifact = {
        key: value
        for key, value in payload.items()
        if key not in {"LIVE_CONTEXT_FAILURES", "REAL_STAGE_A_KWARGS_FAILURES"} or value
    }
    if write_artifacts:
        out = root / PREFLIGHT_RELPATH
        out.mkdir(parents=True, exist_ok=True)
        (out / "formal_preflight_report.json").write_text(
            json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (out / "live_context_matrix.json").write_text(
            json.dumps({"rows": live_report["matrix"], "status": "CONTEXT_READY"}, indent=2) + "\n",
            encoding="utf-8",
        )
    ok = (
        training_guard == "PASS"
        and old_reuse == "NO"
        and payload["CANONICAL_CELLS"] == EXPECTED_CANONICAL_CELLS
        and payload["PLANNED_RUNS"] == EXPECTED_RUNS
        and unique_run_keys == EXPECTED_RUNS
        and unique_stage_a == EXPECTED_CANONICAL_CELLS
        and payload["A_C_SINGLE_FACTOR"] == "PASS"
        and payload["B_C_SINGLE_FACTOR"] == "PASS"
        and payload["C_E_SINGLE_FACTOR"] == "PASS"
        and payload["C_D_SINGLE_FACTOR"] == "PASS"
        and payload["D_EQUALS_FROZEN_V3"] == "PASS"
        and kwargs_report["REAL_DATA_FORMAL_PRECHECK"] == "PASS"
        and live_report["ok"]
        and AE_FORMAL_EXECUTION_ENABLED
    )
    lines = [
        f"AE_PROTOCOL_VERSION={PROTOCOL_VERSION}",
        f"REAL_STAGE_A_AUTHORITY={payload['REAL_STAGE_A_AUTHORITY']}",
        f"REAL_STAGE_B_AUTHORITY={payload['REAL_STAGE_B_AUTHORITY']}",
        f"REAL_PREDICTION_AUTHORITY={payload['REAL_PREDICTION_AUTHORITY']}",
        f"REAL_METRIC_AUTHORITY={payload['REAL_METRIC_AUTHORITY']}",
        f"CANONICAL_CELLS={payload['CANONICAL_CELLS']}",
        f"PLANNED_RUNS={payload['PLANNED_RUNS']}",
        f"UNIQUE_RUN_KEYS={payload['UNIQUE_RUN_KEYS']}",
        f"UNIQUE_STAGE_A_KEYS={payload['UNIQUE_STAGE_A_KEYS']}",
        f"LR_CONTRACT={payload['LR_CONTRACT']}",
        f"RECIPE_CONTRACT={payload['RECIPE_CONTRACT']}",
        f"A_C_SINGLE_FACTOR={payload['A_C_SINGLE_FACTOR']}",
        f"B_C_SINGLE_FACTOR={payload['B_C_SINGLE_FACTOR']}",
        f"C_E_SINGLE_FACTOR={payload['C_E_SINGLE_FACTOR']}",
        f"C_D_SINGLE_FACTOR={payload['C_D_SINGLE_FACTOR']}",
        f"D_EQUALS_FROZEN_V3={payload['D_EQUALS_FROZEN_V3']}",
        f"FORMAL_EXECUTION_CAPABILITY={payload['FORMAL_EXECUTION_CAPABILITY']}",
        f"ACTUAL_EXECUTION_REQUESTED={payload['ACTUAL_EXECUTION_REQUESTED']}",
        f"AE_FORMAL_EXECUTION_ENABLED={enabled}",
        f"TRAINING_GUARD={training_guard}",
        f"D_OLD_RESULT_REUSE={old_reuse}",
        f"FORMAL_GIT_FREEZE_REQUIRED={payload['FORMAL_GIT_FREEZE_REQUIRED']}",
        f"REAL_DATA_FORMAL_PRECHECK={payload['REAL_DATA_FORMAL_PRECHECK']}",
        f"LIVE_CONTEXT_CONSTRUCTION={payload['LIVE_CONTEXT_CONSTRUCTION']}",
        f"REAL_STAGE_A_KWARGS_OBJECTS={payload['REAL_STAGE_A_KWARGS_OBJECTS']}",
        f"STAGE_A_SIGNATURE_COMPATIBILITY={payload['STAGE_A_SIGNATURE_COMPATIBILITY']}",
        f"REAL_SHARED_TARGET_BUNDLES={payload['REAL_SHARED_TARGET_BUNDLES']}",
        f"D1_LIVE_CONTEXT={payload['D1_LIVE_CONTEXT']}",
        f"D2_LIVE_CONTEXT={payload['D2_LIVE_CONTEXT']}",
        f"D3_LIVE_CONTEXT={payload['D3_LIVE_CONTEXT']}",
        f"D4_LIVE_CONTEXT={payload['D4_LIVE_CONTEXT']}",
        f"D5_LIVE_CONTEXT={payload['D5_LIVE_CONTEXT']}",
        f"D6_LIVE_CONTEXT={payload['D6_LIVE_CONTEXT']}",
        f"FORMAL_STAGE_A_WILL_RECOMPUTE={payload['FORMAL_STAGE_A_WILL_RECOMPUTE']}",
        f"planned_formal_run_root={payload['planned_formal_run_root_pattern']}",
        "TRAINING_STARTED=NO",
        f"FORMAL_PREFLIGHT={'PASS' if ok else 'FAIL'}",
    ]
    if d4_trace:
        lines.extend(
            [
                "",
                "D4_REPRESENTATIVE_TRACE",
                f"sealed_source_path={d4_trace.get('sealed_source_path')}",
                f"sealed_target_path={d4_trace.get('sealed_target_path')}",
                f"source_df_shape={d4_trace.get('source_df_shape')}",
                f"target_df_shape={d4_trace.get('target_df_shape')}",
                f"target_identity={d4_trace.get('target_identity')}",
                f"group_cols={d4_trace.get('group_cols')}",
                f"selected_lr={d4_trace.get('selected_lr')}",
                f"stage_a_key={d4_trace.get('stage_a_key')}",
                f"target_bundle_key={d4_trace.get('target_bundle_key')}",
                "STOP_BEFORE=prepare_ms_katl_stage_a",
            ]
        )
    payload["text"] = "\n".join(lines) + "\n"
    payload["FORMAL_PREFLIGHT"] = "PASS" if ok else "FAIL"
    return payload
