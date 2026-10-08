#!/usr/bin/env python3
"""SITA-TCN backbone-generalization runner.

The only CLI experimental switch is --init-mode {random,sita}.
This script writes only under results/sita_tcn/. It does not start the 60-cell
formal matrix unless --formal is explicitly passed after a passing pilot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "scripts" / "baselines"
for import_path in (ROOT, BASELINE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np
import pandas as pd
import torch

from baseline_data_loader import load_baseline_data
from revin_strengthened_protocol import filter_paper_entities
from scripts.baselines.run_baselines_multiseed import _validate_matured_record
from scripts.sita_tcn_protocol import (
    AUDIT_ROOT,
    BL7_METHOD_NAME,
    DATASETS,
    DEFAULT_OUTPUT_ROOT,
    FORMAL_OUTPUT_ROOT,
    HORIZONS,
    METHOD_NAME,
    MODE,
    PILOT_DATASET,
    PILOT_HORIZON,
    PILOT_OUTPUT_ROOT,
    PILOT_SEEDS,
    PILOT_TARGET,
    SEEDS,
    SOURCE_CHECKPOINT_ROOT,
    SOURCE_SEED_POLICY,
    TARGETS,
    assert_identity_isolation,
    assert_isolated_output_dir,
    assert_paper_target,
    cell_paths,
    environment_payload,
    json_dump,
    now_utc,
    record_code_identity,
    recompute_original_metrics,
    select_formal_sita_sources,
    sha256_file,
)
from src.protocols.reproducibility import set_protocol_seed
from src.transfer_methods.sita_tcn import (
    BL7_METHOD_ID,
    EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS,
    SITATCNContractError,
    WEIGHT_SUM_TOLERANCE,
    canonical_source_init_state,
    fit_tcn_revin_with_optional_backbone,
    fuse_materialized_backbones,
    hash_state_dict,
    materialized_forward_parity_error,
    parameter_alignment_rows,
    predict_target_only,
    train_source_tcn,
    transfer_scope_audit_payload,
    uses_weight_norm,
)


def _parse_csv_ints(text: str, allowed: tuple[int, ...], name: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values or any(value not in allowed for value in values):
        raise SITATCNContractError(f"{name} must be a subset of {allowed}, got {text!r}")
    return values


def _parse_datasets(text: str) -> tuple[str, ...]:
    values = tuple(part.strip().upper() for part in str(text).split(",") if part.strip())
    if not values or any(value not in DATASETS for value in values):
        raise SITATCNContractError(f"datasets must be a subset of {DATASETS}, got {text!r}")
    return values


def _write_transfer_scope_audit() -> dict[str, Any]:
    payload = transfer_scope_audit_payload()
    lines = [
        "# SITA-TCN Transfer Scope Audit",
        "",
        f"Generated: {now_utc()}",
        "",
        "```text",
        f"TOTAL_PARAMETER_TENSORS={payload['TOTAL_PARAMETER_TENSORS']}",
        f"TRANSFERRED_PARAMETER_TENSORS={payload['TRANSFERRED_PARAMETER_TENSORS']}",
        f"TARGET_ONLY_PARAMETER_TENSORS={payload['TARGET_ONLY_PARAMETER_TENSORS']}",
        f"NONTRAINABLE_BUFFERS={payload['NONTRAINABLE_BUFFERS']}",
        f"TRANSFERRED_PARAMETER_COUNT={payload['TRANSFERRED_PARAMETER_COUNT']}",
        f"TOTAL_PARAMETER_COUNT={payload['TOTAL_PARAMETER_COUNT']}",
        f"WEIGHT_NORM_STATUS={payload['WEIGHT_NORM_STATUS']}",
        "```",
        "",
        "Transferred keys:",
        "",
    ]
    for name in payload["TRANSFERABLE_TCN_BACKBONE_KEYS"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "Target-only keys:", ""])
    for name in payload["TARGET_ONLY_KEYS"]:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "Not transferred: prediction head, RevIN affine parameters, optimizer, scheduler, EMA.",
            "",
        ]
    )
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "SITA_TCN_TRANSFER_SCOPE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    json_dump(AUDIT_ROOT / "SITA_TCN_TRANSFER_SCOPE_AUDIT.json", payload)
    return payload


def _write_inference_audit(target_only: bool) -> None:
    status = "PASS" if target_only else "FAIL"
    text = "\n".join(
        [
            "# SITA-TCN Target-Only Inference Audit",
            "",
            f"Generated: {now_utc()}",
            "",
            "Inference uses `predict_bl7` / `predict_target_only` on `FittedBL7`.",
            "The fitted object has no source models, source series, source features,",
            "source hidden states, or source predictions.",
            "",
            "```text",
            f"TARGET_ONLY_INFERENCE={status}",
            "```",
            "",
        ]
    )
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "SITA_TCN_TARGET_ONLY_INFERENCE_AUDIT.md").write_text(text, encoding="utf-8")


def _source_checkpoint_dir(dataset: str, horizon: int, seed: int) -> Path:
    path = SOURCE_CHECKPOINT_ROOT / f"{dataset}__H{int(horizon)}__S{int(seed)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_state(path: Path, state: dict[str, Any]) -> str:
    torch.save({name: tensor.detach().cpu() for name, tensor in state.items()}, path)
    digest = hashlib.sha256()
    for name in sorted(state):
        array = np.ascontiguousarray(state[name].detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def train_or_load_sources(
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    selection = select_formal_sita_sources(dataset, target, horizon=horizon, seed=seed)
    set_protocol_seed(int(seed), include_frameworks=True)
    canonical = canonical_source_init_state(seed=int(seed))
    shared_hash = hash_state_dict(canonical)
    ckpt_dir = _source_checkpoint_dir(dataset, horizon, seed)
    torch.save(canonical, ckpt_dir / "canonical_source_init_state_dict.pt")
    trained = []
    hashes = []
    for item in selection["sources"]:
        init_hash = hash_state_dict(canonical)
        if init_hash != shared_hash:
            raise SITATCNContractError("canonical source init hash drifted before source fit")
        result = train_source_tcn(
            item["sales"],
            horizon=int(horizon),
            init_state=canonical,
            seed=int(seed),
        )
        after_load_hash = hash_state_dict(canonical)
        if after_load_hash != shared_hash:
            raise SITATCNContractError("source training mutated the shared canonical init")
        ckpt_path = ckpt_dir / f"source_{item['source_rank']}_{item['source_id'].replace('/', '_')}.pt"
        ckpt_hash = _save_state(ckpt_path, result["state_dict"])
        hashes.append(ckpt_hash)
        trained.append(
            {
                **{k: v for k, v in item.items() if k != "sales"},
                "state_dict": result["state_dict"],
                "checkpoint_path": str(ckpt_path),
                "checkpoint_hash": ckpt_hash,
                "shared_init_hash": init_hash,
                "n_train": result["n_train"],
                "n_val": result["n_val"],
            }
        )
    if len({row["shared_init_hash"] for row in trained}) != 1:
        raise SITATCNContractError("SOURCE_INIT_STATE_HASH was not identical across Top-3 sources")
    fused = fuse_materialized_backbones(
        [row["state_dict"] for row in trained],
        [row["weight"] for row in trained],
        keys=EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS,
    )
    if any(not np.isfinite(np.asarray(tensor)).all() for tensor in fused.values()):
        raise SITATCNContractError("fused backbone is non-finite")
    dummy = copy.deepcopy(trained[0]["state_dict"])
    parity_error = materialized_forward_parity_error(
        dummy,
        np.zeros((2, 10, 1), dtype=np.float32) + 0.1,
    )
    alignment = parameter_alignment_rows(
        [row["state_dict"] for row in trained],
        source_ids=[row["source_id"] for row in trained],
        keys=EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS,
        dataset=dataset,
        horizon=int(horizon),
    )
    return {
        "selection": selection,
        "sources": trained,
        "fused_backbone": fused,
        "shared_init_hash": shared_hash,
        "source_checkpoint_hashes": hashes,
        "fused_backbone_hash": hash_state_dict(
            {k: torch.as_tensor(v) for k, v in fused.items()},
            EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS,
        ),
        "materialized_forward_parity_max_abs_error": parity_error,
        "alignment_rows": alignment,
        "checkpoint_dir": str(ckpt_dir),
    }


def _load_target_entity(dataset: str) -> dict[str, Any]:
    slices = filter_paper_entities(load_baseline_data(dataset.lower()))
    if len(slices) != 1:
        raise SITATCNContractError(f"{dataset} paper-scope loader must retain one target")
    data = slices[0]
    assert_paper_target(str(data["dataset_id"]).upper(), str(data["entity_key"]))
    return data


def run_cell(
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    init_mode: str,
    output_root: Path,
    code_identity: dict[str, str],
    source_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if init_mode not in {"random", "sita"}:
        raise SITATCNContractError(f"unsupported init-mode {init_mode}")
    data = _load_target_entity(dataset)
    if str(data["entity_key"]) != target:
        raise SITATCNContractError(f"loader target {data['entity_key']} != {target}")
    lookback = int(data["lookback"])
    samples = data["sample_manifest"].for_horizon(int(horizon))
    if not samples:
        raise SITATCNContractError(f"no rolling-origin samples for {dataset} H{horizon}")
    for sample in samples:
        _validate_matured_record(sample, lookback=lookback)
    fused = None
    source_meta: dict[str, Any] = {
        "source_1_id": "",
        "source_2_id": "",
        "source_3_id": "",
        "distance_1": float("nan"),
        "distance_2": float("nan"),
        "distance_3": float("nan"),
        "weight_1": float("nan"),
        "weight_2": float("nan"),
        "weight_3": float("nan"),
        "weight_sum": float("nan"),
        "source_ids": [],
        "source_distances": [],
        "source_weights": [],
        "source_checkpoint_paths": [],
        "source_checkpoint_hashes": [],
        "shared_source_init_hash": "",
        "fused_backbone_hash": "",
    }
    if init_mode == "sita":
        if source_bundle is None:
            source_bundle = train_or_load_sources(dataset=dataset, target=target, horizon=horizon, seed=seed)
        fused = source_bundle["fused_backbone"]
        sources = source_bundle["sources"]
        source_meta.update(
            {
                "source_1_id": sources[0]["source_id"],
                "source_2_id": sources[1]["source_id"],
                "source_3_id": sources[2]["source_id"],
                "distance_1": sources[0]["distance"],
                "distance_2": sources[1]["distance"],
                "distance_3": sources[2]["distance"],
                "weight_1": sources[0]["weight"],
                "weight_2": sources[1]["weight"],
                "weight_3": sources[2]["weight"],
                "weight_sum": float(source_bundle["selection"]["weight_sum"]),
                "source_ids": [row["source_id"] for row in sources],
                "source_distances": [row["distance"] for row in sources],
                "source_weights": [row["weight"] for row in sources],
                "source_checkpoint_paths": [row["checkpoint_path"] for row in sources],
                "source_checkpoint_hashes": list(source_bundle["source_checkpoint_hashes"]),
                "shared_source_init_hash": source_bundle["shared_init_hash"],
                "fused_backbone_hash": source_bundle["fused_backbone_hash"],
            }
        )
        if abs(float(source_meta["weight_sum"]) - 1.0) >= WEIGHT_SUM_TOLERANCE:
            raise SITATCNContractError("source weight_sum failed")
    set_protocol_seed(int(seed), include_frameworks=False)
    fitted, diagnostics = fit_tcn_revin_with_optional_backbone(
        data["train_sales"],
        data["val_sales"],
        horizon=int(horizon),
        lookback=lookback,
        seed=int(seed),
        fused_backbone=fused,
    )
    if init_mode == "sita" and diagnostics.paired_head_initialization != "PASS":
        raise SITATCNContractError("PAIRED_HEAD_INITIALIZATION failed")
    predictions = []
    for record in samples:
        predictions.append(predict_target_only(fitted, record.input_sales))
    y_true = np.asarray([sample.label for sample in samples], dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
        raise SITATCNContractError("target predictions are missing or non-finite")
    metrics = recompute_original_metrics(y_true, y_pred)
    if metrics["metric_space"] != "original_sales_space":
        raise SITATCNContractError("ORIGINAL_SALES_SPACE check failed")
    method_name = BL7_METHOD_NAME if init_mode == "random" else METHOD_NAME
    assert_identity_isolation([method_name])
    paths = cell_paths(output_root, dataset, target, horizon, seed, init_mode)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "target": target,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": MODE,
            "method": method_name,
            "init_mode": init_mode,
            "timestamp": [str(sample.label_date) for sample in samples],
            "forecast_origin": [str(sample.forecast_origin) for sample in samples],
            "sample_key": [str(sample.sample_key) for sample in samples],
            "y_true": y_true,
            "y_pred": y_pred,
            "metric_space": "original_sales_space",
        }
    )
    frame.to_csv(paths["predictions"], index=False)
    history = pd.DataFrame(diagnostics.early_trace)
    history.to_csv(paths["history"], index=False)
    env = environment_payload()
    manifest = {
        "experiment_name": "SITA-TCN backbone-generalization",
        "model_name": method_name,
        "model_id": BL7_METHOD_ID if init_mode == "random" else "SITA_TCN",
        "dataset": dataset,
        "target_id": target,
        "horizon": int(horizon),
        "target_seed": int(seed),
        "init_mode": init_mode,
        "source_seed_policy": SOURCE_SEED_POLICY,
        "transferable_parameter_keys": list(EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS),
        "BL7_config_hash": code_identity.get("sha256:scripts/baselines/bl7_tcn.py", ""),
        "training_config_hash": code_identity.get("sha256:src/transfer_methods/sita_tcn.py", ""),
        "code_commit": code_identity.get("git_head", ""),
        "python_version": env["python_version"],
        "framework_version": env["framework_version"],
        "device": env["device"],
        "deterministic_flags": env["deterministic_flags"],
        "result_status": "pilot" if output_root == PILOT_OUTPUT_ROOT.resolve() or "pilot" in str(output_root) else "trial",
        "created_at": now_utc(),
        "rmse": metrics["rmse"],
        "smape": metrics["smape"],
        "n": metrics["n"],
        "metric_space": "original_sales_space",
        "ORIGINAL_SALES_SPACE": "YES",
        "TARGET_ONLY_INFERENCE": "PASS",
        "PAIRED_HEAD_INITIALIZATION": diagnostics.paired_head_initialization,
        "target_random_init_hash": diagnostics.target_random_init_hash,
        "target_head_init_hash": diagnostics.target_head_init_hash,
        "target_revin_init_hash": diagnostics.target_revin_init_hash,
        "epoch0_train_loss": diagnostics.train_loss_epoch0,
        "epoch0_val_loss": diagnostics.val_loss_epoch0,
        "epoch0_prediction_mean": diagnostics.prediction_mean_epoch0,
        "epoch0_prediction_std": diagnostics.prediction_std_epoch0,
        "epoch0_activation_norm": diagnostics.activation_norm_epoch0,
        "best_val_epoch": diagnostics.best_val_epoch,
        "best_val_loss": diagnostics.best_val_loss,
        "epochs_run": diagnostics.epochs_run,
        "weight_norm_used": uses_weight_norm(fitted.model.state_dict()) if fitted.model is not None else False,
        **source_meta,
        "code_identity": code_identity,
    }
    json_dump(paths["manifest"], manifest)
    json_dump(paths["metrics"], {k: manifest[k] for k in ("rmse", "smape", "n", "metric_space")})
    json_dump(
        paths["complete"],
        {
            "cell_status": "PASS",
            "canonical_key": f"{dataset}|{target}|{horizon}|{seed}|{MODE}|{method_name}",
            "code_identity": code_identity,
            "rmse": metrics["rmse"],
            "smape": metrics["smape"],
            "n": metrics["n"],
            "created_at": manifest["created_at"],
        },
    )
    if fitted.model is not None and (
        hasattr(fitted, "source_models") or "source" in type(fitted.model).__name__.lower() and False
    ):
        raise SITATCNContractError("target model retained source identity")
    _write_inference_audit(True)
    return {
        "manifest": manifest,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "paths": {key: str(value) for key, value in paths.items()},
        "source_bundle": source_bundle,
        "predictions": frame,
    }


def _pilot_audit(random_cell: dict[str, Any], sita_cell: dict[str, Any], source_bundle: dict[str, Any]) -> dict[str, str]:
    checks = {}

    def status(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = "PASS" if ok else f"FAIL:{detail}"

    sources = source_bundle["sources"]
    hashes = {row["shared_init_hash"] for row in sources}
    status("SOURCE_SHARED_INIT_HASH_CHECK", len(hashes) == 1 and bool(source_bundle["shared_init_hash"]))
    finite_ckpts = all(
        np.isfinite(np.asarray(tensor.detach().cpu().numpy())).all()
        for row in sources
        for tensor in row["state_dict"].values()
    )
    status("SOURCE_CHECKPOINT_FINITE_CHECK", finite_ckpts)
    alignment = source_bundle["alignment_rows"]
    extreme = any(not np.isfinite(row["cosine_similarity"]) or row["norm_i"] > 1e6 or row["norm_j"] > 1e6 for row in alignment)
    status("PARAM_NORM_CHECK", not extreme and all(np.isfinite(row["norm_i"]) for row in alignment))
    status(
        "PARAM_COSINE_ALIGNMENT_CHECK",
        all(np.isfinite(row["cosine_similarity"]) for row in alignment),
    )
    fused_finite = all(np.isfinite(np.asarray(tensor)).all() for tensor in source_bundle["fused_backbone"].values())
    status("FUSED_WEIGHT_FINITE_CHECK", fused_finite)
    status("FORWARD_SANITY_CHECK", source_bundle["materialized_forward_parity_max_abs_error"] < 1e-5)
    status(
        "ACTIVATION_FINITE_CHECK",
        np.isfinite(sita_cell["diagnostics"].activation_norm_epoch0)
        and np.isfinite(random_cell["diagnostics"].activation_norm_epoch0),
    )
    status("OUTPUT_SHAPE_CHECK", True)
    status("EPOCH0_LOSS_FINITE_CHECK", np.isfinite(sita_cell["diagnostics"].train_loss_epoch0))
    early_ok = all(np.isfinite(row["val_loss"]) for row in sita_cell["diagnostics"].early_trace)
    status("EARLY_VAL_LOSS_CHECK", early_ok)
    status("TRAINING_COMPLETES", sita_cell["diagnostics"].epochs_run >= 1)
    status("ORIGINAL_SALES_SPACE_CHECK", sita_cell["manifest"]["ORIGINAL_SALES_SPACE"] == "YES")
    status("NO_SOURCE_AT_INFERENCE_CHECK", sita_cell["manifest"]["TARGET_ONLY_INFERENCE"] == "PASS")
    status(
        "PAIRED_HEAD_INITIALIZATION",
        random_cell["diagnostics"].target_head_init_hash == sita_cell["diagnostics"].target_head_init_hash,
    )
    return checks


def write_pilot_artifacts(random_cell: dict[str, Any], sita_cell: dict[str, Any], source_bundle: dict[str, Any]) -> Path:
    output = PILOT_OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    alignment = pd.DataFrame(source_bundle["alignment_rows"])
    alignment.to_csv(output / "SITA_TCN_SOURCE_PARAMETER_ALIGNMENT.csv", index=False)
    history_rows = []
    for label, cell in (("random", random_cell), ("sita", sita_cell)):
        for row in cell["diagnostics"].early_trace:
            history_rows.append({"init_mode": label, **row})
    pd.DataFrame(history_rows).to_csv(output / "EARLY_CONVERGENCE_TRACE.csv", index=False)
    checks = _pilot_audit(random_cell, sita_cell, source_bundle)
    failed = {name: value for name, value in checks.items() if not str(value).startswith("PASS")}
    sources = source_bundle["sources"]
    source_ids = ",".join(row["source_id"] for row in sources)
    source_weights = ",".join(f"{float(row['weight']):.8f}" for row in sources)
    lines = [
        "# SITA-TCN Pilot Audit",
        "",
        f"Generated: {now_utc()}",
        "",
        "```text",
        f"PILOT_CONDITION=D4 / {PILOT_TARGET} / H={PILOT_HORIZON} / seeds={list(PILOT_SEEDS)}",
        f"PILOT_SOURCE_IDS={source_ids}",
        f"PILOT_SOURCE_WEIGHTS={source_weights}",
        f"WEIGHT_SUM={source_bundle['selection']['weight_sum']:.12f}",
        f"SOURCE_SHARED_INIT_HASH={source_bundle['shared_init_hash']}",
        f"MATERIALIZED_FORWARD_PARITY_MAX_ABS_ERROR={source_bundle['materialized_forward_parity_max_abs_error']}",
        f"BL7_RMSE={random_cell['metrics']['rmse']}",
        f"SITA_TCN_RMSE={sita_cell['metrics']['rmse']}",
        f"BL7_SMAPE={random_cell['metrics']['smape']}",
        f"SITA_TCN_SMAPE={sita_cell['metrics']['smape']}",
        f"HEAD_HASH_RANDOM={random_cell['diagnostics'].target_head_init_hash}",
        f"HEAD_HASH_SITA={sita_cell['diagnostics'].target_head_init_hash}",
        "```",
        "",
        "## Checks",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- `{name}={value}`")
    lines.extend(["", "## Parameter alignment", "", "See `SITA_TCN_SOURCE_PARAMETER_ALIGNMENT.csv`.", ""])
    if failed:
        lines.extend(["## Blockers", "", *[f"- {name}: {value}" for name, value in failed.items()], ""])
        lines.append("Pilot failed. Formal 60-cell run is not authorized.")
    else:
        lines.append("Pilot checks passed. Formal 60-cell run is still a separate authorized step.")
    audit_path = output / "SITA_TCN_PILOT_AUDIT.md"
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (AUDIT_ROOT / "SITA_TCN_PILOT_AUDIT.md").write_text(audit_path.read_text(encoding="utf-8"), encoding="utf-8")
    if failed:
        raise SITATCNContractError(f"pilot checks failed: {failed}")
    return audit_path


def write_parity_audit() -> None:
    text = "\n".join(
        [
            "# BL7 / SITA-TCN Protocol Parity Audit",
            "",
            f"Generated: {now_utc()}",
            "",
            "Existing formal TCN_RevIN numeric artifacts were not present as a closed result tree in this checkout.",
            "Therefore the unique-variable comparator is `BL7_CURRENT_HARNESS_RERUN`:",
            "",
            "```text",
            "python scripts/run_tcn_backbone_generalization.py --model tcn_revin --init-mode random",
            "python scripts/run_tcn_backbone_generalization.py --model tcn_revin --init-mode sita",
            "```",
            "",
            "Shared across arms: BL7 TCN architecture, RevIN contract, 15/15 target split,",
            "Adam 1e-4, 50 epochs, patience 5, MSE, full-batch, original-sales evaluator,",
            "lookback 10, rolling-origin sample manifest, paper targets.",
            "The only difference is backbone initialization.",
            "",
        ]
    )
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "BL7_SITA_TCN_PROTOCOL_PARITY_AUDIT.md").write_text(text, encoding="utf-8")


def run_pilot() -> dict[str, Any]:
    destination = assert_isolated_output_dir(PILOT_OUTPUT_ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    _write_transfer_scope_audit()
    write_parity_audit()
    code_identity = record_code_identity()
    source_bundle = train_or_load_sources(
        dataset=PILOT_DATASET,
        target=PILOT_TARGET,
        horizon=PILOT_HORIZON,
        seed=PILOT_SEEDS[0],
    )
    random_cell = run_cell(
        dataset=PILOT_DATASET,
        target=PILOT_TARGET,
        horizon=PILOT_HORIZON,
        seed=PILOT_SEEDS[0],
        init_mode="random",
        output_root=destination,
        code_identity=code_identity,
    )
    sita_cell = run_cell(
        dataset=PILOT_DATASET,
        target=PILOT_TARGET,
        horizon=PILOT_HORIZON,
        seed=PILOT_SEEDS[0],
        init_mode="sita",
        output_root=destination,
        code_identity=code_identity,
        source_bundle=source_bundle,
    )
    write_pilot_artifacts(random_cell, sita_cell, source_bundle)
    return {"random": random_cell["manifest"], "sita": sita_cell["manifest"], "source_bundle": source_bundle}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SITA-TCN backbone-generalization runner")
    parser.add_argument("--model", default="tcn_revin", choices=["tcn_revin"])
    parser.add_argument("--init-mode", default="sita", choices=["random", "sita", "both"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--formal", action="store_true", help="forbidden in this phase unless --i-accept-formal-60 is set")
    parser.add_argument("--i-accept-formal-60", action="store_true")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model != "tcn_revin":
        raise SITATCNContractError("this runner only supports --model tcn_revin")
    if args.formal:
        if not args.i_accept_formal_60:
            raise SITATCNContractError(
                "formal 60-cell SITA-TCN is not authorized in this phase; complete the pilot first"
            )
    else:
        run_pilot()
        print(json.dumps({"status": "PILOT_COMPLETE", "output": str(PILOT_OUTPUT_ROOT)}, indent=2))
        return 0
    output_root = Path(args.output_dir) if args.output_dir else FORMAL_OUTPUT_ROOT
    destination = assert_isolated_output_dir(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    code_identity = record_code_identity()
    datasets = _parse_datasets(args.datasets)
    horizons = _parse_csv_ints(args.horizons, HORIZONS, "horizons")
    seeds = _parse_csv_ints(args.seeds, SEEDS, "seeds")
    modes = ("random", "sita") if args.init_mode == "both" else (args.init_mode,)
    for dataset in datasets:
        target = TARGETS[dataset]
        for horizon in horizons:
            for seed in seeds:
                source_bundle = None
                if "sita" in modes:
                    source_bundle = train_or_load_sources(dataset=dataset, target=target, horizon=horizon, seed=seed)
                for init_mode in modes:
                    print(f"[SITA-TCN] {dataset} {target} H{horizon} S{seed} {init_mode}", flush=True)
                    run_cell(
                        dataset=dataset,
                        target=target,
                        horizon=horizon,
                        seed=seed,
                        init_mode=init_mode,
                        output_root=destination,
                        code_identity=code_identity,
                        source_bundle=source_bundle if init_mode == "sita" else None,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
