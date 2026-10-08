#!/usr/bin/env python3
"""Probe whether source initialization matters at the shortest usable window.

The current CNN applies two MaxPooling1D(pool_size=2) layers, so an input
window of 3 collapses to length 1 before the second pool and cannot be built.
Window 4 is the shortest length that still passes both pools.

One dataset, one horizon, one seed. Two target splits share the same
window-4 source models:

- train15: the existing 15/15/180 split, which yields 11 training windows
- short7: 5 training days + 2 validation days + the same last 180 test days,
  which yields 1 training window. Observed history before the test is 7 days.

M6 is fused initialization with RevIN. M4 is native initialization with
RevIN, using M6's pre-training head and RevIN affine values. This is a
probe, not a formal result. The dataset learning rate was selected on the
test set in the frozen protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_mechanism_ablation import prepare_shared_target_bundle
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    STAGE_A_LEARNING_RATE,
    compute_screening_metrics,
)
from src.transfer_methods.ms_katl_v3_frame_assembly import assemble_v3_cell_frames
from src.transfer_methods.sita_m1_m7_ablation import GROUP_SPECS, _nearest_snapshot, _run_group
from src.protocols.reproducibility import set_protocol_seed

OUT_DIR = REPO / "outputs/window4_m4_m6_probe"
DATASET = "D1"
TARGET = "1/10"
HORIZON = 1
SEED = 42
WINDOW = 4
TARGET_LR = 3e-5
EPOCHS = 50
BATCH = 16
SPLITS = {
    "train15": {"train_days": 15, "val_days": 15, "test_days": 180},
    "short7": {"train_days": 5, "val_days": 2, "test_days": 180},
}


def with_day_split(frame: pd.DataFrame, train_days: int, val_days: int, test_days: int) -> pd.DataFrame:
    out = frame.copy(deep=True)
    out.attrs = dict(frame.attrs)
    out.attrs["split_mode"] = "days"
    out.attrs["split_config"] = {
        "train_days": int(train_days),
        "val_days": int(val_days),
        "test_days": int(test_days),
    }
    return out


def score(arm: dict, bundle) -> dict:
    metrics = compute_screening_metrics(
        arm["y_true"],
        arm["y_pred"],
        target_scaler=bundle.target_scaler,
        feature_columns=bundle.target_feature_columns,
    )
    return {
        "rmse_original": float(metrics["rmse_original"]),
        "smape_original": float(metrics["smape_original"]),
        "n_test": int(len(np.asarray(arm["y_true"]).reshape(-1))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared = assemble_v3_cell_frames(
        DATASET, TARGET, horizon=HORIZON, seed=SEED, project_root=REPO, mode="without"
    )
    set_protocol_seed(SEED, include_frameworks=True)
    stage_a = prepare_ms_katl_stage_a(
        source_df=prepared["source_model_df"],
        target_df=prepared["target_model_df"],
        feature_cols=prepared["model_feature_cols"],
        k=3,
        horizon=HORIZON,
        window_size=WINDOW,
        learning_rate=float(STAGE_A_LEARNING_RATE),
        source_epochs=EPOCHS,
        batch_size=BATCH,
        tau_k=1.0,
        group_cols=prepared["group_cols"],
        k3_selection_context=prepared["k3_selection_context"],
        k3_raw_source_context=prepared["k3_raw_source_context"],
        transformation_reuse_context=prepared["transformation_reuse_context"],
        seed=SEED,
    )
    rows = []
    for name, split in SPLITS.items():
        target = with_day_split(prepared["target_model_df"], **split)
        bundle = prepare_shared_target_bundle(
            stage_a,
            target,
            prepared["model_feature_cols"],
            horizon=HORIZON,
            window_size=WINDOW,
            learning_rate=float(STAGE_A_LEARNING_RATE),
            seed=SEED,
            transformation_reuse_context=prepared["transformation_reuse_context"],
        )
        nearest = _nearest_snapshot(bundle, stage_a)
        m6 = _run_group(
            spec=GROUP_SPECS["M6"],
            bundle=bundle,
            nearest=nearest,
            head_init=None,
            revin_init=None,
            learning_rate=TARGET_LR,
            epochs=EPOCHS,
            batch_size=BATCH,
            seed=SEED,
            metric_protocol=None,
        )
        head = [np.array(weight, copy=True) for weight in m6["weight_states"]["init"]["head_target"]]
        revin = m6["weight_states"]["init"]["revin_affine"]
        m4 = _run_group(
            spec=GROUP_SPECS["M4"],
            bundle=bundle,
            nearest=nearest,
            head_init=head,
            revin_init=revin,
            learning_rate=TARGET_LR,
            epochs=EPOCHS,
            batch_size=BATCH,
            seed=SEED,
            metric_protocol=None,
        )
        m4_metrics = score(m4, bundle)
        m6_metrics = score(m6, bundle)
        row = {
            "split": name,
            **split,
            "window": WINDOW,
            "n_train": int(len(np.asarray(bundle.y_train).reshape(-1))),
            "n_val": int(len(np.asarray(bundle.y_val).reshape(-1))),
            "M4_rmse": m4_metrics["rmse_original"],
            "M6_rmse": m6_metrics["rmse_original"],
            "M4_smape": m4_metrics["smape_original"],
            "M6_smape": m6_metrics["smape_original"],
            "n_test": m4_metrics["n_test"],
            "m6_minus_m4_rmse_pct": 100.0 * (m4_metrics["rmse_original"] - m6_metrics["rmse_original"]) / m4_metrics["rmse_original"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {
        "dataset": DATASET,
        "target": TARGET,
        "horizon": HORIZON,
        "seed": SEED,
        "window": WINDOW,
        "epochs": EPOCHS,
        "target_learning_rate": TARGET_LR,
        "note": "Source retrieval still uses the assembled target history. Only the target train/validation split changes. Learning rate comes from the test-selected frozen protocol.",
        "rows": rows,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
