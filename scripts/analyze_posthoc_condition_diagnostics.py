#!/usr/bin/env python3
"""Read-only post-hoc diagnostics for D6 C vs B, D4 SITA-TCN vs TCN, and target series.

This script never trains, never calls model.fit, and never overwrites raw data,
solidified parquet, configs, historical results, or historical outputs.
"""

from __future__ import annotations

import ast
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.protocols.formal_target_scope import (
    resolve_formal_target_window,
    scope_target_to_formal_window,
)
from src.protocols.gate1_transformation import dataset_contract

AE_RESULTS_CSV = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/原始实验结果/ae_uniform_3e4_results.csv"
)
AE_MANIFEST = ROOT / "configs" / "ms_katl_sita_ablation_3e4_manifest.json"
AE_BUNDLE_DIR = ROOT / "outputs" / "sita_ablation_3e4_d1_d4_d5_d6_5seed" / "cell_bundles"
TCN_FORMAL_DIR = ROOT / "results" / "sita_tcn" / "formal"
TCN_SEED_LEVEL_CSV = TCN_FORMAL_DIR / "SITA_TCN_FORMAL_120CELL_RAW.csv"
TCN_CONDITION_STATS_CSV = TCN_FORMAL_DIR / "SITA_TCN_VS_BL7_CONDITION_STATS.csv"
TCN_CELLS_DIR = TCN_FORMAL_DIR / "cells"
REVIN_PROTOCOL = ROOT / "scripts" / "baselines" / "revin_strengthened_protocol.py"
PROMOTE_TARGETS = ROOT / "tools" / "promote_bl6_bl7_formal_results.py"

DATASETS = ("D1", "D4", "D5", "D6")
HORIZONS = (1, 3, 5)
HORIZON_LABELS = {1: "H1", 3: "H3", 5: "H5"}
SEEDS = (42, 43, 44, 45, 46)
AE_ARMS = ("B", "C")
TCN_METHOD = "TCN_RevIN"
SITA_TCN_METHOD = "SITA-TCN"
TCN_INIT = "random"
SITA_INIT = "sita"
VALID_AE_RUN_STATUS = {"OK"}
VALID_AE_PAIR_STATUS = {"PAIR_COMPLETE"}
VALID_TCN_RESULT_STATUS = {"trial"}
MEAN_MATCH_ABS_TOL = 1e-12
Y_TRUE_MATCH_ABS_TOL = 1e-12
MIN_LAG7_PAIRS = 2


class DiagnosticError(RuntimeError):
    """Fail-closed diagnostic contract error."""


def _run_git(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def git_identity() -> dict[str, str]:
    return {
        "git_branch": _run_git(["branch", "--show-current"]),
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_status_short": _run_git(["status", "--short"]),
    }


def fail(message: str) -> None:
    raise DiagnosticError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"required file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def require_columns(frame: pd.DataFrame, columns: Sequence[str], *, where: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        fail(f"{where} missing columns: {missing}; available={list(frame.columns)}")


def unique_sorted(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values})


def horizon_label(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"TOTAL", "H1", "H3", "H5"}:
        return text
    number = int(value)
    if number not in HORIZON_LABELS:
        fail(f"unsupported horizon: {value!r}")
    return HORIZON_LABELS[number]


def is_uniform_3e4(value: object) -> bool:
    text = str(value).strip().lower().replace(" ", "")
    if text in {"3e-4", "3e-04", "0.0003"}:
        return True
    try:
        return math.isclose(float(value), 3e-4, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def literal_dict_assignment(path: Path, name: str) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    payload = ast.literal_eval(node.value)
                    return {str(key): str(value) for key, value in payload.items()}
    fail(f"{path} has no literal assignment for {name}")


def winner_from_delta(delta: float, *, negative_name: str, positive_name: str) -> str:
    if delta < 0:
        return negative_name
    if delta > 0:
        return positive_name
    return "Tie"


def same_direction(deltas: Sequence[float]) -> bool:
    signs = {0 if float(value) == 0 else (1 if float(value) > 0 else -1) for value in deltas}
    return len(signs) == 1


def relative_change_pct(new: float, baseline: float, *, where: str) -> float:
    if float(baseline) == 0:
        fail(f"{where}: baseline RMSE is 0; relative_change_pct is undefined")
    return (float(new) - float(baseline)) / float(baseline) * 100.0


def parse_target_parts(target: str, key_fields: Sequence[str]) -> tuple[str, ...]:
    parts = tuple(str(target).split("/"))
    if len(parts) != len(key_fields):
        fail(
            f"target {target!r} does not match key_fields {tuple(key_fields)} "
            f"(got {parts})"
        )
    return parts


def filter_entity(
    frame: pd.DataFrame,
    *,
    key_fields: Sequence[str],
    values: Sequence[str],
    where: str,
) -> pd.DataFrame:
    missing = [column for column in key_fields if column not in frame.columns]
    if missing:
        fail(f"{where} missing group columns {missing}")
    mask = pd.Series(True, index=frame.index)
    for column, expected in zip(key_fields, values):
        mask &= frame[column].astype(str).eq(str(expected))
    entity = frame.loc[mask].copy()
    if entity.empty:
        fail(f"{where} produced an empty entity slice for {tuple(values)}")
    return entity


def resolve_paper_targets() -> dict[str, str]:
    ae_manifest = load_json(AE_MANIFEST)
    ae_targets = {str(key): str(value) for key, value in ae_manifest["paper_targets"].items()}
    revin = literal_dict_assignment(REVIN_PROTOCOL, "PAPER_TARGETS")
    promote = literal_dict_assignment(PROMOTE_TARGETS, "FORMAL_TARGETS")
    if ae_targets != revin or ae_targets != promote:
        fail(
            "paper target authority mismatch: "
            f"ae_manifest={ae_targets} revin={revin} promote={promote}"
        )
    for dataset, target in ae_targets.items():
        spec = dataset_contract(dataset)
        serialized = {"/".join(key) for key in spec.target_keys}
        if target not in serialized:
            fail(
                f"{dataset} paper target {target!r} is not in dataset_contract "
                f"target_keys={sorted(serialized)}"
            )
    return ae_targets


def resolve_test_intervals() -> dict[str, dict[str, str]]:
    intervals = {}
    for dataset in DATASETS:
        spec = dataset_contract(dataset)
        window = resolve_formal_target_window(dataset)
        paths = resolve_formal_dataset_paths(int(dataset[1:]), repository_root=ROOT)
        manifest = load_json(paths.dataset_manifest_path)
        target_windows = (manifest.get("windows") or {}).get("target") or {}
        manifest_start = str(target_windows.get("blind_start") or "")
        manifest_end = str(target_windows.get("blind_end") or "")
        contract_start = spec.blind_start.isoformat()
        contract_end = spec.blind_end.isoformat()
        if not manifest_start or not manifest_end:
            fail(f"{dataset} sealed dataset manifest is missing windows.target.blind_*")
        if manifest_start != contract_start or manifest_end != contract_end:
            fail(
                f"{dataset} test interval mismatch: manifest={manifest_start}/{manifest_end} "
                f"contract={contract_start}/{contract_end}"
            )
        forecast_days = int(window.forecast_horizon_days)
        if forecast_days != 180:
            fail(f"{dataset} formal forecast window is {forecast_days} days, expected 180")
        intervals[dataset] = {
            "start": contract_start,
            "end": contract_end,
            "train_start": spec.target_train_start.isoformat(),
            "train_end": spec.target_train_end.isoformat(),
            "validation_start": spec.validation_start.isoformat(),
            "validation_end": spec.validation_end.isoformat(),
            "origin": spec.origin.isoformat(),
            "key_fields": list(spec.key_fields),
            "target_path": str(paths.target_path),
            "dataset_manifest_path": str(paths.dataset_manifest_path),
        }
    return intervals


def inspect_ae_schema(frame: pd.DataFrame) -> dict[str, Any]:
    require_columns(
        frame,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "learning_rate",
            "RMSE",
            "run_status",
            "pair_status",
            "matrix_kind",
        ],
        where="ae_uniform_3e4_results.csv",
    )
    payload = {
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "datasets": unique_sorted(frame["dataset"]),
        "targets": unique_sorted(frame["target"]),
        "horizons": unique_sorted(frame["horizon"]),
        "seeds": unique_sorted(frame["seed"]),
        "arms": unique_sorted(frame["arm"]),
        "learning_rate": unique_sorted(frame["learning_rate"]),
        "run_status": unique_sorted(frame["run_status"]),
        "pair_status": unique_sorted(frame["pair_status"]),
        "matrix_kind": unique_sorted(frame["matrix_kind"]),
    }
    print("AE schema:", json.dumps(payload, ensure_ascii=False))
    return payload


def inspect_tcn_schema(frame: pd.DataFrame) -> dict[str, Any]:
    require_columns(
        frame,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "method",
            "init_mode",
            "rmse",
            "result_status",
            "metric_space",
        ],
        where="SITA_TCN_FORMAL_120CELL_RAW.csv",
    )
    payload = {
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "datasets": unique_sorted(frame["dataset"]),
        "targets": unique_sorted(frame["target"]),
        "horizons": unique_sorted(frame["horizon"]),
        "seeds": unique_sorted(frame["seed"]),
        "methods": unique_sorted(frame["method"]),
        "init_mode": unique_sorted(frame["init_mode"]),
        "result_status": unique_sorted(frame["result_status"]),
        "metric_space": unique_sorted(frame["metric_space"]),
    }
    print("TCN seed-level schema:", json.dumps(payload, ensure_ascii=False))
    return payload


def analyze_d6_c_vs_b(
    ae: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    subset = ae[
        (ae["dataset"].astype(str) == "D6")
        & (ae["target"].astype(str) == target)
        & (ae["arm"].astype(str).isin(AE_ARMS))
        & (ae["horizon"].astype(int).isin(HORIZONS))
        & (ae["seed"].astype(int).isin(SEEDS))
    ].copy()
    if subset.empty:
        fail("D6 B/C subset is empty")

    bad_lr = subset.loc[~subset["learning_rate"].map(is_uniform_3e4)]
    if not bad_lr.empty:
        fail(f"D6 B/C contains non-3e-4 learning_rate values: {unique_sorted(bad_lr['learning_rate'])}")
    if (subset["matrix_kind"].astype(str) != "UNIFORM_3E4_PAPER_5SEED").any():
        fail(f"D6 B/C matrix_kind={unique_sorted(subset['matrix_kind'])}")
    bad_run = subset.loc[~subset["run_status"].astype(str).isin(VALID_AE_RUN_STATUS)]
    if not bad_run.empty:
        fail(f"D6 B/C run_status not OK: {unique_sorted(bad_run['run_status'])}")
    bad_pair = subset.loc[~subset["pair_status"].astype(str).isin(VALID_AE_PAIR_STATUS)]
    if not bad_pair.empty:
        fail(f"D6 B/C pair_status not PAIR_COMPLETE: {unique_sorted(bad_pair['pair_status'])}")
    if subset.duplicated(["dataset", "target", "horizon", "seed", "arm"]).any():
        fail("D6 B/C contains duplicate dataset/target/horizon/seed/arm rows")
    if subset["RMSE"].isna().any():
        fail("D6 B/C has missing RMSE")

    expected_keys = {
        (horizon, seed, arm) for horizon in HORIZONS for seed in SEEDS for arm in AE_ARMS
    }
    actual_keys = set(
        zip(
            subset["horizon"].astype(int),
            subset["seed"].astype(int),
            subset["arm"].astype(str),
        )
    )
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra or len(subset) != 30:
        fail(f"D6 B/C completeness failed: n={len(subset)} missing={missing} extra={extra}")

    mismatches: list[str] = []
    seed_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            pair = subset[
                (subset["horizon"].astype(int) == horizon) & (subset["seed"].astype(int) == seed)
            ]
            by_arm = {str(row["arm"]): row for _, row in pair.iterrows()}
            if set(by_arm) != {"B", "C"}:
                fail(f"D6 H{horizon} seed={seed} arms={sorted(by_arm)}")
            rmse_b = float(by_arm["B"]["RMSE"])
            rmse_c = float(by_arm["C"]["RMSE"])
            bundle_name = f"D6|{target.replace('/', '-')}|h{horizon}|s{seed}|without.json"
            bundle_path = AE_BUNDLE_DIR / bundle_name
            if not bundle_path.is_file():
                fail(f"missing AE cell bundle: {bundle_path}")
            bundle = load_json(bundle_path)
            if str(bundle.get("pair_status")) != "PAIR_COMPLETE":
                fail(f"{bundle_path} pair_status={bundle.get('pair_status')}")
            for arm, expected_rmse in (("B", rmse_b), ("C", rmse_c)):
                actual = bundle.get("arms", {}).get(arm, {})
                if str(actual.get("status")) != "OK":
                    fail(f"{bundle_path} arm {arm} status={actual.get('status')}")
                bundle_rmse = float(actual["rmse_original"])
                if not math.isclose(bundle_rmse, expected_rmse, rel_tol=0.0, abs_tol=MEAN_MATCH_ABS_TOL):
                    mismatches.append(
                        f"D6 H{horizon} seed={seed} arm={arm} csv={expected_rmse} bundle={bundle_rmse}"
                    )
            delta = rmse_c - rmse_b
            seed_rows.append(
                {
                    "dataset": "D6",
                    "target": target,
                    "horizon": horizon_label(horizon),
                    "seed": int(seed),
                    "rmse_B": rmse_b,
                    "rmse_C": rmse_c,
                    "delta_rmse_C_minus_B": delta,
                    "relative_change_pct": relative_change_pct(
                        rmse_c, rmse_b, where=f"D6 H{horizon} seed={seed}"
                    ),
                    "winner": winner_from_delta(delta, negative_name="C", positive_name="B"),
                }
            )
    if mismatches:
        fail("AE CSV vs cell_bundle RMSE mismatch:\n" + "\n".join(mismatches))

    seed_frame = pd.DataFrame.from_records(seed_rows)
    summary_rows: list[dict[str, Any]] = []
    print_lines = ["D6 C vs B RMSE"]
    totals = {"C": 0, "B": 0, "Tie": 0}
    all_deltas: list[float] = []
    all_rel: list[float] = []
    all_b: list[float] = []
    all_c: list[float] = []
    for horizon in HORIZONS:
        label = horizon_label(horizon)
        group = seed_frame[seed_frame["horizon"] == label]
        if len(group) != 5:
            fail(f"D6 {label} has {len(group)} seed pairs, expected 5")
        wins = group["winner"].value_counts().to_dict()
        c_wins = int(wins.get("C", 0))
        b_wins = int(wins.get("B", 0))
        ties = int(wins.get("Tie", 0))
        if c_wins + b_wins + ties != 5:
            fail(f"D6 {label} winner counts do not sum to 5")
        deltas = [float(value) for value in group["delta_rmse_C_minus_B"].tolist()]
        rels = [float(value) for value in group["relative_change_pct"].tolist()]
        rmse_b = [float(value) for value in group["rmse_B"].tolist()]
        rmse_c = [float(value) for value in group["rmse_C"].tolist()]
        summary_rows.append(
            {
                "dataset": "D6",
                "target": target,
                "horizon": label,
                "n_seeds": 5,
                "C_wins": c_wins,
                "B_wins": b_wins,
                "ties": ties,
                "mean_rmse_B": float(statistics.mean(rmse_b)),
                "mean_rmse_C": float(statistics.mean(rmse_c)),
                "mean_delta": float(statistics.mean(deltas)),
                "median_delta": float(statistics.median(deltas)),
                "mean_relative_change_pct": float(statistics.mean(rels)),
                "all_seeds_same_direction": same_direction(deltas),
            }
        )
        totals["C"] += c_wins
        totals["B"] += b_wins
        totals["Tie"] += ties
        all_deltas.extend(deltas)
        all_rel.extend(rels)
        all_b.extend(rmse_b)
        all_c.extend(rmse_c)
        print_lines.append(f"{label}: C wins {c_wins}/5, B wins {b_wins}/5, ties {ties}/5")
    if len(all_deltas) != 15:
        fail(f"D6 TOTAL expected 15 seed-pairs, got {len(all_deltas)}")
    summary_rows.append(
        {
            "dataset": "D6",
            "target": target,
            "horizon": "TOTAL",
            "n_seeds": 15,
            "C_wins": totals["C"],
            "B_wins": totals["B"],
            "ties": totals["Tie"],
            "mean_rmse_B": float(statistics.mean(all_b)),
            "mean_rmse_C": float(statistics.mean(all_c)),
            "mean_delta": float(statistics.mean(all_deltas)),
            "median_delta": float(statistics.median(all_deltas)),
            "mean_relative_change_pct": float(statistics.mean(all_rel)),
            "all_seeds_same_direction": same_direction(all_deltas),
        }
    )
    print_lines.append(
        f"TOTAL: C wins {totals['C']}/15, B wins {totals['B']}/15, ties {totals['Tie']}/15"
    )
    summary = pd.DataFrame.from_records(summary_rows)
    return seed_frame, summary, {"print_lines": print_lines, "totals": totals}


def tcn_cell_dirname(dataset: str, target: str, horizon: int, seed: int, init_mode: str) -> str:
    return f"{dataset}__{str(target).replace('/', '_')}__H{int(horizon)}__S{int(seed)}__without__{init_mode}"


def analyze_d4_sita_tcn_vs_tcn(
    raw: pd.DataFrame,
    *,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    subset = raw[
        (raw["dataset"].astype(str) == "D4")
        & (raw["target"].astype(str) == target)
        & (raw["horizon"].astype(int).isin(HORIZONS))
        & (raw["seed"].astype(int).isin(SEEDS))
        & (raw["method"].astype(str).isin({TCN_METHOD, SITA_TCN_METHOD}))
    ].copy()
    if subset.empty:
        fail("D4 TCN/SITA-TCN subset is empty")
    if (subset["mode"].astype(str) != "without").any():
        fail(f"D4 TCN mode={unique_sorted(subset['mode'])}")
    if (subset["metric_space"].astype(str) != "original_sales_space").any():
        fail(f"D4 TCN metric_space={unique_sorted(subset['metric_space'])}")
    bad_status = subset.loc[~subset["result_status"].astype(str).isin(VALID_TCN_RESULT_STATUS)]
    if not bad_status.empty:
        fail(f"D4 TCN unexpected result_status={unique_sorted(bad_status['result_status'])}")
    if "ORIGINAL_SALES_SPACE" in subset.columns and (subset["ORIGINAL_SALES_SPACE"].astype(str) != "YES").any():
        fail("D4 TCN ORIGINAL_SALES_SPACE is not YES")
    if subset.duplicated(["dataset", "target", "horizon", "seed", "method", "init_mode"]).any():
        fail("D4 TCN seed-level rows are duplicated")
    if subset["rmse"].isna().any():
        fail("D4 TCN seed-level RMSE is missing")

    random_arm = subset[
        (subset["method"].astype(str) == TCN_METHOD) & (subset["init_mode"].astype(str) == TCN_INIT)
    ]
    sita_arm = subset[
        (subset["method"].astype(str) == SITA_TCN_METHOD) & (subset["init_mode"].astype(str) == SITA_INIT)
    ]
    expected = {(horizon, seed) for horizon in HORIZONS for seed in SEEDS}
    random_keys = set(zip(random_arm["horizon"].astype(int), random_arm["seed"].astype(int)))
    sita_keys = set(zip(sita_arm["horizon"].astype(int), sita_arm["seed"].astype(int)))
    if random_keys != expected:
        fail(f"D4 TCN_RevIN/random seed mismatch missing={sorted(expected-random_keys)} extra={sorted(random_keys-expected)}")
    if sita_keys != expected:
        fail(f"D4 SITA-TCN/sita seed mismatch missing={sorted(expected-sita_keys)} extra={sorted(sita_keys-expected)}")
    if len(random_arm) != 15 or len(sita_arm) != 15:
        fail(f"D4 expected 15 rows per method, got TCN={len(random_arm)} SITA-TCN={len(sita_arm)}")

    seed_rows: list[dict[str, Any]] = []
    mean_rows: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        tcn_by_seed = {
            int(row["seed"]): float(row["rmse"])
            for _, row in random_arm[random_arm["horizon"].astype(int) == horizon].iterrows()
        }
        sita_by_seed = {
            int(row["seed"]): float(row["rmse"])
            for _, row in sita_arm[sita_arm["horizon"].astype(int) == horizon].iterrows()
        }
        if sorted(tcn_by_seed) != list(SEEDS) or sorted(sita_by_seed) != list(SEEDS):
            fail(f"D4 H{horizon} seed alignment failed tcn={sorted(tcn_by_seed)} sita={sorted(sita_by_seed)}")
        for seed in SEEDS:
            for init_mode, method, expected_rmse in (
                (TCN_INIT, TCN_METHOD, tcn_by_seed[seed]),
                (SITA_INIT, SITA_TCN_METHOD, sita_by_seed[seed]),
            ):
                manifest_path = TCN_CELLS_DIR / tcn_cell_dirname("D4", target, horizon, seed, init_mode) / "cell_manifest.json"
                payload = load_json(manifest_path)
                if str(payload.get("model_name")) != method:
                    fail(f"{manifest_path} model_name={payload.get('model_name')} expected {method}")
                if int(payload.get("horizon")) != horizon or int(payload.get("target_seed")) != seed:
                    fail(f"{manifest_path} identity drifted")
                if str(payload.get("target_id")) != target:
                    fail(f"{manifest_path} target_id={payload.get('target_id')}")
                cell_rmse = float(payload["rmse"])
                if not math.isclose(cell_rmse, expected_rmse, rel_tol=0.0, abs_tol=MEAN_MATCH_ABS_TOL):
                    fail(
                        f"{manifest_path} rmse={cell_rmse} does not match "
                        f"SITA_TCN_FORMAL_120CELL_RAW.csv rmse={expected_rmse}"
                    )
            rmse_tcn = tcn_by_seed[seed]
            rmse_sita = sita_by_seed[seed]
            delta = rmse_sita - rmse_tcn
            seed_rows.append(
                {
                    "dataset": "D4",
                    "target": target,
                    "horizon": horizon_label(horizon),
                    "seed": int(seed),
                    "rmse_tcn": rmse_tcn,
                    "rmse_sita_tcn": rmse_sita,
                    "delta_rmse_sita_tcn_minus_tcn": delta,
                    "relative_change_pct": relative_change_pct(
                        rmse_sita, rmse_tcn, where=f"D4 H{horizon} seed={seed}"
                    ),
                    "winner": winner_from_delta(
                        delta, negative_name="SITA-TCN", positive_name="TCN"
                    ),
                }
            )
        mean_rows[horizon] = {
            "mean_rmse_tcn": float(statistics.mean(tcn_by_seed[seed] for seed in SEEDS)),
            "mean_rmse_sita_tcn": float(statistics.mean(sita_by_seed[seed] for seed in SEEDS)),
        }

    stats = pd.read_csv(TCN_CONDITION_STATS_CSV)
    require_columns(
        stats,
        ["dataset", "target", "horizon", "bl7_rmse_mean", "sita_tcn_rmse_mean", "n_seeds"],
        where="SITA_TCN_VS_BL7_CONDITION_STATS.csv",
    )
    d4_stats = stats[(stats["dataset"].astype(str) == "D4") & (stats["target"].astype(str) == target)]
    diffs: list[str] = []
    for horizon in HORIZONS:
        row = d4_stats[d4_stats["horizon"].astype(int) == horizon]
        if len(row) != 1:
            fail(f"CONDITION_STATS D4 H{horizon} rows={len(row)}")
        rec = row.iloc[0]
        if int(rec["n_seeds"]) != 5:
            fail(f"CONDITION_STATS D4 H{horizon} n_seeds={rec['n_seeds']}")
        recomputed = mean_rows[horizon]
        if not math.isclose(
            float(rec["bl7_rmse_mean"]),
            recomputed["mean_rmse_tcn"],
            rel_tol=0.0,
            abs_tol=MEAN_MATCH_ABS_TOL,
        ):
            diffs.append(
                f"H{horizon} bl7_rmse_mean stats={float(rec['bl7_rmse_mean'])} "
                f"recomputed={recomputed['mean_rmse_tcn']}"
            )
        if not math.isclose(
            float(rec["sita_tcn_rmse_mean"]),
            recomputed["mean_rmse_sita_tcn"],
            rel_tol=0.0,
            abs_tol=MEAN_MATCH_ABS_TOL,
        ):
            diffs.append(
                f"H{horizon} sita_tcn_rmse_mean stats={float(rec['sita_tcn_rmse_mean'])} "
                f"recomputed={recomputed['mean_rmse_sita_tcn']}"
            )
    if diffs:
        fail("D4 seed-level re-aggregation disagrees with CONDITION_STATS:\n" + "\n".join(diffs))

    seed_frame = pd.DataFrame.from_records(seed_rows)
    summary_rows: list[dict[str, Any]] = []
    print_lines = ["D4 SITA-TCN vs TCN RMSE"]
    totals = {"SITA-TCN": 0, "TCN": 0, "Tie": 0}
    all_deltas: list[float] = []
    all_rel: list[float] = []
    all_tcn: list[float] = []
    all_sita: list[float] = []
    for horizon in HORIZONS:
        label = horizon_label(horizon)
        group = seed_frame[seed_frame["horizon"] == label]
        if len(group) != 5:
            fail(f"D4 {label} has {len(group)} seed pairs, expected 5")
        wins = group["winner"].value_counts().to_dict()
        sita_wins = int(wins.get("SITA-TCN", 0))
        tcn_wins = int(wins.get("TCN", 0))
        ties = int(wins.get("Tie", 0))
        deltas = [float(value) for value in group["delta_rmse_sita_tcn_minus_tcn"].tolist()]
        rels = [float(value) for value in group["relative_change_pct"].tolist()]
        rmse_tcn = [float(value) for value in group["rmse_tcn"].tolist()]
        rmse_sita = [float(value) for value in group["rmse_sita_tcn"].tolist()]
        summary_rows.append(
            {
                "dataset": "D4",
                "target": target,
                "horizon": label,
                "n_seeds": 5,
                "sita_tcn_wins": sita_wins,
                "tcn_wins": tcn_wins,
                "ties": ties,
                "mean_rmse_tcn": float(statistics.mean(rmse_tcn)),
                "mean_rmse_sita_tcn": float(statistics.mean(rmse_sita)),
                "mean_delta": float(statistics.mean(deltas)),
                "median_delta": float(statistics.median(deltas)),
                "mean_relative_change_pct": float(statistics.mean(rels)),
                "all_seeds_same_direction": same_direction(deltas),
            }
        )
        totals["SITA-TCN"] += sita_wins
        totals["TCN"] += tcn_wins
        totals["Tie"] += ties
        all_deltas.extend(deltas)
        all_rel.extend(rels)
        all_tcn.extend(rmse_tcn)
        all_sita.extend(rmse_sita)
        print_lines.append(
            f"{label}: SITA-TCN wins {sita_wins}/5, TCN wins {tcn_wins}/5, ties {ties}/5"
        )
    summary_rows.append(
        {
            "dataset": "D4",
            "target": target,
            "horizon": "TOTAL",
            "n_seeds": 15,
            "sita_tcn_wins": totals["SITA-TCN"],
            "tcn_wins": totals["TCN"],
            "ties": totals["Tie"],
            "mean_rmse_tcn": float(statistics.mean(all_tcn)),
            "mean_rmse_sita_tcn": float(statistics.mean(all_sita)),
            "mean_delta": float(statistics.mean(all_deltas)),
            "median_delta": float(statistics.median(all_deltas)),
            "mean_relative_change_pct": float(statistics.mean(all_rel)),
            "all_seeds_same_direction": same_direction(all_deltas),
        }
    )
    print_lines.append(
        f"TOTAL: SITA-TCN wins {totals['SITA-TCN']}/15, TCN wins {totals['TCN']}/15, ties {totals['Tie']}/15"
    )
    return pd.DataFrame.from_records(seed_rows), pd.DataFrame.from_records(summary_rows), {
        "print_lines": print_lines,
        "totals": totals,
    }


def lag7_acf(dates: pd.DatetimeIndex, values: np.ndarray) -> tuple[float, str]:
    series = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates))
    series = series.sort_index()
    if series.index.has_duplicates:
        fail("lag7_acf received duplicated dates")
    calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
    aligned = series.reindex(calendar)
    left = aligned
    right = aligned.shift(7)
    valid = left.notna() & right.notna()
    n_pairs = int(valid.sum())
    if n_pairs < MIN_LAG7_PAIRS:
        return float("nan"), f"lag7_acf NaN: only {n_pairs} paired observations"
    corr = float(left[valid].corr(right[valid]))
    if not math.isfinite(corr):
        return float("nan"), "lag7_acf NaN: Pearson correlation undefined for this series"
    return corr, f"lag7_acf used {n_pairs} calendar-aligned paired observations; missing dates were not zero-filled"


def series_stats(
    *,
    dataset: str,
    target: str,
    scope: str,
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    data_source: str,
    target_filter: str,
    extra_notes: Sequence[str] = (),
) -> dict[str, Any]:
    y = np.asarray(values, dtype=float)
    if y.size == 0:
        fail(f"{dataset}/{target}/{scope} series is empty")
    if not np.isfinite(y).all():
        fail(f"{dataset}/{target}/{scope} series contains non-finite sales")
    n_obs = int(y.size)
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if n_obs >= 2 else float("nan")
    if mean == 0:
        cv = float("nan")
        cv_note = "cv=NaN because mean==0"
    else:
        cv = float(std / mean)
        cv_note = "cv=std/mean with sample std ddof=1"
    zero_ratio = float(np.count_nonzero(y == 0) / n_obs)
    p95 = float(np.percentile(y, 95))
    maximum = float(np.max(y))
    acf, acf_note = lag7_acf(dates, y)
    notes = [
        cv_note,
        "zero_ratio=count(y==0)/n_obs",
        "p95=numpy.percentile(y, 95) linear interpolation on observed y_true",
        acf_note,
        *extra_notes,
    ]
    return {
        "dataset": dataset,
        "target": target,
        "scope": scope,
        "start_date": pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
        "n_obs": n_obs,
        "mean": mean,
        "std": std,
        "cv": cv,
        "zero_ratio": zero_ratio,
        "p95": p95,
        "max": maximum,
        "lag7_acf": acf,
        "data_source": data_source,
        "target_filter": target_filter,
        "notes": " | ".join(notes),
    }


def load_paper_target_series(
    dataset: str,
    target: str,
) -> dict[str, Any]:
    spec = dataset_contract(dataset)
    paths = resolve_formal_dataset_paths(int(dataset[1:]), repository_root=ROOT)
    raw = pd.read_parquet(paths.target_path)
    if "date" not in raw.columns or "sales" not in raw.columns:
        fail(f"{dataset} sealed target.parquet missing date/sales: {list(raw.columns)}")
    scoped = scope_target_to_formal_window(raw, dataset_id=int(dataset[1:]))
    values = parse_target_parts(target, spec.key_fields)
    target_filter = ",".join(f"{field}={value}" for field, value in zip(spec.key_fields, values))
    paper_scoped = filter_entity(
        scoped,
        key_fields=spec.key_fields,
        values=values,
        where=f"{dataset} formal window",
    )
    paper_scoped["date"] = pd.to_datetime(paper_scoped["date"], errors="raise").dt.normalize()
    paper_scoped = paper_scoped.sort_values("date").reset_index(drop=True)
    if paper_scoped["date"].duplicated().any():
        fail(f"{dataset}/{target} formal window has duplicate dates")
    test_start = pd.Timestamp(spec.blind_start).normalize()
    test_end = pd.Timestamp(spec.blind_end).normalize()
    test = paper_scoped.loc[paper_scoped["date"].between(test_start, test_end, inclusive="both")].copy()
    expected = pd.date_range(test_start, test_end, freq="D")
    actual = pd.DatetimeIndex(test["date"])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(test) != 180 or len(missing) or len(extra):
        fail(
            f"{dataset}/{target} formal test calendar is incomplete: "
            f"n={len(test)} missing={list(missing.strftime('%Y-%m-%d'))} extra={list(extra.strftime('%Y-%m-%d'))}"
        )

    pred_path = (
        TCN_CELLS_DIR
        / tcn_cell_dirname(dataset, target, 1, 42, SITA_INIT)
        / "predictions.csv"
    )
    pred = pd.read_csv(pred_path)
    require_columns(pred, ["timestamp", "y_true"], where=str(pred_path))
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], errors="raise").dt.normalize()
    merged = pred.merge(test[["date", "sales"]], left_on="timestamp", right_on="date", how="left")
    if merged["sales"].isna().any():
        fail(f"{dataset}/{target} H1 y_true timestamps fall outside the formal test interval")
    max_abs = float(np.max(np.abs(merged["y_true"].to_numpy(dtype=float) - merged["sales"].to_numpy(dtype=float))))
    if max_abs > Y_TRUE_MATCH_ABS_TOL:
        fail(f"{dataset}/{target} H1 y_true does not match sealed test sales; max_abs={max_abs}")

    full = filter_entity(
        raw,
        key_fields=spec.key_fields,
        values=values,
        where=f"{dataset} full sealed target",
    )
    full["date"] = pd.to_datetime(full["date"], errors="raise").dt.normalize()
    full = full.sort_values("date").reset_index(drop=True)
    if full["date"].duplicated().any():
        fail(f"{dataset}/{target} full sealed history has duplicate dates")
    if full["sales"].isna().any():
        fail(f"{dataset}/{target} full sealed history has NA sales")
    calendar = pd.date_range(full["date"].min(), full["date"].max(), freq="D")
    missing_full = calendar.difference(pd.DatetimeIndex(full["date"]))
    return {
        "dataset": dataset,
        "target": target,
        "key_fields": list(spec.key_fields),
        "target_filter": target_filter,
        "data_source": str(paths.target_path),
        "dataset_manifest_path": str(paths.dataset_manifest_path),
        "test_dates": pd.DatetimeIndex(test["date"]),
        "test_values": test["sales"].to_numpy(dtype=float),
        "full_dates": pd.DatetimeIndex(full["date"]),
        "full_values": full["sales"].to_numpy(dtype=float),
        "full_missing_dates": [stamp.strftime("%Y-%m-%d") for stamp in missing_full],
        "h1_eval_start": pred["timestamp"].min().strftime("%Y-%m-%d"),
        "h1_eval_end": pred["timestamp"].max().strftime("%Y-%m-%d"),
        "h1_eval_n": int(len(pred)),
        "y_true_match_max_abs": max_abs,
        "preprocessing": (
            "Read sealed target.parquet; apply scope_target_to_formal_window "
            "(protocol calendar, no extra zero-fill); filter paper target by "
            f"{target_filter}; take blind_start..blind_end as formal_test_interval."
        ),
    }


def write_readme(
    path: Path,
    *,
    run_time: str,
    identity: Mapping[str, str],
    sources: Mapping[str, Any],
    row_counts: Mapping[str, int],
    paper_targets: Mapping[str, str],
    test_intervals: Mapping[str, Mapping[str, str]],
    series_meta: Sequence[Mapping[str, Any]],
    anomalies: Sequence[str],
) -> None:
    lines = [
        "# Post-hoc condition diagnostics",
        "",
        "Read-only audit. No training. No overwrite of raw/solidified/config/historical outputs.",
        "",
        "## Run identity",
        "",
        f"- script_run_time: `{run_time}`",
        f"- git_branch: `{identity['git_branch']}`",
        f"- git_commit: `{identity['git_commit']}`",
        f"- git_status_short: `{identity['git_status_short'] or '(clean)'}`",
        f"- python_executable: `{sys.executable}`",
        "",
        "## Files read",
        "",
    ]
    for label, value in sources.items():
        lines.append(f"- {label}: `{value}`")
    lines.extend(
        [
            "",
            "## Output row counts",
            "",
        ]
    )
    for name, count in row_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## Paper targets",
            "",
        ]
    )
    for dataset in DATASETS:
        lines.append(f"- {dataset}: `{paper_targets[dataset]}`")
    lines.extend(
        [
            "",
            "## Formal test intervals",
            "",
            "Authority = sealed dataset manifest `windows.target.blind_start/blind_end`, cross-checked against `dataset_contract`.",
            "",
        ]
    )
    for dataset in DATASETS:
        interval = test_intervals[dataset]
        lines.append(
            f"- {dataset}: `{interval['start']}` .. `{interval['end']}` "
            f"(train `{interval['train_start']}`..`{interval['train_end']}`, "
            f"val `{interval['validation_start']}`..`{interval['validation_end']}`)"
        )
    lines.extend(
        [
            "",
            "## Target series sources",
            "",
        ]
    )
    for item in series_meta:
        missing = item["full_missing_dates"]
        lines.append(
            f"- {item['dataset']} `{item['target']}`: sealed `{item['data_source']}`; "
            f"filter `{item['target_filter']}`; H1 evaluated y_true `{item['h1_eval_start']}`..`{item['h1_eval_end']}` "
            f"(n={item['h1_eval_n']}, subset of the 180-day test interval after lookback=10); "
            f"full-history missing calendar dates={missing if missing else 'none'}."
        )
        lines.append(f"  preprocessing: {item['preprocessing']}")
    lines.extend(
        [
            "",
            "## Delta definitions",
            "",
            "- D6 C vs B: `delta_rmse_C_minus_B = RMSE_C - RMSE_B`. delta<0 means C wins; delta>0 means B wins; delta==0 is Tie.",
            "- D4 SITA-TCN vs TCN: `delta_rmse_sita_tcn_minus_tcn = RMSE_SITA_TCN - RMSE_TCN`. delta<0 means SITA-TCN wins; delta>0 means TCN wins; delta==0 is Tie.",
            "- `relative_change_pct` uses the same signed numerator over the baseline RMSE, times 100.",
            "- Ties use exact numeric comparison, no tolerance.",
            "",
            "## Statistic definitions",
            "",
            "- mean: arithmetic mean of observed y_true.",
            "- std: sample standard deviation, ddof=1.",
            "- cv: std / mean. If mean==0, cv=NaN.",
            "- zero_ratio: count(y==0) / n_obs.",
            "- p95: numpy.percentile(y, 95) on observed y_true.",
            "- max: maximum observed y_true.",
            "- lag7_acf: Pearson correlation of y_t and y_{t-7} after sorting onto the official daily calendar. Missing dates are left as NA and are not zero-filled. NaN if fewer than 2 valid pairs.",
            "",
            "## Anomalies / protocol notes",
            "",
        ]
    )
    if anomalies:
        for item in anomalies:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / "audits" / f"posthoc_condition_diagnostics_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = git_identity()
    anomalies: list[str] = []

    print("=== schema inspection ===")
    paper_targets = resolve_paper_targets()
    test_intervals = resolve_test_intervals()
    print("paper_targets:", json.dumps(paper_targets, ensure_ascii=False))
    print(
        "test_intervals:",
        json.dumps({key: {"start": value["start"], "end": value["end"]} for key, value in test_intervals.items()}),
    )

    ae = pd.read_csv(AE_RESULTS_CSV)
    inspect_ae_schema(ae)
    tcn_raw = pd.read_csv(TCN_SEED_LEVEL_CSV)
    inspect_tcn_schema(tcn_raw)
    ae_observed = {
        str(dataset): str(target)
        for dataset, target in ae[["dataset", "target"]].drop_duplicates().itertuples(index=False, name=None)
    }
    tcn_observed = {
        str(dataset): str(target)
        for dataset, target in tcn_raw[["dataset", "target"]].drop_duplicates().itertuples(index=False, name=None)
    }
    if ae_observed != paper_targets:
        fail(f"AE CSV targets {ae_observed} != paper_targets {paper_targets}")
    if tcn_observed != paper_targets:
        fail(f"TCN seed-level targets {tcn_observed} != paper_targets {paper_targets}")

    d6_seed, d6_summary, d6_meta = analyze_d6_c_vs_b(ae, target=paper_targets["D6"])
    d4_seed, d4_summary, d4_meta = analyze_d4_sita_tcn_vs_tcn(tcn_raw, target=paper_targets["D4"])

    series_meta = []
    test_rows = []
    full_rows = []
    for dataset in DATASETS:
        loaded = load_paper_target_series(dataset, paper_targets[dataset])
        series_meta.append(loaded)
        test_rows.append(
            series_stats(
                dataset=dataset,
                target=paper_targets[dataset],
                scope="formal_test_interval",
                dates=loaded["test_dates"],
                values=loaded["test_values"],
                data_source=loaded["data_source"],
                target_filter=loaded["target_filter"],
                extra_notes=[
                    f"formal test interval {test_intervals[dataset]['start']}..{test_intervals[dataset]['end']}",
                    f"H1 evaluation y_true is a lookback-shifted subset {loaded['h1_eval_start']}..{loaded['h1_eval_end']} n={loaded['h1_eval_n']}",
                    "no extra zero-fill; sealed formal calendar already complete for this interval",
                ],
            )
        )
        full_notes = [
            "scope=full_available_target_history from sealed target.parquet paper-target rows only",
            "not mixed with formal_test_interval",
        ]
        if loaded["full_missing_dates"]:
            full_notes.append(
                "calendar gaps left as missing (not zero-filled): "
                + ",".join(loaded["full_missing_dates"])
            )
            anomalies.append(
                f"{dataset} full sealed target history is not a complete daily calendar; "
                f"missing {loaded['full_missing_dates']}. Stats use observed dates only; "
                "lag7_acf uses calendar shift and drops unpaired NA."
            )
        full_rows.append(
            series_stats(
                dataset=dataset,
                target=paper_targets[dataset],
                scope="full_available_target_history",
                dates=loaded["full_dates"],
                values=loaded["full_values"],
                data_source=loaded["data_source"],
                target_filter=loaded["target_filter"],
                extra_notes=full_notes,
            )
        )

    anomalies.append(
        "TCN formal cell result_status is 'trial' in SITA_TCN_FORMAL_120CELL_RAW.csv and cell_manifest.json; "
        "this is the actual formal 5-seed artifact status, not a screening/early-convergence file. "
        "EARLY_CONVERGENCE_TRACE.csv was not used."
    )
    anomalies.append(
        "BL7 in the current backbone-generalization comparison is method=TCN_RevIN with init_mode=random "
        "(cell model_id=BL7_TCN_RevIN). SITA-TCN is method=SITA-TCN with init_mode=sita."
    )
    anomalies.append(
        "dataset_contract for D4/D5/D6 lists multiple sealed target keys; paper analyses used only the "
        "manifest paper_targets and did not pool source products or retired D4 keys 166/311 and 166/313."
    )

    d6_seed.to_csv(output_dir / "cb_d6_seed_level.csv", index=False)
    d6_summary.to_csv(output_dir / "cb_d6_condition_summary.csv", index=False)
    d4_seed.to_csv(output_dir / "tcn_d4_seed_level.csv", index=False)
    d4_summary.to_csv(output_dir / "tcn_d4_condition_summary.csv", index=False)
    test_frame = pd.DataFrame.from_records(test_rows)
    full_frame = pd.DataFrame.from_records(full_rows)
    test_frame.to_csv(output_dir / "target_test_series_stats.csv", index=False)
    full_frame.to_csv(output_dir / "target_full_series_stats.csv", index=False)

    sources = {
        "ae_uniform_3e4_results.csv": str(AE_RESULTS_CSV),
        "ae_manifest": str(AE_MANIFEST),
        "ae_cell_bundles": str(AE_BUNDLE_DIR),
        "tcn_seed_level": str(TCN_SEED_LEVEL_CSV),
        "tcn_condition_stats_crosscheck_only": str(TCN_CONDITION_STATS_CSV),
        "tcn_cells": str(TCN_CELLS_DIR),
        "revin_protocol": str(REVIN_PROTOCOL),
        "promote_formal_targets": str(PROMOTE_TARGETS),
        "sealed_root": str((ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1").resolve()),
    }
    row_counts = {
        "cb_d6_seed_level.csv": int(len(d6_seed)),
        "cb_d6_condition_summary.csv": int(len(d6_summary)),
        "tcn_d4_seed_level.csv": int(len(d4_seed)),
        "tcn_d4_condition_summary.csv": int(len(d4_summary)),
        "target_test_series_stats.csv": int(len(test_frame)),
        "target_full_series_stats.csv": int(len(full_frame)),
        "ae_uniform_3e4_results.csv": int(len(ae)),
        "SITA_TCN_FORMAL_120CELL_RAW.csv": int(len(tcn_raw)),
    }
    write_readme(
        output_dir / "README.md",
        run_time=run_time,
        identity=identity,
        sources=sources,
        row_counts=row_counts,
        paper_targets=paper_targets,
        test_intervals=test_intervals,
        series_meta=series_meta,
        anomalies=anomalies,
    )
    source_trace = {
        "run_time": run_time,
        "output_dir": str(output_dir),
        "git": identity,
        "python_executable": sys.executable,
        "delta_definitions": {
            "cb_d6": "delta_rmse_C_minus_B = RMSE_C - RMSE_B; negative => C wins",
            "tcn_d4": "delta_rmse_sita_tcn_minus_tcn = RMSE_SITA_TCN - RMSE_TCN; negative => SITA-TCN wins",
        },
        "results": {
            "cb_d6_seed_level.csv": {
                "source_path": str(AE_RESULTS_CSV),
                "crosscheck": str(AE_BUNDLE_DIR),
                "filters": {
                    "dataset": "D6",
                    "target": paper_targets["D6"],
                    "horizon": ["H1", "H3", "H5"],
                    "arm": ["B", "C"],
                    "seeds": list(SEEDS),
                    "learning_rate": "0.0003",
                    "run_status": "OK",
                    "pair_status": "PAIR_COMPLETE",
                    "matrix_kind": "UNIFORM_3E4_PAPER_5SEED",
                },
            },
            "cb_d6_condition_summary.csv": {
                "source_path": str(AE_RESULTS_CSV),
                "derived_from": "cb_d6_seed_level.csv",
            },
            "tcn_d4_seed_level.csv": {
                "source_path": str(TCN_SEED_LEVEL_CSV),
                "crosscheck_cells": str(TCN_CELLS_DIR),
                "crosscheck_condition_means": str(TCN_CONDITION_STATS_CSV),
                "filters": {
                    "dataset": "D4",
                    "target": paper_targets["D4"],
                    "horizon": ["H1", "H3", "H5"],
                    "methods": {TCN_METHOD: TCN_INIT, SITA_TCN_METHOD: SITA_INIT},
                    "seeds": list(SEEDS),
                    "result_status": "trial",
                    "metric_space": "original_sales_space",
                },
                "not_used": ["EARLY_CONVERGENCE_TRACE.csv", "SITA_TCN_FORMAL_5SEED_AGGREGATED.csv"],
            },
            "tcn_d4_condition_summary.csv": {
                "source_path": str(TCN_SEED_LEVEL_CSV),
                "derived_from": "tcn_d4_seed_level.csv",
            },
            "target_test_series_stats.csv": {
                "scope": "formal_test_interval",
                "targets": dict(paper_targets),
                "date_range": {
                    dataset: {
                        "start": test_intervals[dataset]["start"],
                        "end": test_intervals[dataset]["end"],
                    }
                    for dataset in DATASETS
                },
                "data_sources": {item["dataset"]: item["data_source"] for item in series_meta},
                "target_filters": {item["dataset"]: item["target_filter"] for item in series_meta},
            },
            "target_full_series_stats.csv": {
                "scope": "full_available_target_history",
                "data_sources": {item["dataset"]: item["data_source"] for item in series_meta},
                "missing_calendar_dates": {
                    item["dataset"]: item["full_missing_dates"] for item in series_meta
                },
            },
        },
    }
    json_dump(output_dir / "source_trace.json", source_trace)

    print()
    for line in d6_meta["print_lines"]:
        print(line)
    print()
    for line in d4_meta["print_lines"]:
        print(line)
    print()
    print(f"OUTPUT_DIR={output_dir}")
    print("NO_TRAINING=YES")
    print("NO_OVERWRITE_HISTORICAL=YES")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
