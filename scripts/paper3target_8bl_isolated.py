#!/usr/bin/env python3
"""Isolated 8BL run for the three paper targets only (no SITA, not full matrix).

Targets:
  D4 166/258
  D5 48/1159414
  D6 CA_1/FOODS_3_080

Matrix: horizons 1,3,5 × seeds 42-46 × 8 baselines
All outputs go under outputs/paper3target_8bl_isolated_20261005/
Does not write into results/formal_* or additional-target 60/10 trees.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baselines"
for path in (REPO_ROOT, BASELINE_DIR, str(REPO_ROOT / "scripts")):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

OUT_ROOT = REPO_ROOT / "outputs" / "paper3target_8bl_isolated_20261005"
FREEZE_EXPECTED = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/"
    "MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN_BL6_BL7_INTEGRATED.xlsx"
)
FREEZE_SOURCE = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/"
    "MS_KATL_Paper_Data_Freeze_v1_6_FINAL_FROZEN.xlsx"
)
PAPER_TARGETS = {
    "d4": "166/258",
    "d5": "48/1159414",
    "d6": "CA_1/FOODS_3_080",
}
DATASETS = ("d4", "d5", "d6")
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
CLASSIC_METHODS = (
    "BL3_LightGBM",
    "BL4_LSTM",
    "BL5_SeasonalNaive",
    "BL6_DLinear",
    "BL7_TCN",
)
MODERN_METHODS = ("PatchTST", "TimesNet")


def ensure_freeze_link() -> None:
    if FREEZE_EXPECTED.exists() or FREEZE_EXPECTED.is_symlink():
        return
    if not FREEZE_SOURCE.is_file():
        raise SystemExit(f"missing freeze source: {FREEZE_SOURCE}")
    FREEZE_EXPECTED.symlink_to(FREEZE_SOURCE)


def run_classic(datasets: tuple[str, ...]) -> None:
    import pandas as pd
    from baseline_data_loader import load_baseline_data
    from run_baselines_multiseed import build_run_id, evaluate_entity_protocol, publish_baseline_run

    root = OUT_ROOT / "classic_bl3_bl7"
    root.mkdir(parents=True, exist_ok=True)
    for dataset_id in datasets:
        target = PAPER_TARGETS[dataset_id]
        slices = [item for item in load_baseline_data(dataset_id) if str(item["entity_key"]) == target]
        if len(slices) != 1:
            raise SystemExit(f"{dataset_id}: expected {target}, got {[s['entity_key'] for s in slices]}")
        run_id = build_run_id("paper3iso", dataset_id, (target,), CLASSIC_METHODS, HORIZONS, SEEDS)
        evaluated = [
            evaluate_entity_protocol(
                data,
                methods=CLASSIC_METHODS,
                horizons=HORIZONS,
                seeds=SEEDS,
                return_raw=True,
                run_id=run_id,
            )
            for data in slices
        ]
        summary = pd.concat([item[0] for item in evaluated], ignore_index=True)
        raw = pd.concat([item[1] for item in evaluated], ignore_index=True)
        expected = len(CLASSIC_METHODS) * len(HORIZONS) * len(SEEDS)
        if len(summary) != expected:
            raise SystemExit(f"{dataset_id} classic expected {expected}, got {len(summary)}")
        dest = root / dataset_id
        dest.mkdir(parents=True, exist_ok=True)
        for child in dest.glob("*"):
            if child.is_file():
                child.unlink()
        publish_baseline_run(summary, raw, output_dir=dest, run_id=run_id, smoke=False)
        print(f"CLASSIC {dataset_id} {target} {len(summary)}/75 DONE", flush=True)


def run_modern(datasets: tuple[str, ...]) -> None:
    import importlib.util

    from baseline_data_loader import load_baseline_data

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    proto = _load("bl9_bl10_formal_protocol_iso", REPO_ROOT / "scripts" / "bl9_bl10_formal_protocol.py")
    modern = _load("run_bl9_bl10_modern_baselines_iso", REPO_ROOT / "scripts" / "run_bl9_bl10_modern_baselines.py")
    code_identity = modern.record_code_identity()
    for dataset_id in datasets:
        dataset = dataset_id.upper()
        target = proto.TARGETS[dataset]
        data = proto.select_paper_entity(load_baseline_data(dataset_id), dataset)
        for method in MODERN_METHODS:
            method_root = OUT_ROOT / ("patchtst" if method == "PatchTST" else "timesnet")
            for horizon in HORIZONS:
                for seed in SEEDS:
                    print(f"RUN {method} {dataset} {target} H{horizon} S{seed}", flush=True)
                    try:
                        result = modern.run_one_cell(
                            method=method,
                            dataset=dataset,
                            data=data,
                            horizon=horizon,
                            seed=seed,
                            code_identity=code_identity,
                            output_root=method_root,
                            smoke=True,
                        )
                        print(f"  {result.get('status', 'DONE')}", flush=True)
                    except Exception:
                        print(traceback.format_exc(), flush=True)
                        raise


def run_msml(datasets: tuple[str, ...]) -> None:
    ensure_freeze_link()
    only_keys = []
    for dataset_id in datasets:
        target = PAPER_TARGETS[dataset_id]
        ds = dataset_id.upper()
        for horizon in HORIZONS:
            for seed in SEEDS:
                only_keys.append(f"{ds}|{target}|{horizon}|{seed}")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_bl8_msml_tl_formal_60cell.py"),
        "--output-root",
        str(OUT_ROOT / "msml_tl"),
        "--only-keys",
        *only_keys,
    ]
    print("+", " ".join(cmd[:6]), f"... ({len(only_keys)} keys)", flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    # subset may fail final 60-cell promotion audit; cells can still be complete
    if result.returncode not in {0, 1}:
        raise SystemExit(result.returncode)
    print(f"MSML exit={result.returncode}", flush=True)


def audit(datasets: tuple[str, ...]) -> None:
    counts = {}
    for dataset_id in datasets:
        summary = OUT_ROOT / "classic_bl3_bl7" / dataset_id / "summary.csv"
        counts[f"{dataset_id}|classic"] = max(0, sum(1 for _ in open(summary)) - 1) if summary.is_file() else 0
        for method, folder in (("PatchTST", "patchtst"), ("TimesNet", "timesnet"), ("MSML-TL", "msml_tl")):
            cells = OUT_ROOT / folder / "cells"
            n = 0
            if cells.is_dir():
                token = PAPER_TARGETS[dataset_id].replace("/", "_")
                n = sum(
                    1
                    for path in cells.glob("*/cell_complete.json")
                    if token in path.parent.name and path.parent.name.startswith(dataset_id.upper() + "__")
                )
            counts[f"{dataset_id}|{method}"] = n
    expected = {
        "classic_per_ds": 75,
        "modern_per_ds": 15,
        "msml_per_ds": 15,
        "total_cells": 3 * (75 + 15 + 15),
    }
    payload = {
        "out_root": str(OUT_ROOT.relative_to(REPO_ROOT)),
        "targets": {k.upper(): v for k, v in PAPER_TARGETS.items()},
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "counts": counts,
        "expected": expected,
        "sita_ran": False,
        "formal_dirs_untouched": True,
    }
    (OUT_ROOT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run", "classic", "modern", "msml", "audit"), default="run")
    parser.add_argument("--dataset", choices=("d4", "d5", "d6", "all"), default="all")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "README.txt").write_text(
        "\n".join(
            [
                "Isolated paper-target 8BL (NO SITA, not full matrix)",
                "D4=166/258 D5=48/1159414 D6=CA_1/FOODS_3_080",
                "horizons=1,3,5 seeds=42-46",
                "methods=SeasonalNaive,LightGBM,LSTM,DLinear,TCN,PatchTST,TimesNet,MSML-TL",
                f"out={OUT_ROOT}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if args.phase in {"run", "classic"}:
        run_classic(datasets)
    if args.phase in {"run", "modern"}:
        run_modern(datasets)
    if args.phase in {"run", "msml"}:
        run_msml(datasets)
    if args.phase in {"run", "audit"}:
        audit(datasets)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
