#!/usr/bin/env python3
"""Run the 8 formal baselines on the three paper targets only.

Targets:
  D4 166/258
  D5 48/1159414
  D6 CA_1/FOODS_3_080

Matrix: horizons 1,3,5 × seeds 42-46 × 8 baselines
  Seasonal Naive / LightGBM / LSTM / DLinear / TCN / PatchTST / TimesNet / MSML-TL

Classic BL3-BL7 → outputs/paper_targets_8bl_20261005/classic_bl3_bl7/
PatchTST/TimesNet → results/formal_bl9_patchtst and results/formal_bl10_timesnet (skip existing)
MSML-TL → outputs/paper_targets_8bl_20261005/formal_bl8_msml_tl/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baselines"
for path in (REPO_ROOT, BASELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUT_ROOT = REPO_ROOT / "outputs" / "paper_targets_8bl_20261005"
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


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def run_classic(datasets: tuple[str, ...]) -> None:
    import pandas as pd
    from baseline_data_loader import load_baseline_data
    from run_baselines_multiseed import build_run_id, evaluate_entity_protocol, publish_baseline_run

    classic_root = OUT_ROOT / "classic_bl3_bl7"
    classic_root.mkdir(parents=True, exist_ok=True)
    for dataset_id in datasets:
        target = PAPER_TARGETS[dataset_id]
        entity_slices = [item for item in load_baseline_data(dataset_id) if str(item["entity_key"]) == target]
        if len(entity_slices) != 1:
            raise SystemExit(
                f"{dataset_id} expected paper target {target}, got {[item['entity_key'] for item in entity_slices]}"
            )
        run_id = build_run_id(
            "paper8bl",
            dataset_id,
            (target,),
            CLASSIC_METHODS,
            HORIZONS,
            SEEDS,
        )
        evaluated = [
            evaluate_entity_protocol(
                data,
                methods=CLASSIC_METHODS,
                horizons=HORIZONS,
                seeds=SEEDS,
                return_raw=True,
                run_id=run_id,
            )
            for data in entity_slices
        ]
        output = pd.concat([item[0] for item in evaluated], ignore_index=True)
        raw = pd.concat([item[1] for item in evaluated], ignore_index=True)
        expected = len(CLASSIC_METHODS) * len(HORIZONS) * len(SEEDS)
        if len(output) != expected:
            raise SystemExit(f"{dataset_id} classic expected {expected} rows, got {len(output)}")
        dest = classic_root / dataset_id
        # overwrite-friendly isolated publish (not smoke)
        if dest.exists():
            for child in dest.glob("*"):
                if child.is_file():
                    child.unlink()
        publish_baseline_run(output, raw, output_dir=dest, run_id=run_id, smoke=False)
        print(f"{dataset_id} classic BL3-BL7 COMPLETE rows={len(output)} target={target}", flush=True)


def run_modern(datasets: tuple[str, ...]) -> None:
    ds_args = [item.upper() for item in datasets]
    _run(
        [
            sys.executable,
            "scripts/run_bl9_bl10_modern_baselines.py",
            "--datasets",
            *ds_args,
            "--horizons",
            *[str(h) for h in HORIZONS],
            "--seeds",
            *[str(s) for s in SEEDS],
            "--methods",
            *MODERN_METHODS,
            "--skip-existing",
        ]
    )


def run_msml(datasets: tuple[str, ...]) -> None:
    only_keys = []
    for dataset_id in datasets:
        target = PAPER_TARGETS[dataset_id]
        ds = dataset_id.upper()
        for horizon in HORIZONS:
            for seed in SEEDS:
                only_keys.append(f"{ds}|{target}|{horizon}|{seed}")
    cmd = [
        sys.executable,
        "scripts/run_bl8_msml_tl_formal_60cell.py",
        "--output-root",
        str(OUT_ROOT / "formal_bl8_msml_tl"),
        "--only-keys",
        *only_keys,
    ]
    print("+", " ".join(cmd[:6]), f"... ({len(only_keys)} keys)", flush=True)
    # subset matrix may fail final 60-cell promotion audit; cells themselves still run
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode not in {0, 1}:
        raise SystemExit(result.returncode)
    print(f"MSML-TL paper-target subset finished exit={result.returncode}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run", "classic", "modern", "msml"), default="run")
    parser.add_argument("--dataset", choices=("d4", "d5", "d6", "all"), default="all")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "README.txt").write_text(
        "\n".join(
            [
                "Paper-target 8BL for D4/D5/D6",
                "targets: D4=166/258 D5=48/1159414 D6=CA_1/FOODS_3_080",
                "horizons=1,3,5 seeds=42,43,44,45,46",
                "methods=SeasonalNaive,LightGBM,LSTM,DLinear,TCN,PatchTST,TimesNet,MSML-TL",
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
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
