#!/usr/bin/env python3
"""Fill missing 5 baselines on the frozen D5/D6 additional-target old 10.

Does not rewrite additional_target_robustness internals, additional_targets_manifest,
source.parquet, old full_run cells, or the new 60-target results.
Does not run SITA. Seasonal Naive / PatchTST / TimesNet already exist in full_run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from additional_target_robustness.protocol import HORIZONS, MANIFEST_SHA256, SEEDS, sha256_file

OLD_MANIFEST = REPO_ROOT / "outputs" / "additional_target_robustness_20261004" / "additional_targets_manifest.json"
OUT_ROOT = REPO_ROOT / "outputs" / "additional_target_robustness_20261004" / "old10_missing_baselines"
MISSING_METHODS = ("LightGBM", "LSTM", "DLinear", "TCN", "MSML-TL")
EXPECTED_PER_METHOD = 10 * len(HORIZONS) * len(SEEDS)
EXPECTED_TOTAL = EXPECTED_PER_METHOD * len(MISSING_METHODS)
OLD_ADDITIONAL = {
    "D5": (
        "48/129296",
        "48/1091365",
        "48/575863",
        "48/1936589",
        "48/2010902",
        "48/162066",
        "48/261052",
        "48/1354380",
        "48/864510",
        "48/567623",
    ),
    "D6": (
        "CA_1/FOODS_3_181",
        "CA_1/FOODS_3_057",
        "CA_1/FOODS_3_784",
        "CA_1/FOODS_3_314",
        "CA_1/FOODS_3_407",
        "CA_1/FOODS_3_284",
        "CA_1/FOODS_3_540",
        "CA_1/FOODS_3_198",
        "CA_1/FOODS_3_773",
        "CA_1/FOODS_3_733",
    ),
}


def _load_bl():
    path = REPO_ROOT / "scripts" / "d5_d6_60target_baselines.py"
    spec = importlib.util.spec_from_file_location("d5_d6_60target_baselines", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_old10(dataset: str) -> list[str]:
    actual = sha256_file(OLD_MANIFEST)
    if actual != MANIFEST_SHA256:
        raise SystemExit(f"HOLD_TARGET_MANIFEST_MISMATCH {actual}")
    records = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))["records"]
    items = [item for item in records if item.get("dataset") == dataset]
    items.sort(key=lambda item: int(item["selected_rank"]))
    identities = [str(item["target_identity"]) for item in items]
    expected = set(OLD_ADDITIONAL[dataset])
    if set(identities) != expected or len(identities) != 10:
        raise SystemExit(f"{dataset} old-10 mismatch live={identities} expected={sorted(expected)}")
    return identities


def patch_dataset(bl, dataset: str) -> Path:
    out = OUT_ROOT / dataset
    bl.DATASETS[dataset]["out"] = out
    bl.DATASETS[dataset]["manifest_sha256"] = MANIFEST_SHA256
    bl.DATASETS[dataset]["manifest_name"] = "additional_targets_manifest.json"
    return out


def audit(bl, dataset: str, selected: tuple[str, ...]) -> dict:
    root = bl._baseline_root(dataset)
    summary = []
    completed_total = 0
    failed_total = 0
    for method in selected:
        slug = bl.METHOD_SLUGS[method]
        cells = root / slug / "cells"
        n_completed = 0
        n_failed = 0
        if cells.is_dir():
            for path in cells.glob("*/DONE.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") == "COMPLETED" and int(payload.get("n_test", 0)) == 180:
                    n_completed += 1
                else:
                    n_failed += 1
        expected = EXPECTED_PER_METHOD
        summary.append({"method": method, "expected": expected, "completed": n_completed, "failed": n_failed})
        completed_total += n_completed
        failed_total += n_failed
    expected_now = EXPECTED_PER_METHOD * len(selected)
    if completed_total == expected_now and failed_total == 0:
        status = "OLD10_MISSING_BL_COMPLETE"
    elif completed_total > 0:
        status = "OLD10_MISSING_BL_PARTIAL"
    else:
        status = "OLD10_MISSING_BL_EMPTY"
    payload = {
        "status": status,
        "dataset": dataset,
        "methods": summary,
        "completed_total": completed_total,
        "failed_total": failed_total,
        "expected_total": expected_now,
        "sita_ran": False,
        "old_full_run_untouched": True,
        "new_60_untouched": True,
        "audited_at": bl._utc_now(),
    }
    bl._atomic_json(root / "baseline_final_audit.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run", "audit"), default="run")
    parser.add_argument("--dataset", choices=("D5", "D6"), required=True)
    parser.add_argument("--methods", default="", help="comma-separated subset of the 5 missing methods")
    args = parser.parse_args()
    selected = tuple(item.strip() for item in args.methods.split(",") if item.strip()) or MISSING_METHODS
    unknown = [method for method in selected if method not in MISSING_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; allowed {MISSING_METHODS}")
    identities = load_old10(args.dataset)
    bl = _load_bl()
    out = patch_dataset(bl, args.dataset)
    root = out / "baselines"
    root.mkdir(parents=True, exist_ok=True)
    log = bl._Log(root / "baseline_run_log.txt")
    log.write(
        f"{args.dataset} old-10 missing BL phase={args.phase} methods={list(selected)} "
        f"targets={identities} out={root.relative_to(REPO_ROOT)}"
    )
    try:
        if args.phase == "run":
            bl.run_cells(
                args.dataset,
                identities=identities,
                methods=selected,
                horizons=HORIZONS,
                seeds=SEEDS,
                log=log,
                msml_blocked="MSML-TL" not in selected,
            )
        payload = audit(bl, args.dataset, selected if args.phase == "audit" else MISSING_METHODS)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        print(payload["status"], flush=True)
        return 0 if payload["status"] in {"OLD10_MISSING_BL_COMPLETE", "OLD10_MISSING_BL_PARTIAL"} else 1
    except (bl.BaselineStop, bl.ProtocolHold, bl.ProtocolError) as exc:
        token = getattr(exc, "token", type(exc).__name__)
        detail = getattr(exc, "detail", str(exc))
        log.write(f"{token}: {detail}")
        bl._atomic_json(root / "HOLD.json", {"status": token, "detail": detail, "written_at": bl._utc_now()})
        print(token, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
