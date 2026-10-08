#!/usr/bin/env python3
"""Fill missing 5 baselines on the frozen D4 additional-target old 10 (store 166).

Does not rewrite additional_target_robustness internals, additional_targets_manifest,
source.parquet, old full_run cells, or the new D4 60-target results.
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
OUT_ROOT = REPO_ROOT / "outputs" / "additional_target_robustness_20261004" / "old10_missing_baselines" / "D4"
MISSING_METHODS = ("LightGBM", "LSTM", "DLinear", "TCN", "MSML-TL")
EXPECTED_PER_METHOD = 10 * len(HORIZONS) * len(SEEDS)
OLD_ADDITIONAL_D4 = (
    "166/246",
    "166/133",
    "166/510",
    "166/244",
    "166/266",
    "166/530",
    "166/297",
    "166/242",
    "166/229",
    "166/307",
)


def _load_bl():
    path = REPO_ROOT / "scripts" / "d4_multistore_baselines.py"
    spec = importlib.util.spec_from_file_location("d4_multistore_baselines", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_old10() -> list[str]:
    actual = sha256_file(OLD_MANIFEST)
    if actual != MANIFEST_SHA256:
        raise SystemExit(f"HOLD_TARGET_MANIFEST_MISMATCH {actual}")
    records = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))["records"]
    items = [item for item in records if item.get("dataset") == "D4"]
    items.sort(key=lambda item: int(item["selected_rank"]))
    identities = [str(item["target_identity"]) for item in items]
    expected = set(OLD_ADDITIONAL_D4)
    if set(identities) != expected or len(identities) != 10:
        raise SystemExit(f"D4 old-10 mismatch live={identities} expected={sorted(expected)}")
    return identities


def patch_bl(bl) -> Path:
    root = OUT_ROOT / "baselines"
    bl.OUT_DIR = OUT_ROOT
    bl.BASELINE_ROOT = root
    bl.MANIFEST_SHA256 = MANIFEST_SHA256
    bl.MANIFEST_PATH = OLD_MANIFEST
    bl.MSML_SOURCE_CACHE = root / "msml_tl" / "source_cnn_cache"
    return root


def audit(bl, selected: tuple[str, ...]) -> dict:
    root = bl.BASELINE_ROOT
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
        summary.append({"method": method, "expected": EXPECTED_PER_METHOD, "completed": n_completed, "failed": n_failed})
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
        "dataset": "D4",
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
    parser.add_argument("--methods", default="", help="comma-separated subset of the 5 missing methods")
    args = parser.parse_args()
    selected = tuple(item.strip() for item in args.methods.split(",") if item.strip()) or MISSING_METHODS
    unknown = [method for method in selected if method not in MISSING_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; allowed {MISSING_METHODS}")
    identities = load_old10()
    bl = _load_bl()
    root = patch_bl(bl)
    root.mkdir(parents=True, exist_ok=True)
    log = bl._Log(root / "baseline_run_log.txt")
    log.write(
        f"D4 old-10 missing BL phase={args.phase} methods={list(selected)} "
        f"targets={identities} out={root.relative_to(REPO_ROOT)}"
    )
    try:
        if args.phase == "run":
            bl.run_cells(
                identities=identities,
                methods=selected,
                horizons=HORIZONS,
                seeds=SEEDS,
                log=log,
                msml_blocked="MSML-TL" not in selected,
            )
        payload = audit(bl, selected if args.phase == "audit" else MISSING_METHODS)
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
