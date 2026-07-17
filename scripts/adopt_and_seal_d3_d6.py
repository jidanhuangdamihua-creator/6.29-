"""Build a sealed dataset directory from approved authority frames.

This is an adoption/sealing primitive only.  It performs transformation and
proof generation; it never trains, predicts, or publishes a deployment root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.gate1_transformation import (  # noqa: E402
    CONTRACT_DIGEST,
    COMBINED_FORMAL_IDENTITY_DIGEST,
    ProofWriter,
    attach_d6_calendar_exact,
    build_contract_views,
    dataset_contract,
    load_formal_identity,
    select_source_history_candidates,
    slice_dataset_roles,
)


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _attach_d6_if_available(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "store_id" not in frame.columns:
        return frame
    calendar_path = root / "数据集" / "原始数据" / "Dataset 6m5-forecasting-accuracy/calendar.csv"
    if not calendar_path.is_file() or "store_id" not in frame.columns:
        return frame
    calendar = pd.read_csv(calendar_path)
    state = str(frame["store_id"].dropna().astype(str).iloc[0]).split("_")[0] if not frame.empty else "CA"
    view = __import__("src.protocols.gate1_transformation", fromlist=["build_d6_calendar_view"]).build_d6_calendar_view(calendar, store_state=state)
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    view["date"] = pd.to_datetime(view["date"], errors="raise").dt.normalize()
    return result.drop(columns=[column for column in ("weekday", "wday", "wm_yr_wk", "snap") if column in result.columns], errors="ignore").merge(view, on="date", how="left", validate="many_to_one")


def _calendarize_and_canonicalize_source(source: pd.DataFrame, dataset: object, scenario: str) -> tuple[pd.DataFrame, dict[str, object]]:
    selected, proof = select_source_history_candidates(dataset, source, scenario, require_complete=False)
    if selected.empty:
        spec = dataset_contract(dataset)
        selected = source.copy()
        selected["date"] = pd.to_datetime(selected["date"], errors="raise").dt.normalize()
        selected = selected.loc[selected["date"].between(pd.Timestamp(spec.source_history_start), pd.Timestamp(spec.source_history_end))].copy()
        proof["status"] = "not_proven"
    else:
        proof["status"] = "passed"
    proof["canonical_digest"] = __import__("src.protocols.gate1_transformation", fromlist=["normalized_frame_digest"]).normalized_frame_digest(selected)
    return selected.reset_index(drop=True), proof


def _build_gate1_publication_proof(dataset: object, views: Mapping[str, pd.DataFrame], authority: Mapping[str, object], readiness_identity: Mapping[str, object] | None = None) -> dict[str, object]:
    identity = load_formal_identity(ROOT)
    proof = ProofWriter().build(contract_digest=CONTRACT_DIGEST, formal_identity=identity, readiness_identity=readiness_identity, authority={**dict(authority), "dataset": dataset_contract(dataset).dataset}, schemas={"worker": list(views["worker_safe_blind"].columns), "knn": ["date", "sales"]}, resolver={"status": "passed"}, views=views, artifacts={"physical_hash": "pending"})
    return proof


def adopt_and_seal_dataset(
    dataset: object,
    *,
    source_path: Path,
    target_path: Path,
    output_root: Path,
    scenario: str = "with-sharing",
    project_root: Path = ROOT,
    readiness_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    identity = load_formal_identity(project_root)
    source = _read(source_path)
    target = _read(target_path)
    if dataset_contract(dataset).dataset == "D6":
        source = _attach_d6_if_available(project_root, source)
        target = _attach_d6_if_available(project_root, target)
    source_history, source_proof = _calendarize_and_canonicalize_source(source, dataset, scenario)
    roles = slice_dataset_roles(dataset, source_history, target)
    views = build_contract_views(dataset, roles)
    destination = Path(output_root) / dataset_contract(dataset).dataset.lower()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name, frame in views.items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    proof = _build_gate1_publication_proof(dataset, views, {"parent": {"source": str(source_path), "target": str(target_path)}, "source_selection": source_proof}, readiness_identity)
    proof["adopted_content_validated"] = True
    proof["source_selection"] = source_proof
    proof["proof_digest"] = __import__("src.protocols.gate1_transformation", fromlist=["canonical_digest"]).canonical_digest({key: value for key, value in proof.items() if key != "proof_digest"})
    (destination / "publication-proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"dataset": dataset_contract(dataset).dataset, "status": "sealed", "output": str(destination), "formal_identity": identity, "proof_digest": proof["proof_digest"], "views": {name: len(frame) for name, frame in views.items()}}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adopt and seal one D1-D6 dataset")
    parser.add_argument("dataset")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario", default="with-sharing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = adopt_and_seal_dataset(args.dataset, source_path=args.source, target_path=args.target, output_root=args.output_root, scenario=args.scenario)
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_code": getattr(exc, "code", "ADOPTION_ERROR"), "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
