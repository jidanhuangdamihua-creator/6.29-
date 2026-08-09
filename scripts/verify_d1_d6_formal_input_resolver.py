#!/usr/bin/env python3
"""Read-only machine verification of the unique D1-D6 formal resolver."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_unified_d1_d6 as unified
from scripts.validate_d1_d6_protocol_inputs import (
    resolve_preflight_formal_input_identity,
)
from src.protocols.formal_input_paths import (
    FORMAL_SEALED_ROOT_RELATIVE,
    formal_dataset_identity,
    resolve_all_formal_dataset_identities,
    resolve_all_formal_dataset_paths,
)
from src.utils.run_artifacts import CodeIdentity
from tools.operations.gate1x_real_input_readiness import (
    resolve_readiness_formal_input_identity,
)


class VerificationFailure(RuntimeError):
    def __init__(self, code: str, *, dataset: str = "ALL", detail: str = "") -> None:
        self.code = code
        self.dataset = dataset
        self.detail = detail
        super().__init__(f"{code} dataset={dataset} detail={detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(paths: Sequence[object]) -> dict[str, tuple[int, int, int, str]]:
    snapshot: dict[str, tuple[int, int, int, str]] = {}
    for resolved in paths:
        for path in (resolved.source_path, resolved.target_path):
            stat = path.stat()
            snapshot[str(path)] = (
                int(pq.ParquetFile(path).metadata.num_rows),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                _sha256(path),
            )
    return snapshot


def _require_equal(
    expected: object,
    actual: object,
    *,
    code: str,
    dataset: str = "ALL",
) -> None:
    if actual != expected:
        raise VerificationFailure(
            code,
            dataset=dataset,
            detail=f"expected={expected!r} actual={actual!r}",
        )


def _verify_exact_matrix(paths: Sequence[object]) -> None:
    expected_root = (ROOT / FORMAL_SEALED_ROOT_RELATIVE).resolve(strict=True)
    for dataset_id, resolved in enumerate(paths, start=1):
        dataset = f"D{dataset_id}"
        _require_equal(expected_root, resolved.sealed_root, code="FORMAL_SEALED_ROOT_MISMATCH", dataset=dataset)
        expected_dataset_root = expected_root / f"dataset{dataset_id}"
        _require_equal(expected_dataset_root, resolved.dataset_root, code="FORMAL_DATASET_ROOT_MISMATCH", dataset=dataset)
        _require_equal(expected_dataset_root / "source.parquet", resolved.source_path, code="FORMAL_SOURCE_PATH_MISMATCH", dataset=dataset)
        _require_equal(expected_dataset_root / "target.parquet", resolved.target_path, code="FORMAL_TARGET_PATH_MISMATCH", dataset=dataset)
        for candidate in (
            resolved.dataset_root,
            resolved.source_path,
            resolved.target_path,
            resolved.dataset_manifest_path,
            resolved.source_schema_path,
            resolved.target_schema_path,
        ):
            try:
                candidate.relative_to(expected_root)
            except ValueError as exc:
                raise VerificationFailure(
                    "FORMAL_PATH_OUTSIDE_SEALED_ROOT",
                    dataset=dataset,
                    detail=str(candidate),
                ) from exc


def _verify_parity(
    paths: Sequence[object],
    identities: Sequence[Mapping[str, object]],
) -> None:
    readiness = tuple(
        resolve_readiness_formal_input_identity(dataset_id, repository_root=ROOT)
        for dataset_id in range(1, 7)
    )
    preflight = tuple(
        resolve_preflight_formal_input_identity(dataset_id, repository_root=ROOT)
        for dataset_id in range(1, 7)
    )
    dry_run = unified.resolve_unified_formal_input_identities(ROOT)
    expected = tuple(dict(item) for item in identities)
    for label, actual in (
        ("readiness", readiness),
        ("preflight", preflight),
        ("unified_dry_run", dry_run),
    ):
        _require_equal(
            expected,
            actual,
            code="FORMAL_INPUT_RESOLVER_PARITY_MISMATCH",
            dataset=label,
        )

    tasks = unified.build_tasks(
        None,
        smoke=False,
        run_dir=Path("/tmp/d1_d6_formal_resolver_verification"),
        repository_root=ROOT,
    )
    by_dataset = {resolved.dataset_id: resolved for resolved in paths}
    for task in tasks:
        resolved = by_dataset[task.dataset_id]
        try:
            source_index = task.cmd.index("--formal-source-path")
            target_index = task.cmd.index("--formal-target-path")
        except ValueError as exc:
            raise VerificationFailure(
                "FORMAL_INPUT_RESOLVER_PARITY_MISMATCH",
                dataset=f"D{task.dataset_id}",
                detail="child runner is missing explicit formal path arguments",
            ) from exc
        _require_equal(str(resolved.source_path), task.cmd[source_index + 1], code="FORMAL_INPUT_RESOLVER_PARITY_MISMATCH", dataset=f"D{task.dataset_id}")
        _require_equal(str(resolved.target_path), task.cmd[target_index + 1], code="FORMAL_INPUT_RESOLVER_PARITY_MISMATCH", dataset=f"D{task.dataset_id}")

    plan = unified.build_run_plan(
        Path("/tmp/d1_d6_formal_resolver_verification"),
        code_identity=CodeIdentity("verification", True, "verification"),
        input_identity={},
    )
    _require_equal(list(expected), plan.get("formal_inputs"), code="FORMAL_INPUT_RESOLVER_PARITY_MISMATCH", dataset="run_plan")


def _verify_frozen_rows_and_hashes(snapshot: Mapping[str, tuple[int, int, int, str]]) -> None:
    expected = {
        (2, "source"): (48654, "04c316a7519e37c6f6712b5c34d25edb38e82833568102a22bd3961081d07409"),
        (2, "target"): (1807, "d2bb78a71cccc0012f0f4f5175d80615565078b0cf7328d6741ab11063ec93c3"),
        (4, "source"): (7935702, "17a1fa5bd1dddfd46bda2a6922ff7821aee2a7e79deca58a94ff7bf20821f7ef"),
        (4, "target"): (3847, "f0b83798ea265c6b79f09487903404c7c75acfcac2657f53e989ef59588e5946"),
    }
    for (dataset_id, role), (rows, digest) in expected.items():
        path = str((ROOT / FORMAL_SEALED_ROOT_RELATIVE / f"dataset{dataset_id}" / f"{role}.parquet").resolve(strict=True))
        actual = snapshot[path]
        _require_equal(rows, actual[0], code="FORMAL_FIXED_ROW_COUNT_MISMATCH", dataset=f"D{dataset_id}:{role}")
        _require_equal(digest, actual[3], code="FORMAL_FIXED_SHA256_MISMATCH", dataset=f"D{dataset_id}:{role}")


def _verify_no_formal_fallback() -> None:
    forbidden = (
        "dataset1-source.parquet",
        "dataset2-source.parquet",
        "dataset3-source.parquet",
        "dataset4-source.parquet",
        "dataset5-source.parquet",
        "dataset6-source.parquet",
        "dataset1-target.parquet",
        "dataset2-target.parquet",
        "d1d2_protocol_v1",
        "数据集/固化数据/dataset",
    )
    formal_consumers = (
        ROOT / "scripts/run_unified_d1_d6.py",
        ROOT / "scripts/run_full_paper_experiments.py",
        ROOT / "scripts/run_d4_experiment.py",
        ROOT / "scripts/run_d5_experiment.py",
        ROOT / "scripts/run_d6_experiment.py",
        ROOT / "scripts/validate_d1_d6_protocol_inputs.py",
        ROOT / "tools/operations/gate1x_real_input_readiness.py",
        ROOT / "src/utils/parquet_data_loader.py",
    )
    for path in formal_consumers:
        text = path.read_text(encoding="utf-8")
        found = [token for token in forbidden if token in text]
        if found:
            raise VerificationFailure(
                "FORMAL_PRODUCTION_FALLBACK_PRESENT",
                dataset="ALL",
                detail=f"path={path} tokens={found}",
            )


def main() -> int:
    try:
        paths = resolve_all_formal_dataset_paths(repository_root=ROOT)
        before = _snapshot(paths)
        _verify_exact_matrix(paths)
        identities = resolve_all_formal_dataset_identities(repository_root=ROOT)
        _verify_parity(paths, identities)
        _verify_frozen_rows_and_hashes(before)
        _verify_no_formal_fallback()
        after = _snapshot(paths)
        if after != before:
            print("FORMAL_INPUT_RESOLVER_VERIFICATION_INVALID — PARQUET MUTATED")
            return 1
    except Exception as exc:
        code = getattr(exc, "code", "FORMAL_INPUT_RESOLVER_VERIFICATION_FAILED")
        dataset = getattr(exc, "dataset", getattr(exc, "dataset_id", "ALL"))
        print(f"{code} dataset={dataset} detail={exc}")
        return 1
    print("D1-D6 FORMAL INPUT RESOLVER VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
