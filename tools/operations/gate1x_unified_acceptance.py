#!/usr/bin/env python3
"""The single read-only final acceptance entry point for Gate 1X."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unified_d1_d6 import execute_formal_dry_run  # noqa: E402
from src.protocols.formal_deployment_manifest import (  # noqa: E402
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    DeploymentManifestError,
    atomic_write_json,
    frozen_artifact_snapshot,
    require_repository_identity,
    sha256_file,
    validate_deployment_manifest,
)
from src.protocols.formal_input_paths import FORMAL_SEALED_ROOT_RELATIVE  # noqa: E402


class AcceptanceFailure(RuntimeError):
    def __init__(self, stage: str, code: str, detail: str | None = None) -> None:
        self.stage = stage
        self.code = code
        super().__init__(code if detail is None else f"{code}: {detail}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-full-tests", action="store_true", required=True)
    return parser


def _git_status(root: Path) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "-z"]
    )


def _formal_outputs(root: Path) -> list[str]:
    outputs = root / "outputs"
    if not outputs.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in outputs.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def _metadata_identity(sealed: Path) -> dict[str, object]:
    paths = [
        sealed / "deployment-manifest.json",
        sealed / "deployment-manifest.sha256",
        sealed / "code-inventory.json",
        *(sealed / f"dataset{i}" / "formal-proof.json" for i in range(1, 7)),
    ]
    return {
        path.relative_to(sealed).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def _snapshot(root: Path, sealed: Path) -> dict[str, object]:
    identity = require_repository_identity(root)
    return {
        "branch": identity["branch"],
        "head": identity["head"],
        "git_status_porcelain_v1_z_hex": _git_status(root).hex(),
        "protected_artifacts": frozen_artifact_snapshot(root),
        "root_metadata": _metadata_identity(sealed),
        "formal_outputs": _formal_outputs(root),
    }


def _run_logged(
    command: Sequence[str],
    *,
    root: Path,
    output: Path,
    name: str,
    env: dict[str, str],
) -> None:
    completed = subprocess.run(
        list(command),
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (output / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    atomic_write_json(
        output / f"{name}.result.json",
        {"command": list(command), "exit_code": completed.returncode},
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise AcceptanceFailure(name, f"{name.upper()}_FAILED", f"exit={completed.returncode}")


def _python_files(root: Path) -> list[str]:
    output = subprocess.check_output(
        [
            "git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "--",
            "src", "scripts", "tools/operations", "tests",
        ],
        text=True,
    )
    return sorted(
        path for path in set(output.splitlines())
        if path.endswith(".py") and "__pycache__" not in Path(path).parts
    )


def _shell_files(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--", "*.sh"], text=True
    )
    return sorted(path for path in output.splitlines() if path)


def accept(
    repository_root: Path,
    *,
    expected_branch: str,
    expected_head: str,
    sealed_root: Path,
    output_dir: Path,
    run_full_tests: bool,
) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    sealed = Path(sealed_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    try:
        output.relative_to(Path("/tmp").resolve())
    except ValueError as exc:
        raise AcceptanceFailure("identity", "OUTPUT_DIR_NOT_TMP", str(output)) from exc
    if sealed != (root / FORMAL_SEALED_ROOT_RELATIVE).resolve(strict=True):
        raise AcceptanceFailure("identity", "SEALED_ROOT_MISMATCH")
    require_repository_identity(
        root, expected_branch=expected_branch, expected_head=expected_head
    )
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise AcceptanceFailure("identity", "OUTPUT_DIR_NOT_EMPTY", str(output))

    before = _snapshot(root, sealed)
    atomic_write_json(output / "snapshot-before.json", before)
    try:
        preflight = validate_deployment_manifest(root, sealed_root=sealed)
        atomic_write_json(
            output / "preflight.json",
            {
                key: value
                for key, value in preflight.items()
                if key not in {"manifest", "proofs"}
            },
        )
        plan = execute_formal_dry_run(
            output / "formal-dry-run", project_root=root
        )
        if (
            plan.get("preflight_status") != "ready"
            or plan.get("datasets_ready") != 6
            or plan.get("cell_count") != 300
            or plan.get("unique_cell_count") != 300
            or plan.get("training_started") is not False
            or plan.get("results_created") is not False
            or plan.get("publication_performed") is not False
        ):
            raise AcceptanceFailure("dry-run", "FORMAL_DRY_RUN_ASSERTION_FAILED")

        env = os.environ.copy()
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(output / "pycache"),
                "PYTHONPATH": str(root),
            }
        )
        python = root / ".venv" / "bin" / "python"
        if run_full_tests:
            _run_logged(
                [str(python), "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"],
                root=root,
                output=output,
                name="full-tests",
                env=env,
            )
        python_files = _python_files(root)
        _run_logged(
            [str(python), "-m", "py_compile", *python_files],
            root=root,
            output=output,
            name="py-compile",
            env=env,
        )
        shell_files = _shell_files(root)
        if shell_files:
            _run_logged(
                ["bash", "-n", *shell_files],
                root=root,
                output=output,
                name="shell-syntax",
                env=env,
            )
        _run_logged(
            ["git", "diff", "--check"],
            root=root,
            output=output,
            name="git-diff-check",
            env=env,
        )
    except DeploymentManifestError as exc:
        raise AcceptanceFailure("preflight", exc.code, str(exc)) from exc

    after = _snapshot(root, sealed)
    atomic_write_json(output / "snapshot-after.json", after)
    if after["protected_artifacts"] != before["protected_artifacts"]:
        raise AcceptanceFailure("immutability", "PROTECTED_ARTIFACT_MUTATED")
    if after["root_metadata"] != before["root_metadata"]:
        raise AcceptanceFailure("immutability", "ROOT_METADATA_MUTATED")
    if after["formal_outputs"] != before["formal_outputs"]:
        raise AcceptanceFailure("immutability", "FORMAL_OUTPUTS_MUTATED")
    if after["head"] != before["head"]:
        raise AcceptanceFailure("immutability", "HEAD_MUTATED")
    if after["git_status_porcelain_v1_z_hex"] != before["git_status_porcelain_v1_z_hex"]:
        raise AcceptanceFailure("immutability", "WORKING_TREE_MUTATED")
    report = {
        "status": "accepted",
        "preflight_status": "ready",
        "datasets_ready": 6,
        "datasets_total": 6,
        "cells": 300,
        "unique_cells": 300,
        "training_started": False,
        "results_created": False,
        "publication_performed": False,
        "artifact_immutability": True,
        "working_tree_exact": True,
    }
    atomic_write_json(output / "acceptance.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        accept(
            args.repository_root,
            expected_branch=args.expected_branch,
            expected_head=args.expected_head,
            sealed_root=args.sealed_root,
            output_dir=args.output_dir,
            run_full_tests=args.run_full_tests,
        )
    except AcceptanceFailure as exc:
        traceback.print_exc()
        print("REJECTED")
        print(f"stage={exc.stage}")
        print(f"error_code={exc.code}")
        return 1
    except Exception as exc:
        traceback.print_exc()
        print("REJECTED")
        print("stage=unexpected")
        print(f"error_code={getattr(exc, 'code', type(exc).__name__)}")
        return 1
    print("ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
