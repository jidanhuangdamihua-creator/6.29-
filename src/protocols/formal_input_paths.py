"""Fail-closed resolver for the only formal D1-D6 input tree."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


FORMAL_SEALED_ROOT_RELATIVE = Path("数据集/固化数据/d1_d6_sealed_v1")
FORMAL_DATASET_IDS = tuple(range(1, 7))


class FormalInputPathError(RuntimeError):
    """A formal input path could not be resolved without fallback."""

    def __init__(
        self,
        code: str,
        *,
        dataset_id: object,
        path: Path | None = None,
        detail: str | None = None,
    ) -> None:
        self.code = str(code)
        self.dataset_id = dataset_id
        self.path = path
        message = f"{self.code} dataset={dataset_id}"
        if path is not None:
            message += f" path={path}"
        if detail:
            message += f" detail={detail}"
        super().__init__(message)


@dataclass(frozen=True)
class FormalDatasetPaths:
    dataset_id: int
    sealed_root: Path
    dataset_root: Path
    source_path: Path
    target_path: Path
    dataset_manifest_path: Path
    source_schema_path: Path
    target_schema_path: Path
    formal_proof_path: Path
    root_manifest_path: Path
    code_inventory_path: Path

    def as_dict(self) -> dict[str, Path | int]:
        return {
            "dataset_id": self.dataset_id,
            "sealed_root": self.sealed_root,
            "dataset_root": self.dataset_root,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "dataset_manifest_path": self.dataset_manifest_path,
            "source_schema_path": self.source_schema_path,
            "target_schema_path": self.target_schema_path,
            "formal_proof_path": self.formal_proof_path,
            "root_manifest_path": self.root_manifest_path,
            "code_inventory_path": self.code_inventory_path,
        }


def _repository_root(repository_root: Path | None) -> Path:
    candidate = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise FormalInputPathError(
            "FORMAL_SEALED_ROOT_MISSING",
            dataset_id="unknown",
            path=candidate,
            detail="repository root does not exist",
        ) from exc


def _resolve_required(
    path: Path,
    *,
    code: str,
    dataset_id: int,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FormalInputPathError(
            code,
            dataset_id=dataset_id,
            path=path,
        ) from exc
    if require_file and not resolved.is_file():
        raise FormalInputPathError(code, dataset_id=dataset_id, path=resolved)
    if require_directory and not resolved.is_dir():
        raise FormalInputPathError(code, dataset_id=dataset_id, path=resolved)
    return resolved


def _require_within_sealed_root(
    path: Path,
    *,
    sealed_root: Path,
    dataset_id: int,
) -> None:
    try:
        path.relative_to(sealed_root)
    except ValueError as exc:
        raise FormalInputPathError(
            "FORMAL_PATH_OUTSIDE_SEALED_ROOT",
            dataset_id=dataset_id,
            path=path,
        ) from exc


def resolve_formal_dataset_paths(
    dataset_id: int,
    *,
    repository_root: Path | None = None,
) -> FormalDatasetPaths:
    """Resolve one dataset from the fixed sealed root, with no fallback."""

    if isinstance(dataset_id, bool):
        dataset = -1
    else:
        try:
            dataset = int(dataset_id)
        except (TypeError, ValueError) as exc:
            raise FormalInputPathError(
                "FORMAL_DATASET_ID_INVALID",
                dataset_id=dataset_id,
            ) from exc
    if dataset not in FORMAL_DATASET_IDS or str(dataset_id).strip() != str(dataset):
        raise FormalInputPathError(
            "FORMAL_DATASET_ID_INVALID",
            dataset_id=dataset_id,
        )
    if ".." in FORMAL_SEALED_ROOT_RELATIVE.parts:
        raise FormalInputPathError(
            "FORMAL_PATH_OUTSIDE_SEALED_ROOT",
            dataset_id=dataset,
            path=FORMAL_SEALED_ROOT_RELATIVE,
        )

    root = _repository_root(repository_root)
    sealed_root = _resolve_required(
        root / FORMAL_SEALED_ROOT_RELATIVE,
        code="FORMAL_SEALED_ROOT_MISSING",
        dataset_id=dataset,
        require_directory=True,
    )
    dataset_root = _resolve_required(
        sealed_root / f"dataset{dataset}",
        code="FORMAL_SEALED_ROOT_MISSING",
        dataset_id=dataset,
        require_directory=True,
    )
    _require_within_sealed_root(
        dataset_root,
        sealed_root=sealed_root,
        dataset_id=dataset,
    )

    required = {
        "source_path": ("source.parquet", "FORMAL_SOURCE_MISSING"),
        "target_path": ("target.parquet", "FORMAL_TARGET_MISSING"),
        "dataset_manifest_path": ("manifest.json", "FORMAL_MANIFEST_MISSING"),
        "source_schema_path": ("source_schema.json", "FORMAL_SCHEMA_MISSING"),
        "target_schema_path": ("target_schema.json", "FORMAL_SCHEMA_MISSING"),
    }
    resolved: dict[str, Path] = {}
    for field, (filename, code) in required.items():
        path = _resolve_required(
            dataset_root / filename,
            code=code,
            dataset_id=dataset,
            require_file=True,
        )
        _require_within_sealed_root(
            path,
            sealed_root=sealed_root,
            dataset_id=dataset,
        )
        resolved[field] = path

    return FormalDatasetPaths(
        dataset_id=dataset,
        sealed_root=sealed_root,
        dataset_root=dataset_root,
        source_path=resolved["source_path"],
        target_path=resolved["target_path"],
        dataset_manifest_path=resolved["dataset_manifest_path"],
        source_schema_path=resolved["source_schema_path"],
        target_schema_path=resolved["target_schema_path"],
        formal_proof_path=dataset_root / "formal-proof.json",
        root_manifest_path=sealed_root / "deployment-manifest.json",
        code_inventory_path=sealed_root / "code-inventory.json",
    )


def resolve_all_formal_dataset_paths(
    *,
    repository_root: Path | None = None,
) -> tuple[FormalDatasetPaths, ...]:
    """Resolve all twelve parquet paths before a formal consumer proceeds."""

    return tuple(
        resolve_formal_dataset_paths(dataset, repository_root=repository_root)
        for dataset in FORMAL_DATASET_IDS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_dataset_identity(paths: FormalDatasetPaths) -> dict[str, object]:
    """Return the shared source/target identity reported by every consumer."""

    return {
        "dataset": f"D{paths.dataset_id}",
        "source_path": str(paths.source_path),
        "target_path": str(paths.target_path),
        "source_sha256": _sha256(paths.source_path),
        "target_sha256": _sha256(paths.target_path),
        "source_size": paths.source_path.stat().st_size,
        "target_size": paths.target_path.stat().st_size,
    }


def resolve_all_formal_dataset_identities(
    *,
    repository_root: Path | None = None,
) -> tuple[dict[str, object], ...]:
    return tuple(
        formal_dataset_identity(paths)
        for paths in resolve_all_formal_dataset_paths(repository_root=repository_root)
    )


def require_explicit_formal_paths(
    dataset_id: int,
    *,
    source_path: Path,
    target_path: Path,
    repository_root: Path | None = None,
) -> FormalDatasetPaths:
    """Verify child-runner arguments are exactly the resolver-selected files."""

    resolved = resolve_formal_dataset_paths(
        dataset_id,
        repository_root=repository_root,
    )
    try:
        explicit = {
            "source": Path(source_path).resolve(strict=True),
            "target": Path(target_path).resolve(strict=True),
        }
    except OSError as exc:
        raise FormalInputPathError(
            "FORMAL_INPUT_RESOLVER_PARITY_MISMATCH",
            dataset_id=dataset_id,
            detail=f"explicit child path is missing: {exc}",
        ) from exc
    expected = {"source": resolved.source_path, "target": resolved.target_path}
    if explicit != expected:
        raise FormalInputPathError(
            "FORMAL_INPUT_RESOLVER_PARITY_MISMATCH",
            dataset_id=dataset_id,
            detail=f"expected={expected} explicit={explicit}",
        )
    return resolved


def formal_input_paths(project_root: Path, dataset_id: int) -> dict[str, Path]:
    """Compatibility view backed exclusively by the unique resolver."""

    paths = resolve_formal_dataset_paths(
        dataset_id,
        repository_root=project_root,
    )
    return {"source": paths.source_path, "target": paths.target_path}
