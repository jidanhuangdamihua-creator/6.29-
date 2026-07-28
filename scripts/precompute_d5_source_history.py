"""Precompute and validate the frozen D5 180-day source-history frame.

This script is intentionally offline-only.  It does not modify the D5 runtime
loader or run any experiment cell.  The generated parquet and manifest are
written below the sealed D5 dataset directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Iterator, Mapping, Sequence

import pandas as pd
import psutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import SOURCE_HISTORY_DAYS
from src.protocols.experiment_protocol import get_experiment_protocol
from src.protocols.gate1_transformation import dataset_contract
from src.protocols.source_history import (
    build_exact_source_history_candidate_frame,
    source_history_frame_digest,
)
from src.utils.d5_calendar_reconstruction import (
    load_d5_authorities,
    reconstruct_d5_source_history_calendar,
)
from src.utils.parquet_data_loader import _coerce_known_model_candidate_columns


SEALED_D5_RELATIVE_DIR = Path("数据集/固化数据/d1_d6_sealed_v1/dataset5")
RAW_D5_RELATIVE_DIR = Path("数据集/原始数据/Dataset 5Favorita")
OUTPUT_FILENAME = "prepared_180day_source.parquet"
MANIFEST_FILENAME = "prepared_180day_source_manifest.json"
DISK_SPACE_SAFETY_FACTOR = 3.0
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.2
MIN_ESTIMATED_OUTPUT_BYTES = 64 * 1024 * 1024


def _iso_date(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"invalid date value: {value!r}")
    return timestamp.normalize().date().isoformat()


def _display_path(path: Path, *, repository_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Return the ordered, JSON-safe pandas column schema."""
    return [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in frame.dtypes.items()
    ]


def _file_record(path: Path, *, repository_root: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required input file does not exist: {path}")
    return {
        "path": _display_path(path, repository_root=repository_root),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def build_output_manifest(
    *,
    source_path: Path,
    auxiliary_files: Mapping[str, Path],
    output_path: Path,
    frame: pd.DataFrame,
    source_history_start: object,
    source_history_end: object,
    source_history_days: int,
    key_fields: Sequence[str],
    generated_at: str,
    source_history_frame_digest: str,
    synthetic_row_count: int,
    repository_root: Path | None = None,
    elapsed_seconds: float | None = None,
    memory_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the manifest after the output parquet has been atomically written."""
    root = Path(repository_root or PROJECT_ROOT).resolve()
    output_path = Path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"output parquet does not exist: {output_path}")

    artifact = {
        "path": _display_path(output_path, repository_root=root),
        "sha256": sha256_file(output_path),
        "size_bytes": int(output_path.stat().st_size),
        "row_count": int(len(frame)),
        "schema": frame_schema(frame),
    }
    manifest: dict[str, object] = {
        "manifest_version": "d5_precomputed_source_history_v1",
        "dataset_id": "D5",
        "generated_at": str(generated_at),
        "source": _file_record(Path(source_path), repository_root=root),
        "auxiliary_csvs": {
            str(name): _file_record(Path(path), repository_root=root)
            for name, path in sorted(auxiliary_files.items())
        },
        "source_history": {
            "source_history_days": int(source_history_days),
            "source_history_start": _iso_date(source_history_start),
            "source_history_end": _iso_date(source_history_end),
            "key_fields": [str(field) for field in key_fields],
            "source_history_frame_digest": str(source_history_frame_digest),
            "synthetic_row_count": int(synthetic_row_count),
        },
        "artifact": artifact,
    }
    if elapsed_seconds is not None or memory_report is not None:
        manifest["execution"] = {
            "elapsed_seconds": (
                round(float(elapsed_seconds), 3)
                if elapsed_seconds is not None
                else None
            ),
            "memory_report": dict(memory_report) if memory_report is not None else None,
        }
    return manifest


def validate_precomputed_output(output_path: Path, manifest: Mapping[str, object]) -> None:
    """Re-read the output and fail if its bytes, row count, or schema drifted."""
    output_path = Path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"precomputed output is missing: {output_path}")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("manifest artifact section is missing")

    actual_hash = sha256_file(output_path)
    expected_hash = str(artifact.get("sha256", ""))
    if actual_hash != expected_hash:
        raise ValueError(
            "precomputed output SHA-256 mismatch: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    expected_size = int(artifact.get("size_bytes", -1))
    actual_size = int(output_path.stat().st_size)
    if actual_size != expected_size:
        raise ValueError(
            "precomputed output size mismatch: "
            f"expected={expected_size} actual={actual_size}"
        )

    frame = pd.read_parquet(output_path)
    expected_rows = int(artifact.get("row_count", -1))
    if len(frame) != expected_rows:
        raise ValueError(
            "precomputed output row count mismatch: "
            f"expected={expected_rows} actual={len(frame)}"
        )
    expected_schema = artifact.get("schema")
    actual_schema = frame_schema(frame)
    if actual_schema != expected_schema:
        raise ValueError(
            "precomputed output schema mismatch: "
            f"expected={expected_schema!r} actual={actual_schema!r}"
        )


@dataclass
class _MemoryStep:
    name: str
    before_rss_bytes: int
    after_rss_bytes: int = 0
    peak_rss_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.name,
            "before_rss_bytes": int(self.before_rss_bytes),
            "after_rss_bytes": int(self.after_rss_bytes),
            "peak_rss_bytes": int(self.peak_rss_bytes),
            "before_rss_mib": round(self.before_rss_bytes / 2**20, 2),
            "after_rss_mib": round(self.after_rss_bytes / 2**20, 2),
            "peak_rss_mib": round(self.peak_rss_bytes / 2**20, 2),
        }


class MemoryMonitor:
    """Sample this process RSS and retain per-step and total peaks."""

    def __init__(self, interval_seconds: float = MEMORY_SAMPLE_INTERVAL_SECONDS) -> None:
        if interval_seconds <= 0:
            raise ValueError("memory sample interval must be positive")
        self._interval_seconds = float(interval_seconds)
        self._process = psutil.Process(os.getpid())
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_step: _MemoryStep | None = None
        self._steps: list[_MemoryStep] = []
        self._total_peak_bytes = 0

    def _rss_bytes(self) -> int:
        return int(self._process.memory_info().rss)

    def _sample(self) -> int:
        rss = self._rss_bytes()
        with self._lock:
            self._total_peak_bytes = max(self._total_peak_bytes, rss)
            if self._current_step is not None:
                self._current_step.peak_rss_bytes = max(
                    self._current_step.peak_rss_bytes,
                    rss,
                )
        return rss

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("memory monitor already started")
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="d5-precompute-memory-monitor",
            daemon=True,
        )
        self._thread.start()

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        if self._thread is None:
            raise RuntimeError("memory monitor must be started before a step")
        if self._current_step is not None:
            raise RuntimeError(f"memory step is already active: {self._current_step.name}")
        before = self._sample()
        self._current_step = _MemoryStep(
            name=str(name),
            before_rss_bytes=before,
            peak_rss_bytes=before,
        )
        try:
            yield
        finally:
            after = self._sample()
            with self._lock:
                assert self._current_step is not None
                self._current_step.after_rss_bytes = after
                self._current_step.peak_rss_bytes = max(
                    self._current_step.peak_rss_bytes,
                    after,
                )
                self._steps.append(self._current_step)
                self._current_step = None

    def stop(self) -> dict[str, object]:
        if self._thread is None:
            raise RuntimeError("memory monitor has not been started")
        self._sample()
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self._interval_seconds * 5))
        self._sample()
        with self._lock:
            steps = [step.to_dict() for step in self._steps]
            total_peak = int(self._total_peak_bytes)
        return {
            "sample_interval_seconds": self._interval_seconds,
            "steps": steps,
            "total_peak_rss_bytes": total_peak,
            "total_peak_rss_mib": round(total_peak / 2**20, 2),
        }


def estimate_output_size_bytes(source_path: Path) -> int:
    """Use the sealed source parquet size as a conservative output estimate."""
    source_size = int(Path(source_path).stat().st_size)
    return max(source_size, MIN_ESTIMATED_OUTPUT_BYTES)


def check_disk_space(
    output_directory: Path,
    *,
    estimated_output_bytes: int,
    safety_factor: float = DISK_SPACE_SAFETY_FACTOR,
) -> dict[str, object]:
    """Fail before data loading when the output volume lacks safety space."""
    output_directory = Path(output_directory)
    if not output_directory.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output_directory}")
    if estimated_output_bytes <= 0 or safety_factor < 1:
        raise ValueError("invalid disk-space estimate or safety factor")
    usage = shutil.disk_usage(output_directory)
    required = int(math.ceil(estimated_output_bytes * safety_factor))
    if int(usage.free) < required:
        raise RuntimeError(
            "insufficient disk space for D5 precompute: "
            f"free={usage.free} required={required} "
            f"(estimate={estimated_output_bytes}, safety_factor={safety_factor:g})"
        )
    return {
        "output_directory": str(output_directory),
        "free_bytes": int(usage.free),
        "estimated_output_bytes": int(estimated_output_bytes),
        "required_free_bytes": required,
        "safety_factor": float(safety_factor),
    }


def _atomic_write_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    serializable_frame = frame.copy(deep=False)
    # Runtime attrs contain Timestamp objects, eligible-key metadata, and
    # other audit state.  They are restored by the future runtime loader from
    # the manifest; they must not be serialized into parquet metadata here.
    serializable_frame.attrs = {}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        serializable_frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _resolve_path(value: Path | None, *, repository_root: Path, default: Path) -> Path:
    path = Path(value) if value is not None else default
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _default_paths(repository_root: Path) -> tuple[Path, Path, Path, Path]:
    sealed_dir = repository_root / SEALED_D5_RELATIVE_DIR
    raw_dir = repository_root / RAW_D5_RELATIVE_DIR
    return (
        raw_dir,
        sealed_dir / "source.parquet",
        sealed_dir / OUTPUT_FILENAME,
        sealed_dir / MANIFEST_FILENAME,
    )


def _print_memory_report(report: Mapping[str, object]) -> None:
    print("\n[D5 precompute memory report]")
    for item in report.get("steps", []):
        if not isinstance(item, Mapping):
            continue
        print(
            "  {step}: before={before:.2f} MiB after={after:.2f} MiB peak={peak:.2f} MiB".format(
                step=item.get("step", "unknown"),
                before=float(item.get("before_rss_mib", 0.0)),
                after=float(item.get("after_rss_mib", 0.0)),
                peak=float(item.get("peak_rss_mib", 0.0)),
            )
        )
    print(
        "  total_peak: {peak:.2f} MiB".format(
            peak=float(report.get("total_peak_rss_mib", 0.0))
        )
    )


def run_precompute(
    *,
    repository_root: Path = PROJECT_ROOT,
    raw_dir: Path | None = None,
    source_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Run the one-time D5 source-history precomputation."""
    root = Path(repository_root).resolve()
    default_raw, default_source, default_output, default_manifest = _default_paths(root)
    raw_dir = _resolve_path(raw_dir, repository_root=root, default=default_raw)
    source_path = _resolve_path(source_path, repository_root=root, default=default_source)
    output_path = _resolve_path(output_path, repository_root=root, default=default_output)
    manifest_path = _resolve_path(
        manifest_path,
        repository_root=root,
        default=default_manifest,
    )

    if not source_path.is_file():
        raise FileNotFoundError(f"sealed D5 source parquet does not exist: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing D5 precompute output or manifest: "
            f"output={output_path} manifest={manifest_path}"
        )
    disk_report = check_disk_space(
        output_path.parent,
        estimated_output_bytes=estimate_output_size_bytes(source_path),
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    monitor = MemoryMonitor()
    memory_report: dict[str, object] = {}
    try:
        monitor.start()
        with monitor.step("load_d5_authorities"):
            authorities = load_d5_authorities(raw_dir, use_holidays=True)

        source_history_end = pd.Timestamp(dataset_contract("D5").origin).normalize()
        source_history_start = source_history_end - pd.Timedelta(
            days=int(SOURCE_HISTORY_DAYS) - 1
        )
        source_history_dates = pd.date_range(
            source_history_start,
            source_history_end,
            freq="D",
        )

        with monitor.step("读取source.parquet后"):
            source_filters = [
                ("date", ">=", source_history_start),
                ("date", "<=", source_history_end),
            ]
            source_df = pd.read_parquet(source_path, filters=source_filters)
            source_df = _coerce_known_model_candidate_columns(
                source_df,
                dataset_id=5,
                role="source",
            )
            source_df["date"] = pd.to_datetime(source_df["date"], errors="coerce")
            if source_df["date"].isna().any():
                raise ValueError("D5 source dataframe contains invalid date values")
            source_df = source_df.loc[
                source_df["date"].between(
                    source_history_start,
                    source_history_end,
                    inclusive="both",
                )
            ].copy()

        with monitor.step("calendarization重建后"):
            reconstructed, reconstruction_report = reconstruct_d5_source_history_calendar(
                source_df,
                expected_dates=source_history_dates,
                authorities=authorities,
            )

        key_fields = tuple(get_experiment_protocol(5).source_pool_rule.key_fields)
        with monitor.step("candidate frame构建后"):
            eligibility = build_exact_source_history_candidate_frame(
                reconstructed,
                key_fields=key_fields,
                origin=source_history_end,
                source_history_days=int(SOURCE_HISTORY_DAYS),
            )
            candidate_frame = eligibility.candidate_frame
            candidate_digest = source_history_frame_digest(
                candidate_frame,
                key_fields=key_fields,
            )

        with monitor.step("写入prepared parquet"):
            _atomic_write_parquet(candidate_frame, output_path)
        elapsed_before_manifest = time.perf_counter() - started
        memory_report = monitor.stop()
        manifest = build_output_manifest(
            source_path=source_path,
            auxiliary_files={
                name: root / evidence.path
                if not Path(evidence.path).is_absolute()
                else Path(evidence.path)
                for name, evidence in authorities.files.items()
                if evidence.used
            },
            output_path=output_path,
            frame=candidate_frame,
            source_history_start=source_history_start,
            source_history_end=source_history_end,
            source_history_days=int(SOURCE_HISTORY_DAYS),
            key_fields=key_fields,
            generated_at=generated_at,
            source_history_frame_digest=candidate_digest,
            synthetic_row_count=int(reconstruction_report.synthetic_row_count),
            repository_root=root,
            elapsed_seconds=elapsed_before_manifest,
            memory_report=memory_report,
        )
        _atomic_write_json(manifest, manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            persisted_manifest = json.load(handle)
        validate_precomputed_output(output_path, persisted_manifest)
    finally:
        if not memory_report and monitor._thread is not None:
            memory_report = monitor.stop()
        if memory_report:
            _print_memory_report(memory_report)

    elapsed_seconds = time.perf_counter() - started
    print(f"\n[D5 precompute] elapsed_seconds={elapsed_seconds:.3f}")
    print(f"[D5 precompute] output={output_path}")
    print(f"[D5 precompute] manifest={manifest_path}")
    print("[D5 precompute] disk-space preflight:")
    print(json.dumps(disk_report, ensure_ascii=False, indent=2))
    print("[D5 precompute] manifest:")
    print(json.dumps(persisted_manifest, ensure_ascii=False, indent=2))
    return persisted_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute the sealed D5 180-day source-history candidate frame."
    )
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_precompute(
        repository_root=args.repository_root,
        raw_dir=args.raw_dir,
        source_path=args.source_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
