from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_dir(root: Path, label: str) -> Path:
    """Atomically create a unique timestamped experiment run directory."""
    safe_label = str(label).replace("/", "_").replace(" ", "_")
    runs_root = Path(root) / "outputs" / "runs"
    for _ in range(100):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = runs_root / f"{run_id}_{safe_label}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique run directory below {runs_root}")


def reserve_new_output_dir(path: Path) -> Path:
    """Create a caller-selected output directory only when it does not exist."""
    output_dir = Path(path)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"output directory already exists and will not be reused: {output_dir}"
        ) from exc
    return output_dir
