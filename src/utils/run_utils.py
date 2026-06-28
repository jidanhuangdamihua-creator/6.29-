from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_dir(root: Path, label: str) -> Path:
    """Create a timestamped experiment run directory."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = str(label).replace("/", "_").replace(" ", "_")
    run_dir = Path(root) / "outputs" / "runs" / f"{run_id}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
