"""Validate current metric protocol against the documented paper protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_reproduction_protocol import assess_metric_alignment, get_results_output_paths, load_paper_protocol


def main() -> None:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    protocol = load_paper_protocol(cfg)
    metric = assess_metric_alignment(protocol)
    output_paths = get_results_output_paths(ROOT, protocol)
    output_paths["alignment_dir"].mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "alignment_status": metric["metric_alignment_status"],
                "paper_metric_space": metric["paper_metric_space"],
                "current_metric_space": metric["current_metric_space"],
                "notes": metric["metric_alignment_notes"],
            }
        ]
    )
    out_path = output_paths["alignment_dir"] / "metric_alignment_status.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("Metric alignment validation completed")
    print(out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()