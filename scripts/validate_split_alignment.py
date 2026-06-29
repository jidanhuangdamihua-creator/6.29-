"""Validate data split protocol against the documented paper reconstruction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.experiment_runner import prepare_base_data_for_experiments
from paper_reproduction_protocol import assess_split_alignment, get_results_output_paths, load_paper_protocol


def main() -> None:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    protocol = load_paper_protocol(cfg)
    output_paths = get_results_output_paths(ROOT, protocol)
    output_paths["alignment_dir"].mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name, dataset_path in cfg["dataset_paths"].items():
        base = prepare_base_data_for_experiments(
            dataset_name=dataset_name,
            data_path=dataset_path,
            config=cfg,
            verbose_mode="summary",
        )
        split = assess_split_alignment(protocol=protocol, base_data=base)
        rows.append({"dataset": dataset_name, **split})

    df = pd.DataFrame(rows)
    out_path = output_paths["alignment_dir"] / "split_alignment_status.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("Split alignment validation completed")
    print(out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()