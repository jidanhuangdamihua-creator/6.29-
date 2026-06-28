"""Validate source/pretrained-model protocol for paper vs extended tracks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_reproduction_protocol import (
    MULTI_SOURCE_TL_METHODS,
    assess_source_pretrained_alignment,
    get_extended_source_counts,
    get_paper_source_counts,
    get_results_output_paths,
    load_paper_protocol,
    resolve_experiment_track,
)


def main() -> None:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    protocol = load_paper_protocol(cfg)
    output_paths = get_results_output_paths(ROOT, protocol)
    output_paths["alignment_dir"].mkdir(parents=True, exist_ok=True)

    rows = []
    for method_name in ["No-TL", "SS-TL", *sorted(MULTI_SOURCE_TL_METHODS)]:
        requested_counts = [0] if method_name == "No-TL" else ([1] if method_name == "SS-TL" else get_paper_source_counts(protocol) + get_extended_source_counts(protocol))
        for requested_count in requested_counts:
            track = resolve_experiment_track(method_name, int(requested_count), protocol)
            actual_count = 0 if method_name == "No-TL" else (1 if method_name == "SS-TL" else int(requested_count))
            status = assess_source_pretrained_alignment(
                method_name=method_name,
                requested_source_count=int(requested_count),
                actual_pretrained_model_count=actual_count,
                protocol=protocol,
                experiment_track=track,
            )
            rows.append(
                {
                    "method": method_name,
                    "requested_source_count": int(requested_count),
                    "experiment_track": track,
                    **status,
                }
            )

    df = pd.DataFrame(rows)
    out_path = output_paths["alignment_dir"] / "source_pretrained_protocol_status.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("Source/pretrained protocol validation completed")
    print(out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()