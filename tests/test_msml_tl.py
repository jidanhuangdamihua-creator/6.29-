"""
Test script for Module 8: MSML-TL

Minimal runnable test for multi-source multi-layer transfer learning.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from environment import setup_logging
from data_preprocessing import build_source_target_split, extract_datetime_features, load_dataset
from msml_tl import run_msml_tl


def main() -> None:
    """Run a minimal MSML-TL test on Dataset1."""
    setup_logging(log_level="INFO", log_file=None)

    config = Config(config_file="config.yaml", supply_chain_file="supply_chain.yaml", verbose=True)
    csv_path = str(Path(str(config.dataset.path)) / "train.csv")

    df = load_dataset(dataset_name="Dataset1", data_path=csv_path)
    df = extract_datetime_features(df)

    source_df, target_df = build_source_target_split(df, config)

    feature_cols = ["sales", "year", "month", "week", "day"]
    feature_cols = [c for c in feature_cols if c in source_df.columns and c in target_df.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns for MSML-TL.")

    result = run_msml_tl(
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        k=3,
        horizon=1,
        window_size=10,
        weight_mode="inverse_distance",
        source_epochs=2,
        target_epochs=2,
        batch_size=16,
    )

    print("\nMSML-TL Completed Successfully")
    print()

    print("Selected Sources:")
    for i, info in enumerate(result["source_models_info"], start=1):
        print(
            f"  {i}. source_key={info['source_key']} "
            f"distance={info['distance']:.4f} "
            f"weight={info['weight']:.4f}"
        )
    print()

    print(f"Fused Layers:\n  {result['meta']['fused_layers']}")
    print()

    print(f"Frozen Layers:\n  {result['frozen_layers']}")
    print()

    print("Fused Result:")
    print(f"  RMSE: {result['fused_result']['rmse']:.4f}")
    print(f"  Accuracy: {result['fused_result']['accuracy']:.4f}")
    print(f"  Prediction Shape: {result['fused_result']['prediction_shape']}")


if __name__ == "__main__":
    main()
