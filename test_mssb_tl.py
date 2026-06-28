"""
Test script for Module 7: MSSB-TL

Minimal runnable test for multi-source switching-based transfer learning.
"""

from __future__ import annotations

from pathlib import Path

from config import Config
from environment import setup_logging
from data_preprocessing import build_source_target_split, extract_datetime_features, load_dataset
from mssb_tl import run_mssb_tl


def main() -> None:
    """Run a minimal MSSB-TL test on Dataset1."""
    setup_logging(log_level="INFO", log_file=None)

    config = Config(config_file="config.yaml", supply_chain_file="supply_chain.yaml", verbose=True)
    csv_path = str(Path(str(config.dataset.path)) / "train.csv")

    df = load_dataset(dataset_name="Dataset1", data_path=csv_path)
    df = extract_datetime_features(df)

    source_df, target_df = build_source_target_split(df, config)

    feature_cols = ["sales", "year", "month", "week", "day"]
    feature_cols = [c for c in feature_cols if c in source_df.columns and c in target_df.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns for MSSB-TL.")

    result = run_mssb_tl(
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

    print("MSSB-TL Completed Successfully")
    print()

    print("Selected Sources:")
    selected_sources = result["meta"]["selected_sources"]
    for i, src in enumerate(selected_sources, start=1):
        print(
            f"{i}. source_key={src['source_key']} "
            f"distance={src['distance']:.4f} "
            f"weight={src['weight']:.4f}"
        )
    print()

    print("Individual Source Results:")
    for i, one in enumerate(result["individual_results"], start=1):
        print(
            f"{i}. source_key={one['source_key']} "
            f"val_rmse={one['val_rmse']:.4f} "
            f"val_accuracy={one['val_accuracy']:.4f} "
            f"test_rmse={one['test_rmse']:.4f} "
            f"test_accuracy={one['test_accuracy']:.4f} "
            f"prediction_shape={one['prediction_shape']}"
        )
    print()

    best = result["best_source_result"]
    print("Best Source Selected:")
    print(f"source_key={best['source_key']}")
    print(f"val_rmse={best['val_rmse']:.4f}")
    print(f"test_rmse={best['test_rmse']:.4f}")
    print(f"test_accuracy={best['test_accuracy']:.4f}")
    print()

    print("Final Result:")
    print(f"RMSE: {result['final_result']['rmse']:.4f}")
    print(f"Accuracy: {result['final_result']['accuracy']:.4f}")
    print(f"Prediction Shape: {result['final_result']['prediction_shape']}")


if __name__ == "__main__":
    main()
