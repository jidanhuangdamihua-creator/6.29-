#!/usr/bin/env python3
"""Diagnose Dataset4 store-SKU date coverage for the 510-day threshold."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/d4_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "数据集" / "原始数据" / "Dataset 4叮咚数据集" / "data" / "train.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "d4_date_coverage_diagnostics"
THRESHOLD = 510
SCAN_START, SCAN_END, SCAN_STEP = 300, 601, 30
CLIFF_PP = 10


def read_d4_table(path: Path) -> pd.DataFrame:
    rename_map = {
        "product_id": "sku_id",
        "dt": "date",
        "second_category_id": "category",
    }
    needed_raw = ["store_id", "product_id", "dt", "second_category_id"]
    needed_clean = ["store_id", "sku_id", "date", "category"]

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        probe_cols = set(pq.ParquetFile(path).schema_arrow.names)
        columns = list(dict.fromkeys(c for c in needed_raw + needed_clean if c in probe_cols))
        df = pd.read_parquet(path, columns=columns or None)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required = {"store_id", "sku_id", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "category" not in df.columns:
        df["category"] = pd.NA

    return df[["store_id", "sku_id", "date", "category"]].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_d4_table(args.data_path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    bad_keys = df[["store_id", "sku_id"]].isna().any(axis=1).sum()
    if bad_keys:
        print(f"WARN: {bad_keys:,} rows have null store_id/sku_id and will be excluded by groupby.")

    coverage = (
        df.groupby(["store_id", "sku_id"])["date"]
        .agg(min_date="min", max_date="max", observed_days="nunique")
        .reset_index()
    )
    coverage["span_days"] = (coverage["max_date"] - coverage["min_date"]).dt.days + 1
    coverage["density"] = coverage["observed_days"] / coverage["span_days"]

    total = len(coverage)
    if total == 0:
        print("ERR: coverage table is empty. Check whether the D4 table loaded correctly.")
        return 1

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    print("=== Coverage distribution (span_days / observed_days / density) ===")
    print(coverage[["span_days", "observed_days", "density"]].quantile(quantiles).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label in zip(
        axes,
        ["span_days", "observed_days"],
        ["span_days", "observed_days"],
    ):
        ax.hist(coverage[col], bins=60, color="#378ADD", edgecolor="white", linewidth=0.4)
        ax.axvline(THRESHOLD, color="#D85A30", linewidth=1.5, linestyle="--", label=f"{THRESHOLD} days")
        ax.set_xlabel(col)
        ax.set_ylabel("entity count")
        ax.set_title(f"D4 store-SKU {label} distribution")
        ax.legend()
    plt.tight_layout()
    hist_path = args.output_dir / "D4_coverage_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close(fig)

    survivors = coverage[coverage["span_days"] >= THRESHOLD]
    n_surv = len(survivors)
    survival_rate = n_surv / total * 100

    print(f"\n=== Survival rate (span_days >= {THRESHOLD} days) ===")
    print(f"Total entities: {total:,}")
    print(f"Surviving entities: {n_surv:,}")
    print(f"Survival rate: {survival_rate:.1f}%")
    if n_surv:
        print(f"Surviving entity density median: {survivors['density'].median():.3f}")
    else:
        print("Surviving entity density median: N/A")

    if df["category"].notna().any():
        cat_all = df.groupby(["store_id", "sku_id"])["category"].first().reset_index()
        cat_surv = survivors.merge(cat_all, on=["store_id", "sku_id"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, data, title in zip(
            axes,
            [
                cat_all["category"].value_counts(normalize=True),
                cat_surv["category"].value_counts(normalize=True),
            ],
            ["All category distribution", f"Survivor category distribution (>={THRESHOLD} days)"],
        ):
            data.plot(kind="bar", ax=ax, color="#378ADD", edgecolor="white")
            ax.set_title(title)
            ax.set_ylabel("share")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        cat_path = args.output_dir / "D4_category_distribution.png"
        plt.savefig(cat_path, dpi=150)
        plt.close(fig)
    else:
        cat_path = None

    print("\n=== Survivor store Top-10 share ===")
    print(survivors["store_id"].value_counts(normalize=True).head(10).to_string())

    thresholds = list(range(SCAN_START, SCAN_END, SCAN_STEP))
    rates = [(coverage["span_days"] >= t).mean() * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds, rates, marker="o", markersize=5, color="#378ADD", linewidth=1.5)
    ax.axvline(THRESHOLD, color="#D85A30", linewidth=1.5, linestyle="--", label=f"current {THRESHOLD}")
    ax.axhline(5, color="#993C1D", linewidth=1, linestyle=":", label="warning 5%")
    ax.axhline(15, color="#1D9E75", linewidth=1, linestyle=":", label="healthy 15%")
    ax.set_xlabel("date_coverage threshold (days)")
    ax.set_ylabel("survival rate (%)")
    ax.set_title("Threshold scan: survival rate vs date_coverage threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    scan_path = args.output_dir / "D4_threshold_scan.png"
    plt.savefig(scan_path, dpi=150)
    plt.close(fig)

    print("\n=== Threshold scan details with cliff detection ===")
    for i, (t, r) in enumerate(zip(thresholds, rates)):
        marker = " <- current" if t == THRESHOLD else ""
        cliff = ""
        if i > 0:
            drop = rates[i - 1] - r
            if drop > CLIFF_PP:
                cliff = f"  CLIFF: {thresholds[i - 1]}->{t} days, drop {drop:.1f}pp"
        print(f"  {t:4d} days: {r:5.1f}%{marker}{cliff}")

    print("\n=== Output files ===")
    print(hist_path)
    if cat_path is not None:
        print(cat_path)
    print(scan_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
