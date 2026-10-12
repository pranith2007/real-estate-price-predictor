
"""
Real Estate Price Predictor - Data Preparation
Processes housing data in chunks, cleans prices,
creates features, and exports Parquet files.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_data(input_file, output_dir, price_col):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: calculate price statistics without loading
    # the entire dataset into memory.
    count = 0
    total = 0.0
    total_sq = 0.0

    for chunk in pd.read_csv(input_file, chunksize=100_000):
        if price_col not in chunk.columns:
            raise ValueError(
                f"Price column '{price_col}' not found. "
                f"Available columns: {chunk.columns.tolist()}"
            )

        prices = pd.to_numeric(chunk[price_col], errors="coerce")
        prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
        prices = prices[prices > 0]

        count += len(prices)
        total += prices.sum()
        total_sq += (prices ** 2).sum()

    if count < 2:
        raise ValueError("Not enough valid positive prices.")

    mean = total / count
    variance = max(total_sq / count - mean**2, 0)
    std = np.sqrt(variance)
    lower, upper = mean - 3 * std, mean + 3 * std

    print(f"Mean price: {mean:.2f}")
    print(f"Standard deviation: {std:.2f}")
    print(f"Outlier limits: {lower:.2f} to {upper:.2f}")

    samples = []
    rows_saved = 0

    # Second pass: clean, engineer features and save chunks.
    for number, chunk in enumerate(
        pd.read_csv(input_file, chunksize=100_000)
    ):
        chunk[price_col] = pd.to_numeric(
            chunk[price_col], errors="coerce"
        )
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=[price_col])
        chunk = chunk[chunk[price_col] > 0]
        chunk = chunk[chunk[price_col].between(lower, upper)].copy()

        # Feature 1: age of home
        if "year_built" in chunk.columns:
            year = pd.to_numeric(
                chunk["year_built"], errors="coerce"
            )
            chunk["age_of_home"] = (2026 - year).clip(lower=0)

        # Feature 2: distance to city center, in kilometers
        if {"latitude", "longitude"}.issubset(chunk.columns):
            lat1 = np.radians(chunk["latitude"])
            lon1 = np.radians(chunk["longitude"])
            lat2, lon2 = np.radians(17.3850), np.radians(78.4867)

            a = (
                np.sin((lat1 - lat2) / 2) ** 2
                + np.cos(lat1) * np.cos(lat2)
                * np.sin((lon1 - lon2) / 2) ** 2
            )
            chunk["distance_to_city_km"] = (
                6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
            )

        # Feature 3: price per square foot
        if "area_sqft" in chunk.columns:
            area = pd.to_numeric(
                chunk["area_sqft"], errors="coerce"
            )
            chunk["price_per_sqft"] = (
                chunk[price_col] / area.where(area > 0)
            )

        samples.extend(
            chunk[price_col].sample(
                min(5000, len(chunk)), random_state=42
            ).tolist()
        )

        chunk.to_parquet(
            output_dir / f"part_{number:05d}.parquet",
            index=False,
            compression="snappy",
        )
        rows_saved += len(chunk)

    if not samples:
        raise ValueError("No rows remain after cleaning.")

    # Plot the sampled price distribution.
    sns.histplot(pd.Series(samples), bins=50, kde=True)
    plt.title("House Price Distribution After Cleaning")
    plt.xlabel("House price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "price_distribution.png")
    plt.close()

    print(f"Cleaned rows saved: {rows_saved}")
    print(f"Parquet files and chart saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="cleaned_housing")
    parser.add_argument("--price-col", default="price")
    args = parser.parse_args()

    prepare_data(args.input, args.output, args.price_col)
