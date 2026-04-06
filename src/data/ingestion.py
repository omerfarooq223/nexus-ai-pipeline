"""Data ingestion and cleaning pipeline for large noisy datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Load, clean, optimize types, and persist dataset."""
    df = pd.read_csv(input_path)

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    text_like_cols = [col for col in df.columns if "text" in col or "query" in col]
    for col in text_like_cols:
        df[col] = df[col].astype(str).str.strip()

    df = df.drop_duplicates()
    df = df.dropna(how="all")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def recommend_sql_indexes(table_name: str, columns: list[str]) -> list[str]:
    """Generate SQL index statements based on high-value filter columns."""
    statements = []
    for col in columns:
        idx_name = f"idx_{table_name}_{col}"
        statements.append(f"CREATE INDEX {idx_name} ON {table_name} ({col});")
    return statements


def main() -> None:
    """CLI for dataset ingestion."""
    parser = argparse.ArgumentParser(description="Clean and optimize dataset")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    cleaned = clean_dataset(args.input, args.output)
    print(f"Rows after cleaning: {len(cleaned)}")


if __name__ == "__main__":
    main()
