"""Convenience script for ingestion pipeline execution."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure script works when run directly from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run the ingestion pipeline from the command line."""
    from src.data.ingestion import clean_dataset

    cleaned_df = clean_dataset(
        input_path="data/noisy_dataset.csv",
        output_path="data/processed/cleaned_dataset.parquet",
    )
    print(f"Cleaned rows: {len(cleaned_df)}")


if __name__ == "__main__":
    main()
