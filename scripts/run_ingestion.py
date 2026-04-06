"""Convenience script for ingestion pipeline execution."""

from src.data.ingestion import clean_dataset

if __name__ == "__main__":
    cleaned_df = clean_dataset(
        input_path="data/noisy_dataset.csv",
        output_path="data/processed/cleaned_dataset.parquet",
    )
    print(f"Cleaned rows: {len(cleaned_df)}")
