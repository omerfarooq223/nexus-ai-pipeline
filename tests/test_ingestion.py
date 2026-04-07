from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.ingestion import clean_dataset, recommend_sql_indexes


def test_clean_dataset_removes_duplicates_and_strips_text(tmp_path: Path) -> None:
    input_path = tmp_path / "dirty.csv"
    output_path = tmp_path / "cleaned.parquet"

    dirty = pd.DataFrame(
        [
            {"Text": " hello ", "value": 1, "timestamp": "2024-01-01"},
            {"Text": "hello", "value": 1, "timestamp": "2024-01-01"},
            {"Text": None, "value": None, "timestamp": None},
        ]
    )
    dirty.to_csv(input_path, index=False)

    cleaned = clean_dataset(str(input_path), str(output_path))

    assert output_path.exists()
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text"] == "hello"


def test_recommend_sql_indexes_formats_expected_statements() -> None:
    statements = recommend_sql_indexes("events", ["created_at", "label"])

    assert statements == [
        "CREATE INDEX idx_events_created_at ON events (created_at);",
        "CREATE INDEX idx_events_label ON events (label);",
    ]