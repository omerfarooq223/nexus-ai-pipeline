"""Standalone bootstrapper for the MySQL schema and inference indexes."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure script works when run directly from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mysql.connector  # noqa: E402
from mysql.connector import Error  # noqa: E402


TABLE_NAME = "inference_results"
INDEX_QUERIES = [
    (
        "idx_inference_created_at",
        "CREATE INDEX idx_inference_created_at ON inference_results (created_at)",
    ),
    (
        "idx_inference_image_label",
        "CREATE INDEX idx_inference_image_label ON inference_results (image_label)",
    ),
    (
        "idx_inference_sentiment_label",
        "CREATE INDEX idx_inference_sentiment_label ON inference_results (sentiment_label)",
    ),
    (
        "idx_inference_query_text",
        "CREATE FULLTEXT INDEX idx_inference_query_text ON inference_results (query_text)",
    ),
]


def connect(settings, database: str | None = None):
    """Open a MySQL connection using project settings."""
    connection_kwargs = {
        "host": settings.mysql_host,
        "port": settings.mysql_port,
        "user": settings.mysql_user,
        "password": settings.mysql_password,
    }
    if database is not None:
        connection_kwargs["database"] = database
    return mysql.connector.connect(**connection_kwargs)


def ensure_database(settings) -> None:
    """Create the target database if it does not exist."""
    conn = None
    cursor = None
    try:
        conn = connect(settings)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}`")
        conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


def ensure_table(settings) -> None:
    """Create the inference table with the expected columns."""
    conn = None
    cursor = None
    try:
        conn = connect(settings, settings.mysql_database)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_results (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_name VARCHAR(255) NOT NULL,
                query_text TEXT NOT NULL,
                image_label VARCHAR(255) NOT NULL,
                image_confidence FLOAT NOT NULL,
                face_count INT NOT NULL,
                edge_density FLOAT NOT NULL,
                token_count INT NOT NULL,
                sentiment_label VARCHAR(32) NOT NULL,
                sentiment_score FLOAT NOT NULL,
                combined_summary TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


def ensure_index(settings, index_name: str, create_sql: str) -> None:
    """Create an index only when it is missing."""
    conn = None
    cursor = None
    try:
        conn = connect(settings, settings.mysql_database)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(1)
            FROM information_schema.statistics
            WHERE table_schema = %s
              AND table_name = %s
              AND index_name = %s
            """,
            (settings.mysql_database, TABLE_NAME, index_name),
        )
        index_exists = cursor.fetchone()[0] > 0
        if not index_exists:
            cursor.execute(create_sql)
            conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


def main() -> None:
    """Bootstrap the database schema and indexes."""
    try:
        from src.config import settings

        ensure_database(settings)
        ensure_table(settings)
        for index_name, create_sql in INDEX_QUERIES:
            ensure_index(settings, index_name, create_sql)
    except Error as exc:
        raise SystemExit(f"Failed to bootstrap schema or indexes: {exc}") from exc

    print(f"Database ready: {settings.mysql_database}")


if __name__ == "__main__":
    main()