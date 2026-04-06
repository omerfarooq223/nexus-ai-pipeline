"""MySQL client and query helpers for inference persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import logging

import mysql.connector
from mysql.connector import Error

from src.config import settings


logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    """Typed payload used for inserts into MySQL."""

    image_name: str
    query_text: str
    image_label: str
    image_confidence: float
    face_count: int
    edge_density: float
    token_count: int
    sentiment_label: str
    sentiment_score: float
    combined_summary: str


class MySQLClient:
    """Thin MySQL wrapper with explicit SQL statements."""

    def __init__(self) -> None:
        self.conn = None

    def connect(self) -> None:
        """Open a database connection."""
        if self.conn is None or not self.conn.is_connected():
            self.conn = mysql.connector.connect(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
            )

    def close(self) -> None:
        """Close database connection if open."""
        if self.conn and self.conn.is_connected():
            self.conn.close()

    def insert_inference(self, record: InferenceRecord) -> int:
        """Insert one inference row and return the inserted ID."""
        self.connect()
        sql = """
            INSERT INTO inference_results (
                image_name,
                query_text,
                image_label,
                image_confidence,
                face_count,
                edge_density,
                token_count,
                sentiment_label,
                sentiment_score,
                combined_summary
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            record.image_name,
            record.query_text,
            record.image_label,
            record.image_confidence,
            record.face_count,
            record.edge_density,
            record.token_count,
            record.sentiment_label,
            record.sentiment_score,
            record.combined_summary,
        )

        cursor = None
        try:
            logger.info("Inserting inference for image_name=%s", record.image_name)
            cursor = self.conn.cursor()
            cursor.execute(sql, values)
            self.conn.commit()
            inserted_id = cursor.lastrowid
            return int(inserted_id)
        except Error as exc:
            if self.conn and self.conn.is_connected():
                self.conn.rollback()
            logger.error(
                "Insert failed for image_name=%s: %s",
                record.image_name,
                exc,
                exc_info=True,
            )
            raise
        finally:
            if cursor is not None:
                cursor.close()
            self.close()

    def healthcheck(self) -> Dict[str, Any]:
        """Run a simple query to validate database availability."""
        cursor = None
        try:
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return {"ok": bool(result and result[0] == 1)}
        except Error as err:
            logger.warning("Healthcheck failed: %s", err)
            return {"ok": False, "error": str(err)}
        finally:
            if cursor is not None:
                cursor.close()
            self.close()