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
    """Thin MySQL wrapper with explicit SQL statements using connection pooling."""

    _pool = None

    @classmethod
    def init_pool(cls, pool_name: str = "mypool", pool_size: int = 5) -> None:
        """Initialize the database connection pool."""
        if cls._pool is None:
            try:
                cls._pool = mysql.connector.pooling.MySQLConnectionPool(
                    pool_name=pool_name,
                    pool_size=pool_size,
                    host=settings.mysql_host,
                    port=settings.mysql_port,
                    user=settings.mysql_user,
                    password=settings.mysql_password,
                    database=settings.mysql_database,
                )
                logger.info("Database connection pool initialized")
            except Error as exc:
                logger.error("Failed to initialize database connection pool: %s", exc)
                raise

    @classmethod
    def get_connection(cls):
        """Get a connection from the pool."""
        if cls._pool is None:
            cls.init_pool()
        return cls._pool.get_connection()

    def insert_inference(self, record: InferenceRecord) -> int:
        """Insert one inference row and return the inserted ID."""
        conn = self.get_connection()
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
            cursor = conn.cursor()
            cursor.execute(sql, values)
            conn.commit()
            inserted_id = cursor.lastrowid
            return int(inserted_id)
        except Error as exc:
            if conn and conn.is_connected():
                conn.rollback()
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
            if conn is not None and conn.is_connected():
                conn.close()

    def healthcheck(self) -> Dict[str, Any]:
        """Run a simple query to validate database availability."""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return {"ok": bool(result and result[0] == 1)}
        except Error as err:
            logger.warning("Healthcheck failed: %s", err)
            return {"ok": False, "error": str(err)}
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()