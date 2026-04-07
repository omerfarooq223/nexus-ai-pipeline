from __future__ import annotations

import pytest

pytest.importorskip("mysql.connector")

from src.db.mysql_client import MySQLClient, init_db


def test_init_db_uses_healthcheck(monkeypatch) -> None:
    monkeypatch.setattr(MySQLClient, "init_pool", lambda: None)
    monkeypatch.setattr(MySQLClient, "healthcheck", lambda self: {"ok": True})

    assert init_db() == {"ok": True}