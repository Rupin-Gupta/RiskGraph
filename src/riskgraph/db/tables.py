"""Postgres tables for daily risk results (SPEC §4.8) and the run writer.

Rows carry a run_id: the date for a base run, date+<hash> for a run with overrides, so
incident runs never overwrite the base run for the same date (ADR-006).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import date
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    insert,
)
from sqlalchemy.engine import Engine

metadata = MetaData()


def _common() -> list[Column[Any]]:
    return [
        Column("id", Integer, primary_key=True),
        Column("run_id", String, nullable=False, index=True),
        Column("date", Date, nullable=False, index=True),
    ]


risk_results = Table(
    "risk_results",
    metadata,
    *_common(),
    Column("desk", String, nullable=False),  # desk name or "firm"
    Column("metric", String, nullable=False),
    Column("value", Float, nullable=False),  # USD unless the metric name says otherwise
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("run_id", "desk", "metric"),
)

limit_status = Table(
    "limit_status",
    metadata,
    *_common(),
    Column("scope", String, nullable=False),  # desk name or "firm"
    Column("metric", String, nullable=False),
    Column("value", Float, nullable=False),
    Column("limit", Float, nullable=False),
    Column("utilization", Float, nullable=False),  # value / limit
    Column("status", String, nullable=False),  # ok | warning | breach
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("run_id", "scope", "metric"),
)


def database_url() -> str:
    """DATABASE_URL, else the local Compose Postgres using POSTGRES_PASSWORD. Never log it."""
    if url := os.environ.get("DATABASE_URL"):
        return url
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("set DATABASE_URL or POSTGRES_PASSWORD (see .env.example)")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    return f"postgresql+psycopg://riskgraph:{quote_plus(password)}@{host}:5432/riskgraph"


def get_engine() -> Engine:
    return create_engine(database_url())


def write_run(
    engine: Engine,
    run_id: str,
    day: date,
    risk_rows: Sequence[dict[str, Any]],
    limit_rows: Sequence[dict[str, Any]],
) -> None:
    """Replace this run_id's rows in both tables in one transaction (reruns are idempotent)."""
    with engine.begin() as conn:
        for table, rows in ((risk_results, risk_rows), (limit_status, limit_rows)):
            conn.execute(delete(table).where(table.c.run_id == run_id))
            if rows:
                conn.execute(insert(table), [r | {"run_id": run_id, "date": day} for r in rows])
