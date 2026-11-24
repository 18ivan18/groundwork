"""Shared fixtures."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from groundwork.config import Settings
from groundwork.db import databases
from groundwork.db.pool import Pool, open_pool
from groundwork.ingestion.corpora_bundle import LOCAL_CORPORA_ENV


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's shell variables from changing test results."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    monkeypatch.delenv(LOCAL_CORPORA_ENV, raising=False)


# Postgres for the `db` tests: `docker compose up db` serves this. The tests use their own
# database next to the app's, created and migrated on first use, and empty it before each test.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://groundwork:groundwork@localhost:5434/groundwork_test"
)


@pytest.fixture(scope="session")
def database_url() -> str:
    """The migrated test database. Skips the test without Postgres, or fails it under
    REQUIRE_DB=1, so CI cannot pass by skipping."""
    try:
        databases.create_and_migrate(TEST_DATABASE_URL)
    except psycopg.OperationalError as error:
        if os.environ.get("REQUIRE_DB") == "1":
            raise
        pytest.skip(f"No Postgres at TEST_DATABASE_URL: {error}")
    return TEST_DATABASE_URL


async def recreate(pool: Pool) -> None:
    """Empty the test database and start its ids again, as in a newly created one."""
    async with pool.connection() as connection:
        await connection.execute(f"TRUNCATE {databases.EMPTIED_TABLES} RESTART IDENTITY CASCADE")


@asynccontextmanager
async def empty_database(database_url: str) -> AsyncIterator[Pool]:
    """A pool on the test database, emptied first."""
    opened = await open_pool(database_url, max_size=4)
    try:
        await databases.empty_database(opened, databases.database_name(database_url))
        yield opened
    finally:
        await opened.close()


@pytest.fixture
async def pool(database_url: str) -> AsyncIterator[Pool]:
    async with empty_database(database_url) as opened:
        yield opened


@pytest.fixture
async def hnsw_pool(database_url: str, pool: Pool) -> AsyncIterator[Pool]:
    """One connection that has to use the HNSW index. The test tables are tiny, so the planner
    would rightly sort every chunk by distance; taking the sort away leaves the graph."""
    options = "-c enable_seqscan=off -c enable_sort=off"
    opened = await open_pool(make_conninfo(database_url, options=options), max_size=1)
    try:
        yield opened
    finally:
        await opened.close()
