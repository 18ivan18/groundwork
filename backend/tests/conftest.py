"""Shared fixtures."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from groundwork.config import Settings
from groundwork.db.pool import Pool, open_pool


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's shell variables from changing test results."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)


# Postgres for the `db` tests: `docker compose up db` serves this. The tests use their own
# database next to the app's, created and migrated on first use, and empty it before each test.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://groundwork:groundwork@localhost:5434/groundwork_test"
)


@pytest.fixture(scope="session")
def database_url() -> str:
    """The migrated test database. Skips the test without Postgres, or fails it under
    REQUIRE_DB=1, so CI cannot pass by skipping."""
    admin = make_conninfo(TEST_DATABASE_URL, dbname="postgres")
    name = conninfo_to_dict(TEST_DATABASE_URL)["dbname"]
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as connection:
            exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
            if not exists:
                connection.execute(f'CREATE DATABASE "{name}"')
    except psycopg.OperationalError as error:
        if os.environ.get("REQUIRE_DB") == "1":
            raise
        pytest.skip(f"No Postgres at TEST_DATABASE_URL: {error}")
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    try:
        command.upgrade(AlembicConfig(str(Path(__file__).parents[1] / "alembic.ini")), "head")
    finally:
        if previous is None:
            del os.environ["DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = previous
    return TEST_DATABASE_URL


@asynccontextmanager
async def empty_database(database_url: str) -> AsyncIterator[Pool]:
    """A pool on the test database, emptied first."""
    opened = await open_pool(database_url, max_size=4)
    async with opened.connection() as connection:
        await connection.execute("TRUNCATE indexes, documents, collections, users CASCADE")
        # Partial HNSW indexes outlive the rows they were built for.
        cursor = await connection.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'chunks_hnsw_%'"
        )
        for (index_name,) in await cursor.fetchall():
            await connection.execute(f'DROP INDEX "{index_name}"')
    try:
        yield opened
    finally:
        await opened.close()


@pytest.fixture
async def pool(database_url: str) -> AsyncIterator[Pool]:
    async with empty_database(database_url) as opened:
        yield opened
