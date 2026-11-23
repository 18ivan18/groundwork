"""Databases besides the app's, on the same server: created, migrated and emptied here.

The retrieval eval and the CI gate build their indexes in the eval database, named after the app's,
so they never touch the app's data. The tests have a database of their own too.
"""

import asyncio
from urllib.parse import quote, urlsplit

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from groundwork.db.pool import Pool, open_pool


class WrongDatabaseError(RuntimeError):
    """Refusing to empty a database other than the one named."""


def database_name(database_url: str) -> str:
    name = conninfo_to_dict(database_url).get("dbname")
    if not name:
        raise ValueError("The database URL names no database")
    return str(name)


def eval_database_name(app_database_url: str) -> str:
    return f"{database_name(app_database_url)}_eval"


def eval_database_url(app_database_url: str) -> str:
    """The eval database, on the app's server: `groundwork` -> `groundwork_eval`."""
    name = eval_database_name(app_database_url)
    url = urlsplit(app_database_url)._replace(path=f"/{quote(name, safe='')}").geturl()
    # A URL can also name its database in the query string, which would win over the path.
    if database_name(url) != name:
        raise ValueError("The database URL must name its database in the path")
    return url


def create_and_migrate(database_url: str) -> None:
    """Create the database if it is missing and bring it to the latest migration. Blocking."""
    name = database_name(database_url)
    admin = make_conninfo(database_url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True, connect_timeout=3) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
        if not exists:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    config = Config()
    config.set_main_option("script_location", "groundwork.db:migrations")
    # Read by migrations/env.py, in place of DATABASE_URL.
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")


async def empty_database(pool: Pool, name: str) -> None:
    """Delete every document and index. Raises WrongDatabaseError, deleting nothing, unless the
    pool is connected to the database called `name`."""
    async with pool.connection() as connection, connection.transaction():
        cursor = await connection.execute("SELECT current_database()")
        row = await cursor.fetchone()
        if row != (name,):
            raise WrongDatabaseError(f"Refusing to empty {row and row[0]}: it is not {name}")
        await connection.execute("TRUNCATE indexes, documents, collections, users CASCADE")
        # Partial HNSW indexes outlive the rows they were built for.
        cursor = await connection.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'chunks_hnsw_%'"
        )
        for (index_name,) in await cursor.fetchall():
            await connection.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(index_name)))


async def empty_eval_database(pool: Pool, app_database_url: str) -> None:
    """Delete what the last eval run left. Refuses any database but the app's eval database."""
    await empty_database(pool, eval_database_name(app_database_url))


async def open_eval_database(app_database_url: str) -> Pool:
    """A pool on the eval database, created and migrated if needed, and emptied."""
    url = eval_database_url(app_database_url)
    await asyncio.to_thread(create_and_migrate, url)
    pool = await open_pool(url)
    try:
        await empty_eval_database(pool, app_database_url)
    except BaseException:
        await pool.close()
        raise
    return pool
