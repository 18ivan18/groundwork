"""The async connection pool. Every connection has the pgvector types registered."""

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

Pool = AsyncConnectionPool[AsyncConnection[TupleRow]]


def sqlalchemy_url(database_url: str) -> str:
    """Alembic goes through SQLAlchemy, which needs the driver named in the URL."""
    scheme, rest = database_url.split("://", 1)
    return f"postgresql+psycopg://{rest}" if scheme in ("postgres", "postgresql") else database_url


async def _configure(connection: AsyncConnection[TupleRow]) -> None:
    await register_vector_async(connection)


async def open_pool(database_url: str, *, min_size: int = 1, max_size: int = 10) -> Pool:
    """Open a pool and wait until its first connection works, so a wrong URL fails at startup."""
    pool: Pool = AsyncConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        open=False,
    )
    await pool.open(wait=True, timeout=10)
    return pool
