import pytest

from groundwork.db.pool import Pool
from groundwork.db.users import sync_users
from groundwork.domain.users import User

pytestmark = pytest.mark.db


async def rows(pool: Pool) -> list[tuple[str, str, list[str]]]:
    async with pool.connection() as connection:
        cursor = await connection.execute("SELECT id, display_name, groups FROM users ORDER BY id")
        return [(row[0], row[1], row[2]) for row in await cursor.fetchall()]


async def test_users_are_inserted_then_updated_and_never_removed(pool: Pool) -> None:
    await sync_users(pool, {"bo": User("bo", "Bo", frozenset({"everyone", "hr"}))})
    await sync_users(pool, {"cy": User("cy", "Cy", frozenset({"everyone"}))})
    await sync_users(pool, {"bo": User("bo", "Bo B.", frozenset({"everyone"}))})

    assert await rows(pool) == [
        ("bo", "Bo B.", ["everyone"]),
        ("cy", "Cy", ["everyone"]),
    ]
