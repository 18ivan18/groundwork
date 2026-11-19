"""The users table mirrors conf/users.yaml, so documents can reference their owner."""

from collections.abc import Mapping

from groundwork.db.pool import Pool
from groundwork.domain.users import User


async def sync_users(pool: Pool, users: Mapping[str, User]) -> None:
    """Insert or update every user in the file. Users removed from the file stay, so documents
    they own keep a valid owner."""
    async with (
        pool.connection() as connection,
        connection.transaction(),
        connection.cursor() as cursor,
    ):
        await cursor.executemany(
            "INSERT INTO users (id, display_name, groups) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET display_name = EXCLUDED.display_name, "
            "groups = EXCLUDED.groups",
            [(u.user_id, u.display_name, sorted(u.groups)) for u in users.values()],
        )
