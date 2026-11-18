"""Alembic environment: runs the hand-written SQL migrations against DATABASE_URL."""

from alembic import context
from sqlalchemy import create_engine, pool

from groundwork.config import Settings
from groundwork.db.pool import sqlalchemy_url


def run_migrations_online() -> None:
    engine = create_engine(sqlalchemy_url(Settings().database_url), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=None, transaction_per_migration=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("Offline mode is not supported: the migrations are plain SQL already.")
run_migrations_online()
