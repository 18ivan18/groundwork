"""EXPLAIN ANALYZE the dense query the store runs, for docs/RETRIEVAL.md.

Run from backend/: uv run python scripts/explain_search.py [--index NAME] [--query TEXT]

Prints four plans for one index (the active one unless --index names another):
1. the plain top-20 query,
2. the same with a selective collection filter and pgvector's iterative scan,
3. the plain query at two hnsw.ef_search values, with the overlap of their results,
4. the sixth EXECUTE of a prepared statement, the first that may use a generic plan. psycopg
   prepares a query on its fifth run on a connection, so this is what the API's pool runs.
"""

import argparse
import asyncio
import re
from typing import Any

import numpy as np
from psycopg import AsyncClientCursor, AsyncConnection

from groundwork.config import Settings
from groundwork.container import build_embedder
from groundwork.db.pool import open_pool
from groundwork.domain.models import SearchFilters
from groundwork.retrieval.indexes import active_index, get_index
from groundwork.retrieval.pgvector_store import PgVectorStore

N = 20


def plan_text(rows: list[Any]) -> str:
    """The plan, with the query vector's 384 numbers shown as <query>."""
    return re.sub(r"'\[[-0-9.e, ]+\]'", "<query>", "\n".join(row[0] for row in rows))


async def explain(connection: AsyncConnection[Any], sql: str, params: list[Any]) -> str:
    cursor = await connection.execute(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) {sql}", params)
    return plan_text(await cursor.fetchall())


async def ids(connection: AsyncConnection[Any], sql: str, params: list[Any]) -> list[str]:
    cursor = await connection.execute(sql, params)
    # Column 0 is the document id and the chunk ordinal follows the document columns.
    return [f"{row[0]}:{row[10]}" for row in await cursor.fetchall()]


def numbered(sql: str) -> str:
    """psycopg's %s placeholders as PREPARE's $1, $2, ..."""
    count = iter(range(1, 100))
    return re.sub(r"%s", lambda _: f"${next(count)}", sql)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index")
    parser.add_argument("--query", default="How many weeks of parental leave do employees get?")
    parser.add_argument("--collection", default="northwind")
    args = parser.parse_args()

    settings = Settings()
    embedder = build_embedder(settings)
    vector = embedder.embed_query(args.query)
    pool = await open_pool(settings.database_url)
    try:
        index = await (get_index(pool, args.index) if args.index else active_index(pool))
        assert index is not None, "no such index"
        store = PgVectorStore(pool, index)
        print(f"Index {index.name} (id {index.id}), {await store.count_chunks()} chunks\n")
        plain_sql, plain = store.search_sql(vector, N)
        filtered_sql, filtered = store.search_sql(
            vector, N, SearchFilters(collections=(args.collection,))
        )
        async with pool.connection() as connection:
            await connection.execute(plain_sql, plain)  # warm the cache
            print("## 1. Top 20, no filter\n")
            print(await explain(connection, plain_sql, plain))

            print(f"\n## 2. Top 20 in {args.collection}, iterative scan\n")
            async with connection.transaction():
                await connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
                print(await explain(connection, filtered_sql, filtered))
            async with connection.transaction():
                await connection.execute("SET LOCAL hnsw.iterative_scan = off")
                found = await ids(connection, filtered_sql, filtered)
                print(f"\nWithout the iterative scan: {len(found)} rows of {N}.")

            print("\n## 3. hnsw.ef_search\n")
            async with connection.transaction():
                await connection.execute("SET LOCAL enable_indexscan = off")
                exact = await ids(connection, plain_sql, plain)
            for ef in (40, 100):
                async with connection.transaction():
                    await connection.execute(f"SET LOCAL hnsw.ef_search = {ef}")
                    plan = await explain(connection, plain_sql, plain)
                    found = await ids(connection, plain_sql, plain)
                time = re.search(r"Execution Time: .*", plan)
                overlap = len(set(found) & set(exact))
                print(f"ef_search {ef}: {overlap}/{N} of the exact top 20; {time and time[0]}")

            print("\n## 4. Sixth execution of the prepared statement\n")
            # EXECUTE takes no bind parameters, so the values are written into the statement.
            literal = [str(v.tolist()) if isinstance(v, np.ndarray) else v for v in plain]
            cursor = AsyncClientCursor(connection)
            await cursor.execute(f"PREPARE search AS {numbered(plain_sql)}")
            execute = "EXECUTE search(" + ", ".join(["%s"] * len(literal)) + ")"
            for _ in range(5):
                await cursor.execute(execute, literal)
            await cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) {execute}", literal)
            print(plan_text(await cursor.fetchall()))
            await cursor.execute("DEALLOCATE search")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
